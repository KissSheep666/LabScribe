from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..schema import BBox, Detection, Frame, Observation
from .detector import Detector, build_detector
from .ocr import OCR, build_ocr
from .stability import ReadingStabilizer
from .tracker import Tracker, build_tracker
from .utils import relative_bbox

log = logging.getLogger(__name__)


@dataclass
class _ObjectState:
    stabilizer: ReadingStabilizer
    last_ocr_frame: int = -10**9
    last_ocr_ts: float = -1e9
    last_reading_conf: float = 0.0
    raw_reading: float | None = None
    stable_now: bool = False
    stable_value: float | None = None


class VisionPipeline:
    """detections -> stable track IDs -> throttled OCR -> observations."""

    def __init__(
        self,
        cfg: Config,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
        ocr: OCR | None = None,
    ):
        self.cfg = cfg
        self.detector = detector or build_detector(cfg)
        self.tracker = tracker or build_tracker(cfg)
        self.ocr = ocr or build_ocr(cfg)
        reading_cfg = cfg.section("vision.reading")
        self.stable_samples = int(reading_cfg.get("stable_samples", 3))
        self.tolerance_abs = float(reading_cfg.get("tolerance_abs", 0.25))
        self.tolerance_rel = float(reading_cfg.get("tolerance_rel", 0.01))
        self.max_gap_s = float(reading_cfg.get("max_gap_s", 2.0))
        self.ocr_every_n = max(1, int(cfg.get("vision.ocr_every_n_frames", 3)))
        self.ocr_min_interval = float(cfg.get("vision.ocr_min_interval_s", 0.05))
        self.min_detection = float(cfg.get("vision.confidence.min_detection", 0.45))
        self.display_ratio = list(cfg.get("vision.display_roi", [0.20, 0.52, 0.80, 0.86]))
        self.proximity_margin = float(cfg.get("vision.proximity_margin_px", 70))
        self.occlusion_ratio = float(cfg.get("vision.display_occlusion_ratio", 0.30))
        self.object_id_map = dict(cfg.get("vision.object_id_map", {}) or {})
        self._states: dict[str, _ObjectState] = {}
        self.trace: list[dict[str, Any]] = []
        self.last_observations: list[Observation] = []

    def _state(self, object_id: str) -> _ObjectState:
        state = self._states.get(object_id)
        if state is None:
            state = _ObjectState(
                stabilizer=ReadingStabilizer(
                    stable_samples=self.stable_samples,
                    tolerance_abs=self.tolerance_abs,
                    tolerance_rel=self.tolerance_rel,
                    max_gap_s=self.max_gap_s,
                )
            )
            self._states[object_id] = state
        return state

    def process(self, frame: Frame) -> list[Observation]:
        raw = self.detector.detect(frame)
        detections = self.tracker.update([d for d in raw if d.confidence >= self.min_detection])
        actors = [d for d in detections if d.label == "actor"]
        instruments = [d for d in detections if d.label == "instrument"]
        observations: list[Observation] = []
        for inst in instruments:
            observations.append(self._observe(frame, inst, actors))
        self.last_observations = observations
        return observations

    def _observe(self, frame: Frame, inst: Detection, actors: list[Detection]) -> Observation:
        object_id = self.object_id_map.get(inst.label)
        if not object_id:
            object_id = f"{inst.label}-{inst.track_id or '0'}"
        display_roi = self._display_roi(inst)
        state = self._state(object_id)

        closest_actor = self._closest_actor(inst, actors)
        actor_present = closest_actor is not None

        occluded = False
        if closest_actor is not None:
            occluded = closest_actor.bbox.coverage(display_roi) >= self.occlusion_ratio

        due_ocr = (
            not occluded
            and frame.index - state.last_ocr_frame >= self.ocr_every_n
            and frame.ts - state.last_ocr_ts >= self.ocr_min_interval
        )
        if due_ocr:
            result = self.ocr.read(frame.image, display_roi)
            state.last_ocr_frame = frame.index
            state.last_ocr_ts = frame.ts
            state.raw_reading = result.value if result.ok else None
            state.last_reading_conf = result.confidence if result.ok else 0.0
            self.trace.append(
                {
                    "frame_index": frame.index,
                    "frame_id": frame.frame_id,
                    "ts": frame.ts,
                    "text": result.text,
                    "value": result.value,
                    "confidence": result.confidence,
                    "ok": result.ok,
                }
            )
            stable = state.stabilizer.update(
                frame.ts,
                result.value if (result.ok and result.value is not None) else None,
                result.confidence if result.ok else 0.0,
            )
            state.stable_now = stable.stable
            state.stable_value = stable.value if stable.stable else None
        elif occluded:
            stable = state.stabilizer.update(frame.ts, None, 0.0)
            state.stable_now = False
            state.stable_value = None

        reading = state.stable_value if state.stable_now else None
        reading_confidence = state.last_reading_conf if reading is not None else None
        return Observation(
            frame_id=frame.frame_id,
            frame_index=frame.index,
            ts=frame.ts,
            object_id=object_id,
            object_label=inst.label,
            bbox=inst.bbox,
            actor_id=closest_actor.track_id if closest_actor else None,
            actor_present=actor_present,
            action="interacting" if actor_present else "idle",
            confidence=float(inst.confidence),
            reading=reading,
            raw_reading=state.raw_reading,
            reading_confidence=reading_confidence,
            reading_stable=bool(state.stable_now),
            occluded=occluded,
            meta={
                "display_roi": display_roi.as_list(),
                "detector": self.detector.name,
                "tracker": self.tracker.name,
                "ocr": self.ocr.name,
                "track_id": inst.track_id,
                "raw_reading": state.raw_reading,
            },
        )

    def _display_roi(self, inst: Detection) -> BBox:
        roi = inst.meta.get("display_roi")
        if roi:
            return BBox.from_seq(roi)
        return relative_bbox(inst.bbox, self.display_ratio)

    def _closest_actor(self, inst: Detection, actors: list[Detection]) -> Detection | None:
        if not actors:
            return None
        zone = inst.bbox.expand(self.proximity_margin)
        best: Detection | None = None
        best_score = 0.0
        for actor in actors:
            overlap = actor.bbox.intersection(zone)
            if overlap <= 0:
                continue
            score = overlap / max(1.0, inst.bbox.area)
            if score > best_score:
                best_score = score
                best = actor
        return best

    def close(self) -> None:
        self.detector.close()
        self.ocr.close()
