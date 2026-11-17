from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..config import Config
from ..schema import BBox, Detection

log = logging.getLogger(__name__)


class Tracker(ABC):
    name = "base"

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Detection]:
        raise NotImplementedError


class IoUTracker(Tracker):
    """Small greedy IoU tracker used when ByteTrack is unavailable."""

    name = "iou"

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 12):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[str, dict] = {}
        self._counter = 0

    def update(self, detections: list[Detection]) -> list[Detection]:
        for track in self._tracks.values():
            track["age"] += 1
        assigned: set[str] = set()
        for det in detections:
            best_id = None
            best_iou = self.iou_threshold
            for track_id, track in self._tracks.items():
                if track_id in assigned or track["label"] != det.label:
                    continue
                iou = det.bbox.iou(track["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_id is None:
                self._counter += 1
                best_id = f"{det.label}-{self._counter}"
                self._tracks[best_id] = {
                    "label": det.label,
                    "bbox": det.bbox,
                    "age": 0,
                }
            else:
                self._tracks[best_id]["bbox"] = det.bbox
                self._tracks[best_id]["age"] = 0
            det.track_id = best_id
            assigned.add(best_id)
        for track_id in [tid for tid, track in self._tracks.items() if track["age"] > self.max_age]:
            del self._tracks[track_id]
        return detections


class ByteTrackTracker(Tracker):
    """Adapter around ``supervision.ByteTrack`` (same algorithm as ByteTrack)."""

    name = "bytetrack"

    def __init__(self, cfg: Config):
        import supervision as sv  # lazy heavy import

        self.sv = sv
        self.tracker = sv.ByteTrack()
        self.display_ratio = list(cfg.get("vision.display_roi", [0.20, 0.52, 0.80, 0.86]))

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            self.tracker.update_with_detections(self.sv.Detections.empty())
            return []
        import numpy as np

        roles = sorted({d.label for d in detections})
        role_to_id = {role: i for i, role in enumerate(roles)}
        sv_dets = self.sv.Detections(
            xyxy=np.array([d.bbox.as_list() for d in detections], dtype=np.float32),
            confidence=np.array([d.confidence for d in detections], dtype=np.float32),
            class_id=np.array([role_to_id[d.label] for d in detections], dtype=int),
        )
        tracked = self.tracker.update_with_detections(sv_dets)
        out: list[Detection] = []
        if tracked is None or len(tracked) == 0:
            return out
        for i in range(len(tracked)):
            xyxy = tracked.xyxy[i].tolist()
            role = roles[int(tracked.class_id[i])] if tracked.class_id is not None else "instrument"
            track_id = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            bbox = BBox(*xyxy)
            meta = {}
            if role == "instrument":
                rx1, ry1, rx2, ry2 = (float(v) for v in self.display_ratio)
                meta["display_roi"] = [
                    bbox.x1 + rx1 * bbox.width,
                    bbox.y1 + ry1 * bbox.height,
                    bbox.x1 + rx2 * bbox.width,
                    bbox.y1 + ry2 * bbox.height,
                ]
            out.append(
                Detection(
                    label=role,
                    confidence=float(tracked.confidence[i]),
                    bbox=bbox,
                    track_id=f"{role}-{track_id}" if track_id is not None else None,
                    meta=meta,
                )
            )
        return out


def build_tracker(cfg: Config) -> Tracker:
    mode = str(cfg.get("vision.tracker", "auto")).lower()
    if mode == "iou":
        return IoUTracker()
    if mode in ("bytetrack", "auto"):
        try:
            tracker = ByteTrackTracker(cfg)
            if mode == "auto":
                log.warning("vision.tracker=auto: using ByteTrack backend")
            return tracker
        except Exception as exc:  # noqa: BLE001
            if mode == "bytetrack":
                raise
            log.warning("ByteTrack unavailable (%s); falling back to IoU tracker", exc)
    return IoUTracker()
