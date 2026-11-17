"""Deterministic synthetic lab scenes.

The renderer draws an instrument panel with a digital display, a person/hand
entering the interaction zone, digital readings that step over time, and
optional occlusion / detector-dropout windows.  It is intentionally rendered
with the same colors and font used by the offline ``marker`` detector and
``template`` OCR, so the whole vision pipeline can be exercised without a
camera or GPU.  Production deployments swap the detector / tracker / OCR
backends behind the same interfaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..schema import BBox, Frame
from .fonts import get_font
from .utils import bbox_to_int, relative_bbox

# BGR colors shared with the offline detector.
COLOR_INSTRUMENT = (200, 90, 30)
COLOR_INSTRUMENT_DARK = (150, 65, 20)
COLOR_INSTRUMENT_LIGHT = (225, 125, 60)
COLOR_ACTOR = (40, 140, 240)
COLOR_SKIN = (70, 175, 250)
COLOR_BACKGROUND = (38, 34, 31)
COLOR_BENCH = (52, 47, 43)
COLOR_DISPLAY = (8, 8, 8)
COLOR_DIGIT = (245, 245, 245)

DEFAULT_DISPLAY_RATIO = (0.20, 0.52, 0.80, 0.86)


@dataclass
class ScenarioSpec:
    scenario_id: str = "demo"
    seed: int = 7
    width: int = 480
    height: int = 320
    fps: int = 10
    duration: float = 12.0
    object_id: str = "instrument-1"
    object_label: str = "instrument"
    unit: str = "mL"
    instrument_bbox: list[float] = field(default_factory=lambda: [250.0, 60.0, 450.0, 230.0])
    display_ratio: list[float] = field(default_factory=lambda: list(DEFAULT_DISPLAY_RATIO))
    label_text: str = "TITRATOR"
    reading_steps: list[dict[str, float]] = field(default_factory=lambda: [{"t": 0.0, "value": 0.0}])
    actor_intervals: list[dict[str, float]] = field(default_factory=list)
    occlusions: list[dict[str, float]] = field(default_factory=list)
    detector_dropouts: list[int] = field(default_factory=list)
    ocr_dropouts: list[int] = field(default_factory=list)
    noise: float = 5.0
    blur: float = 0.0
    ground_truth: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return max(1, int(round(self.duration * self.fps)))

    @property
    def display_bbox(self) -> BBox:
        return relative_bbox(BBox.from_seq(self.instrument_bbox), self.display_ratio)

    def reading_at(self, t: float) -> float:
        value = float(self.reading_steps[0]["value"])
        for step in self.reading_steps:
            if float(step["t"]) <= t + 1e-9:
                value = float(step["value"])
            else:
                break
        return value

    def actor_active(self, t: float) -> bool:
        return any(float(iv["t_start"]) <= t < float(iv["t_end"]) for iv in self.actor_intervals)

    def occluded_at(self, t: float) -> bool:
        return any(float(o["t_start"]) <= t <= float(o["t_end"]) for o in self.occlusions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "object_id": self.object_id,
            "object_label": self.object_label,
            "unit": self.unit,
            "instrument_bbox": [float(v) for v in self.instrument_bbox],
            "display_ratio": [float(v) for v in self.display_ratio],
            "label_text": self.label_text,
            "reading_steps": self.reading_steps,
            "actor_intervals": self.actor_intervals,
            "occlusions": self.occlusions,
            "detector_dropouts": self.detector_dropouts,
            "ocr_dropouts": self.ocr_dropouts,
            "noise": self.noise,
            "blur": self.blur,
            "ground_truth": self.ground_truth,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ScenarioSpec":
        known = set(ScenarioSpec.__dataclass_fields__)  # type: ignore[attr-defined]
        return ScenarioSpec(**{k: v for k, v in data.items() if k in known})


def make_demo_scenario(fps: int = 10) -> ScenarioSpec:
    spec = ScenarioSpec(
        scenario_id="demo",
        seed=7,
        fps=fps,
        duration=14.0,
        label_text="TITRATOR",
        instrument_bbox=[250.0, 58.0, 452.0, 232.0],
        reading_steps=[
            {"t": 0.0, "value": 0.0},
            {"t": 4.0, "value": 3.5},
            {"t": 8.4, "value": 7.2},
            {"t": 10.9, "value": 15.2},
        ],
        actor_intervals=[
            {"t_start": 1.2, "t_end": 5.6},
            {"t_start": 7.2, "t_end": 12.2},
        ],
        occlusions=[{"t_start": 8.1, "t_end": 9.1}],
        noise=5.0,
        blur=0.5,
        notes={"purpose": "demo", "experiment_id": "exp-titration"},
    )
    spec.ground_truth = _ground_truth(spec)
    return spec


def make_scenario(
    seed: int,
    scenario_id: str,
    occluded: bool = False,
    fps: int = 10,
    duration: float | None = None,
) -> ScenarioSpec:
    """Build a deterministic, randomized-but-plausible scenario."""
    rng = random.Random(seed)
    width, height = 480, 320
    iw = rng.randint(168, 208)
    ih = rng.randint(140, 166)
    ix = rng.randint(212, 272)
    iy = rng.randint(38, 62)
    ix = min(ix, width - iw - 6)
    iy = min(iy, height - ih - 6)
    instrument = [float(ix), float(iy), float(ix + iw), float(iy + ih)]

    initial = round(rng.choice([0.0, 0.0, 0.0, 0.5, 1.0, 2.0]), 1)
    dur = float(duration) if duration else round(rng.uniform(9.0, 11.5), 2)

    t_start = round(rng.uniform(0.8, 1.8), 2)
    length = round(rng.uniform(3.6, 4.7), 2)
    intervals = [{"t_start": t_start, "t_end": round(t_start + length, 2)}]
    if rng.random() < 0.45:
        gap = round(rng.uniform(2.7, 3.6), 2)
        t1 = round(intervals[-1]["t_end"] + gap, 2)
        length2 = round(rng.uniform(2.5, 3.8), 2)
        intervals.append({"t_start": t1, "t_end": round(t1 + length2, 2)})

    # Put reading changes inside interaction windows.
    windows = [
        (iv["t_start"] + 0.9, iv["t_end"] - 0.6)
        for iv in intervals
        if iv["t_end"] - iv["t_start"] > 2.2
    ]
    change_times: list[float] = []
    for lo, hi in windows:
        if hi > lo and rng.random() < 0.9:
            change_times.append(round(rng.uniform(lo, hi), 2))
    if not change_times:
        change_times = [round(intervals[0]["t_start"] + 1.4, 2)]
    change_times = sorted(set(change_times))[:2]

    steps: list[dict[str, float]] = [{"t": 0.0, "value": initial}]
    value = initial
    for t in change_times:
        value = min(13.5, round(value + rng.choice([1.5, 2.0, 2.5, 3.0, 3.5]), 1))
        steps.append({"t": float(t), "value": float(value)})

    # Roughly one scenario in seven ends with an out-of-range reading so the
    # anomaly path is exercised and scored like any other event type.
    inject_anomaly = seed % 7 == 0
    if inject_anomaly:
        steps[-1]["value"] = 15.2

    last_end = max(iv["t_end"] for iv in intervals)
    dur = max(dur, last_end + 3.0, steps[-1]["t"] + 2.0)

    occlusions: list[dict[str, float]] = []
    if occluded and steps[1:]:
        target = steps[-1]["t"]
        # Keep the occlusion around an actual display change.
        occlusions.append({"t_start": round(target - 0.7, 2), "t_end": round(target + 0.9, 2)})

    dropouts: list[int] = []
    for _ in range(rng.randint(1, 2)):
        start = rng.randint(8, max(9, int(dur * fps) - 10))
        length_frames = rng.randint(2, 4)
        dropouts.extend(range(start, start + length_frames))

    ocr_dropouts: list[int] = []
    for _ in range(rng.randint(0, 2)):
        start = rng.randint(6, max(7, int(dur * fps) - 6))
        ocr_dropouts.extend(range(start, start + rng.randint(2, 3)))

    spec = ScenarioSpec(
        scenario_id=scenario_id,
        seed=seed,
        fps=fps,
        duration=round(dur, 2),
        instrument_bbox=instrument,
        label_text=rng.choice(["TITRATOR", "ANALYZER", "DISPENSER"]),
        reading_steps=steps,
        actor_intervals=intervals,
        occlusions=occlusions,
        detector_dropouts=sorted(set(dropouts)),
        ocr_dropouts=sorted(set(ocr_dropouts)),
        noise=round(rng.uniform(3.0, 9.0), 1),
        blur=rng.choice([0.0, 0.0, 0.4, 0.8]),
        notes={"purpose": "eval", "occluded": occluded, "inject_anomaly": inject_anomaly},
    )
    spec.ground_truth = _ground_truth(spec, min_confidence=0.55)
    return spec


def _ground_truth(spec: ScenarioSpec, min_confidence: float = 0.55) -> list[dict[str, Any]]:
    idle_s = 1.2
    events: list[dict[str, Any]] = []
    for iv in spec.actor_intervals:
        start = float(iv["t_start"])
        end = float(iv["t_end"])
        events.append(
            {
                "type": "session_start",
                "t": round(start, 2),
                "window": [round(start - 0.2, 2), round(start + 1.2, 2)],
                "reading": spec.reading_at(start),
                "occluded": False,
            }
        )
        events.append(
            {
                "type": "session_end",
                "t": round(end + idle_s, 2),
                "window": [round(end + 0.4, 2), round(end + 2.5, 2)],
                "reading": spec.reading_at(end),
                "occluded": False,
            }
        )
    for step in spec.reading_steps[1:]:
        t = float(step["t"])
        covering = [
            o
            for o in spec.occlusions
            if float(o["t_start"]) - 0.2 <= t <= float(o["t_end"]) + 0.2
        ]
        occluded = bool(covering)
        # An occluded change cannot be observed until the hand clears and the
        # reading stabilizes again, so the annotation window extends past it.
        window_end = t + 1.8
        if covering:
            window_end = max(window_end, max(float(o["t_end"]) for o in covering) + 1.5)
        events.append(
            {
                "type": "reading_change",
                "t": round(t, 2),
                "window": [round(t - 0.2, 2), round(window_end, 2)],
                "reading": float(step["value"]),
                "occluded": occluded,
            }
        )
        if float(step["value"]) > 14.0:
            events.append(
                {
                    "type": "anomaly",
                    "t": round(t, 2),
                    "window": [round(t - 0.2, 2), round(window_end, 2)],
                    "reading": float(step["value"]),
                    "occluded": occluded,
                }
            )
    events.sort(key=lambda e: (e["t"], e["type"]))
    return events


class SyntheticSource:
    """Frame iterator for a synthetic scenario."""

    def __init__(self, spec: ScenarioSpec):
        self.spec = spec

    def __len__(self) -> int:
        return self.spec.frame_count

    def iter_frames(self):
        for index in range(self.spec.frame_count):
            yield self.render(index)

    def render(self, index: int) -> Frame:
        spec = self.spec
        t = index / float(spec.fps)
        img = self._background(index)
        inst = BBox.from_seq(spec.instrument_bbox)
        self._draw_instrument(img, inst)
        disp = spec.display_bbox
        self._draw_display(img, disp, index, t)
        if spec.actor_active(t):
            self._draw_actor(img, inst, disp, t)
        img = self._degrade(img, index)
        return Frame(
            frame_id=f"{spec.scenario_id}-{index:05d}",
            index=index,
            ts=round(t, 4),
            image=img,
            meta={"scenario_id": spec.scenario_id, "reading": spec.reading_at(t)},
        )

    # -- rendering pieces -------------------------------------------------
    def _background(self, index: int) -> np.ndarray:
        spec = self.spec
        img = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
        img[:] = COLOR_BACKGROUND
        bench_y = int(spec.height * 0.78)
        cv2.rectangle(img, (0, bench_y), (spec.width, spec.height), COLOR_BENCH, -1)
        cv2.line(img, (0, bench_y), (spec.width, bench_y), (72, 66, 60), 2)
        rng = np.random.default_rng(spec.seed * 7919 + index)
        glow = rng.integers(0, 5, size=(spec.height, spec.width, 1), dtype=np.uint8)
        img = cv2.add(img, np.repeat(glow, 3, axis=2))
        return img

    def _draw_instrument(self, img: np.ndarray, inst: BBox) -> None:
        x1, y1, x2, y2 = bbox_to_int(inst, self.spec.width, self.spec.height)
        cv2.rectangle(img, (x1 + 6, y1 + 8), (x2 + 6, y2 + 8), (20, 18, 16), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_INSTRUMENT, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_INSTRUMENT_DARK, 3)
        cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), COLOR_INSTRUMENT_LIGHT, 1)
        cv2.putText(
            img,
            self.spec.label_text,
            (x1 + 16, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (248, 248, 248),
            1,
            cv2.LINE_AA,
        )
        # Status LEDs drawn as gray so they never collide with role colors.
        cv2.circle(img, (x2 - 18, y1 + 18), 5, (120, 120, 120), -1)
        cv2.circle(img, (x2 - 36, y1 + 18), 5, (95, 95, 95), -1)

    def _draw_display(self, img: np.ndarray, disp: BBox, index: int, t: float) -> None:
        x1, y1, x2, y2 = bbox_to_int(disp, self.spec.width, self.spec.height)
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (18, 18, 18), -1)
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (70, 70, 70), 1)
        dw, dh = x2 - x1, y2 - y1
        if not dw or not dh:
            return
        panel = Image.new("RGB", (dw, dh), (8, 8, 8))
        if index not in set(self.spec.ocr_dropouts):
            text = f"{self.spec.reading_at(t):0.1f}"
            font = get_font(int(dh * 0.62))
            draw = ImageDraw.Draw(panel)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pos = ((dw - tw) / 2.0 - bbox[0], (dh - th) / 2.0 - bbox[1])
            draw.text(pos, text, font=font, fill=COLOR_DIGIT)
        img[y1:y2, x1:x2] = cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)

    def _draw_actor(self, img: np.ndarray, inst: BBox, disp: BBox, t: float) -> None:
        spec = self.spec
        x1 = inst.x2 - 0.30 * inst.width
        x2 = min(inst.x2 + 0.52 * inst.width, spec.width - 6)
        y1 = inst.y1 + 0.12 * inst.height
        y2 = min(inst.y2 + 0.24 * inst.height, spec.height - 6)
        # Body + head, both warm colors picked up as the "actor" role.
        cv2.rectangle(img, (int(x1), int(y1 + 18)), (int(x2), int(y2)), COLOR_ACTOR, -1)
        head_center = (int((x1 + x2) / 2), int(y1 + 8))
        cv2.circle(img, head_center, 14, COLOR_SKIN, -1)
        cv2.rectangle(
            img,
            (int(x1), int(y1 + 18)),
            (int(x2), int(y2)),
            (25, 100, 190),
            2,
        )
        # Hand near the display; during an occlusion window it covers it.
        occluding = spec.occluded_at(t)
        if occluding:
            hx1 = disp.x1 + 0.30 * disp.width
            hy1 = disp.y1 - 0.05 * disp.height
            hx2 = disp.x2 + 0.06 * disp.width
            hy2 = disp.y2 + 0.10 * disp.height
        else:
            hx1 = disp.x2 - 0.10 * disp.width
            hy1 = disp.y1 + 0.22 * disp.height
            hx2 = disp.x2 + 0.14 * disp.width
            hy2 = disp.y2 + 0.04 * disp.height
        cv2.rectangle(img, (int(hx1), int(hy1)), (int(hx2), int(hy2)), COLOR_SKIN, -1)
        cv2.rectangle(img, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (60, 150, 215), 2)

    def _degrade(self, img: np.ndarray, index: int) -> np.ndarray:
        spec = self.spec
        if spec.blur > 0:
            img = cv2.GaussianBlur(img, (3, 3), spec.blur)
        if spec.noise > 0:
            rng = np.random.default_rng(spec.seed * 104729 + index)
            noise = rng.normal(0.0, spec.noise, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img
