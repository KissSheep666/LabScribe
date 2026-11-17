from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def new_event_id() -> str:
    return new_id("evt")


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[float]:
        return [round(float(v), 2) for v in (self.x1, self.y1, self.x2, self.y2)]

    def expand(self, px: float) -> "BBox":
        return BBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)

    def intersection(self, other: "BBox") -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def coverage(self, other: "BBox") -> float:
        """Fraction of ``other`` covered by this box."""
        return self.intersection(other) / other.area if other.area > 0 else 0.0

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            min(max(self.x1, 0.0), width - 1.0),
            min(max(self.y1, 0.0), height - 1.0),
            min(max(self.x2, 1.0), float(width)),
            min(max(self.y2, 1.0), float(height)),
        )

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(x, y, x + w, y + h)

    @staticmethod
    def from_seq(seq: Any) -> "BBox":
        if isinstance(seq, BBox):
            return seq
        if isinstance(seq, dict):
            return BBox(seq["x1"], seq["y1"], seq["x2"], seq["y2"])
        values = list(seq)
        if len(values) != 4:
            raise ValueError(f"bbox needs 4 values, got {values!r}")
        return BBox(*(float(v) for v in values))


@dataclass(slots=True)
class Detection:
    label: str                    # pipeline role: actor | instrument
    confidence: float
    bbox: BBox
    track_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Frame:
    frame_id: str
    index: int
    ts: float                     # seconds (media/simulation clock)
    image: np.ndarray             # BGR uint8
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class OCRResult:
    text: str
    value: float | None
    confidence: float
    ok: bool = True

    @staticmethod
    def failed(reason: str = "") -> "OCRResult":
        return OCRResult(text="", value=None, confidence=0.0, ok=False)


@dataclass(slots=True)
class Observation:
    """One instrument-centric observation aggregated from a single frame."""

    frame_id: str
    frame_index: int
    ts: float
    object_id: str
    object_label: str
    bbox: BBox
    actor_id: str | None
    actor_present: bool
    action: str                   # interacting | idle
    confidence: float
    reading: float | None = None
    raw_reading: float | None = None
    reading_confidence: float | None = None
    reading_stable: bool = False
    occluded: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabEvent:
    event_id: str
    type: str                     # session_start | reading_change | session_end | anomaly
    ts: float
    object_id: str
    object_label: str
    actor_id: str | None
    action: str
    reading: float | None
    confidence: float
    source_frame: str
    session_id: str | None = None
    status: str = "confirmed"     # confirmed | pending | human_confirmed
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "ts": round(float(self.ts), 3),
            "object_id": self.object_id,
            "object_label": self.object_label,
            "actor_id": self.actor_id,
            "action": self.action,
            "reading": self.reading,
            "confidence": round(float(self.confidence), 4),
            "source_frame": self.source_frame,
            "session_id": self.session_id,
            "status": self.status,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LabEvent":
        return LabEvent(
            event_id=d.get("event_id") or new_event_id(),
            type=d["type"],
            ts=float(d.get("ts", 0.0)),
            object_id=d.get("object_id", "instrument-1"),
            object_label=d.get("object_label", "instrument"),
            actor_id=d.get("actor_id"),
            action=d.get("action", "interacting"),
            reading=d.get("reading"),
            confidence=float(d.get("confidence", 1.0)),
            source_frame=d.get("source_frame", ""),
            session_id=d.get("session_id"),
            status=d.get("status", "confirmed"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass(slots=True)
class PendingCandidate:
    """A low-confidence observation that may represent an event."""

    kind: str                     # candidate event type
    reason: str
    object_id: str
    object_label: str
    confidence: float
    ts: float
    source_frame: str
    actor_id: str | None = None
    action: str = "interacting"
    reading: float | None = None
    session_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def draft_event(self) -> LabEvent:
        status = "human_confirmed"
        session_id = self.session_id
        if self.kind == "session_start" and not session_id:
            session_id = new_id("sess")
        return LabEvent(
            event_id=new_event_id(),
            type=self.kind,
            ts=self.ts,
            object_id=self.object_id,
            object_label=self.object_label,
            actor_id=self.actor_id,
            action=self.action,
            reading=self.reading,
            confidence=self.confidence,
            source_frame=self.source_frame,
            session_id=session_id,
            status=status,
            meta={**self.meta, "confirmed_from_pending": True},
        )


@dataclass(slots=True)
class ExperimentRecord:
    record_id: str
    experiment_id: str
    session_id: str
    step_id: str | None
    event_id: str
    source_frame: str
    object_id: str
    reading: float | None
    note: str
    created_at: float
    agent_run_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "event_id": self.event_id,
            "source_frame": self.source_frame,
            "object_id": self.object_id,
            "reading": self.reading,
            "note": self.note,
            "created_at": self.created_at,
            "agent_run_id": self.agent_run_id,
            "meta": self.meta,
        }


@dataclass(slots=True)
class SOPStep:
    id: str
    name: str
    instruction: str
    instrument: str | None = None
    expected_reading: dict[str, Any] | None = None
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "instruction": self.instruction,
            "instrument": self.instrument,
            "expected_reading": self.expected_reading,
            "duration_s": self.duration_s,
        }


@dataclass(slots=True)
class SOP:
    experiment_id: str
    name: str
    version: str
    steps: list[SOPStep]

    def get_step(self, step_id: str | None) -> SOPStep | None:
        if not step_id:
            return None
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def index_of(self, step_id: str) -> int:
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        return -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
        }


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BBox):
        return value.as_list()
    return str(value)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
