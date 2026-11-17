from __future__ import annotations

from labscribe.schema import BBox, Observation


def make_obs(
    ts: float,
    action: str = "interacting",
    reading: float | None = None,
    confidence: float = 0.9,
    reading_confidence: float | None = None,
    stable: bool = True,
    frame_index: int = 0,
    object_id: str = "instrument-1",
) -> Observation:
    return Observation(
        frame_id=f"frame-{frame_index:04d}",
        frame_index=frame_index,
        ts=ts,
        object_id=object_id,
        object_label="instrument",
        bbox=BBox(100, 50, 220, 160),
        actor_id="actor-1" if action == "interacting" else None,
        actor_present=action == "interacting",
        action=action,
        confidence=confidence,
        reading=reading,
        raw_reading=reading,
        reading_confidence=(
            reading_confidence
            if reading_confidence is not None
            else (confidence if reading is not None else None)
        ),
        reading_stable=stable,
        occluded=False,
        meta={},
    )
