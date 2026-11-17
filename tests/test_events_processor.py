from __future__ import annotations

from labscribe.events.processor import EventProcessor
from labscribe.events.state_machine import RuleStateMachine
from labscribe.schema import LabEvent, PendingCandidate, new_event_id
from labscribe.store.frames import FrameStore
from labscribe.vision.synthetic import SyntheticSource, make_demo_scenario

from _helpers import make_obs


def _processor(cfg, db):
    return EventProcessor(cfg, RuleStateMachine(cfg), db, FrameStore(db, cfg.frames_dir))


def test_processor_persists_event_and_source_frame(cfg, db):
    processor = _processor(cfg, db)
    spec = make_demo_scenario()
    frame = SyntheticSource(spec).render(14)
    processor.frame_store.keep(frame)
    for i in range(3):
        observation = make_obs(i * 0.1, frame_index=frame.index, reading=0.0)
        observation.frame_id = frame.frame_id
        processor.handle_observations([observation])
    events = db.list_events(10)
    assert [e["type"] for e in events] == ["session_start"]
    event = events[0]
    assert event["source_frame"] == frame.frame_id
    row = db.get_frame(frame.frame_id)
    assert row is not None
    from pathlib import Path

    assert Path(row["path"]).exists()
    trace = processor.frame_store.path_for(frame.frame_id)
    assert trace is not None


def test_low_confidence_pending_can_be_confirmed(cfg, db):
    processor = _processor(cfg, db)
    candidate = PendingCandidate(
        kind="reading_change",
        reason="reading confidence 0.40 < 0.55",
        object_id="instrument-1",
        object_label="instrument",
        confidence=0.4,
        ts=4.0,
        source_frame="demo-00040",
        actor_id="actor-1",
        action="interacting",
        reading=2.0,
        session_id="sess-test",
        meta={"previous_reading": 1.0},
    )
    item = db.add_pending(candidate)
    assert item["status"] == "open"
    assert item["draft_event"]["type"] == "reading_change"
    assert len(db.list_pending()) == 1

    result = processor.confirm_pending(item["pending_id"])
    assert result is not None
    assert result["status"] == "confirmed"
    event = db.get_event(result["event"]["event_id"])
    assert event is not None
    assert event["status"] == "human_confirmed"
    assert event["reading"] == 2.0
    assert db.list_pending() == []


def test_pending_items_coalesce(cfg, db):
    processor = _processor(cfg, db)
    for i in range(4):
        candidate = PendingCandidate(
            kind="session_start",
            reason="low confidence",
            object_id="instrument-1",
            object_label="instrument",
            confidence=0.3 + i * 0.02,
            ts=i * 0.1,
            source_frame=f"frame-{i}",
        )
        db.add_pending(candidate, coalesce_s=2.0)
    items = db.list_pending()
    assert len(items) == 1
    assert items[0]["confidence"] == 0.36


def test_unknown_event_type_is_rejected(cfg, db):
    processor = _processor(cfg, db)
    import pytest

    with pytest.raises(ValueError):
        processor.ingest_event({"type": "made_up"})


def test_manual_ingest_records_event(cfg, db):
    processor = _processor(cfg, db)
    event = processor.ingest_event(
        {
            "type": "session_start",
            "ts": 1.0,
            "object_id": "instrument-1",
            "reading": 0.0,
            "confidence": 0.9,
            "source_frame": "manual-1",
            "session_id": "sess-manual",
        }
    )
    assert isinstance(event, LabEvent)
    assert db.get_event(event.event_id)["meta"]["source"] == "manual_ingest"
