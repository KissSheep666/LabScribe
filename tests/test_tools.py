from __future__ import annotations

import pytest

from labscribe.agent.tools import ToolContext
from labscribe.schema import LabEvent, new_event_id


def test_step_order_validation_and_progress(tools):
    ctx = ToolContext()
    info = tools.call("get_sop_step", {"experiment_id": "exp-titration", "step_id": "next"}, ctx)
    assert info["ok"] is True
    assert info["step"]["id"] == "S1"
    assert info["completed_steps"] == []

    out_of_order = tools.call(
        "validate_step_order", {"experiment_id": "exp-titration", "step_id": "S2"}, ctx
    )
    assert out_of_order["ok"] is True
    assert out_of_order["allowed"] is False
    assert out_of_order["expected_step_id"] == "S1"

    refused = tools.call(
        "advance_sop_step", {"experiment_id": "exp-titration", "step_id": "S2"}, ctx
    )
    assert refused["advanced"] is False

    advanced = tools.call(
        "advance_sop_step", {"experiment_id": "exp-titration", "step_id": "S1"}, ctx
    )
    assert advanced["advanced"] is True
    assert advanced["next_step_id"] == "S2"

    repeated = tools.call(
        "advance_sop_step", {"experiment_id": "exp-titration", "step_id": "S1"}, ctx
    )
    assert repeated["advanced"] is False

    info = tools.call("get_sop_step", {"experiment_id": "exp-titration", "step_id": "next"}, ctx)
    assert info["step"]["id"] == "S2"
    assert info["completed_steps"] == ["S1"]


def test_reading_range_validation(tools):
    within = tools.call(
        "validate_reading",
        {"experiment_id": "exp-titration", "step_id": "S3", "reading": 7.2},
    )
    assert within["checked"] is True and within["within_range"] is True
    outside = tools.call(
        "validate_reading",
        {"experiment_id": "exp-titration", "step_id": "S4", "reading": 15.2},
    )
    assert outside["checked"] is True and outside["within_range"] is False
    no_range = tools.call(
        "validate_reading",
        {"experiment_id": "exp-titration", "step_id": "S1", "reading": 1.0},
    )
    assert no_range["checked"] is False


def test_record_is_traceable_and_idempotent(cfg, db, tools):
    event = LabEvent.from_dict(
        {
            "type": "reading_change",
            "ts": 4.8,
            "object_id": "instrument-1",
            "object_label": "instrument",
            "actor_id": "actor-1",
            "action": "interacting",
            "reading": 3.5,
            "confidence": 0.9,
            "source_frame": "demo-00048",
            "session_id": "sess-test",
        }
    )
    db.save_event(event)
    ctx = ToolContext(event=event, run_id="run-test")
    result = tools.call(
        "write_experiment_record",
        {
            "experiment_id": "exp-titration",
            "event_id": event.event_id,
            "summary": "读数变更为 3.5 mL",
            "step_id": "S1",
        },
        ctx,
    )
    assert result["ok"] is True and result["created"] is True
    assert result["source_frame"] == "demo-00048"
    assert result["frame_url"] == "/api/frames/demo-00048"
    record = db.get_record(result["record_id"])
    assert record["event_id"] == event.event_id
    assert record["source_frame"] == event.source_frame
    assert record["agent_run_id"] == "run-test"

    duplicate = tools.call(
        "write_experiment_record",
        {
            "experiment_id": "exp-titration",
            "event_id": event.event_id,
            "summary": "重复写入",
            "step_id": "S1",
        },
        ctx,
    )
    assert duplicate["created"] is False
    assert duplicate["deduped"] is True
    assert duplicate["record_id"] == result["record_id"]


def test_unknown_event_id_is_rejected(tools):
    result = tools.call(
        "write_experiment_record",
        {"experiment_id": "exp-titration", "event_id": "evt-missing", "summary": "x"},
    )
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_unknown_tool_and_bad_arguments_are_reported(tools):
    assert tools.call("does_not_exist", {})["ok"] is False
    assert tools.call("validate_step_order", {"experiment_id": "exp-titration"})["ok"] is False
    assert tools.call("raise_alert", {"severity": "nope", "message": "x"})["ok"] is False


def test_alert_links_to_event(tools):
    event_id = new_event_id()
    result = tools.call(
        "raise_alert",
        {"severity": "critical", "message": "读数 15.2 超限", "event_id": event_id},
    )
    assert result["ok"] is True
    assert result["alert"]["event_id"] == event_id
    assert result["alert"]["severity"] == "critical"
