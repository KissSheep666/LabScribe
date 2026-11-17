from __future__ import annotations

from labscribe.agent.llm import LLMError
from labscribe.agent.loop import LabAgent
from labscribe.schema import LabEvent


def _event(event_type: str = "session_start", **overrides) -> LabEvent:
    data = {
        "type": event_type,
        "ts": 1.4,
        "object_id": "instrument-1",
        "object_label": "instrument",
        "actor_id": "actor-1",
        "action": "interacting",
        "reading": 0.0,
        "confidence": 0.95,
        "source_frame": "demo-00014",
        "session_id": "sess-agent",
        "meta": {},
    }
    data.update(overrides)
    return LabEvent.from_dict(data)


def test_rules_agent_creates_record_and_tool_trace(cfg, db, tools):
    agent = LabAgent(cfg, db, tools, mode="rules")
    event = _event()
    db.save_event(event)
    result = agent.handle_event(event)
    assert result.status == "completed"
    record = db.record_for_event(event.event_id)
    assert record is not None
    assert record["source_frame"] == event.source_frame
    assert record["step_id"] == "S1"
    run = db.get_run(result.run_id)
    tool_names = [s["tool"] for s in run["steps"] if s["kind"] == "tool"]
    assert "validate_step_order" in tool_names
    assert "write_experiment_record" in tool_names
    assert result.tool_calls >= 3


def test_rules_agent_alerts_on_anomaly(cfg, db, tools):
    agent = LabAgent(cfg, db, tools, mode="rules")
    event = _event(
        event_type="anomaly",
        reading=15.2,
        meta={"limits": {"min": 0.0, "max": 14.0}},
    )
    db.save_event(event)
    result = agent.handle_event(event)
    assert result.status == "completed"
    alerts = db.alerts_for_event(event.event_id)
    assert alerts and alerts[0]["severity"] == "critical"
    assert db.record_for_event(event.event_id) is not None


def test_agent_safety_net_writes_missing_record(cfg, db, tools):
    class NoopLLM:
        def chat(self, *args, **kwargs):
            raise LLMError("not used")

    cfg.set("agent.fallback_to_rules", False)
    agent = LabAgent(cfg, db, tools, mode="llm", llm=NoopLLM())
    event = _event(event_type="session_end", reading=3.5)
    db.save_event(event)
    result = agent.handle_event(event)
    assert result.status == "error"
    # The safety net still guarantees traceability for every event.
    assert db.record_for_event(event.event_id) is not None
    assert "自动补录" in db.record_for_event(event.event_id)["note"]


def test_llm_failure_falls_back_to_rules(cfg, db, tools):
    class BrokenLLM:
        def chat(self, *args, **kwargs):
            raise LLMError("simulated outage")

    agent = LabAgent(cfg, db, tools, mode="llm", llm=BrokenLLM())
    event = _event()
    db.save_event(event)
    result = agent.handle_event(event)
    assert result.status == "fallback_rules"
    assert result.error and "simulated outage" in result.error
    record = db.record_for_event(event.event_id)
    assert record is not None
    assert "回退" in result.summary


def test_rule_agent_advances_sop_on_valid_reading(cfg, db, tools):
    agent = LabAgent(cfg, db, tools, mode="rules")
    # Complete S1 and S2 by recording a session start and end first.
    start = _event()
    db.save_event(start)
    agent.handle_event(start)
    end = _event(event_type="session_end", ts=6.0, reading=0.0)
    db.save_event(end)
    agent.handle_event(end)
    reading = _event(event_type="reading_change", ts=8.0, reading=1.0)
    reading.meta = {"previous_reading": 0.0}
    db.save_event(reading)
    agent.handle_event(reading)
    completed = db.completed_steps("exp-titration", "progress:exp-titration")
    assert completed[:2] == ["S1", "S2"]


def test_planner_advice_matches_sop_flow(cfg, db, tools):
    from labscribe.agent.tools import ToolContext

    agent = LabAgent(cfg, db, tools, mode="rules")
    reading = _event(event_type="reading_change", reading=1.0)
    db.save_event(reading)
    advice = agent._planner_advice(reading, ToolContext(event=reading))
    actions = [a["tool"] for a in advice["recommended_actions"]]
    assert "validate_step_order" in actions
    assert "validate_reading" in actions
    assert "write_experiment_record" in actions
    assert "advance_sop_step" in actions

    anomaly = _event(
        event_type="anomaly", reading=15.2, meta={"limits": {"min": 0.0, "max": 14.0}}
    )
    db.save_event(anomaly)
    advice = agent._planner_advice(anomaly, ToolContext(event=anomaly))
    actions = [a["tool"] for a in advice["recommended_actions"]]
    assert "raise_alert" in actions
    assert "write_experiment_record" in actions
    assert "advance_sop_step" not in actions
