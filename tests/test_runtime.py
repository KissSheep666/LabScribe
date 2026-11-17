from __future__ import annotations

from pathlib import Path

from labscribe.runtime import Runtime
from labscribe.vision.synthetic import make_demo_scenario


def test_runtime_demo_rules_end_to_end(cfg):
    runtime = Runtime(cfg, agent_mode="rules")
    runtime.start(scenario=make_demo_scenario(fps=10), agent_mode="rules")
    assert runtime.wait_idle(120), "runtime did not become idle"
    assert runtime.last_error is None

    events = runtime.db.list_events(200)
    types = [e["type"] for e in events]
    assert types.count("session_start") == 2
    assert types.count("session_end") == 2
    assert types.count("reading_change") == 3
    assert types.count("anomaly") == 1
    assert len(events) == 8

    records = runtime.db.list_records(200)
    assert len(records) == 8
    for record in records:
        assert record["source_frame"]
        assert runtime.db.get_frame(record["source_frame"]) is not None
        frame_path = Path(runtime.db.get_frame(record["source_frame"])["path"])
        assert frame_path.exists()

    alerts = runtime.db.list_alerts(20)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"
    assert "15.2" in alerts[0]["message"]

    completed = runtime.db.completed_steps("exp-titration", "progress:exp-titration")
    assert completed == ["S1", "S2", "S3"]

    runs = runtime.db.list_runs(100)
    assert len(runs) == 8
    assert all(r["status"] == "completed" for r in runs)

    # Every record can be resolved back to its event and frame.
    for record in records[:3]:
        event = runtime.db.get_event(record["event_id"])
        assert event is not None
        assert event["source_frame"] == record["source_frame"]
        assert runtime.frame_store.path_for(record["source_frame"])

    assert runtime.frames_processed == 140
    runtime.close()


def test_runtime_can_rerun_a_scenario_without_state_bleed(cfg):
    runtime = Runtime(cfg, agent_mode="rules")
    runtime.start(scenario=make_demo_scenario(fps=10), agent_mode="rules")
    assert runtime.wait_idle(120)
    first_count = len(runtime.db.list_events(200))
    runtime.start(scenario=make_demo_scenario(fps=10), agent_mode="rules")
    assert runtime.wait_idle(120)
    second_count = len(runtime.db.list_events(200))
    assert second_count == first_count * 2
    # A fresh rule machine starts new sessions instead of continuing old ones.
    starts = [e for e in runtime.db.list_events(200) if e["type"] == "session_start"]
    assert len({e["session_id"] for e in starts}) == 4
    runtime.close()
