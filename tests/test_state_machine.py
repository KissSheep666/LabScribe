from __future__ import annotations

from labscribe.events.state_machine import RuleStateMachine

from _helpers import make_obs


def test_session_start_requires_consecutive_frames(cfg):
    machine = RuleStateMachine(cfg)
    assert machine.process(make_obs(0.0)).events == []
    assert machine.process(make_obs(0.1)).events == []
    output = machine.process(make_obs(0.2))
    assert [e.type for e in output.events] == ["session_start"]
    event = output.events[0]
    assert event.session_id
    assert event.action == "interacting"


def test_session_end_after_idle_threshold(cfg):
    machine = RuleStateMachine(cfg)
    for i in range(3):
        machine.process(make_obs(i * 0.1))
    assert machine.process(make_obs(0.4, action="idle")).events == []
    output = machine.process(make_obs(1.4, action="idle"))
    assert [e.type for e in output.events] == ["session_end"]
    assert output.events[0].session_id


def test_reading_change_is_emitted_once(cfg):
    machine = RuleStateMachine(cfg)
    assert machine.process(make_obs(0.0, action="idle", reading=1.0)).events == []
    output = machine.process(make_obs(0.1, action="idle", reading=2.0))
    assert [e.type for e in output.events] == ["reading_change"]
    assert output.events[0].reading == 2.0
    assert output.events[0].meta["previous_reading"] == 1.0
    assert machine.process(make_obs(0.2, action="idle", reading=2.0)).events == []


def test_anomaly_emitted_for_out_of_range_reading(cfg):
    machine = RuleStateMachine(cfg)
    machine.process(make_obs(0.0, action="idle", reading=0.0))
    output = machine.process(make_obs(0.1, action="idle", reading=15.2))
    types = [e.type for e in output.events]
    assert "reading_change" in types
    assert "anomaly" in types
    anomaly = next(e for e in output.events if e.type == "anomaly")
    assert anomaly.meta["rule"] == "reading_out_of_range"
    # The same out-of-range value is not re-alerted every frame.
    again = machine.process(make_obs(0.2, action="idle", reading=15.2))
    assert [e.type for e in again.events] == []


def test_low_confidence_reading_goes_to_pending(cfg):
    machine = RuleStateMachine(cfg)
    machine.process(make_obs(0.0, reading=1.0, reading_confidence=0.9))
    output = machine.process(make_obs(0.1, reading=2.0, reading_confidence=0.4))
    assert output.events == []
    assert [p.kind for p in output.pending] == ["reading_change"]
    assert output.pending[0].reading == 2.0


def test_low_confidence_start_goes_to_pending_without_transition(cfg):
    machine = RuleStateMachine(cfg)
    outputs = [machine.process(make_obs(i * 0.1, confidence=0.3)) for i in range(3)]
    assert outputs[0].pending == [] and outputs[1].pending == []
    assert [p.kind for p in outputs[2].pending] == ["session_start"]
    assert outputs[2].events == []
    # A later confident frame triggers the real transition using accumulated evidence.
    output = machine.process(make_obs(0.3, confidence=0.95))
    assert [e.type for e in output.events] == ["session_start"]
