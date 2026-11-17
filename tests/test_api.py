from __future__ import annotations

import time

from fastapi.testclient import TestClient

from labscribe.api.app import create_app
from labscribe.schema import PendingCandidate


def _wait_done(client: TestClient, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/status").json()
        if (
            status["vision_done"]
            and not status["running"]
            and not status["agent_running"]
            and status["agent_queue"] == 0
        ):
            return status
        time.sleep(0.25)
    raise AssertionError(f"runtime did not finish: {status}")


def test_dashboard_and_api_flow(cfg):
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["ok"] is True
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["running"] is False

        response = client.post(
            "/api/runtime/start",
            json={"seed": 321, "occluded": False, "agent_mode": "rules"},
        )
        assert response.status_code == 200, response.text
        _wait_done(client)
        time.sleep(0.5)

        events = client.get("/api/events?limit=100").json()["events"]
        assert events, "expected events from the scenario"
        assert {e["type"] for e in events} >= {"session_start", "session_end"}

        records = client.get("/api/records?limit=100").json()["records"]
        assert records
        record = records[0]
        trace = client.get(f"/api/records/{record['record_id']}/trace").json()
        assert trace["trace_chain"] == ["record", "event", "frame"]
        assert trace["event"]["event_id"] == record["event_id"]
        assert trace["frame"]["url"] == f"/api/frames/{record['source_frame']}"

        frame = client.get(f"/api/frames/{record['source_frame']}")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert len(frame.content) > 1000

        event_detail = client.get(f"/api/events/{trace['event']['event_id']}").json()
        assert event_detail["event"]["event_id"] == trace["event"]["event_id"]
        assert event_detail["frame"]["frame_id"] == record["source_frame"]

        runs = client.get("/api/agent/runs").json()["runs"]
        assert runs
        run_detail = client.get(f"/api/agent/runs/{runs[0]['run_id']}").json()["run"]
        assert run_detail["steps"]

        sessions = client.get("/api/sessions").json()["sessions"]
        assert sessions
        session_detail = client.get(f"/api/sessions/{sessions[0]['session_id']}").json()
        assert "completed_steps" in session_detail

        stats = client.get("/api/stats").json()
        assert stats["events"] >= len(events)
        assert stats["records"] >= len(records)

        # Manual event injection goes through the same processor/agent path.
        ingested = client.post(
            "/api/ingest",
            json={
                "type": "session_start",
                "ts": 88.0,
                "reading": 1.0,
                "confidence": 0.9,
                "source_frame": "manual-frame",
            },
        )
        assert ingested.status_code == 200
        event_id = ingested.json()["event"]["event_id"]
        assert client.get(f"/api/events/{event_id}").status_code == 200


def test_pending_confirmation_triggers_agent(cfg):
    app = create_app(cfg)
    with TestClient(app) as client:
        runtime = app.state.runtime
        candidate = PendingCandidate(
            kind="reading_change",
            reason="low confidence test",
            object_id="instrument-1",
            object_label="instrument",
            confidence=0.42,
            ts=4.0,
            source_frame="pending-frame",
            actor_id="actor-1",
            action="interacting",
            reading=2.5,
            session_id="sess-pending",
            meta={"previous_reading": 1.0},
        )
        item = runtime.db.add_pending(candidate)
        assert client.get("/api/pending").json()["count"] == 1

        response = client.post(f"/api/pending/{item['pending_id']}/confirm")
        assert response.status_code == 200, response.text
        event_id = response.json()["event"]["event_id"]
        assert client.get(f"/api/events/{event_id}").status_code == 200
        assert client.get("/api/pending").json()["count"] == 0

        deadline = time.time() + 10
        while time.time() < deadline:
            if runtime.db.record_for_event(event_id):
                break
            time.sleep(0.2)
        record = runtime.db.record_for_event(event_id)
        assert record is not None
        assert record["source_frame"] == "pending-frame"


def test_eval_endpoint(cfg):
    app = create_app(cfg)
    with TestClient(app) as client:
        response = client.get("/api/eval/latest")
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            assert "event_metrics" in response.json()
