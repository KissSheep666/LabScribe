from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Config
from ..paths import resolve_path
from ..runtime import Runtime
from ..schema import LabEvent, new_event_id
from ..vision.synthetic import make_demo_scenario, make_scenario

STATIC_DIR = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    scenario: str = "demo"
    seed: int | None = None
    occluded: bool = False
    agent_mode: str | None = None
    realtime: bool | None = None
    source: str | None = None


class IngestRequest(BaseModel):
    type: str = Field(..., description="session_start|reading_change|session_end|anomaly")
    ts: float = 0.0
    object_id: str = "instrument-1"
    object_label: str = "instrument"
    actor_id: str | None = None
    action: str = "interacting"
    reading: float | None = None
    confidence: float = 0.9
    source_frame: str = ""
    session_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


def create_app(cfg: Config | None = None, runtime: Runtime | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    rt = runtime or Runtime(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.get("api.autostart_runtime", False):
            rt.start()
        yield
        rt.stop()

    app = FastAPI(title="LabScribe", version=__version__, lifespan=lifespan)
    app.state.runtime = rt
    app.state.config = cfg

    def get_rt(request: Request) -> Runtime:
        return request.app.state.runtime

    # -- UI ---------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        path = STATIC_DIR / "index.html"
        if not path.exists():
            return HTMLResponse("<h1>LabScribe</h1><p>dashboard not installed</p>", status_code=500)
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "labscribe", "version": __version__}

    # -- runtime ----------------------------------------------------------
    @app.get("/api/status")
    def status(request: Request) -> dict[str, Any]:
        return get_rt(request).status()

    @app.post("/api/runtime/start")
    def start_runtime(req: StartRequest, request: Request) -> dict[str, Any]:
        rt = get_rt(request)
        try:
            if req.source and req.source != "synthetic":
                cfg.set("vision.source", req.source)
                return rt.start(agent_mode=req.agent_mode, realtime=req.realtime)
            if req.seed is not None:
                scenario = make_scenario(
                    seed=req.seed,
                    scenario_id=f"ui-{req.seed}",
                    occluded=req.occluded,
                    fps=int(cfg.get("vision.fps", 10)),
                )
            else:
                scenario = make_demo_scenario(fps=int(cfg.get("vision.fps", 10)))
                scenario.seed = int(cfg.get("synthetic.seed", 7))
            return rt.start(
                scenario=scenario,
                agent_mode=req.agent_mode,
                realtime=req.realtime,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/runtime/stop")
    def stop_runtime(request: Request) -> dict[str, Any]:
        return get_rt(request).stop()

    # -- events / records -------------------------------------------------
    @app.get("/api/events")
    def list_events(
        request: Request,
        limit: int = Query(100, ge=1, le=1000),
        since: float | None = None,
        type: str | None = None,
        session_id: str | None = None,
        object_id: str | None = None,
    ) -> dict[str, Any]:
        events = get_rt(request).db.list_events(
            limit=limit, since=since, event_type=type, session_id=session_id, object_id=object_id
        )
        return {"events": events, "count": len(events)}

    @app.get("/api/events/{event_id}")
    def get_event(event_id: str, request: Request) -> dict[str, Any]:
        db = get_rt(request).db
        event = db.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        frame = db.get_frame(event.get("source_frame") or "")
        records = [r for r in db.list_records(limit=1000) if r.get("event_id") == event_id]
        return {
            "event": event,
            "frame": _frame_view(frame),
            "records": records,
            "agent_runs": db.runs_for_event(event_id),
            "alerts": db.alerts_for_event(event_id),
        }

    @app.get("/api/records")
    def list_records(
        request: Request,
        limit: int = Query(100, ge=1, le=1000),
        session_id: str | None = None,
    ) -> dict[str, Any]:
        records = get_rt(request).db.list_records(limit=limit, session_id=session_id)
        return {"records": records, "count": len(records)}

    @app.get("/api/records/{record_id}")
    def get_record(record_id: str, request: Request) -> dict[str, Any]:
        record = get_rt(request).db.get_record(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="record not found")
        return {"record": record}

    @app.get("/api/records/{record_id}/trace")
    def trace_record(record_id: str, request: Request) -> dict[str, Any]:
        db = get_rt(request).db
        record = db.get_record(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="record not found")
        event = db.get_event(record.get("event_id") or "")
        frame = db.get_frame(record.get("source_frame") or "")
        return {
            "record": record,
            "event": event,
            "frame": _frame_view(frame),
            "agent_runs": db.runs_for_event(record.get("event_id") or ""),
            "trace_chain": ["record", "event", "frame"],
        }

    @app.get("/api/frames/{frame_id}")
    def get_frame(frame_id: str, request: Request):
        frame = request.app.state.runtime.db.get_frame(frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="frame not found")
        path = Path(frame["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="frame file missing")
        return FileResponse(path, media_type="image/jpeg", filename=path.name)

    # -- pending / alerts -------------------------------------------------
    @app.get("/api/pending")
    def list_pending(request: Request, status: str = "open") -> dict[str, Any]:
        items = get_rt(request).db.list_pending(status=status)
        return {"pending": items, "count": len(items)}

    @app.post("/api/pending/{pending_id}/confirm")
    def confirm_pending(pending_id: str, request: Request) -> dict[str, Any]:
        rt = get_rt(request)
        result = rt.processor.confirm_pending(pending_id)
        if result is None:
            raise HTTPException(status_code=404, detail="pending item not found or already resolved")
        event = LabEvent.from_dict(result["event"])
        rt.dispatch_event(event)
        return result

    @app.post("/api/pending/{pending_id}/reject")
    def reject_pending(pending_id: str, request: Request) -> dict[str, Any]:
        ok = get_rt(request).processor.reject_pending(pending_id)
        if not ok:
            raise HTTPException(status_code=404, detail="pending item not found or already resolved")
        return {"pending_id": pending_id, "status": "rejected"}

    @app.get("/api/alerts")
    def list_alerts(request: Request, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        alerts = get_rt(request).db.list_alerts(limit=limit)
        return {"alerts": alerts, "count": len(alerts)}

    # -- sessions / agent runs -------------------------------------------
    @app.get("/api/sessions")
    def list_sessions(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        sessions = get_rt(request).db.list_sessions(limit=limit)
        return {"sessions": sessions, "count": len(sessions)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, request: Request) -> dict[str, Any]:
        rt = get_rt(request)
        session = rt.db.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        experiment_id = str(cfg.get("agent.experiment_id", "exp-titration"))
        return {
            "session": session,
            "events": rt.db.list_events(limit=500, session_id=session_id),
            "records": rt.db.list_records(limit=500, session_id=session_id),
            "completed_steps": rt.db.completed_steps(
                experiment_id, f"progress:{experiment_id}"
            ),
        }

    @app.get("/api/agent/runs")
    def list_runs(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        runs = get_rt(request).db.list_runs(limit=limit)
        return {"runs": runs, "count": len(runs)}

    @app.get("/api/agent/runs/{run_id}")
    def get_run(run_id: str, request: Request) -> dict[str, Any]:
        run = get_rt(request).db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run": run}

    # -- misc -------------------------------------------------------------
    @app.get("/api/stats")
    def stats(request: Request) -> dict[str, Any]:
        return get_rt(request).db.stats()

    @app.get("/api/eval/latest")
    def latest_eval(request: Request):
        path = resolve_path(cfg.get("eval.report_json", "reports/eval_report.json"))
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail="no evaluation report; run `labscribe eval`")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.post("/api/ingest")
    def ingest(req: IngestRequest, request: Request) -> dict[str, Any]:
        rt = get_rt(request)
        data = req.model_dump()
        data["event_id"] = new_event_id()
        try:
            event = rt.processor.ingest_event(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rt.dispatch_event(event)
        return {"event": event.to_dict()}

    return app


def _frame_view(frame: dict | None) -> dict[str, Any] | None:
    if not frame:
        return None
    return {
        "frame_id": frame.get("frame_id"),
        "ts": frame.get("ts"),
        "width": frame.get("width"),
        "height": frame.get("height"),
        "meta": frame.get("meta"),
        "url": f"/api/frames/{frame.get('frame_id')}",
    }
