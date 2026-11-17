from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Config
from ..sop import SOPRegistry
from ..schema import ExperimentRecord, LabEvent, new_id
from ..store.db import Database

ALLOWED_SEVERITIES = {"info", "warning", "critical"}


@dataclass
class ToolContext:
    event: LabEvent | None = None
    run_id: str | None = None
    mode: str = "tool"

    def session_id(self, explicit: str | None = None) -> str | None:
        if explicit:
            return explicit
        if self.event and self.event.session_id:
            return self.event.session_id
        return None


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_sop_step",
            "description": "Look up an SOP step by id, or the next incomplete step when step_id='next'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "step_id": {
                        "type": "string",
                        "description": "Step id such as 'S2', or 'next' for the next incomplete step.",
                    },
                    "session_id": {"type": "string"},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_step_order",
            "description": "Check whether a step may be executed now according to the SOP order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "step_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["experiment_id", "step_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_reading",
            "description": "Check a reading against the expected range of an SOP step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "step_id": {"type": "string"},
                    "reading": {"type": "number"},
                },
                "required": ["experiment_id", "step_id", "reading"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_experiment_record",
            "description": (
                "Append a traceable experiment record. The record automatically inherits "
                "source_frame from the referenced event. Provide the real event_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "event_id": {"type": "string"},
                    "summary": {"type": "string", "description": "One concise factual sentence."},
                    "step_id": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["experiment_id", "event_id", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_sop_step",
            "description": "Mark an SOP step complete after its work is verified; refuses out-of-order steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "step_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "event_id": {"type": "string"},
                },
                "required": ["experiment_id", "step_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "raise_alert",
            "description": "Raise an anomaly alert for a human operator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "message": {"type": "string"},
                    "event_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["severity", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_events",
            "description": "Read recent confirmed events for context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "object_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_context",
            "description": "Get session state, completed SOP steps, recent events and records.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
            },
        },
    },
]


class LabTools:
    """Deterministic tool implementations shared by LLM and rule agents."""

    def __init__(self, cfg: Config, db: Database, sops: SOPRegistry):
        self.cfg = cfg
        self.db = db
        self.sops = sops
        self._handlers: dict[str, Callable[[dict[str, Any], ToolContext], dict[str, Any]]] = {
            "get_sop_step": self.get_sop_step,
            "validate_step_order": self.validate_step_order,
            "validate_reading": self.validate_reading,
            "write_experiment_record": self.write_experiment_record,
            "advance_sop_step": self.advance_sop_step,
            "raise_alert": self.raise_alert,
            "get_recent_events": self.get_recent_events,
            "get_session_context": self.get_session_context,
        }

    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def call(self, name: str, arguments: dict[str, Any], context: ToolContext | None = None) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        ctx = context or ToolContext()
        try:
            return handler(arguments or {}, ctx)
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as a tool error
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- helpers ----------------------------------------------------------
    def _require(self, args: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            if args.get(key) in (None, ""):
                return key
        return None

    def _session(self, args: dict[str, Any], ctx: ToolContext) -> str:
        return ctx.session_id(args.get("session_id")) or "default"

    @staticmethod
    def _progress_key(experiment_id: str) -> str:
        """SOP progress spans vision sessions for one experiment.

        A camera session ends when the operator walks away, but the experiment's
        SOP step progress must survive that break.  Progress is therefore keyed
        per experiment rather than per vision session.
        """
        return f"progress:{experiment_id}"

    @staticmethod
    def _trim_event(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "type": event.get("type"),
            "ts": event.get("ts"),
            "object_id": event.get("object_id"),
            "actor_id": event.get("actor_id"),
            "reading": event.get("reading"),
            "confidence": event.get("confidence"),
            "session_id": event.get("session_id"),
            "source_frame": event.get("source_frame"),
        }

    # -- tools ------------------------------------------------------------
    def get_sop_step(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "experiment_id")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        experiment_id = args["experiment_id"]
        sop = self.sops.get(experiment_id)
        if sop is None:
            return {"ok": False, "error": f"unknown experiment_id: {experiment_id}"}
        session_id = self._session(args, ctx)
        progress = self._progress_key(experiment_id)
        completed = self.db.completed_steps(experiment_id, progress)
        requested = str(args.get("step_id") or "next")
        if requested == "next":
            step = next((s for s in sop.steps if s.id not in completed), None)
        else:
            step = sop.get_step(requested)
            if step is None:
                return {"ok": False, "error": f"unknown step_id: {requested}"}
        next_step = next((s for s in sop.steps if s.id not in completed), None)
        return {
            "ok": True,
            "experiment_id": experiment_id,
            "experiment_name": sop.name,
            "session_id": session_id,
            "step": step.to_dict() if step else None,
            "steps_total": len(sop.steps),
            "completed_steps": completed,
            "next_step_id": next_step.id if next_step else None,
            "message": "all SOP steps completed" if next_step is None else "",
        }

    def validate_step_order(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "experiment_id", "step_id")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        experiment_id = args["experiment_id"]
        step_id = str(args["step_id"])
        sop = self.sops.get(experiment_id)
        if sop is None:
            return {"ok": False, "error": f"unknown experiment_id: {experiment_id}"}
        step = sop.get_step(step_id)
        if step is None:
            return {"ok": False, "error": f"unknown step_id: {step_id}"}
        session_id = self._session(args, ctx)
        progress = self._progress_key(experiment_id)
        completed = self.db.completed_steps(experiment_id, progress)
        expected = next((s for s in sop.steps if s.id not in completed), None)
        if expected is None:
            return {
                "ok": True,
                "allowed": True,
                "step_id": step_id,
                "expected_step_id": None,
                "completed_steps": completed,
                "message": "all steps already completed; repeat allowed",
            }
        if step_id == expected.id:
            return {
                "ok": True,
                "allowed": True,
                "step_id": step_id,
                "expected_step_id": expected.id,
                "completed_steps": completed,
                "message": f"step {step_id} is next in order",
            }
        if step_id in completed:
            return {
                "ok": True,
                "allowed": True,
                "already_completed": True,
                "step_id": step_id,
                "expected_step_id": expected.id,
                "completed_steps": completed,
                "message": f"step {step_id} was already completed; repeat allowed",
            }
        return {
            "ok": True,
            "allowed": False,
            "step_id": step_id,
            "expected_step_id": expected.id,
            "completed_steps": completed,
            "message": f"out of order: expected step {expected.id} ({expected.name})",
        }

    def validate_reading(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "experiment_id", "step_id", "reading")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        experiment_id = args["experiment_id"]
        step_id = str(args["step_id"])
        try:
            reading = float(args["reading"])
        except (TypeError, ValueError):
            return {"ok": False, "error": "reading must be numeric"}
        sop = self.sops.get(experiment_id)
        if sop is None:
            return {"ok": False, "error": f"unknown experiment_id: {experiment_id}"}
        step = sop.get_step(step_id)
        if step is None:
            return {"ok": False, "error": f"unknown step_id: {step_id}"}
        expected = step.expected_reading
        if not expected:
            return {
                "ok": True,
                "checked": False,
                "within_range": None,
                "reading": reading,
                "step_id": step_id,
                "message": f"step {step_id} has no expected reading range",
            }
        low = float(expected.get("min", 0.0))
        high = float(expected.get("max", 1e9))
        within = low <= reading <= high
        return {
            "ok": True,
            "checked": True,
            "within_range": within,
            "reading": reading,
            "step_id": step_id,
            "expected": {"min": low, "max": high, "unit": expected.get("unit")},
            "deviation": round(reading - min(max(reading, low), high), 4),
            "message": "reading within expected range" if within else "reading outside expected range",
        }

    def write_experiment_record(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "experiment_id", "event_id", "summary")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        event_id = str(args["event_id"])
        event = self.db.get_event(event_id)
        if event is None:
            return {"ok": False, "error": f"event_id not found: {event_id}"}
        summary = str(args["summary"]).strip()
        if not summary:
            return {"ok": False, "error": "summary must not be empty"}
        step_id = args.get("step_id")
        if step_id:
            sop = self.sops.get(args["experiment_id"])
            if sop is not None and sop.get_step(str(step_id)) is None:
                return {"ok": False, "error": f"unknown step_id: {step_id}"}
            step_id = str(step_id)
        note = str(args.get("note") or "").strip()
        full_note = summary if not note else f"{summary}\n{note}"
        session_id = event.get("session_id") or ctx.session_id() or "adhoc"
        record = ExperimentRecord(
            record_id=new_id("rec"),
            experiment_id=str(args["experiment_id"]),
            session_id=session_id,
            step_id=step_id,
            event_id=event_id,
            source_frame=event.get("source_frame") or "",
            object_id=event.get("object_id") or "",
            reading=event.get("reading"),
            note=full_note[:1000],
            created_at=time.time(),
            agent_run_id=ctx.run_id,
            meta={"event_type": event.get("type"), "agent_mode": ctx.mode},
        )
        dedupe_key = f"{event_id}|{step_id or '-'}"
        saved, created = self.db.save_record(record, dedupe_key)
        frame = self.db.get_frame(record.source_frame) if record.source_frame else None
        return {
            "ok": True,
            "created": created,
            "deduped": not created,
            "record_id": saved.get("record_id"),
            "session_id": saved.get("session_id"),
            "step_id": saved.get("step_id"),
            "source_frame": saved.get("source_frame"),
            "frame_url": f"/api/frames/{saved.get('source_frame')}" if saved.get("source_frame") else None,
            "frame_available": bool(frame),
            "message": "record appended" if created else "record already existed (idempotent)",
        }

    def advance_sop_step(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "experiment_id", "step_id")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        experiment_id = args["experiment_id"]
        step_id = str(args["step_id"])
        sop = self.sops.get(experiment_id)
        if sop is None:
            return {"ok": False, "error": f"unknown experiment_id: {experiment_id}"}
        if sop.get_step(step_id) is None:
            return {"ok": False, "error": f"unknown step_id: {step_id}"}
        session_id = self._session(args, ctx)
        progress = self._progress_key(experiment_id)
        order = self.validate_step_order(
            {"experiment_id": experiment_id, "step_id": step_id, "session_id": session_id}, ctx
        )
        if not order.get("allowed"):
            return {
                "ok": True,
                "advanced": False,
                "reason": order.get("message", "step order violation"),
                "expected_step_id": order.get("expected_step_id"),
                "completed_steps": order.get("completed_steps", []),
            }
        if order.get("already_completed"):
            return {
                "ok": True,
                "advanced": False,
                "reason": f"step {step_id} already completed",
                "next_step_id": order.get("expected_step_id"),
                "completed_steps": order.get("completed_steps", []),
            }
        self.db.mark_step_complete(experiment_id, progress, step_id, time.time())
        completed = self.db.completed_steps(experiment_id, progress)
        next_step = next((s for s in sop.steps if s.id not in completed), None)
        return {
            "ok": True,
            "advanced": True,
            "step_id": step_id,
            "session_id": session_id,
            "next_step_id": next_step.id if next_step else None,
            "completed_steps": completed,
            "message": f"step {step_id} marked complete",
        }

    def raise_alert(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        missing = self._require(args, "severity", "message")
        if missing:
            return {"ok": False, "error": f"missing argument: {missing}"}
        severity = str(args["severity"]).lower()
        if severity not in ALLOWED_SEVERITIES:
            return {"ok": False, "error": f"invalid severity: {severity}"}
        message = str(args["message"]).strip()
        if not message:
            return {"ok": False, "error": "message must not be empty"}
        event_id = args.get("event_id") or (ctx.event.event_id if ctx.event else None)
        session_id = ctx.session_id(args.get("session_id"))
        alert = self.db.save_alert(
            severity=severity,
            message=message[:500],
            event_id=event_id,
            session_id=session_id,
            meta={"run_id": ctx.run_id},
        )
        return {"ok": True, "alert": alert}

    def get_recent_events(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        limit = max(1, min(50, int(args.get("limit", 10) or 10)))
        events = self.db.list_events(
            limit=limit,
            object_id=args.get("object_id"),
            session_id=ctx.session_id(args.get("session_id")),
        )
        return {"ok": True, "events": [self._trim_event(e) for e in events]}

    def get_session_context(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        session_id = ctx.session_id(args.get("session_id"))
        if not session_id:
            sessions = self.db.list_sessions(1)
            session_id = sessions[0]["session_id"] if sessions else None
        experiment_id = str(args.get("experiment_id") or self.cfg.get("agent.experiment_id", "exp-titration"))
        session = self.db.get_session(session_id) if session_id else None
        events = self.db.list_events(limit=20, session_id=session_id) if session_id else []
        records = self.db.list_records(limit=10, session_id=session_id) if session_id else []
        completed = self.db.completed_steps(experiment_id, self._progress_key(experiment_id))
        sop = self.sops.get(experiment_id)
        next_step = None
        if sop:
            candidate = next((s for s in sop.steps if s.id not in completed), None)
            next_step = candidate.to_dict() if candidate else None
        return {
            "ok": True,
            "session_id": session_id,
            "session": session,
            "experiment_id": experiment_id,
            "completed_steps": completed,
            "next_step": next_step,
            "recent_events": [self._trim_event(e) for e in events],
            "recent_records": [
                {
                    "record_id": r.get("record_id"),
                    "step_id": r.get("step_id"),
                    "event_id": r.get("event_id"),
                    "note": r.get("note"),
                    "created_at": r.get("created_at"),
                }
                for r in records
            ],
        }

    def sop_summary(self, experiment_id: str) -> dict[str, Any]:
        sop = self.sops.get(experiment_id)
        return sop.to_dict() if sop else {"experiment_id": experiment_id, "steps": []}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
