from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from ..schema import LabEvent, Observation, new_event_id
from ..store.db import Database
from ..store.frames import FrameStore
from .bus import EventBus
from .state_machine import RuleStateMachine

log = logging.getLogger(__name__)

ALLOWED_EVENT_TYPES = {"session_start", "reading_change", "session_end", "anomaly"}


class EventProcessor:
    """Runs the state machine and persists events / pending reviews."""

    def __init__(
        self,
        cfg: Config,
        machine: RuleStateMachine,
        db: Database,
        frame_store: FrameStore,
        bus: EventBus | None = None,
    ):
        self.cfg = cfg
        self.machine = machine
        self.db = db
        self.frame_store = frame_store
        self.bus = bus
        self.experiment_id = str(cfg.get("agent.experiment_id", "exp-default"))
        self.coalesce_s = float(cfg.get("events.pending_coalesce_s", 2.0))

    def handle_observations(self, observations: list[Observation]) -> list[LabEvent]:
        emitted: list[LabEvent] = []
        for obs in observations:
            output = self.machine.process(obs)
            for event in output.events:
                self.record_event(event)
                emitted.append(event)
            for candidate in output.pending:
                self.frame_store.persist(candidate.source_frame)
                item = self.db.add_pending(candidate, coalesce_s=self.coalesce_s)
                if self.bus:
                    self.bus.publish("pending", item)
        return emitted

    def record_event(self, event: LabEvent) -> LabEvent:
        if event.type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"unknown event type: {event.type}")
        self.frame_store.persist(event.source_frame)
        self.db.save_event(event)
        if event.session_id:
            if event.type == "session_start":
                self.db.upsert_session(
                    event.session_id, event.object_id, event.ts, self.experiment_id
                )
            if event.type in ("session_start", "reading_change"):
                self.db.touch_session(event.session_id, reading=event.reading)
            if event.type == "session_end":
                self.db.touch_session(event.session_id, reading=event.reading, ended=event.ts)
        if self.bus:
            self.bus.publish("event", event.to_dict())
        return event

    def ingest_event(self, data: dict[str, Any]) -> LabEvent:
        """Manually injected event (tests, external sensors)."""
        event = LabEvent.from_dict(data)
        if event.type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"unknown event type: {event.type}")
        if not event.event_id:
            event.event_id = new_event_id()
        event.status = "confirmed"
        event.meta.setdefault("source", "manual_ingest")
        return self.record_event(event)

    def confirm_pending(self, pending_id: str) -> dict[str, Any] | None:
        item = self.db.get_pending(pending_id)
        if item is None or item.get("status") != "open":
            return None
        draft = item.get("draft_event")
        if not draft:
            return None
        event = LabEvent.from_dict(draft)
        event.event_id = new_event_id()
        event.status = "human_confirmed"
        event.meta["confirmed_from_pending"] = True
        event.meta["pending_id"] = pending_id
        self.record_event(event)
        self.db.resolve_pending(pending_id, "confirmed")
        if self.bus:
            self.bus.publish("pending_resolved", {"pending_id": pending_id, "status": "confirmed"})
        return {"pending_id": pending_id, "status": "confirmed", "event": event.to_dict()}

    def reject_pending(self, pending_id: str) -> bool:
        ok = self.db.resolve_pending(pending_id, "rejected")
        if ok and self.bus:
            self.bus.publish("pending_resolved", {"pending_id": pending_id, "status": "rejected"})
        return ok
