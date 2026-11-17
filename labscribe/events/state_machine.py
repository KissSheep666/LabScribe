from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..schema import LabEvent, Observation, PendingCandidate, new_event_id, new_id


@dataclass
class MachineOutput:
    events: list[LabEvent] = field(default_factory=list)
    pending: list[PendingCandidate] = field(default_factory=list)


@dataclass
class _ObjectState:
    phase: str = "idle"
    session_id: str | None = None
    interact_frames: int = 0
    last_interact_ts: float | None = None
    last_reading: float | None = None
    last_anomaly_reading: float | None = None


class RuleStateMachine:
    """Folds frame observations into session / reading / anomaly events.

    Rules are intentionally transparent and tunable via config:

    * ``session_start``  — actor present at the instrument for N consecutive frames
    * ``reading_change`` — a stabilized display value differs from the last one
    * ``session_end``    — no actor for ``session_end_idle_s`` seconds
    * ``anomaly``        — stable reading outside ``events.limits`` for the object

    Low-confidence readings never update state; they are emitted as pending
    candidates for human confirmation instead.
    """

    def __init__(self, cfg: Config):
        self.min_conf = float(cfg.get("events.min_confidence", 0.55))
        self.start_frames = max(1, int(cfg.get("events.session_start_frames", 3)))
        self.end_idle_s = float(cfg.get("events.session_end_idle_s", 1.2))
        self.reading_tol = float(cfg.get("events.reading_tolerance", 0.4))
        self.limits: dict[str, dict[str, float]] = dict(cfg.get("events.limits", {}) or {})
        self._states: dict[str, _ObjectState] = {}

    def _state(self, object_id: str) -> _ObjectState:
        return self._states.setdefault(object_id, _ObjectState())

    def process(self, obs: Observation) -> MachineOutput:
        out = MachineOutput()
        state = self._state(obs.object_id)
        detection_conf = float(obs.confidence)

        if detection_conf >= self.min_conf:
            self._session_transitions(obs, state, out)
        else:
            # Low-confidence observations never change state, but they still
            # count toward a candidate event that a human can confirm.
            if obs.action == "interacting" and state.phase == "idle":
                state.interact_frames += 1
            elif obs.action != "interacting":
                state.interact_frames = 0
            candidate = self._session_candidate(obs, state)
            if candidate is not None:
                out.pending.append(candidate)

        reading_conf = obs.reading_confidence if obs.reading_confidence is not None else 1.0
        effective = min(detection_conf, float(reading_conf))

        if obs.reading_stable and obs.reading is not None:
            if state.last_reading is None:
                if effective >= self.min_conf:
                    state.last_reading = obs.reading
            elif abs(obs.reading - state.last_reading) > self.reading_tol:
                if effective >= self.min_conf:
                    event = self._event(
                        "reading_change",
                        obs,
                        effective,
                        reading=obs.reading,
                        meta={
                            "previous_reading": state.last_reading,
                            "reading_confidence": obs.reading_confidence,
                            "machine": "rule",
                        },
                    )
                    out.events.append(event)
                    state.last_reading = obs.reading
                else:
                    out.pending.append(
                        PendingCandidate(
                            kind="reading_change",
                            reason=f"reading confidence {effective:.2f} < {self.min_conf:.2f}",
                            object_id=obs.object_id,
                            object_label=obs.object_label,
                            confidence=effective,
                            ts=obs.ts,
                            source_frame=obs.frame_id,
                            actor_id=obs.actor_id,
                            action=obs.action,
                            reading=obs.reading,
                            session_id=state.session_id,
                            meta={
                                "previous_reading": state.last_reading,
                                "reading_confidence": obs.reading_confidence,
                                "raw_reading": obs.raw_reading,
                            },
                        )
                    )

        limits = self.limits.get(obs.object_id)
        if limits and obs.reading is not None and effective >= 0.3:
            low = float(limits.get("min", 0.0))
            high = float(limits.get("max", 1e9))
            if obs.reading < low or obs.reading > high:
                already_flagged = (
                    state.last_anomaly_reading is not None
                    and abs(obs.reading - state.last_anomaly_reading) <= self.reading_tol
                )
                if not already_flagged:
                    meta = {
                        "rule": "reading_out_of_range",
                        "min": low,
                        "max": high,
                        "reading_confidence": obs.reading_confidence,
                    }
                    if effective >= self.min_conf:
                        out.events.append(
                            self._event("anomaly", obs, effective, reading=obs.reading, meta=meta)
                        )
                        state.last_anomaly_reading = obs.reading
                    else:
                        out.pending.append(
                            PendingCandidate(
                                kind="anomaly",
                                reason=f"out-of-range reading with confidence {effective:.2f}",
                                object_id=obs.object_id,
                                object_label=obs.object_label,
                                confidence=effective,
                                ts=obs.ts,
                                source_frame=obs.frame_id,
                                actor_id=obs.actor_id,
                                action=obs.action,
                                reading=obs.reading,
                                session_id=state.session_id,
                                meta=meta,
                            )
                        )
            elif (
                state.last_anomaly_reading is not None
                and abs(obs.reading - state.last_anomaly_reading) > self.reading_tol
            ):
                state.last_anomaly_reading = None

        return out

    # -- transitions ------------------------------------------------------
    def _session_transitions(self, obs: Observation, state: _ObjectState, out: MachineOutput) -> None:
        if obs.action == "interacting":
            if state.phase == "idle":
                state.interact_frames += 1
                if state.interact_frames >= self.start_frames:
                    state.phase = "active"
                    state.session_id = new_id("sess")
                    state.last_interact_ts = obs.ts
                    if obs.reading is not None:
                        state.last_reading = obs.reading
                    out.events.append(
                        self._event(
                            "session_start",
                            obs,
                            obs.confidence,
                            reading=obs.reading,
                            session_id=state.session_id,
                            meta={
                                "restart": False,
                                "start_frames": self.start_frames,
                                "machine": "rule",
                            },
                        )
                    )
            else:
                state.interact_frames = 0
                state.last_interact_ts = obs.ts
        else:
            state.interact_frames = 0
            if (
                state.phase == "active"
                and state.last_interact_ts is not None
                and obs.ts - state.last_interact_ts >= self.end_idle_s
            ):
                out.events.append(
                    self._event(
                        "session_end",
                        obs,
                        obs.confidence,
                        reading=state.last_reading,
                        session_id=state.session_id,
                        meta={
                            "idle_s": round(obs.ts - state.last_interact_ts, 3),
                            "machine": "rule",
                        },
                    )
                )
                state.phase = "idle"
                state.session_id = None
                state.last_interact_ts = None

    def _session_candidate(self, obs: Observation, state: _ObjectState) -> PendingCandidate | None:
        reason = f"detection confidence {obs.confidence:.2f} < {self.min_conf:.2f}"
        if (
            state.phase == "idle"
            and obs.action == "interacting"
            and state.interact_frames >= self.start_frames
        ):
            return PendingCandidate(
                kind="session_start",
                reason=reason,
                object_id=obs.object_id,
                object_label=obs.object_label,
                confidence=obs.confidence,
                ts=obs.ts,
                source_frame=obs.frame_id,
                actor_id=obs.actor_id,
                action=obs.action,
                reading=obs.reading,
                meta={"machine": "rule", "low_confidence": True},
            )
        if (
            state.phase == "active"
            and obs.action != "interacting"
            and state.last_interact_ts is not None
            and obs.ts - state.last_interact_ts >= self.end_idle_s
        ):
            return PendingCandidate(
                kind="session_end",
                reason=reason,
                object_id=obs.object_id,
                object_label=obs.object_label,
                confidence=obs.confidence,
                ts=obs.ts,
                source_frame=obs.frame_id,
                actor_id=obs.actor_id,
                action=obs.action,
                reading=state.last_reading,
                session_id=state.session_id,
                meta={"machine": "rule", "low_confidence": True},
            )
        return None

    def _event(
        self,
        event_type: str,
        obs: Observation,
        confidence: float,
        reading: float | None = None,
        session_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> LabEvent:
        state = self._state(obs.object_id)
        payload = dict(meta or {})
        payload.setdefault("occluded", obs.occluded)
        if self.limits.get(obs.object_id):
            payload.setdefault("limits", self.limits[obs.object_id])
        return LabEvent(
            event_id=new_event_id(),
            type=event_type,
            ts=obs.ts,
            object_id=obs.object_id,
            object_label=obs.object_label,
            actor_id=obs.actor_id,
            action=obs.action,
            reading=reading,
            confidence=float(confidence),
            source_frame=obs.frame_id,
            session_id=session_id if session_id is not None else state.session_id,
            meta=payload,
        )
