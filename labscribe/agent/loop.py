from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..schema import LabEvent, new_id
from ..store.db import Database
from .llm import LLMClient, LLMError
from .tools import LabTools, ToolContext

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    run_id: str
    status: str
    mode: str
    summary: str
    event_id: str | None = None
    error: str | None = None
    tool_calls: int = 0
    steps: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "mode": self.mode,
            "summary": self.summary,
            "event_id": self.event_id,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "duration_ms": round(self.duration_ms, 2),
        }


class _StepLogger:
    def __init__(self, db: Database, run_id: str):
        self.db = db
        self.run_id = run_id
        self.seq = 0
        self.tool_calls = 0

    def log(self, kind: str, **kwargs: Any) -> dict:
        step = self.db.save_step(self.run_id, self.seq, kind, **kwargs)
        self.seq += 1
        return step

    def call(
        self, tools: LabTools, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = tools.call(name, args, ctx)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.log("tool", tool=name, arguments=args, result=result, latency_ms=latency_ms)
        self.tool_calls += 1
        return result


class LabAgent:
    """Event-driven agent loop with deterministic tools and a rules fallback."""

    def __init__(
        self,
        cfg: Config,
        db: Database,
        tools: LabTools,
        mode: str | None = None,
        llm: LLMClient | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.tools = tools
        self.mode = (mode or str(cfg.get("agent.mode", "llm"))).lower()
        self.experiment_id = str(cfg.get("agent.experiment_id", "exp-titration"))
        self.max_steps = int(cfg.get("agent.max_steps", 8))
        self.fallback_to_rules = bool(cfg.get("agent.fallback_to_rules", True))
        self.llm = llm
        if self.mode == "llm" and self.llm is None:
            self.llm = LLMClient(cfg)

    def handle_event(self, event: LabEvent) -> AgentResult:
        started = time.perf_counter()
        run_id = new_id("run")
        if self.mode == "off":
            return AgentResult(run_id, "skipped", "off", "agent disabled", event.event_id)
        self.db.save_run(
            run_id,
            event.event_id,
            self.mode,
            time.time(),
            event_type=event.type,
            object_id=event.object_id,
        )
        ctx = ToolContext(event=event, run_id=run_id, mode=self.mode)
        logger = _StepLogger(self.db, run_id)
        try:
            if self.mode == "llm":
                result = self._run_llm(event, run_id, ctx, logger)
            else:
                result = self._run_rules(event, run_id, ctx, logger)
        except LLMError as exc:
            if self.mode == "llm" and self.fallback_to_rules:
                log.warning("LLM agent failed (%s); using rule fallback", exc)
                result = self._run_rules(event, run_id, ctx, logger)
                result.status = "fallback_rules"
                result.summary = f"LLM 不可用，已回退到规则 Agent：{exc}"
                result.error = str(exc)
            else:
                result = AgentResult(
                    run_id, "error", self.mode, "agent failed", event.event_id, error=str(exc)
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("agent run failed")
            result = AgentResult(
                run_id, "error", self.mode, "agent failed", event.event_id, error=str(exc)
            )

        try:
            self._ensure_traceability(event, ctx, logger)
        except Exception:  # noqa: BLE001
            log.exception("safety net failed for event %s", event.event_id)

        result.tool_calls = max(result.tool_calls, logger.tool_calls)
        result.steps = logger.seq
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        self.db.finish_run(run_id, result.status, result.summary, result.error)
        return result

    # -- LLM mode ---------------------------------------------------------
    def _run_llm(
        self, event: LabEvent, run_id: str, ctx: ToolContext, logger: _StepLogger
    ) -> AgentResult:
        if self.llm is None:
            raise LLMError("LLM client is not configured")
        context = self.tools.get_session_context(
            {"experiment_id": self.experiment_id, "session_id": event.session_id}, ctx
        )
        logger.log("context", content="session context loaded", result=context)
        advice = self._planner_advice(event, ctx)
        logger.log("planner", content="deterministic plan derived from SOP", result=advice)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Event:\n{json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"
                    f"Session context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
                    f"Planner guidance (deterministic, derived from the SOP):\n"
                    f"{json.dumps(advice, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        tool_calls = 0
        for _ in range(self.max_steps):
            response = self.llm.chat(messages, tools=self.tools.schemas())
            logger.log(
                "assistant",
                content=response.content,
                latency_ms=response.latency_ms,
                result={"tool_calls": [tc.name for tc in response.tool_calls]},
            )
            if not response.tool_calls:
                return AgentResult(
                    run_id,
                    "completed",
                    "llm",
                    response.content or "Agent finished",
                    event.event_id,
                    tool_calls=tool_calls,
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.raw_tool_calls,
                }
            )
            for call in response.tool_calls:
                if call.error:
                    result = {"ok": False, "error": call.error}
                    logger.log("tool", tool=call.name, arguments=call.arguments, result=result)
                else:
                    result = logger.call(self.tools, call.name, call.arguments, ctx)
                tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
        return AgentResult(
            run_id,
            "max_steps",
            "llm",
            f"达到最大推理步数 {self.max_steps}",
            event.event_id,
            tool_calls=tool_calls,
        )

    def _system_prompt(self) -> str:
        sop = self.tools.sop_summary(self.experiment_id)
        return (
            "You are LabScribe, an agent that converts vision events from a lab bench into a "
            "compliant, traceable experiment log.\n\n"
            f"Experiment: {self.experiment_id}\n"
            f"SOP:\n{json.dumps(sop, ensure_ascii=False, indent=2)}\n\n"
            "You receive exactly one event per turn. Use the tools; do not answer with facts "
            "that are not in the event or tool results.\n\n"
            "Operating procedure:\n"
            "1. session_start: validate the next SOP step's order, then append one record for "
            "that step (include the initial reading if present). Do NOT advance a step merely "
            "because a session started.\n"
            "2. reading_change: validate step order, validate the reading against the step's "
            "expected range, and append one record. If the reading is within range (or the step "
            "has no range), call advance_sop_step. If it is outside the range, raise a warning "
            "alert and do not advance.\n"
            "3. session_end: append one record with the final reading. Advance the current step "
            "only if it has no expected reading and its work is complete.\n"
            "4. anomaly: raise a critical alert and append one record. Never ignore an anomaly.\n\n"
            "Hard rules:\n"
            "- Use only the given event_id. write_experiment_record inherits source_frame from "
            "the event; never invent frame ids, readings or timestamps.\n"
            "- At most one record per event and step.\n"
            "- Record summaries must be one concise Chinese sentence containing the concrete "
            "instrument/action/reading.\n"
            "- Execute the planner guidance unless a tool result contradicts it; you remain "
            "responsible for checking tool outputs.\n"
            "- Finish with one short Chinese sentence. Do not call unnecessary tools."
        )

    def _planner_advice(self, event: LabEvent, ctx: ToolContext) -> dict[str, Any]:
        """Deterministic plan shown to the LLM as guidance.

        The LLM still chooses and executes the tool calls; the plan keeps the
        multi-step SOP flow from stalling on conservative reasoning.
        """
        experiment_id = self.experiment_id
        step_info = self.tools.call(
            "get_sop_step",
            {"experiment_id": experiment_id, "step_id": "next"},
            ctx,
        )
        step = step_info.get("step") if step_info.get("ok") else None
        step_id = step["id"] if step else None
        base = {"experiment_id": experiment_id, "event_id": event.event_id}
        actions: list[dict[str, Any]] = []
        notes: list[str] = []

        if step_id:
            actions.append(
                {
                    "tool": "validate_step_order",
                    "arguments": {**base, "step_id": step_id, "session_id": event.session_id},
                    "why": "verify the current step is next in order",
                }
            )

        if event.type in ("session_start", "reading_change", "session_end"):
            reading_check = None
            if event.type == "reading_change" and step_id and event.reading is not None:
                reading_check = self.tools.call(
                    "validate_reading",
                    {"experiment_id": experiment_id, "step_id": step_id, "reading": event.reading},
                    ctx,
                )
                actions.append(
                    {
                        "tool": "validate_reading",
                        "arguments": {
                            "experiment_id": experiment_id,
                            "step_id": step_id,
                            "reading": event.reading,
                        },
                        "why": "check the stabilized reading against the step range",
                    }
                )
            actions.append(
                {
                    "tool": "write_experiment_record",
                    "arguments": {
                        **base,
                        "summary": self._rule_summary(event),
                        "step_id": step_id,
                    },
                    "why": "one traceable record for this event",
                }
            )
            if event.type == "reading_change":
                checked = bool(reading_check and reading_check.get("checked"))
                within = bool(reading_check and reading_check.get("within_range"))
                limits = (event.meta or {}).get("limits") or {}
                hard_out = bool(
                    limits
                    and event.reading is not None
                    and (
                        event.reading < float(limits.get("min", -1e9))
                        or event.reading > float(limits.get("max", 1e9))
                    )
                )
                if step_id and (not checked or within):
                    actions.append(
                        {
                            "tool": "advance_sop_step",
                            "arguments": {
                                **base,
                                "step_id": step_id,
                                "session_id": event.session_id,
                            },
                            "why": "reading satisfies the current step (or the step has no range)",
                        }
                    )
                elif step_id and hard_out:
                    notes.append(
                        "Reading is outside the global limits; the companion anomaly event "
                        "will raise the critical alert. Do not advance the step."
                    )
                elif step_id:
                    actions.append(
                        {
                            "tool": "raise_alert",
                            "arguments": {
                                "severity": "warning",
                                "message": (
                                    f"读数 {event.reading} 超出步骤 {step_id} 的允许范围 "
                                    f"{reading_check.get('expected') if reading_check else ''}"
                                ),
                                "event_id": event.event_id,
                                "session_id": event.session_id,
                            },
                            "why": "reading is outside the SOP step range",
                        }
                    )
            elif event.type == "session_end":
                if step_id and not (step or {}).get("expected_reading"):
                    actions.append(
                        {
                            "tool": "advance_sop_step",
                            "arguments": {
                                **base,
                                "step_id": step_id,
                                "session_id": event.session_id,
                            },
                            "why": "step has no reading requirement and the interaction ended",
                        }
                    )
                else:
                    notes.append("Do not advance a step that has an expected reading range.")
            else:
                notes.append("Do not advance a step merely because a session started.")

        elif event.type == "anomaly":
            limits = (event.meta or {}).get("limits") or {}
            actions = [
                {
                    "tool": "raise_alert",
                    "arguments": {
                        "severity": "critical",
                        "message": (
                            f"异常读数 {event.reading} 超出允许范围 "
                            f"[{limits.get('min', '?')}, {limits.get('max', '?')}]"
                        ),
                        "event_id": event.event_id,
                        "session_id": event.session_id,
                    },
                    "why": "anomaly events must alert a human",
                },
                {
                    "tool": "write_experiment_record",
                    "arguments": {
                        **base,
                        "summary": self._rule_summary(event),
                        "step_id": step_id,
                    },
                    "why": "one traceable record for this event",
                },
            ]

        return {
            "current_step": step,
            "completed_steps": step_info.get("completed_steps", []),
            "recommended_actions": actions,
            "notes": notes,
        }

    # -- deterministic fallback ------------------------------------------
    def _run_rules(
        self, event: LabEvent, run_id: str, ctx: ToolContext, logger: _StepLogger
    ) -> AgentResult:
        experiment_id = self.experiment_id
        step_info = logger.call(
            self.tools,
            "get_sop_step",
            {"experiment_id": experiment_id, "step_id": "next"},
            ctx,
        )
        step = step_info.get("step") if step_info.get("ok") else None
        step_id = step["id"] if step else None
        summary = self._rule_summary(event)

        if event.type in ("session_start", "reading_change", "session_end"):
            order: dict[str, Any] = {}
            if step_id:
                order = logger.call(
                    self.tools,
                    "validate_step_order",
                    {
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "session_id": event.session_id,
                    },
                    ctx,
                )
            reading_check: dict[str, Any] = {}
            if event.type == "reading_change" and step_id and event.reading is not None:
                reading_check = logger.call(
                    self.tools,
                    "validate_reading",
                    {
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "reading": event.reading,
                    },
                    ctx,
                )
            logger.call(
                self.tools,
                "write_experiment_record",
                {
                    "experiment_id": experiment_id,
                    "event_id": event.event_id,
                    "summary": summary,
                    "step_id": step_id,
                    "note": f"event={event.type}; confidence={event.confidence:.2f}",
                },
                ctx,
            )
            if event.type == "reading_change":
                within = reading_check.get("within_range")
                checked = reading_check.get("checked", False)
                hard_limits = (event.meta or {}).get("limits") or {}
                hard_out = bool(
                    hard_limits
                    and event.reading is not None
                    and (
                        event.reading < float(hard_limits.get("min", -1e9))
                        or event.reading > float(hard_limits.get("max", 1e9))
                    )
                )
                if checked and within is False and not hard_out:
                    logger.call(
                        self.tools,
                        "raise_alert",
                        {
                            "severity": "warning",
                            "message": (
                                f"读数 {event.reading} 超出步骤 {step_id} 的允许范围 "
                                f"{reading_check.get('expected')}"
                            ),
                            "event_id": event.event_id,
                        },
                        ctx,
                    )
                elif step_id and order.get("allowed") and (not checked or within):
                    logger.call(
                        self.tools,
                        "advance_sop_step",
                        {
                            "experiment_id": experiment_id,
                            "step_id": step_id,
                            "session_id": event.session_id,
                            "event_id": event.event_id,
                        },
                        ctx,
                    )
            elif event.type == "session_end" and step_id and order.get("allowed"):
                if not (step or {}).get("expected_reading"):
                    logger.call(
                        self.tools,
                        "advance_sop_step",
                        {
                            "experiment_id": experiment_id,
                            "step_id": step_id,
                            "session_id": event.session_id,
                            "event_id": event.event_id,
                        },
                        ctx,
                    )

        elif event.type == "anomaly":
            limits = (event.meta or {}).get("limits") or {}
            logger.call(
                self.tools,
                "raise_alert",
                {
                    "severity": "critical",
                    "message": (
                        f"异常读数 {event.reading} 超出允许范围 "
                        f"[{limits.get('min', '?')}, {limits.get('max', '?')}]"
                    ),
                    "event_id": event.event_id,
                    "session_id": event.session_id,
                },
                ctx,
            )
            logger.call(
                self.tools,
                "write_experiment_record",
                {
                    "experiment_id": experiment_id,
                    "event_id": event.event_id,
                    "summary": summary,
                    "step_id": step_id,
                    "note": "anomaly auto-logged",
                },
                ctx,
            )

        return AgentResult(
            run_id,
            "completed",
            "rules",
            f"{event.type} handled by rule agent",
            event.event_id,
        )

    # -- safety net -------------------------------------------------------
    def _ensure_traceability(
        self, event: LabEvent, ctx: ToolContext, logger: _StepLogger
    ) -> None:
        if self.db.record_for_event(event.event_id) is None:
            logger.call(
                self.tools,
                "write_experiment_record",
                {
                    "experiment_id": self.experiment_id,
                    "event_id": event.event_id,
                    "summary": f"自动补录：{self._rule_summary(event)}",
                    "note": "safety net: agent did not write a record",
                },
                ctx,
            )
        if event.type == "anomaly" and not self.db.alerts_for_event(event.event_id):
            limits = (event.meta or {}).get("limits") or {}
            logger.call(
                self.tools,
                "raise_alert",
                {
                    "severity": "critical",
                    "message": (
                        f"异常读数 {event.reading} 超出允许范围 "
                        f"[{limits.get('min', '?')}, {limits.get('max', '?')}]"
                    ),
                    "event_id": event.event_id,
                    "session_id": event.session_id,
                },
                ctx,
            )

    @staticmethod
    def _rule_summary(event: LabEvent) -> str:
        reading = event.reading
        if event.type == "session_start":
            if reading is not None:
                return f"开始使用 {event.object_id}，初始读数 {reading}"
            return f"开始使用 {event.object_id}，暂无稳定读数"
        if event.type == "reading_change":
            previous = (event.meta or {}).get("previous_reading")
            if previous is not None:
                return f"{event.object_id} 读数变更：{previous} → {reading}"
            return f"{event.object_id} 读数变更为 {reading}"
        if event.type == "session_end":
            return f"操作结束，{event.object_id} 最终读数 {reading if reading is not None else '无'}"
        if event.type == "anomaly":
            return f"异常：{event.object_id} 读数 {reading} 超出允许范围"
        return f"{event.type} on {event.object_id}"
