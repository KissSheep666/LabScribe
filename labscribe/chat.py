"""Ask questions about the archived lab records.

The chat layer is deliberately read-only: it retrieves evidence from the
database, then asks the LLM to answer *using only that evidence*. Every
answer carries the record / event / frame ids it was built from, so a human
can verify it with one click.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .agent.llm import LLMClient, LLMError
from .config import Config
from .sop import SOPRegistry
from .store.db import Database

_STEP_RE = re.compile(r"\bS\d{1,2}\b", re.I)
_SESSION_RE = re.compile(r"sess-[0-9a-f]{6,}", re.I)
_RECORD_RE = re.compile(r"rec-[0-9a-f]{6,}", re.I)
_EVENT_RE = re.compile(r"evt-[0-9a-f]{6,}", re.I)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Keyword buckets used for cheap intent routing (no embeddings offline).
_INTENTS = [
    ("anomaly", ("异常", "超限", "超出", "告警", "报警", "critical", "warning", "anomaly")),
    ("step", ("步骤", "第几步", "SOP", "流程", "进度", "做到哪")),
    ("session", ("会话", "这次", "本次", "上一次", "最近一次", "多久")),
    ("trend", ("趋势", "历史", "统计", "平均", "几次", "多少", "对比", "规律")),
    ("reading", ("读数", "数是多少", "数值", "多少毫升", "mL", "ml")),
]


@dataclass
class ChatAnswer:
    answer: str
    mode: str                       # llm | search | error
    sources: list[dict[str, Any]] = field(default_factory=list)
    context_summary: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode,
            "sources": self.sources,
            "context_summary": self.context_summary,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


class ArchiveChat:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        sops: SOPRegistry,
        llm: LLMClient | None = None,
        use_llm: bool | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.sops = sops
        self.experiment_id = str(cfg.get("agent.experiment_id", "exp-titration"))
        self.llm = llm
        if use_llm is None:
            use_llm = str(cfg.get("agent.mode", "llm")) != "off" and bool(cfg.api_key())
        if self.llm is None and use_llm:
            self.llm = LLMClient(cfg)

    # -- public -----------------------------------------------------------
    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> ChatAnswer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return ChatAnswer("请先输入一个问题。", "error", error="empty question")
        query = self._plan(question)
        context = self._collect(query)
        sources = context.pop("sources")
        summary = context.get("summary", {})

        if self.llm is None:
            answer = self._search_only_answer(context, sources)
            return ChatAnswer(
                answer,
                "search",
                sources,
                summary,
                (time.perf_counter() - started) * 1000.0,
            )
        try:
            text = self._ask_llm(question, context, history or [])
            mode = "llm"
        except LLMError as exc:
            text = self._search_only_answer(context, sources)
            mode = "search"
            return ChatAnswer(
                text,
                mode,
                sources,
                summary,
                (time.perf_counter() - started) * 1000.0,
                error=f"LLM 不可用，已降级为检索：{exc}",
            )
        return ChatAnswer(
            text,
            mode,
            sources,
            summary,
            (time.perf_counter() - started) * 1000.0,
        )

    # -- query planning ---------------------------------------------------
    def _plan(self, question: str) -> dict[str, Any]:
        intents = [
            name for name, words in _INTENTS if any(w.lower() in question.lower() for w in words)
        ]
        step_ids = [s.upper() for s in _STEP_RE.findall(question)]
        session_ids = _SESSION_RE.findall(question)
        record_ids = _RECORD_RE.findall(question)
        event_ids = _EVENT_RE.findall(question)

        # Pull a plausible reading threshold, e.g. "超过 14 的" or "15.2 正常吗"
        numbers = [float(n) for n in _NUMBER_RE.findall(question)]
        return {
            "question": question,
            "intents": intents,
            "step_ids": step_ids,
            "session_ids": session_ids,
            "record_ids": record_ids,
            "event_ids": event_ids,
            "numbers": numbers,
            "text_terms": self._terms(question),
        }

    @staticmethod
    def _terms(question: str) -> list[str]:
        """Very small Chinese/English term extractor for LIKE search."""
        cleaned = re.sub(r"[？?，,。！!、:：；;\s]+", " ", question)
        tokens: list[str] = []
        for chunk in cleaned.split():
            chunk = chunk.strip()
            if len(chunk) >= 2 and not chunk.isdigit():
                tokens.append(chunk[:12])
        for word in (
            "超限", "异常", "告警", "读数", "滴定", "终点", "准备", "加样",
            "清洗", "复位", "会话", "步骤", "仪器",
        ):
            if word in question:
                tokens.append(word)
        # Keep order, drop duplicates.
        return list(dict.fromkeys(tokens))[:6]

    # -- retrieval --------------------------------------------------------
    def _collect(self, query: dict[str, Any]) -> dict[str, Any]:
        db = self.db
        intents = set(query["intents"])
        sources: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        # 1. Explicit ids always win.
        for rid in query["record_ids"][:3]:
            rec = db.get_record(rid)
            if rec:
                records.append(rec)
        for eid in query["event_ids"][:3]:
            rec = db.record_for_event(eid)
            if rec:
                records.append(rec)

        # 2. Targeted searches.
        for sid in query["session_ids"][:2]:
            records.extend(db.search_records(session_id=sid, limit=30))
        for step in query["step_ids"][:2]:
            records.extend(db.search_records(step_id=step, limit=20))
        for term in query["text_terms"][:4]:
            records.extend(db.search_records(text=term, limit=10))

        # 3. Intent-driven fallbacks.
        if not records or "trend" in intents or "anomaly" in intents:
            records.extend(db.list_records(limit=30))

        # Dedupe, newest first, cap.
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for rec in sorted(records, key=lambda r: r.get("created_at", 0), reverse=True):
            rid = rec.get("record_id")
            if rid and rid not in seen:
                seen.add(rid)
                unique.append(rec)
        records = unique[:20]

        alerts = db.list_alerts(limit=20)
        sessions = db.list_sessions(limit=10)
        sop = self.sops.get(self.experiment_id)
        completed = db.completed_steps(self.experiment_id, f"progress:{self.experiment_id}")
        readings = {
            step.id: db.readings_for_step(self.experiment_id, step.id, limit=50)
            for step in (sop.steps if sop else [])
        }

        # 4. Build citations for the records we will show the model.
        for rec in records:
            event = db.get_event(rec.get("event_id") or "")
            sources.append(
                {
                    "kind": "record",
                    "record_id": rec.get("record_id"),
                    "event_id": rec.get("event_id"),
                    "step_id": rec.get("step_id"),
                    "session_id": rec.get("session_id"),
                    "reading": rec.get("reading"),
                    "note": (rec.get("note") or "").splitlines()[0][:160],
                    "created_at": rec.get("created_at"),
                    "source_frame": rec.get("source_frame"),
                    "frame_url": f"/api/frames/{rec.get('source_frame')}"
                    if rec.get("source_frame")
                    else None,
                    "event_type": (event or {}).get("type"),
                }
            )
        for alert in alerts[:5]:
            sources.append(
                {
                    "kind": "alert",
                    "severity": alert.get("severity"),
                    "message": alert.get("message"),
                    "event_id": alert.get("event_id"),
                    "frame_url": None,
                }
            )

        stats = db.stats()
        summary = {
            "experiment_id": self.experiment_id,
            "records_total": stats.get("records", 0),
            "events_total": stats.get("events", 0),
            "alerts_total": stats.get("alerts", 0),
            "sessions_total": stats.get("sessions", 0),
            "completed_steps": completed,
            "next_step": next((s.id for s in sop.steps if s.id not in completed), None) if sop else None,
        }

        # Deterministic aggregates — the model may quote these, never recompute.
        aggregates: dict[str, Any] = {}
        for step_id, values in readings.items():
            if values:
                aggregates[step_id] = {
                    "samples": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": round(sum(values) / len(values), 2),
                    "last": values[0],
                }
        out_of_range = [
            r for r in records if r.get("reading") is not None and self._out_of_range(r)
        ]

        return {
            "question": query["question"],
            "intents": query["intents"],
            "sop": self._sop_brief(sop),
            "completed_steps": completed,
            "next_step": summary["next_step"],
            "records": [self._record_brief(r) for r in records],
            "recent_alerts": alerts[:5],
            "sessions": sessions[:5],
            "reading_stats": aggregates,
            "out_of_range_records": [self._record_brief(r) for r in out_of_range[:10]],
            "totals": {k: stats.get(k) for k in ("records", "events", "alerts", "sessions")},
            "sources": sources,
            "summary": summary,
        }

    def _out_of_range(self, record: dict[str, Any]) -> bool:
        limits = (self.cfg.get("events.limits", {}) or {}).get(record.get("object_id") or "")
        if not limits or record.get("reading") is None:
            return False
        return not (float(limits["min"]) <= float(record["reading"]) <= float(limits["max"]))

    def _sop_brief(self, sop) -> list[dict[str, Any]]:
        if not sop:
            return []
        return [
            {
                "id": s.id,
                "name": s.name,
                "expected_reading": s.expected_reading,
            }
            for s in sop.steps
        ]

    @staticmethod
    def _record_brief(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_id": record.get("record_id"),
            "event_id": record.get("event_id"),
            "step_id": record.get("step_id"),
            "session_id": record.get("session_id"),
            "reading": record.get("reading"),
            "note": record.get("note"),
            "created_at": record.get("created_at"),
            "source_frame": record.get("source_frame"),
            "agent_mode": (record.get("meta") or {}).get("agent_mode"),
        }

    # -- answering --------------------------------------------------------
    def _ask_llm(
        self, question: str, context: dict[str, Any], history: list[dict[str, str]]
    ) -> str:
        if self.llm is None:
            raise LLMError("LLM not configured")
        system = (
            "你是 LabScribe 的实验档案助手。你只能依据下面提供的档案数据回答，"
            "不得编造读数、时间、步骤或记录编号。\n\n"
            "规则：\n"
            "1. 每个结论都要能对应到具体的 record_id / event_id，回答中带上这些编号。\n"
            "2. 如果档案里没有答案，直接说「档案中没有相关记录」，不要推测。\n"
            "3. 数字必须原样引用档案中的值，不要做未说明的换算。\n"
            "4. 回答用中文，2–5 句话；可以分点，但不要冗长。\n"
            "5. 如果用户问「是否正常 / 是否超限」，先判断，再给出依据（范围或统计）。"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in history[-6:]:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": str(turn["content"])[:2000]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"档案数据（唯一事实来源）：\n"
                    f"{json.dumps(context, ensure_ascii=False, default=str)}"
                ),
            }
        )
        response = self.llm.chat(messages, tools=None)
        return response.content.strip() or "档案中没有足够信息回答这个问题。"

    def _search_only_answer(self, context: dict[str, Any], sources: list[dict[str, Any]]) -> str:
        records = context.get("records") or []
        if not records:
            return "档案中没有相关记录。可以先运行 `python -m labscribe simulate` 生成一批示例数据。"
        lines = [f"检索到 {len(records)} 条相关记录（未启用 LLM，仅返回检索结果）："]
        for rec in records[:8]:
            ts = rec.get("created_at")
            when = time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "—"
            reading = rec.get("reading")
            lines.append(
                f"- [{when}] {rec.get('step_id') or '--'} "
                f"{'读数 ' + str(reading) if reading is not None else '无读数'}："
                f"{(rec.get('note') or '').splitlines()[0][:90]} "
                f"({rec.get('record_id')})"
            )
        return "\n".join(lines)
