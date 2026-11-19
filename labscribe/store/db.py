from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..schema import ExperimentRecord, LabEvent, PendingCandidate, json_dumps, new_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,
    ts REAL,
    path TEXT,
    width INTEGER,
    height INTEGER,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    type TEXT,
    object_id TEXT,
    object_label TEXT,
    actor_id TEXT,
    action TEXT,
    reading REAL,
    confidence REAL,
    source_frame TEXT,
    ts REAL,
    session_id TEXT,
    status TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE TABLE IF NOT EXISTS pending (
    pending_id TEXT PRIMARY KEY,
    kind TEXT,
    object_id TEXT,
    reason TEXT,
    confidence REAL,
    source_frame TEXT,
    ts REAL,
    status TEXT,
    payload TEXT,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending(status);
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    dedupe_key TEXT UNIQUE,
    experiment_id TEXT,
    session_id TEXT,
    step_id TEXT,
    event_id TEXT,
    source_frame TEXT,
    object_id TEXT,
    reading REAL,
    note TEXT,
    created_at REAL,
    agent_run_id TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_event ON records(event_id);
CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    severity TEXT,
    message TEXT,
    event_id TEXT,
    session_id TEXT,
    created_at REAL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    event_id TEXT,
    mode TEXT,
    status TEXT,
    started REAL,
    ended REAL,
    error TEXT,
    summary TEXT,
    event_type TEXT,
    object_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_event ON agent_runs(event_id);
CREATE TABLE IF NOT EXISTS agent_steps (
    step_id TEXT PRIMARY KEY,
    run_id TEXT,
    seq INTEGER,
    kind TEXT,
    tool TEXT,
    arguments TEXT,
    result TEXT,
    latency_ms REAL,
    created_at REAL,
    content TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON agent_steps(run_id);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    object_id TEXT,
    experiment_id TEXT,
    started REAL,
    ended REAL,
    last_reading REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS step_state (
    experiment_id TEXT,
    session_id TEXT,
    step_id TEXT,
    completed_at REAL,
    PRIMARY KEY (experiment_id, session_id, step_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Database:
    """Thread-safe SQLite store (WAL) for events, records and agent traces."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- low level --------------------------------------------------------
    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _query_one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # -- frames -----------------------------------------------------------
    def save_frame(self, frame_id: str, path: str, ts: float, width: int, height: int, meta: dict | None = None) -> None:
        self._exec(
            "INSERT OR REPLACE INTO frames(frame_id, ts, path, width, height, meta) VALUES (?,?,?,?,?,?)",
            (frame_id, ts, path, width, height, json_dumps(meta or {})),
        )

    def get_frame(self, frame_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM frames WHERE frame_id=?", (frame_id,))
        return self._row_dict(row, "meta") if row else None

    # -- events -----------------------------------------------------------
    def save_event(self, event: LabEvent) -> bool:
        payload = json_dumps(event.to_dict())
        cur = self._exec(
            """INSERT OR IGNORE INTO events
               (event_id, type, object_id, object_label, actor_id, action, reading,
                confidence, source_frame, ts, session_id, status, payload)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id,
                event.type,
                event.object_id,
                event.object_label,
                event.actor_id,
                event.action,
                event.reading,
                event.confidence,
                event.source_frame,
                event.ts,
                event.session_id,
                event.status,
                payload,
            ),
        )
        return cur.rowcount > 0

    def get_event(self, event_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM events WHERE event_id=?", (event_id,))
        return self._event_dict(row) if row else None

    def list_events(
        self,
        limit: int = 100,
        since: float | None = None,
        event_type: str | None = None,
        session_id: str | None = None,
        object_id: str | None = None,
    ) -> list[dict]:
        clauses = []
        params: list[Any] = []
        if since is not None:
            clauses.append("ts > ?")
            params.append(since)
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if object_id:
            clauses.append("object_id = ?")
            params.append(object_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ?",
            (*params, int(limit)),
        )
        return [self._event_dict(r) for r in rows]

    def count_events(self) -> dict[str, int]:
        rows = self._query("SELECT type, COUNT(*) AS n FROM events GROUP BY type")
        return {r["type"]: int(r["n"]) for r in rows}

    # -- pending ----------------------------------------------------------
    def add_pending(self, candidate: PendingCandidate, coalesce_s: float = 2.0) -> dict:
        now = time.time()
        draft = candidate.draft_event()
        payload = json_dumps(draft.to_dict())
        with self._lock:
            row = self._query_one(
                """SELECT * FROM pending
                   WHERE status='open' AND object_id=? AND kind=? AND ts >= ?
                   ORDER BY ts DESC LIMIT 1""",
                (candidate.object_id, candidate.kind, candidate.ts - coalesce_s),
            )
            if row is not None:
                confidence = max(float(row["confidence"]), candidate.confidence)
                self._exec(
                    """UPDATE pending SET reason=?, confidence=?, source_frame=?, ts=?, payload=?,
                       updated_at=? WHERE pending_id=?""",
                    (
                        candidate.reason,
                        confidence,
                        candidate.source_frame,
                        candidate.ts,
                        payload,
                        now,
                        row["pending_id"],
                    ),
                )
                updated = self._query_one("SELECT * FROM pending WHERE pending_id=?", (row["pending_id"],))
                return self._pending_dict(updated)  # type: ignore[arg-type]
            pending_id = new_id("pnd")
            self._exec(
                """INSERT INTO pending
                   (pending_id, kind, object_id, reason, confidence, source_frame, ts, status, payload, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    pending_id,
                    candidate.kind,
                    candidate.object_id,
                    candidate.reason,
                    candidate.confidence,
                    candidate.source_frame,
                    candidate.ts,
                    "open",
                    payload,
                    now,
                ),
            )
            created = self._query_one("SELECT * FROM pending WHERE pending_id=?", (pending_id,))
            return self._pending_dict(created)  # type: ignore[arg-type]

    def list_pending(self, status: str = "open", limit: int = 100) -> list[dict]:
        rows = self._query(
            "SELECT * FROM pending WHERE status=? ORDER BY ts DESC LIMIT ?", (status, int(limit))
        )
        return [self._pending_dict(r) for r in rows]

    def get_pending(self, pending_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM pending WHERE pending_id=?", (pending_id,))
        return self._pending_dict(row) if row else None

    def resolve_pending(self, pending_id: str, status: str = "confirmed") -> bool:
        cur = self._exec(
            "UPDATE pending SET status=?, updated_at=? WHERE pending_id=? AND status='open'",
            (status, time.time(), pending_id),
        )
        return cur.rowcount > 0

    # -- records ----------------------------------------------------------
    def save_record(self, record: ExperimentRecord, dedupe_key: str) -> tuple[dict, bool]:
        payload = json_dumps(record.to_dict())
        with self._lock:
            cur = self._exec(
                """INSERT OR IGNORE INTO records
                   (record_id, dedupe_key, experiment_id, session_id, step_id, event_id, source_frame,
                    object_id, reading, note, created_at, agent_run_id, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.record_id,
                    dedupe_key,
                    record.experiment_id,
                    record.session_id,
                    record.step_id,
                    record.event_id,
                    record.source_frame,
                    record.object_id,
                    record.reading,
                    record.note,
                    record.created_at,
                    record.agent_run_id,
                    payload,
                ),
            )
            if cur.rowcount > 0:
                return record.to_dict(), True
            row = self._query_one("SELECT * FROM records WHERE dedupe_key=?", (dedupe_key,))
            return (json.loads(row["payload"]) if row else record.to_dict()), False

    def get_record(self, record_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM records WHERE record_id=?", (record_id,))
        return json.loads(row["payload"]) if row else None

    def list_records(self, limit: int = 100, session_id: str | None = None) -> list[dict]:
        if session_id:
            rows = self._query(
                "SELECT * FROM records WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, int(limit)),
            )
        else:
            rows = self._query("SELECT * FROM records ORDER BY created_at DESC LIMIT ?", (int(limit),))
        return [json.loads(r["payload"]) for r in rows]

    def search_records(
        self,
        text: str | None = None,
        session_id: str | None = None,
        step_id: str | None = None,
        object_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Filter archived records. ``text`` matches note/step/event/object."""
        clauses: list[str] = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if step_id:
            clauses.append("step_id = ?")
            params.append(step_id)
        if object_id:
            clauses.append("object_id = ?")
            params.append(object_id)
        if since_ts is not None:
            clauses.append("created_at >= ?")
            params.append(since_ts)
        if until_ts is not None:
            clauses.append("created_at <= ?")
            params.append(until_ts)
        if text:
            like = f"%{text}%"
            clauses.append(
                "(note LIKE ? OR step_id LIKE ? OR event_id LIKE ? OR object_id LIKE ?)"
            )
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM records {where} ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        )
        return [json.loads(r["payload"]) for r in rows]

    def readings_for_step(self, experiment_id: str, step_id: str, limit: int = 200) -> list[float]:
        rows = self._query(
            """SELECT reading FROM records
               WHERE experiment_id=? AND step_id=? AND reading IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (experiment_id, step_id, int(limit)),
        )
        return [float(r["reading"]) for r in rows]

    def record_for_event(self, event_id: str, step_id: str | None = None) -> dict | None:
        if step_id:
            row = self._query_one(
                "SELECT * FROM records WHERE event_id=? AND step_id=?", (event_id, step_id)
            )
        else:
            row = self._query_one(
                "SELECT * FROM records WHERE event_id=? ORDER BY created_at LIMIT 1", (event_id,)
            )
        return json.loads(row["payload"]) if row else None

    # -- alerts -----------------------------------------------------------
    def save_alert(
        self,
        severity: str,
        message: str,
        event_id: str | None = None,
        session_id: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        alert_id = new_id("alt")
        payload = {
            "alert_id": alert_id,
            "severity": severity,
            "message": message,
            "event_id": event_id,
            "session_id": session_id,
            "created_at": time.time(),
            "meta": meta or {},
        }
        self._exec(
            """INSERT INTO alerts(alert_id, severity, message, event_id, session_id, created_at, payload)
               VALUES (?,?,?,?,?,?,?)""",
            (
                alert_id,
                severity,
                message,
                event_id,
                session_id,
                payload["created_at"],
                json_dumps(payload),
            ),
        )
        return payload

    def list_alerts(self, limit: int = 100) -> list[dict]:
        rows = self._query("SELECT payload FROM alerts ORDER BY created_at DESC LIMIT ?", (int(limit),))
        return [json.loads(r["payload"]) for r in rows]

    def alerts_for_event(self, event_id: str) -> list[dict]:
        rows = self._query("SELECT payload FROM alerts WHERE event_id=?", (event_id,))
        return [json.loads(r["payload"]) for r in rows]

    # -- sessions ---------------------------------------------------------
    def upsert_session(
        self, session_id: str, object_id: str, started: float, experiment_id: str | None = None
    ) -> None:
        now = time.time()
        self._exec(
            """INSERT INTO sessions(session_id, object_id, experiment_id, started, ended, last_reading, updated_at)
               VALUES (?,?,?,?,NULL,NULL,?)
               ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at""",
            (session_id, object_id, experiment_id, started, now),
        )

    def touch_session(self, session_id: str, reading: float | None = None, ended: float | None = None) -> None:
        if reading is not None:
            self._exec(
                "UPDATE sessions SET last_reading=?, updated_at=? WHERE session_id=?",
                (reading, time.time(), session_id),
            )
        if ended is not None:
            self._exec(
                "UPDATE sessions SET ended=?, updated_at=? WHERE session_id=?",
                (ended, time.time(), session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self._query(
            "SELECT * FROM sessions ORDER BY started DESC LIMIT ?", (int(limit),)
        )
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM sessions WHERE session_id=?", (session_id,))
        return dict(row) if row else None

    # -- SOP step state ---------------------------------------------------
    def mark_step_complete(
        self, experiment_id: str, session_id: str, step_id: str, completed_at: float
    ) -> None:
        self._exec(
            """INSERT OR REPLACE INTO step_state(experiment_id, session_id, step_id, completed_at)
               VALUES (?,?,?,?)""",
            (experiment_id, session_id, step_id, completed_at),
        )

    def completed_steps(self, experiment_id: str, session_id: str) -> list[str]:
        rows = self._query(
            "SELECT step_id FROM step_state WHERE experiment_id=? AND session_id=? ORDER BY completed_at",
            (experiment_id, session_id),
        )
        return [r["step_id"] for r in rows]

    # -- agent traces -----------------------------------------------------
    def save_run(
        self,
        run_id: str,
        event_id: str | None,
        mode: str,
        started: float,
        event_type: str | None = None,
        object_id: str | None = None,
    ) -> None:
        self._exec(
            """INSERT OR REPLACE INTO agent_runs
               (run_id, event_id, mode, status, started, ended, error, summary, event_type, object_id)
               VALUES (?,?,?,?,?,NULL,NULL,NULL,?,?)""",
            (run_id, event_id, mode, "running", started, event_type, object_id),
        )

    def finish_run(self, run_id: str, status: str, summary: str = "", error: str | None = None) -> None:
        self._exec(
            "UPDATE agent_runs SET status=?, summary=?, error=?, ended=? WHERE run_id=?",
            (status, summary, error, time.time(), run_id),
        )

    def save_step(
        self,
        run_id: str,
        seq: int,
        kind: str,
        tool: str | None = None,
        arguments: Any = None,
        result: Any = None,
        latency_ms: float | None = None,
        content: str | None = None,
    ) -> dict:
        step_id = new_id("stp")
        created = time.time()
        payload = {
            "step_id": step_id,
            "run_id": run_id,
            "seq": seq,
            "kind": kind,
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "content": content,
            "created_at": created,
        }
        self._exec(
            """INSERT INTO agent_steps
               (step_id, run_id, seq, kind, tool, arguments, result, latency_ms, created_at, content)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                step_id,
                run_id,
                seq,
                kind,
                tool,
                json_dumps(arguments) if arguments is not None else None,
                json_dumps(result) if result is not None else None,
                payload["latency_ms"],
                created,
                content,
            ),
        )
        return payload

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self._query("SELECT * FROM agent_runs ORDER BY started DESC LIMIT ?", (int(limit),))
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        row = self._query_one("SELECT * FROM agent_runs WHERE run_id=?", (run_id,))
        if not row:
            return None
        run = dict(row)
        steps = self._query("SELECT * FROM agent_steps WHERE run_id=? ORDER BY seq", (run_id,))
        run["steps"] = [
            {
                "step_id": s["step_id"],
                "seq": s["seq"],
                "kind": s["kind"],
                "tool": s["tool"],
                "arguments": json.loads(s["arguments"]) if s["arguments"] else None,
                "result": json.loads(s["result"]) if s["result"] else None,
                "latency_ms": s["latency_ms"],
                "content": s["content"],
                "created_at": s["created_at"],
            }
            for s in steps
        ]
        return run

    def runs_for_event(self, event_id: str) -> list[dict]:
        rows = self._query("SELECT * FROM agent_runs WHERE event_id=? ORDER BY started", (event_id,))
        return [dict(r) for r in rows]

    # -- stats ------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        def count(table: str, where: str = "") -> int:
            row = self._query_one(f"SELECT COUNT(*) AS n FROM {table} {where}")
            return int(row["n"]) if row else 0

        latency_row = self._query_one("SELECT AVG(latency_ms) AS avg_ms FROM agent_steps WHERE latency_ms IS NOT NULL")
        return {
            "frames": count("frames"),
            "events": count("events"),
            "events_by_type": self.count_events(),
            "pending_open": count("pending", "WHERE status='open'"),
            "records": count("records"),
            "alerts": count("alerts"),
            "sessions": count("sessions"),
            "agent_runs": count("agent_runs"),
            "agent_avg_step_latency_ms": round(float(latency_row["avg_ms"] or 0.0), 2)
            if latency_row
            else 0.0,
        }

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _row_dict(row: sqlite3.Row, json_field: str | None = None) -> dict:
        data = dict(row)
        if json_field and data.get(json_field):
            try:
                data[json_field] = json.loads(data[json_field])
            except (TypeError, json.JSONDecodeError):
                pass
        return data

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict:
        if row["payload"]:
            try:
                return json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                pass
        return dict(row)

    @staticmethod
    def _pending_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        if data.get("payload"):
            try:
                data["draft_event"] = json.loads(data["payload"])
            except (TypeError, json.JSONDecodeError):
                data["draft_event"] = None
        data.pop("payload", None)
        return data

    def close(self) -> None:
        with self._lock:
            self._conn.close()
