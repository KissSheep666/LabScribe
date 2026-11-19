from __future__ import annotations

from fastapi.testclient import TestClient

from labscribe.api.app import create_app
from labscribe.chat import ArchiveChat
from labscribe.schema import ExperimentRecord, LabEvent, new_id
from labscribe.store import Database


def _seed(db: Database, readings=(0.0, 3.5, 7.2, 15.2)):
    events = [
        ("session_start", 0.0),
        ("reading_change", 3.5),
        ("reading_change", 7.2),
        ("anomaly", 15.2),
    ]
    for step, (etype, reading) in zip(["S1", "S2", "S3", "S4"], events):
        event = LabEvent(
            event_id=new_id("evt"),
            type=etype,
            ts=10.0,
            object_id="instrument-1",
            object_label="instrument",
            actor_id="actor-1",
            action="interacting",
            reading=reading,
            confidence=0.9,
            source_frame=f"demo-{step.lower()}",
            session_id="sess-chat",
        )
        db.save_event(event)
        record = ExperimentRecord(
            record_id=new_id("rec"),
            experiment_id="exp-titration",
            session_id="sess-chat",
            step_id=step,
            event_id=event.event_id,
            source_frame=event.source_frame,
            object_id="instrument-1",
            reading=reading,
            note=f"{step} 读数 {reading}",
            created_at=1000.0 + len(step),
            meta={"agent_mode": "rules"},
        )
        db.save_record(record, f"{event.event_id}|{step}")
    db.save_alert("critical", "异常读数 15.2 超出允许范围 [0.0, 14.0]",
                  event_id=db.list_events(1)[0]["event_id"], session_id="sess-chat")


def test_plan_extracts_steps_and_intents(cfg, db, sops):
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    plan = chat._plan("S4 的读数超过 14 了吗？跟上一次会话对比")
    assert "S4" in plan["step_ids"]
    assert "reading" in plan["intents"]
    assert "trend" in plan["intents"]
    assert 14.0 in plan["numbers"]


def test_retrieval_finds_matching_records(cfg, db, sops):
    _seed(db)
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    context = chat._collect(chat._plan("S4 读数是多少"))
    steps = {r["step_id"] for r in context["records"]}
    assert "S4" in steps
    assert any(s["kind"] == "record" for s in context["sources"])
    assert context["summary"]["records_total"] == 4


def test_aggregates_are_computed_deterministically(cfg, db, sops):
    _seed(db)
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    context = chat._collect(chat._plan("S4 历史读数范围"))
    stats = context["reading_stats"]["S4"]
    assert stats["min"] == 15.2 and stats["max"] == 15.2 and stats["samples"] == 1
    assert context["out_of_range_records"]


def test_search_mode_without_llm(cfg, db, sops):
    _seed(db)
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    answer = chat.ask("有没有超限的读数？")
    assert answer.mode == "search"
    assert "检索到" in answer.answer
    assert answer.sources


def test_empty_archive_answers_gracefully(cfg, db, sops):
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    answer = chat.ask("最近一次实验做到哪一步？")
    assert answer.mode == "search"
    assert "没有相关记录" in answer.answer


def test_empty_question_rejected(cfg, db, sops):
    chat = ArchiveChat(cfg, db, sops, use_llm=False)
    answer = chat.ask("   ")
    assert answer.mode == "error"
    assert answer.error


def test_llm_failure_degrades_to_search(cfg, db, sops):
    from labscribe.agent.llm import LLMError

    class Broken:
        def chat(self, *a, **k):
            raise LLMError("no key")

    _seed(db)
    chat = ArchiveChat(cfg, db, sops, llm=Broken())
    answer = chat.ask("有没有超限的读数？")
    assert answer.mode == "search"
    assert answer.error and "no key" in answer.error
    assert answer.sources

class _FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat(self, messages, tools=None, **kwargs):
        from labscribe.agent.llm import LLMResponse

        self.prompts.append(messages)
        return LLMResponse(content="依据 rec-x，读数 15.2 超出 [0.0, 14.0]。", latency_ms=5.0)


def test_llm_receives_retrieved_context_only(cfg, db, sops):
    _seed(db)
    fake = _FakeLLM()
    chat = ArchiveChat(cfg, db, sops, llm=fake)
    answer = chat.ask("有没有超限的读数？")
    assert answer.mode == "llm"
    assert "15.2" in answer.answer
    user_prompt = fake.prompts[-1][-1]["content"]
    assert "rec-" in user_prompt          # retrieved records are embedded
    assert "唯一事实来源" in user_prompt
    assert "不得编造" in fake.prompts[-1][0]["content"]


def test_chat_endpoints(cfg, db, sops):
    _seed(db)
    runtime_app = create_app(cfg)
    with TestClient(runtime_app) as client:
        assert client.get("/chat").status_code == 200
        assert "<title>实验档案问答" in client.get("/chat").text
        suggest = client.get("/api/chat/suggestions").json()
        assert suggest["suggestions"]
        assert isinstance(suggest["has_data"], bool)
        resp = client.post(
            "/api/chat", json={"question": "有没有超限的读数？", "history": []}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["mode"] in ("llm", "search")
        assert isinstance(data["sources"], list)


def test_chat_rejects_blank_question(cfg):
    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"question": "", "history": []})
        assert resp.status_code == 422
