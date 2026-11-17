from __future__ import annotations

from labscribe.eval import build_dataset, load_dataset, run_evaluation, spec_from_record
from labscribe.paths import resolve_path


def test_shipped_dataset_has_100_annotated_events():
    path = resolve_path("data/eval/annotations.jsonl")
    assert path.exists(), "the 100-event annotated dataset must be shipped"
    meta, records = load_dataset(path)
    assert meta["count"] == 100
    assert len(records) == 100
    assert meta["occluded_scenarios"] == 20

    all_events = [event for record in records for event in record["events"]]
    assert len(all_events) >= 400
    types = {event["type"] for event in all_events}
    assert {"session_start", "session_end", "reading_change", "anomaly"} <= types
    for event in all_events:
        assert len(event["window"]) == 2
        assert event["window"][1] >= event["window"][0]
    assert sum(1 for event in all_events if event["occluded"]) > 0


def test_dataset_generation_is_deterministic(tmp_path):
    first = build_dataset(count=5, out_path=tmp_path / "a.jsonl", seed=999)
    second = build_dataset(count=5, out_path=tmp_path / "b.jsonl", seed=999)
    assert first["count"] == second["count"] == 5
    _, records_a = load_dataset(tmp_path / "a.jsonl")
    _, records_b = load_dataset(tmp_path / "b.jsonl")
    assert [r["events"] for r in records_a] == [r["events"] for r in records_b]
    assert [r["spec"]["reading_steps"] for r in records_a] == [
        r["spec"]["reading_steps"] for r in records_b
    ]


def test_evaluation_smoke(cfg, tmp_path):
    dataset = tmp_path / "eval.jsonl"
    build_dataset(count=3, out_path=dataset)
    report = run_evaluation(cfg, dataset_path=dataset, write_reports=False)
    assert report["dataset"]["scenarios"] == 3
    assert report["event_metrics"]["overall"]["tp"] > 0
    assert report["event_metrics"]["overall"]["recall"] >= 80.0
    assert report["ocr"]["raw_samples"] > 0
    assert report["latency"]["mean_s"] is not None
    assert "recall_pct" in report["occlusion"]
    assert report["runtime"]["frames"] > 0
    assert len(report["per_scenario"]) == 3


def test_scenario_spec_roundtrip(tmp_path):
    build_dataset(count=1, out_path=tmp_path / "one.jsonl")
    _, records = load_dataset(tmp_path / "one.jsonl")
    spec = spec_from_record(records[0])
    assert spec.scenario_id.startswith("eval-")
    assert spec.frame_count > 0
    assert spec.ground_truth == records[0]["events"]
