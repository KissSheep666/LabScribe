from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..events.state_machine import RuleStateMachine
from ..paths import resolve_path
from ..schema import LabEvent
from ..vision.detector import Detector, MarkerDetector
from ..vision.ocr import TemplateOCR
from ..vision.pipeline import VisionPipeline
from ..vision.synthetic import ScenarioSpec, SyntheticSource
from ..vision.tracker import IoUTracker
from .dataset import ensure_dataset, load_dataset, spec_from_record

SIMPLE_MATCH_TOLERANCE = {
    "session_start": 1.0,
    "reading_change": 1.5,
    "session_end": 1.5,
    "anomaly": 1.5,
}


class DropoutDetector(Detector):
    """Simulates frames where the detector misses objects entirely."""

    def __init__(self, inner: Detector, frames: set[int]):
        self.inner = inner
        self.frames = frames

    def detect(self, frame):
        if frame.index in self.frames:
            return []
        return self.inner.detect(frame)

    def close(self) -> None:
        self.inner.close()


@dataclass
class ScenarioResult:
    scenario_id: str
    occluded: bool
    gt_events: list[dict[str, Any]]
    predicted: list[dict[str, Any]]
    pending: int
    frames: int
    ocr_samples: int
    ocr_correct: int
    target_samples: int
    target_correct: int
    processing_ms: float
    dropped_frames: int


@dataclass
class EvalSummary:
    report: dict[str, Any] = field(default_factory=dict)
    scenarios: list[ScenarioResult] = field(default_factory=list)


def run_scenario(spec: ScenarioSpec, cfg: Config) -> ScenarioResult:
    source = SyntheticSource(spec)
    detector: Detector = DropoutDetector(
        MarkerDetector(cfg), set(spec.detector_dropouts)
    )
    pipeline = VisionPipeline(
        cfg, detector=detector, tracker=IoUTracker(), ocr=TemplateOCR(cfg)
    )
    machine = RuleStateMachine(cfg)
    events: list[dict[str, Any]] = []
    pending_count = 0
    frames = 0
    processing_s = 0.0
    target_samples = 0
    target_correct = 0
    for frame in source.iter_frames():
        frames += 1
        started = time.perf_counter()
        observations = pipeline.process(frame)
        processing_s += time.perf_counter() - started
        for observation in observations:
            output = machine.process(observation)
            for event in output.events:
                events.append(
                    {
                        "event_id": event.event_id,
                        "type": event.type,
                        "ts": event.ts,
                        "reading": event.reading,
                        "confidence": event.confidence,
                    }
                )
            pending_count += len(output.pending)
        # Raw OCR accuracy on visible, non-blanked frames.
        for trace in pipeline.trace:
            if trace.get("_counted"):
                continue
            trace["_counted"] = True
            if not trace.get("ok"):
                continue
            ts = float(trace["ts"])
            if spec.occluded_at(ts) or trace["frame_index"] in set(spec.ocr_dropouts):
                continue
            target_samples += 1
            ground_truth_value = spec.reading_at(ts)
            if trace.get("value") is not None and abs(trace["value"] - ground_truth_value) < 1e-6:
                target_correct += 1
    ocr_samples = len(pipeline.trace)
    ocr_correct = sum(
        1
        for t in pipeline.trace
        if t.get("ok")
        and t.get("value") is not None
        and abs(t["value"] - spec.reading_at(float(t["ts"]))) < 1e-6
    )
    return ScenarioResult(
        scenario_id=spec.scenario_id,
        occluded=bool(spec.notes.get("occluded")),
        gt_events=list(spec.ground_truth),
        predicted=events,
        pending=pending_count,
        frames=frames,
        ocr_samples=ocr_samples,
        ocr_correct=ocr_correct,
        target_samples=target_samples,
        target_correct=target_correct,
        processing_ms=processing_s * 1000.0,
        dropped_frames=len(spec.detector_dropouts),
    )


def _match_scenario(result: ScenarioResult) -> dict[str, Any]:
    """Greedy one-to-one matching by type, inside the ground-truth window."""
    predictions = sorted(result.predicted, key=lambda e: e["ts"])
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    latencies: list[float] = []
    reading_correct = 0
    reading_compared = 0
    for gt in result.gt_events:
        candidates = [
            (idx, pred)
            for idx, pred in enumerate(predictions)
            if idx not in used and pred["type"] == gt["type"]
        ]
        if not candidates:
            missed.append(gt)
            continue
        lo, hi = gt["window"]
        in_window = [(i, p) for i, p in candidates if lo <= p["ts"] <= hi]
        pool = in_window or candidates
        idx, pred = min(pool, key=lambda item: abs(item[1]["ts"] - gt["t"]))
        if not in_window:
            # Outside the annotation window is treated as a miss + false positive.
            missed.append(gt)
            continue
        used.add(idx)
        latency = max(0.0, pred["ts"] - gt["t"])
        latencies.append(latency)
        reading_ok = None
        if gt["type"] in ("reading_change", "anomaly") and gt.get("reading") is not None:
            reading_compared += 1
            reading_ok = pred.get("reading") is not None and abs(
                float(pred["reading"]) - float(gt["reading"])
            ) <= 0.15
            if reading_ok:
                reading_correct += 1
        matches.append(
            {
                "gt": gt,
                "pred": pred,
                "latency_s": round(latency, 3),
                "reading_ok": reading_ok,
            }
        )
    false_positives = [p for i, p in enumerate(predictions) if i not in used]
    return {
        "matches": matches,
        "missed": missed,
        "false_positives": false_positives,
        "latencies": latencies,
        "reading_compared": reading_compared,
        "reading_correct": reading_correct,
    }


def _pct(value: float) -> float:
    return round(100.0 * value, 2)


def run_evaluation(
    cfg: Config | None = None,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    write_reports: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    cfg = cfg or Config.load()
    path = resolve_path(dataset_path) if dataset_path else ensure_dataset(cfg)
    meta, records = load_dataset(path)
    if limit:
        records = records[:limit]

    results: list[ScenarioResult] = []
    started = time.perf_counter()
    for i, record in enumerate(records, 1):
        spec = spec_from_record(record)
        results.append(run_scenario(spec, cfg))
        if verbose and (i % 10 == 0 or i == len(records)):
            elapsed = time.perf_counter() - started
            print(f"[eval] {i}/{len(records)} scenarios ({elapsed:.1f}s)", flush=True)

    by_type: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    all_latencies: list[float] = []
    latencies_by_type: dict[str, list[float]] = {}
    gt_occluded = 0
    matched_occluded = 0
    missed_occluded = 0
    reading_compared = 0
    reading_correct = 0
    ocr_samples = total_target_samples = total_target_correct = 0
    ocr_correct = 0
    total_frames = 0
    total_processing_ms = 0.0
    total_pending = 0
    per_scenario: list[dict[str, Any]] = []

    for result in results:
        match_info = _match_scenario(result)
        gt_count = len(result.gt_events)
        matched = len(match_info["matches"])
        fp = len(match_info["false_positives"])
        fn = gt_count - matched
        total_tp += matched
        total_fp += fp
        total_fn += fn
        all_latencies.extend(match_info["latencies"])
        reading_compared += match_info["reading_compared"]
        reading_correct += match_info["reading_correct"]
        ocr_samples += result.ocr_samples
        ocr_correct += result.ocr_correct
        total_target_samples += result.target_samples
        total_target_correct += result.target_correct
        total_frames += result.frames
        total_processing_ms += result.processing_ms
        total_pending += result.pending

        for gt in result.gt_events:
            if gt.get("occluded"):
                gt_occluded += 1
        for match in match_info["matches"]:
            if match["gt"].get("occluded"):
                matched_occluded += 1
        for missed in match_info["missed"]:
            if missed.get("occluded"):
                missed_occluded += 1

        for event_type in set(e["type"] for e in result.gt_events) | set(
            e["type"] for e in result.predicted
        ):
            bucket = by_type.setdefault(event_type, {"tp": 0, "fp": 0, "fn": 0, "latencies": []})
            for match in match_info["matches"]:
                if match["gt"]["type"] == event_type:
                    bucket["tp"] += 1
                    bucket["latencies"].append(match["latency_s"])
                    latencies_by_type.setdefault(event_type, []).append(match["latency_s"])
            bucket["fp"] += sum(1 for e in match_info["false_positives"] if e["type"] == event_type)
            bucket["fn"] += sum(1 for e in match_info["missed"] if e["type"] == event_type)

        per_scenario.append(
            {
                "scenario_id": result.scenario_id,
                "occluded": result.occluded,
                "gt": gt_count,
                "matched": matched,
                "false_positives": fp,
                "missed": fn,
                "pending": result.pending,
                "reading_ok": f"{match_info['reading_correct']}/{match_info['reading_compared']}"
                if match_info["reading_compared"]
                else "n/a",
                "latency_mean_s": round(statistics.mean(match_info["latencies"]), 3)
                if match_info["latencies"]
                else None,
            }
        )

    def metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _pct(precision),
            "recall": _pct(recall),
            "f1": _pct(f1),
        }

    overall = metrics(total_tp, total_fp, total_fn)
    by_type_out: dict[str, Any] = {}
    for event_type, bucket in sorted(by_type.items()):
        m = metrics(bucket["tp"], bucket["fp"], bucket["fn"])
        m["latency_mean_s"] = (
            round(statistics.mean(bucket["latencies"]), 3) if bucket["latencies"] else None
        )
        m["latency_p95_s"] = (
            round(_percentile(bucket["latencies"], 95), 3) if bucket["latencies"] else None
        )
        by_type_out[event_type] = m

    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(path),
            "name": meta.get("dataset"),
            "scenarios": len(results),
            "gt_events": sum(len(r.gt_events) for r in results),
            "occluded_gt_events": gt_occluded,
            "seed": meta.get("seed"),
        },
        "system": {
            "detector": "marker",
            "tracker": "iou",
            "ocr": "template",
            "min_event_confidence": float(cfg.get("events.min_confidence", 0.55)),
            "stable_samples": int(cfg.get("vision.reading.stable_samples", 3)),
            "ocr_every_n_frames": int(cfg.get("vision.ocr_every_n_frames", 3)),
        },
        "event_metrics": {
            "overall": {**overall, "accuracy_pct": overall["recall"]},
            "by_type": by_type_out,
        },
        "ocr": {
            "raw_samples": ocr_samples,
            "raw_sample_accuracy_pct": _pct(ocr_correct / ocr_samples) if ocr_samples else 0.0,
            "visible_frames_checked": total_target_samples,
            "visible_frame_accuracy_pct": _pct(total_target_correct / total_target_samples)
            if total_target_samples
            else 0.0,
            "event_reading_accuracy_pct": _pct(reading_correct / reading_compared)
            if reading_compared
            else 0.0,
            "event_readings_compared": reading_compared,
        },
        "latency": {
            "mean_s": round(statistics.mean(all_latencies), 3) if all_latencies else None,
            "p95_s": round(_percentile(all_latencies, 95), 3) if all_latencies else None,
            "by_type": {
                t: {
                    "mean_s": round(statistics.mean(v), 3),
                    "p95_s": round(_percentile(v, 95), 3),
                    "n": len(v),
                }
                for t, v in latencies_by_type.items()
            },
        },
        "occlusion": {
            "gt_events": gt_occluded,
            "matched": matched_occluded,
            "missed": missed_occluded,
            "recall_pct": _pct(matched_occluded / gt_occluded) if gt_occluded else 0.0,
        },
        "runtime": {
            "frames": total_frames,
            "scenarios": len(results),
            "pending_candidates": total_pending,
            "mean_frame_processing_ms": round(total_processing_ms / total_frames, 3)
            if total_frames
            else 0.0,
            "eval_wall_clock_s": round(elapsed, 2),
        },
        "per_scenario": per_scenario,
    }

    if write_reports:
        json_path = resolve_path(cfg.get("eval.report_json", "reports/eval_report.json"))
        md_path = resolve_path(cfg.get("eval.report_md", "reports/eval_report.md"))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LabScribe Evaluation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Dataset",
        "",
        f"- scenarios: **{report['dataset']['scenarios']}**",
        f"- ground-truth events: **{report['dataset']['gt_events']}**",
        f"- occluded ground-truth events: **{report['dataset']['occluded_gt_events']}**",
        f"- backends: `{report['system']['detector']}` / `{report['system']['tracker']}` / `{report['system']['ocr']}`",
        "",
        "## Event detection",
        "",
        "| type | TP | FP | FN | precision | recall | F1 | mean latency (s) | p95 latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = report["event_metrics"]["overall"]
    lines.append(
        f"| **overall** | {overall['tp']} | {overall['fp']} | {overall['fn']} | "
        f"{overall['precision']}% | {overall['recall']}% | {overall['f1']}% | "
        f"{report['latency']['mean_s']} | {report['latency']['p95_s']} |"
    )
    for event_type, m in report["event_metrics"]["by_type"].items():
        lines.append(
            f"| {event_type} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']}% | "
            f"{m['recall']}% | {m['f1']}% | {m['latency_mean_s']} | {m['latency_p95_s']} |"
        )
    lines += [
        "",
        "## OCR / readings",
        "",
        f"- raw OCR sample accuracy: **{report['ocr']['raw_sample_accuracy_pct']}%** "
        f"({report['ocr']['raw_samples']} samples)",
        f"- visible-frame reading accuracy: **{report['ocr']['visible_frame_accuracy_pct']}%** "
        f"({report['ocr']['visible_frames_checked']} frames)",
        f"- event reading accuracy: **{report['ocr']['event_reading_accuracy_pct']}%** "
        f"({report['ocr']['event_readings_compared']} readings compared)",
        "",
        "## Occlusion",
        "",
        f"- occluded GT events: **{report['occlusion']['gt_events']}**",
        f"- matched: **{report['occlusion']['matched']}**, missed: **{report['occlusion']['missed']}**",
        f"- occlusion recall: **{report['occlusion']['recall_pct']}%**",
        "",
        "## Runtime",
        "",
        f"- frames processed: **{report['runtime']['frames']}**",
        f"- mean frame processing: **{report['runtime']['mean_frame_processing_ms']} ms**",
        f"- pending (low-confidence) candidates: **{report['runtime']['pending_candidates']}**",
        f"- eval wall clock: **{report['runtime']['eval_wall_clock_s']} s**",
        "",
    ]
    return "\n".join(lines)
