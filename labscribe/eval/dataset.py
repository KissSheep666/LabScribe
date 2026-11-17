from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config
from ..paths import resolve_path
from ..vision.synthetic import ScenarioSpec, make_scenario


def build_dataset(
    count: int = 100,
    out_path: str | Path | None = None,
    seed: int = 20260101,
) -> dict[str, Any]:
    """Create a deterministic 100-scenario event dataset.

    Each record contains the full scene specification (readings, sessions,
    occlusions, detector dropouts) and the ground-truth event list.  Twenty
    percent of scenes hide the display during a reading change so occlusion
    recall can be reported separately.
    """
    records: list[dict[str, Any]] = []
    for i in range(count):
        scenario_seed = seed + i
        occluded = i % 5 == 0
        spec = make_scenario(
            seed=scenario_seed,
            scenario_id=f"eval-{i:03d}",
            occluded=occluded,
        )
        records.append(
            {
                "scenario_id": spec.scenario_id,
                "seed": scenario_seed,
                "occluded": occluded,
                "fps": spec.fps,
                "duration": spec.duration,
                "detector_dropouts": spec.detector_dropouts,
                "spec": spec.to_dict(),
                "events": spec.ground_truth,
            }
        )
    payload = {
        "dataset": "labsynthetic-events-v1",
        "count": len(records),
        "seed": seed,
        "occluded_scenarios": sum(1 for r in records if r["occluded"]),
        "records": records,
    }
    path = resolve_path(out_path or "data/eval/annotations.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {
                "dataset": payload["dataset"],
                "count": payload["count"],
                "seed": payload["seed"],
                "occluded_scenarios": payload["occluded_scenarios"],
            },
            ensure_ascii=False,
        ) + "\n")
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"path": str(path), **{k: payload[k] for k in ("dataset", "count", "seed", "occluded_scenarios")}}


def load_dataset(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p = Path(path)
    lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty dataset: {p}")
    meta = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    return meta, records


def ensure_dataset(cfg: Config, force: bool = False) -> Path:
    path = cfg.dataset_path
    if force or not path.exists():
        build_dataset(
            count=int(cfg.get("eval.count", 100)),
            out_path=path,
        )
    return path


def spec_from_record(record: dict[str, Any]) -> ScenarioSpec:
    spec = ScenarioSpec.from_dict(record["spec"])
    if not spec.ground_truth:
        spec.ground_truth = record.get("events", [])
    return spec
