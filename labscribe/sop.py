from __future__ import annotations

from pathlib import Path

import yaml

from .schema import SOP, SOPStep


class SOPRegistry:
    """Loads SOP YAML files from a directory, keyed by experiment_id."""

    def __init__(self, sop_dir: str | Path):
        self.sop_dir = Path(sop_dir)
        self._cache: dict[str, SOP] = {}
        self.reload()

    def reload(self) -> None:
        self._cache.clear()
        if not self.sop_dir.exists():
            return
        for path in sorted(self.sop_dir.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                sop = _sop_from_dict(data)
                self._cache[sop.experiment_id] = sop
            except Exception:  # noqa: BLE001 - a bad SOP must not kill the server
                continue

    def get(self, experiment_id: str) -> SOP | None:
        return self._cache.get(experiment_id)

    def require(self, experiment_id: str) -> SOP:
        sop = self.get(experiment_id)
        if sop is None:
            raise KeyError(f"SOP not found for experiment '{experiment_id}'")
        return sop

    def list(self) -> list[dict]:
        return [
            {"experiment_id": s.experiment_id, "name": s.name, "version": s.version, "steps": len(s.steps)}
            for s in self._cache.values()
        ]


def _sop_from_dict(data: dict) -> SOP:
    steps = []
    for raw in data.get("steps") or []:
        steps.append(
            SOPStep(
                id=str(raw["id"]),
                name=str(raw.get("name", raw["id"])),
                instruction=str(raw.get("instruction", "")),
                instrument=raw.get("instrument"),
                expected_reading=raw.get("expected_reading"),
                duration_s=raw.get("duration_s"),
            )
        )
    if not steps:
        raise ValueError("SOP has no steps")
    return SOP(
        experiment_id=str(data["experiment_id"]),
        name=str(data.get("name", data["experiment_id"])),
        version=str(data.get("version", "1.0")),
        steps=steps,
    )
