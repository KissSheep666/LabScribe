from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, resolve_path

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"name": "LabScribe", "data_dir": "data"},
    "vision": {
        "detector": "auto",
        "tracker": "auto",
        "ocr": "auto",
        "detector_model": "yolo11n.pt",
        "class_map": {"person": "actor"},
        "object_id_map": {"instrument": "instrument-1"},
        "display_roi": [0.20, 0.52, 0.80, 0.86],
        "ocr_every_n_frames": 3,
        "ocr_min_interval_s": 0.05,
        "reading": {
            "stable_samples": 3,
            "tolerance_abs": 0.25,
            "tolerance_rel": 0.01,
            "max_gap_s": 2.0,
        },
        "confidence": {"min_detection": 0.45, "min_ocr": 0.50},
        "proximity_margin_px": 70,
        "display_occlusion_ratio": 0.30,
        "source": "synthetic",
        "camera_index": 0,
        "video_path": None,
        "image_dir": None,
        "fps": 10,
        "max_frames": 0,
        "realtime": False,
    },
    "synthetic": {"scenario": "demo", "seed": 7, "speed": 2.0},
    "events": {
        "min_confidence": 0.55,
        "session_start_frames": 3,
        "session_end_idle_s": 1.2,
        "reading_tolerance": 0.4,
        "pending_coalesce_s": 2.0,
        "limits": {"instrument-1": {"min": 0.0, "max": 14.0}},
    },
    "agent": {
        "mode": "llm",
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.1,
        "max_steps": 8,
        "timeout_s": 60,
        "max_tokens": 900,
        "experiment_id": "exp-titration",
        "sop_dir": "data/sop",
        "fallback_to_rules": True,
    },
    "store": {
        "db_path": "data/labscribe.db",
        "frames_dir": "data/frames",
        "max_buffered_frames": 240,
    },
    "api": {"host": "127.0.0.1", "port": 8000, "autostart_runtime": False},
    "eval": {
        "dataset_path": "data/eval/annotations.jsonl",
        "count": 100,
        "report_json": "reports/eval_report.json",
        "report_md": "reports/eval_report.md",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Existing environment variables always win."""
    candidates = [path] if path else [PROJECT_ROOT / ".env", Path.cwd() / ".env"]
    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        try:
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


class Config:
    def __init__(self, data: dict[str, Any] | None = None, source_path: Path | None = None):
        self.data = _deep_merge(DEFAULT_CONFIG, data or {})
        self.source_path = source_path
        self._apply_env()

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        load_dotenv()
        cfg_path: Path | None = None
        if path:
            cfg_path = resolve_path(path)
        else:
            candidates = [PROJECT_ROOT / "configs" / "default.yaml", Path.cwd() / "configs" / "default.yaml"]
            cfg_path = next((p for p in candidates if p.exists()), None)
        data: dict[str, Any] = {}
        if cfg_path and cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(data, cfg_path)

    def _apply_env(self) -> None:
        mapping = {
            "LABSCRIBE_AGENT_MODE": "agent.mode",
            "LABSCRIBE_LLM_MODEL": "agent.model",
            "LABSCRIBE_LLM_BASE_URL": "agent.base_url",
            "DEEPSEEK_BASE_URL": "agent.base_url",
            "LABSCRIBE_DB_PATH": "store.db_path",
            "LABSCRIBE_FRAMES_DIR": "store.frames_dir",
            "LABSCRIBE_VISION_SOURCE": "vision.source",
            "LABSCRIBE_EXPERIMENT_ID": "agent.experiment_id",
        }
        for env_key, dotted in mapping.items():
            value = os.environ.get(env_key)
            if value:
                self.set(dotted, value)

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    # -- derived paths ----------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return resolve_path(self.get("app.data_dir", "data"))  # type: ignore[return-value]

    @property
    def db_path(self) -> Path:
        return resolve_path(self.get("store.db_path", "data/labscribe.db"))  # type: ignore[return-value]

    @property
    def frames_dir(self) -> Path:
        return resolve_path(self.get("store.frames_dir", "data/frames"))  # type: ignore[return-value]

    @property
    def sop_dir(self) -> Path:
        return resolve_path(self.get("agent.sop_dir", "data/sop"))  # type: ignore[return-value]

    @property
    def dataset_path(self) -> Path:
        return resolve_path(self.get("eval.dataset_path", "data/eval/annotations.jsonl"))  # type: ignore[return-value]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.db_path.parent, self.frames_dir, self.sop_dir):
            path.mkdir(parents=True, exist_ok=True)

    def api_key(self) -> str | None:
        for key in ("LABSCRIBE_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(key)
            if value:
                return value
        return self.get("agent.api_key")

    def public_dict(self) -> dict[str, Any]:
        """Config safe to expose through the API (no secrets)."""
        data = copy.deepcopy(self.data)
        return data
