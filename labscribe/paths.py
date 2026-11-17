from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root."""
    if path in (None, ""):
        return None
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
