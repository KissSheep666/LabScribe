from __future__ import annotations

import pytest

from labscribe.agent.tools import LabTools
from labscribe.config import Config
from labscribe.sop import SOPRegistry
from labscribe.store import Database


@pytest.fixture
def cfg(tmp_path) -> Config:
    config = Config.load(
        overrides={
            "store": {
                "db_path": str(tmp_path / "labscribe.db"),
                "frames_dir": str(tmp_path / "frames"),
            },
            "agent": {"mode": "rules", "fallback_to_rules": True},
            "vision": {
                "detector": "marker",
                "tracker": "iou",
                "ocr": "template",
                "source": "synthetic",
                "realtime": False,
            },
            "api": {"autostart_runtime": False},
        }
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def db(cfg) -> Database:
    database = Database(cfg.db_path)
    yield database
    database.close()


@pytest.fixture
def sops(cfg) -> SOPRegistry:
    return SOPRegistry(cfg.sop_dir)


@pytest.fixture
def tools(cfg, db, sops) -> LabTools:
    return LabTools(cfg, db, sops)
