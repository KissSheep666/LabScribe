from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import cv2

from ..config import Config
from ..paths import resolve_path
from ..schema import Frame
from .synthetic import ScenarioSpec, SyntheticSource, make_demo_scenario

log = logging.getLogger(__name__)


class CameraSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index = int(cfg.get("vision.camera_index", 0))

    def iter_frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.index}")
        started = time.perf_counter()
        index = 0
        fps = float(self.cfg.get("vision.fps", 10) or 10)
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(
                    frame_id=f"cam-{index:06d}",
                    index=index,
                    ts=round(time.perf_counter() - started, 4),
                    image=image,
                )
                index += 1
                if fps > 0:
                    time.sleep(max(0.0, 1.0 / fps))
        finally:
            cap.release()


class VideoFileSource:
    def __init__(self, cfg: Config, path: str | Path):
        self.cfg = cfg
        self.path = Path(path)

    def iter_frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cfg.get("vision.fps", 10) or 10)
        index = 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield Frame(
                    frame_id=f"video-{index:06d}",
                    index=index,
                    ts=round(index / max(1e-6, fps), 4),
                    image=image,
                )
                index += 1
        finally:
            cap.release()


class ImageDirSource:
    def __init__(self, cfg: Config, directory: str | Path):
        self.cfg = cfg
        self.directory = Path(directory)
        self.fps = float(cfg.get("vision.fps", 10) or 10)

    def iter_frames(self) -> Iterator[Frame]:
        paths = sorted(
            p
            for p in self.directory.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            yield Frame(
                frame_id=f"img-{index:06d}",
                index=index,
                ts=round(index / max(1e-6, self.fps), 4),
                image=image,
                meta={"path": str(path)},
            )


def build_source(cfg: Config, scenario: ScenarioSpec | None = None):
    source = str(cfg.get("vision.source", "synthetic")).lower()
    if source == "synthetic" or scenario is not None:
        if scenario is None:
            scenario = make_demo_scenario(fps=int(cfg.get("vision.fps", 10)))
        return SyntheticSource(scenario)
    if source == "camera":
        return CameraSource(cfg)
    if source == "video":
        path = resolve_path(cfg.get("vision.video_path"))
        if path is None:
            raise ValueError("vision.video_path is required for source=video")
        return VideoFileSource(cfg, path)
    if source == "images":
        path = resolve_path(cfg.get("vision.image_dir"))
        if path is None:
            raise ValueError("vision.image_dir is required for source=images")
        return ImageDirSource(cfg, path)
    raise ValueError(f"unknown vision.source: {source}")
