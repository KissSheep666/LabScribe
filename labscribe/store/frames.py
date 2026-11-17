from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import cv2

from ..schema import Frame
from .db import Database

log = logging.getLogger(__name__)


class FrameStore:
    """Keeps a small in-memory ring of frames and persists frames on demand.

    Only frames referenced by an event or a pending review item are written to
    disk, so a 24/7 camera feed does not fill the disk.  The persisted JPEG is
    the single artifact a human needs to verify a record.
    """

    def __init__(self, db: Database, frames_dir: str | Path, max_buffered: int = 240):
        self.db = db
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffered = max(8, int(max_buffered))
        self._buffer: OrderedDict[str, Frame] = OrderedDict()
        self._persisted: dict[str, str] = {}

    def keep(self, frame: Frame) -> None:
        self._buffer[frame.frame_id] = frame
        self._buffer.move_to_end(frame.frame_id)
        while len(self._buffer) > self.max_buffered:
            self._buffer.popitem(last=False)

    def persist(self, frame_id: str) -> str | None:
        if not frame_id:
            return None
        cached = self._persisted.get(frame_id)
        if cached:
            return cached
        frame = self._buffer.get(frame_id)
        if frame is None:
            row = self.db.get_frame(frame_id)
            return row["path"] if row else None
        safe = frame_id.replace("/", "_").replace("\\", "_")
        path = self.frames_dir / f"{safe}.jpg"
        if not path.exists():
            ok = cv2.imwrite(str(path), frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not ok:
                log.warning("failed to write frame %s", path)
                return None
        self.db.save_frame(
            frame_id=frame_id,
            path=str(path),
            ts=frame.ts,
            width=frame.width,
            height=frame.height,
            meta=frame.meta,
        )
        self._persisted[frame_id] = str(path)
        return str(path)

    def path_for(self, frame_id: str) -> str | None:
        cached = self._persisted.get(frame_id)
        if cached:
            return cached
        row = self.db.get_frame(frame_id)
        if not row:
            return None
        path = Path(row["path"])
        return str(path) if path.exists() else None

    def image(self, frame_id: str):
        frame = self._buffer.get(frame_id)
        if frame is not None:
            return frame.image
        path = self.path_for(frame_id)
        return cv2.imread(path) if path else None
