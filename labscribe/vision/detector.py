from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..schema import BBox, Detection, Frame

log = logging.getLogger(__name__)


class Detector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MarkerDetector(Detector):
    """Offline detector based on high-saturation color markers.

    Blue = instrument body, orange = person/hand.  This backend makes the demo
    and the evaluation harness fully deterministic while exercising the same
    detector -> tracker -> OCR interfaces as the production backend.
    """

    name = "marker"

    # OpenCV HSV ranges: (hue_lo, sat_lo, val_lo), (hue_hi, sat_hi, val_hi)
    RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
        "instrument": ((100, 90, 50), (130, 255, 255)),
        "actor": ((5, 100, 80), (25, 255, 255)),
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.display_ratio = list(cfg.get("vision.display_roi", [0.20, 0.52, 0.80, 0.86]))
        self.min_area = 250.0

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        detections: list[Detection] = []
        for role, (lo, hi) in self.RANGES.items():
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            boxes = [
                cv2.boundingRect(c)
                for c in contours
                if cv2.contourArea(c) >= self.min_area
            ]
            if not boxes:
                continue
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            bbox = BBox(float(x1), float(y1), float(x2), float(y2)).clip(w, h)
            area_ratio = (bbox.area / float(w * h)) if w and h else 0.0
            confidence = float(min(0.99, 0.62 + 3.0 * area_ratio))
            meta: dict[str, Any] = {}
            if role == "instrument":
                meta["display_roi"] = _display_roi(bbox, self.display_ratio).as_list()
            detections.append(
                Detection(label=role, confidence=round(confidence, 4), bbox=bbox, meta=meta)
            )
        return detections


class Yolo11Detector(Detector):
    """Production detector adapter for Ultralytics YOLO11 models."""

    name = "yolo11"

    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # lazy heavy import

        self.cfg = cfg
        model_path = cfg.get("vision.detector_model", "yolo11n.pt")
        self.model = YOLO(model_path)
        self.class_map: dict[str, str] = dict(cfg.get("vision.class_map", {}) or {})
        self.display_ratio = list(cfg.get("vision.display_roi", [0.20, 0.52, 0.80, 0.86]))
        self.min_conf = float(cfg.get("vision.confidence.min_detection", 0.45))

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(frame.image, conf=self.min_conf, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_label = str(names.get(cls_id, cls_id))
                role = self.class_map.get(raw_label)
                if role is None and not self.class_map and raw_label == "person":
                    role = "actor"
                if role not in ("actor", "instrument"):
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                bbox = BBox(x1, y1, x2, y2).clip(frame.width, frame.height)
                meta: dict[str, Any] = {"raw_label": raw_label}
                if role == "instrument":
                    meta["display_roi"] = _display_roi(bbox, self.display_ratio).as_list()
                detections.append(
                    Detection(
                        label=role,
                        confidence=float(box.conf[0].item()),
                        bbox=bbox,
                        meta=meta,
                    )
                )
        return detections


def _display_roi(bbox: BBox, ratio: list[float]) -> BBox:
    rx1, ry1, rx2, ry2 = (float(v) for v in ratio)
    return BBox(
        bbox.x1 + rx1 * bbox.width,
        bbox.y1 + ry1 * bbox.height,
        bbox.x1 + rx2 * bbox.width,
        bbox.y1 + ry2 * bbox.height,
    )


def build_detector(cfg: Config) -> Detector:
    mode = str(cfg.get("vision.detector", "auto")).lower()
    if mode == "marker":
        return MarkerDetector(cfg)
    if mode in ("yolo11", "yolo", "auto"):
        try:
            detector = Yolo11Detector(cfg)
            if mode == "auto":
                log.warning("vision.detector=auto: using YOLO11 backend (%s)", cfg.get("vision.detector_model"))
            return detector
        except Exception as exc:  # noqa: BLE001
            if mode == "yolo11":
                raise
            log.warning("YOLO11 unavailable (%s); falling back to marker detector", exc)
    return MarkerDetector(cfg)
