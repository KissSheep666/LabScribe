from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from ..schema import BBox, OCRResult
from .fonts import get_font
from .utils import bbox_to_int

log = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class OCR(ABC):
    name = "base"

    @abstractmethod
    def read(self, image: np.ndarray, roi: BBox) -> OCRResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class TemplateOCR(OCR):
    """Offline digit OCR: segment glyphs, match normalized 0-9 templates.

    Designed for bright digits on a dark digital display.  It is deliberately
    deterministic and dependency-free so the demo and the evaluation harness
    produce reproducible numbers; production uses :class:`PaddleOCRBackend`.
    """

    name = "template"

    def __init__(self, cfg: Config):
        self.min_conf = float(cfg.get("vision.confidence.min_ocr", 0.5))
        self._templates: dict[str, np.ndarray] | None = None

    def _ensure_templates(self) -> dict[str, np.ndarray]:
        if self._templates is not None:
            return self._templates
        templates: dict[str, np.ndarray] = {}
        for digit in "0123456789":
            canvas = Image.new("L", (120, 160), 0)
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), digit, font=get_font(120), fill=255)
            arr = np.array(canvas)
            ys, xs = np.where(arr > 80)
            if len(ys) == 0:
                continue
            glyph = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            glyph = cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_AREA)
            templates[digit] = (glyph > 110).astype(np.uint8)
        self._templates = templates
        return templates

    def read(self, image: np.ndarray, roi: BBox) -> OCRResult:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox_to_int(roi, w, h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return OCRResult.failed("empty display roi")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        scale = 120.0 / max(1, gray.shape[0])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.resize(hsv, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Bright, unsaturated pixels = display digits.  Orange skin/gloves are
        # saturated and therefore excluded even if they occlude the display.
        mask = ((gray > 145) & (hsv[:, :, 1] < 125)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if int(mask.sum()) < 12:
            return OCRResult.failed("no glyph pixels")

        segments = _column_segments(mask)
        height = mask.shape[0]
        stats = [
            s for s in (_segment_stats(mask, seg) for seg in segments)
            if s is not None and s["height"] >= max(3.0, 0.05 * height) and s["pixels"] >= 8
        ]
        if not stats:
            return OCRResult.failed("no segments")

        max_height = max(s["height"] for s in stats)
        baselines = [s["row2"] for s in stats if s["height"] >= 0.55 * max_height]
        baseline = float(np.median(baselines)) if baselines else float(mask.shape[0])
        templates = self._ensure_templates()

        chars: list[str] = []
        scores: list[float] = []
        for seg in stats:
            glyph = _crop_ink(mask, (seg["x1"], seg["x2"]))
            if glyph is None or glyph.size == 0:
                continue
            gh, gw = glyph.shape
            is_dot = (
                seg["height"] < 0.42 * max_height
                and seg["width"] <= max(1.35 * seg["height"], 0.30 * max_height)
                and seg["row2"] >= 0.5 * baseline
            )
            if is_dot:
                chars.append(".")
                scores.append(0.92)
                continue
            binary = cv2.resize(glyph.astype(np.float32), (24, 32), interpolation=cv2.INTER_AREA)
            binary = (binary > 0.4).astype(np.uint8)
            ranked = sorted(
                (
                    (_mask_iou(binary, template), digit)
                    for digit, template in templates.items()
                ),
                reverse=True,
            )
            best_score, best_digit = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score < 0.42:
                return OCRResult.failed(f"low glyph match ({best_score:.2f})")
            chars.append(best_digit)
            # Reward a clear margin over the runner-up digit.
            margin = max(0.0, best_score - second_score)
            scores.append(min(0.99, best_score * (0.82 + 0.18 * min(1.0, margin / 0.25))))

        text = "".join(chars)
        if not text or text.count(".") > 1:
            return OCRResult.failed(f"unparsable text {text!r}")
        if not re.fullmatch(r"\d{1,3}(?:\.\d{1,2})?", text):
            return OCRResult.failed(f"unparsable text {text!r}")
        try:
            value = float(text)
        except ValueError:
            return OCRResult.failed(f"unparsable text {text!r}")
        confidence = float(np.mean(scores)) if scores else 0.0
        return OCRResult(text=text, value=value, confidence=round(confidence, 4), ok=True)


class PaddleOCRBackend(OCR):
    """Production OCR adapter for PaddleOCR."""

    name = "paddle"

    def __init__(self, cfg: Config):
        from paddleocr import PaddleOCR  # lazy heavy import

        lang = str(cfg.get("vision.ocr_lang", "en"))
        try:
            self.engine = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
        except (TypeError, ValueError):
            self.engine = PaddleOCR(lang=lang)
        self.min_conf = float(cfg.get("vision.confidence.min_ocr", 0.5))

    def read(self, image: np.ndarray, roi: BBox) -> OCRResult:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox_to_int(roi, w, h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return OCRResult.failed("empty display roi")
        try:
            raw = self.engine.ocr(crop, cls=False)
        except TypeError:
            raw = self.engine.ocr(crop)
        texts, scores = _extract_paddle_result(raw)
        if not texts:
            return OCRResult.failed("paddle found no text")
        text = " ".join(texts)
        match = _NUMBER_RE.search(text.replace(",", "."))
        if not match:
            return OCRResult.failed(f"non-numeric text {text!r}")
        value = float(match.group(0))
        confidence = float(np.mean(scores)) if scores else 0.5
        return OCRResult(text=text, value=value, confidence=round(confidence, 4), ok=True)


def _extract_paddle_result(raw) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key in ("rec_texts", "texts"):
                if key in item and isinstance(item[key], list):
                    texts.extend(str(t) for t in item[key])
            for key in ("rec_scores", "scores"):
                if key in item and isinstance(item[key], list):
                    scores.extend(float(s) for s in item[key])
            stack.extend(v for k, v in item.items() if k not in ("rec_texts", "texts", "rec_scores", "scores"))
        elif isinstance(item, (list, tuple)):
            if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (int, float)):
                texts.append(item[0])
                scores.append(float(item[1]))
            else:
                stack.extend(item)
    return texts, scores


def _column_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = (mask.sum(axis=0) > 0).astype(np.int8)
    height = mask.shape[0]
    segments: list[list[int]] = []
    start = None
    for x, active in enumerate(cols):
        if active and start is None:
            start = x
        elif not active and start is not None:
            segments.append([start, x])
            start = None
    if start is not None:
        segments.append([start, len(cols)])
    if not segments:
        return []
    merged = [segments[0]]
    gap_limit = max(1, int(0.025 * height))
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] <= gap_limit:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(a, b) for a, b in merged if b > a]


def _segment_stats(mask: np.ndarray, seg: tuple[int, int]) -> dict | None:
    x1, x2 = seg
    sub = mask[:, x1:x2]
    rows = np.where(sub.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return {
        "x1": x1,
        "x2": x2,
        "width": x2 - x1,
        "row1": int(rows.min()),
        "row2": int(rows.max()),
        "height": int(rows.max() - rows.min() + 1),
        "pixels": int(sub.sum()),
    }


def _segment_height(mask: np.ndarray, seg: tuple[int, int]) -> int:
    stats = _segment_stats(mask, seg)
    return stats["height"] if stats else 0


def _crop_ink(mask: np.ndarray, seg: tuple[int, int]) -> np.ndarray | None:
    sub = mask[:, seg[0] : seg[1]]
    rows = np.where(sub.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return sub[rows.min() : rows.max() + 1, :]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def build_ocr(cfg: Config) -> OCR:
    mode = str(cfg.get("vision.ocr", "auto")).lower()
    if mode == "template":
        return TemplateOCR(cfg)
    if mode in ("paddle", "auto"):
        try:
            engine = PaddleOCRBackend(cfg)
            if mode == "auto":
                log.warning("vision.ocr=auto: using PaddleOCR backend")
            return engine
        except Exception as exc:  # noqa: BLE001
            if mode == "paddle":
                raise
            log.warning("PaddleOCR unavailable (%s); falling back to template OCR", exc)
    return TemplateOCR(cfg)
