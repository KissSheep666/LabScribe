from __future__ import annotations

from ..schema import BBox


def relative_bbox(parent: BBox, ratio: list[float] | tuple[float, ...]) -> BBox:
    """Compute a child bbox from fractional coordinates of ``parent``."""
    rx1, ry1, rx2, ry2 = (float(v) for v in ratio)
    return BBox(
        parent.x1 + rx1 * parent.width,
        parent.y1 + ry1 * parent.height,
        parent.x1 + rx2 * parent.width,
        parent.y1 + ry2 * parent.height,
    )


def bbox_to_int(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    clipped = bbox.clip(width, height)
    x1, y1 = int(round(clipped.x1)), int(round(clipped.y1))
    x2, y2 = int(round(clipped.x2)), int(round(clipped.y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2
