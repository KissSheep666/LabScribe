"""Render one synthetic scenario as a labelled contact sheet.

Usage:
    python scripts/show_scenario.py
    python scripts/show_scenario.py --seed 105 --occluded --out docs/examples/scenario.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from labscribe.config import Config
from labscribe.paths import PROJECT_ROOT
from labscribe.vision.fonts import get_font
from labscribe.vision.synthetic import SyntheticSource, make_scenario


def _pick_moments(spec, limit: int) -> list[float]:
    """Moments worth looking at: session edges, reading changes, occlusion."""
    moments = {0.4}
    for step in spec.reading_steps[1:]:
        moments.add(round(float(step["t"]), 2))
        moments.add(round(float(step["t"]) + 0.45, 2))
    for iv in spec.actor_intervals:
        moments.add(round(float(iv["t_start"]) + 0.35, 2))
        moments.add(round(float(iv["t_end"]) + 1.4, 2))
    for occ in spec.occlusions:
        moments.add(round((float(occ["t_start"]) + float(occ["t_end"])) / 2, 2))
    valid = sorted(m for m in moments if 0.0 <= m < spec.duration)
    if len(valid) <= limit:
        return valid
    # Keep temporal spread: sample evenly across the timeline.
    idx = np.linspace(0, len(valid) - 1, limit).round().astype(int)
    return [valid[i] for i in sorted(set(idx))]


def _draw_frame(image_bgr: np.ndarray, spec, t: float, caption: str) -> Image.Image:
    rgb = image_bgr[:, :, ::-1].copy()
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(13)
    small = get_font(11)

    # Instrument box and display ROI.
    ix1, iy1, ix2, iy2 = (int(v) for v in spec.instrument_bbox)
    draw.rectangle([ix1, iy1, ix2, iy2], outline=(90, 200, 255), width=2)
    roi = spec.display_bbox
    draw.rectangle(
        [int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)],
        outline=(255, 200, 60),
        width=2,
    )
    draw.text((ix1 + 4, iy1 - 16), "instrument", font=small, fill=(140, 220, 255))
    draw.text((int(roi.x1) + 3, int(roi.y1) - 15), "display ROI", font=small, fill=(255, 210, 90))

    # Occlusion window indicator.
    if spec.occluded_at(t):
        draw.rectangle([0, 0, pil.width - 1, pil.height - 1], outline=(255, 90, 90), width=3)
        draw.text((8, 6), "OCCLUDED", font=font, fill=(255, 120, 120))

    # Caption bar.
    draw.rectangle([0, pil.height - 20, pil.width, pil.height], fill=(10, 14, 19))
    draw.text((6, pil.height - 18), caption, font=small, fill=(210, 226, 240))
    return pil


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--occluded", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out", default="docs/examples/scenario-preview.png")
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args(argv)

    cfg = Config.load()
    spec = make_scenario(
        seed=args.seed,
        scenario_id=f"show-{args.seed}",
        occluded=args.occluded,
        fps=int(cfg.get("vision.fps", 10)),
    )
    source = SyntheticSource(spec)

    print("=== scenario ===")
    print(f"id={spec.scenario_id} seed={spec.seed} fps={spec.fps} duration={spec.duration}s "
          f"frames={spec.frame_count}")
    print(f"instrument bbox={spec.instrument_bbox} display_ratio={spec.display_ratio}")
    print(f"readings={spec.reading_steps}")
    print(f"actor_intervals={spec.actor_intervals}")
    print(f"occlusions={spec.occlusions} detector_dropouts={len(spec.detector_dropouts)} "
          f"ocr_dropouts={len(spec.ocr_dropouts)}")
    print("\n=== ground truth events ===")
    for e in spec.ground_truth:
        print(json.dumps(e, ensure_ascii=False))

    moments = _pick_moments(spec, args.frames)
    cells: list[Image.Image] = []
    for t in moments:
        frame = source.render(int(round(t * spec.fps)))
        hits = [
            e["type"]
            for e in spec.ground_truth
            if abs(e["t"] - t) <= 0.55
        ]
        caption = f"t={t:.1f}s  reading={spec.reading_at(t):.1f}"
        if hits:
            caption += "  <" + ", ".join(sorted(set(hits))) + ">"
        if spec.occluded_at(t):
            caption += "  [occluded]"
        cells.append(_draw_frame(frame.image, spec, t, caption))

    cols = max(1, min(args.cols, len(cells)))
    rows = (len(cells) + cols - 1) // cols
    w, h = cells[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (10, 14, 19))
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % cols) * w, (i // cols) * h))

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nwrote {args.out} ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
