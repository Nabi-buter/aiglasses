#!/usr/bin/env python3
"""Build a SYNTHETIC dataset to smoke-test the pipeline end to end.

It draws cartoon glasses on the astronaut photo that ships with scikit-image
(public domain, NASA). "ai_glasses" get a camera dot + LED at the hinge and
thicker temples; "regular_glasses" get a plain frame; "no_glasses" nothing.

This proves the code works. It does NOT produce a model you can use on real
people: train on real photos for that (see docs/DATA.md).

    python tests/make_synthetic.py --out /tmp/synth --n 30
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import glasses_features as gf  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402


def base_face() -> tuple[Image.Image, list[tuple[float, float]]]:
    from skimage import data
    img = Image.fromarray(data.astronaut()).resize((1024, 1024), Image.LANCZOS)
    with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(gf.DEFAULT_MODEL_DIR / gf.FACE_FILE)))) as fl:
        res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    lm = [(p.x * img.width, p.y * img.height) for p in res.face_landmarks[0]]
    return img, lm


def draw_glasses(img: Image.Image, lm, kind: str, rng: random.Random) -> Image.Image:
    img = img.copy()
    if kind == "no_glasses":
        return img
    d = ImageDraw.Draw(img)
    P = lambda i: lm[i]  # noqa: E731
    fw = abs(P(454)[0] - P(234)[0])
    color = rng.choice([(20, 20, 20), (60, 35, 20), (150, 120, 60), (40, 40, 70)])
    width = rng.randint(3, 6)
    lens_w, lens_h = fw * rng.uniform(0.30, 0.36), fw * rng.uniform(0.20, 0.26)
    eyes = [((P(33)[0] + P(133)[0]) / 2, (P(159)[1] + P(145)[1]) / 2),
            ((P(263)[0] + P(362)[0]) / 2, (P(386)[1] + P(374)[1]) / 2)]
    boxes = []
    for cx, cy in eyes:
        box = [cx - lens_w / 2, cy - lens_h / 2, cx + lens_w / 2, cy + lens_h / 2]
        d.rounded_rectangle(box, radius=lens_h * 0.35, outline=color, width=width)
        boxes.append(box)
    (l0, _, l1, _), (r0, _, r1, _) = sorted(boxes)
    top = min(b[1] for b in boxes) + lens_h * 0.2
    d.line([(l1, top), (r0, top)], fill=color, width=width)
    temple_w = width if kind == "regular_glasses" else width + rng.randint(5, 8)
    d.line([(l0, top), (P(234)[0] - fw * 0.03, top + 3)], fill=color, width=temple_w)
    d.line([(r1, top), (P(454)[0] + fw * 0.03, top + 3)], fill=color, width=temple_w)
    if kind == "ai_glasses":
        rad = fw * rng.uniform(0.022, 0.03)
        sides = rng.choice([("l",), ("r",), ("l", "r")])
        for s in sides:
            x = l0 + rad * 1.6 if s == "l" else r1 - rad * 1.6
            y = top + rad * 0.2
            d.ellipse([x - rad * 1.5, y - rad * 1.5, x + rad * 1.5, y + rad * 1.5], fill=(90, 90, 95))
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(10, 10, 15))
            d.ellipse([x - rad * 0.35, y - rad * 0.55, x, y - rad * 0.2], fill=(200, 200, 220))
        lx = r1 - rad * 4 if sides == ("l",) else l0 + rad * 4
        d.ellipse([lx - rad * 0.5, top - rad * 0.5, lx + rad * 0.5, top + rad * 0.5], fill=(235, 235, 240))
    return img


def jitter(img: Image.Image, face_box, rng: random.Random) -> Image.Image:
    """Random rotation / zoom / shift / lighting that keeps the whole face in frame."""
    img = img.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, fillcolor=(30, 30, 30))
    fx0, fy0, fx1, fy1 = face_box
    m = 0.25 * (fx1 - fx0)
    fx0, fy0, fx1, fy1 = fx0 - m, fy0 - m, fx1 + m, fy1 + m
    side = rng.uniform(max(fx1 - fx0, fy1 - fy0), img.width)
    x0 = rng.uniform(max(0, fx1 - side), min(fx0, img.width - side))
    y0 = rng.uniform(max(0, fy1 - side), min(fy0, img.height - side))
    out = rng.randint(480, 900)
    img = img.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((out, out), Image.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30, help="클래스당 장 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    img, lm = base_face()
    xs, ys = [p[0] for p in lm], [p[1] for p in lm]
    face_box = (min(xs), min(ys), max(xs), max(ys))
    for kind in ("ai_glasses", "regular_glasses", "no_glasses"):
        (args.out / kind).mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            out = jitter(draw_glasses(img, lm, kind, rng), face_box, rng)
            out.save(args.out / kind / f"{kind}_{i:03d}.jpg", quality=rng.randint(80, 95))
    print(f"synthetic dataset -> {args.out}")


if __name__ == "__main__":
    main()
