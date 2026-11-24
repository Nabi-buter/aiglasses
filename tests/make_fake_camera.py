#!/usr/bin/env python3
"""Write a .y4m clip that Chromium can use as a fake webcam
(--use-file-for-fake-video-capture). Each input image becomes one "person"
placed side by side; the people drift slowly so the tracker has to follow them.

    python tests/make_fake_camera.py a.jpg b.jpg --out cam.y4m
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_i420(rgb: np.ndarray) -> bytes:
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    sub = lambda c: c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean(axis=(1, 3))  # noqa: E731
    out = [np.clip(p, 0, 255).astype(np.uint8) for p in (y, sub(u), sub(v))]
    return b"".join(p.tobytes() for p in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))
    n = len(args.images)
    cell = min(W // n, H) - 40
    faces = [Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS) for p in args.images]
    with open(args.out, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{args.fps}:1 Ip A1:1 C420jpeg\n".encode())
        for i in range(args.frames):
            canvas = Image.new("RGB", (W, H), (46, 52, 60))
            for k, im in enumerate(faces):
                dx = int(12 * math.sin(2 * math.pi * (i / args.frames + k * 0.3)))
                dy = int(8 * math.cos(2 * math.pi * (i / args.frames + k * 0.3)))
                x = k * (W // n) + (W // n - cell) // 2 + dx
                canvas.paste(im, (x, (H - cell) // 2 + dy))
            f.write(b"FRAME\n" + rgb_to_i420(np.asarray(canvas)))
    print(f"fake camera -> {args.out} ({args.frames} frames, {W}x{H})")


if __name__ == "__main__":
    main()
