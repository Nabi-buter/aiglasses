#!/usr/bin/env python3
"""Build the built-in "is this person wearing glasses?" model (web/models/presence.json).

It works without any data from you, so the web app can tell glasses from no glasses out of the box.
(Your own head.json, trained on your photos, is what tells AI glasses from regular glasses.)

Data: Open Images V7 validation, test and (part of) train splits.
  - Annotations: CC BY 4.0, Google LLC
  - Images: listed as CC BY 2.0
  Only the learned weights are saved. No images or crops are written to the repository.

Labels come from the human-drawn boxes:
  - glasses:     a "Human face" box with a Glasses / Sunglasses / Goggles box over its eyes
  - no glasses:  a "Human face" box with no eyewear box, in an image whose eyewear labels were
                 verified by a person (so the eyewear boxes there are exhaustive),
                 or in an image verified to contain no glasses
Each annotated face is matched to the face our own face finder detects, and the feature is the
same one the app computes (glasses_features.extract_faces == web/pipeline.js featureFor).

    python training/build_presence.py --cache ~/.cache/aiglasses-oi --splits validation,test,train \
        --out web/models/presence.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glasses_features as gf  # noqa: E402

ANN = "https://storage.googleapis.com/openimages/v5/{split}-annotations-{kind}.csv"
ANN_TRAIN_BBOX = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"   # 2.2 GB
IMG = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
FACE = "/m/0dzct"
EYEWEAR = {"/m/0jyfg", "/m/017ftj", "/m/02_n6y"}   # Glasses, Sunglasses, Goggles
GLASSES = "/m/0jyfg"
MIN_FACE_FRAC = 0.10      # annotated face width / image width
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(dest)
    return dest


def read_csv(path: Path):
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def label_faces(cache: Path, split: str):
    """-> {image_id: [(face_box, label)]}  label 1 = glasses, 0 = none; box = (x0, y0, x1, y1) normalised."""
    boxes = fetch(ANN_TRAIN_BBOX if split == "train" else ANN.format(split=split, kind="bbox"), cache / f"{split}-bbox.csv")
    labels = fetch(ANN.format(split=split, kind="human-imagelabels-boxable"), cache / f"{split}-labels.csv")
    verified_eyewear, verified_no_glasses, verified_eyewear_any = set(), set(), set()
    for r in read_csv(labels):
        if r["LabelName"] in EYEWEAR:
            if r["Confidence"] == "1":
                verified_eyewear.add(r["ImageID"])
            verified_eyewear_any.add(r["ImageID"])
            if r["LabelName"] == GLASSES and r["Confidence"] == "0":
                verified_no_glasses.add(r["ImageID"])
    faces, eyewear = defaultdict(list), defaultdict(list)
    for r in read_csv(boxes):
        b = tuple(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax"))
        if r["LabelName"] == FACE and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0" and b[2] - b[0] >= MIN_FACE_FRAC:
            faces[r["ImageID"]].append(b)
        elif r["LabelName"] in EYEWEAR:
            eyewear[r["ImageID"]].append((b, r["IsGroupOf"] == "1"))
    out = {}
    for img, fl in faces.items():
        positive_img = img in verified_eyewear
        negative_img = img in verified_no_glasses and img not in verified_eyewear
        if not (positive_img or negative_img):
            continue        # eyewear boxes are only exhaustive on verified images
        res = []
        for f in fl:
            fw = f[2] - f[0]
            label, ambiguous = 0, False
            for g, group in eyewear.get(img, []):
                cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                inside = f[0] <= cx <= f[2] and f[1] <= cy <= f[3]
                if not inside:
                    continue
                if group:
                    ambiguous = True
                elif (g[2] - g[0]) >= 0.35 * fw and cy <= f[1] + 0.7 * (f[3] - f[1]):
                    label = 1
                else:
                    ambiguous = True
            if label == 1 or not ambiguous:
                res.append((f, label))
        if res:
            out[img] = res
    return out


def match(det_rect, ann):
    """Index of the annotated face that this detected face belongs to, or None."""
    cx, cy = (det_rect[0] + det_rect[2]) / 2, (det_rect[1] + det_rect[3]) / 2
    dw = det_rect[2] - det_rect[0]
    best, best_d = None, 1e9
    for i, (f, _) in enumerate(ann):
        fw = f[2] - f[0]
        if f[0] <= cx <= f[2] and f[1] <= cy <= f[3] and 0.45 <= dw / fw <= 1.6:
            d = abs(cx - (f[0] + f[2]) / 2) + abs(cy - (f[1] + f[3]) / 2)
            if d < best_d:
                best, best_d = i, d
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="where annotations, images and features are cached")
    ap.add_argument("--out", type=Path, default=Path("web/models/presence.json"))
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--max-images", type=int, default=0, help="limit per split (0 = all)")
    ap.add_argument("--max-train-images", type=int, default=10000, help="limit for the (large) train split")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-images", action="store_true", help="keep downloaded images (default: delete after use)")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    feat_cache_path = args.cache / f"features_{gf.CROP_VERSION.replace('/', '_')}.npz"
    fcache = {}
    if feat_cache_path.exists():
        with np.load(feat_cache_path, allow_pickle=False) as z:
            fcache = {k: z[k] for k in z.files}

    X, y, groups = [], [], []
    for split in args.splits.split(","):
        ann = label_faces(args.cache, split)
        ids = sorted(ann)
        limit = args.max_train_images if split == "train" else args.max_images
        if limit:
            ids = ids[:limit]
        n_pos = sum(lbl for i in ids for _, lbl in ann[i])
        n_all = sum(len(ann[i]) for i in ids)
        print(f"{split}: {len(ids)} images, {n_all} labelled faces ({n_pos} with glasses)")

        todo = [i for i in ids if f"{split}/{i}/done" not in fcache]

        def dl(i):
            try:
                return i, fetch(IMG.format(split=split, id=i), args.cache / "img" / split / f"{i}.jpg")
            except Exception as e:  # noqa: BLE001
                return i, e

        with gf.FeatureExtractor() as fx, ThreadPoolExecutor(args.workers) as pool:
            for n, (i, path) in enumerate(pool.map(dl, todo), 1):
                if isinstance(path, Exception):
                    print(f"  ! {i}: {path}")
                    continue
                try:
                    with Image.open(path) as im:
                        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                    for k, face in enumerate(fx.extract_faces(rgb)):
                        j = match(face["rect"], ann[i])
                        if j is None or face["feature"] is None or face["face_px"] < 48:
                            continue
                        fcache[f"{split}/{i}/{j}/{k}"] = face["feature"]
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {i}: {e}")
                fcache[f"{split}/{i}/done"] = np.zeros(0, np.float32)
                if not args.keep_images:
                    path.unlink(missing_ok=True)
                if n % 100 == 0 or n == len(todo):
                    print(f"  {split}: {n}/{len(todo)} images", end="\r", flush=True)
                if n % 500 == 0:
                    np.savez(feat_cache_path, **fcache)
        print()
        np.savez(feat_cache_path, **fcache)

        for key, feat in fcache.items():
            parts = key.split("/")
            if parts[0] != split or parts[-1] == "done" or parts[1] not in ann or feat.size != gf.FEATURE_DIM:
                continue
            X.append(feat); y.append(ann[parts[1]][int(parts[2])][1]); groups.append(parts[1])

    X, y, groups = np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)
    print(f"faces used: {len(y)} ({int(y.sum())} glasses, {int((1 - y).sum())} none) from {len(set(groups))} images")
    if len(y) < 100 or y.min() == y.max():
        sys.exit("not enough labelled faces")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(cv.split(X, y, groups))
    best = None
    for C in C_GRID:
        p = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        acc = float(((p >= 0.5) == y).mean())
        auc = float(roc_auc_score(y, p))
        print(f"  C={C:<5} CV accuracy {acc:.4f}  AUC {auc:.4f}")
        if best is None or acc > best[1]:
            best = (C, acc, auc, p)
    C, acc, auc, p = best
    # Threshold: balanced accuracy is fine here — a missed "glasses" only means the box stays grey,
    # and with a head.json trained with no_glasses the head decides anyway.
    t = 0.5
    tp = float(((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()))
    tn = float(((p < t) & (y == 0)).sum() / max(1, (y == 0).sum()))
    print(f"chosen C={C}: CV accuracy {acc:.4f}  AUC {auc:.4f}  glasses recall {tp:.4f}  no-glasses recall {tn:.4f}")

    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, y)
    w, b = model.coef_[0], model.intercept_[0]
    out = {
        "format": "ai-glasses-presence/v1",
        "embedder": gf.EMBEDDER_FILE, "crop": gf.CROP_VERSION, "feature_dim": gf.FEATURE_DIM,
        "classes": ["no_glasses", "glasses"],
        "W": [[round(float(v), 6) for v in -w / 2], [round(float(v), 6) for v in w / 2]],
        "b": [round(float(-b / 2), 6), round(float(b / 2), 6)],
        "threshold": t, "C": C,
        "cv": {"accuracy": round(acc, 4), "auc": round(auc, 4), "glasses_recall": round(tp, 4),
               "no_glasses_recall": round(tn, 4), "n": int(len(y)), "n_glasses": int(y.sum()),
               "n_images": int(len(set(groups))), "folds": 5},
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "data": "Open Images V7 (" + args.splits + "): annotations CC BY 4.0 Google LLC, images CC BY 2.0",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"saved {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
