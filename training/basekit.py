"""Shared training / evaluation / export for the SHARED model (web/models/head.json)
and the SHARED data (web/models/base_data.*).

Used by build_base.py (public photos) and merge_contributions.py (public + every user's
contribution). Needs only numpy + scikit-learn, so it runs in GitHub Actions without MediaPipe.

base_data.* layout (also read by web/train.js and train.py --with-base):
  base_data.bin   int8 row-major n×dim, then float32 scale[n]    (value = int8 * scale)
  base_data.json  y / group / source / flip per row (+ name tables)
Rows whose source starts with "contrib:" came from user contributions.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["ai_glasses", "no_glasses", "regular_glasses"]   # == train.py (sorted)
CROP_VERSION = "band+hinges/v1"                              # == glasses_features.CROP_VERSION
EMBEDDER_FILE = "mobilenet_v3_large.tflite"
DIM = 2560
CONTRIB = "contrib:"


def quantize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = (np.maximum(np.abs(X).max(1), 1e-8) / 127.0).astype(np.float32)
    q = np.clip(np.round(X / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


def load_base_data(models: Path) -> dict:
    meta = json.loads((models / "base_data.json").read_text())
    if meta["crop"] != CROP_VERSION or meta["dim"] != DIM:
        raise SystemExit("base_data 의 crop 버전/차원이 코드와 다릅니다.")
    raw = (models / "base_data.bin").read_bytes()
    n = meta["n"]
    q = np.frombuffer(raw, np.int8, n * DIM).reshape(n, DIM)
    scale = np.frombuffer(raw, np.float32, n, offset=n * DIM)
    return {
        "X": dequantize(q, scale), "q": q, "scale": scale,
        "y": np.array([CLASSES.index(meta["classes"][i]) for i in meta["y"]]),
        "group": np.array([meta["groups"][g] for g in meta["group"]]),
        "source": np.array([meta["sources"][s] for s in meta["source"]]),
        "flip": np.array(meta["flip"], dtype=int), "meta": meta,
    }


def save_base_data(models: Path, X, y, group, source, flip, extra: dict | None = None, q=None, scale=None):
    if q is None:
        q, scale = quantize(X)
    (models / "base_data.bin").write_bytes(np.ascontiguousarray(q).tobytes() + np.asarray(scale, np.float32).tobytes())
    gnames, snames = sorted(set(group.tolist())), sorted(set(source.tolist()))
    gi, si = {g: i for i, g in enumerate(gnames)}, {s: i for i, s in enumerate(snames)}
    meta = {
        "format": "ai-glasses-basedata/v1", "crop": CROP_VERSION, "dim": DIM, "n": int(len(y)),
        "classes": CLASSES, "y": [int(v) for v in y], "group": [gi[g] for g in group], "groups": gnames,
        "source": [si[s] for s in source], "sources": snames, "flip": [int(v) for v in flip],
        "encoding": "int8 row-major n×dim, then float32 scale[n]; value = int8 * scale",
        "credits": "Open Images V7 (CC BY 4.0 annotations, CC BY 2.0 images); Wikimedia Commons (see docs/BASE_DATA_SOURCES.md); "
                   "user contributions (see docs/CONTRIBUTIONS.md)",
    }
    meta.update(extra or {})
    (models / "base_data.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))


def _rates(t, p, pos, neg):
    r = lambda m: None if not m.any() else round(float((p[m] >= t).mean()), 4)  # noqa: E731
    return r(pos), r(neg)


def fit_head(X, y, group, source, flip, *, max_false_alarm=0.05, C_grid=(0.1, 0.3, 1.0, 3.0),
             base=True, trainer="", note="", log=print) -> dict:
    """Grouped 5-fold CV over C, threshold so that at most `max_false_alarm` of regular-glasses
    faces get a red box, then a final fit on everything. Returns head.json content."""
    orig = flip == 0
    n_folds = int(min(5, *(len(set(group[y == c])) for c in range(len(CLASSES)) if (y == c).any())))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=0)
    folds = list(cv.split(X, y, group))
    best = None
    for C in C_grid:
        P = np.zeros((len(y), len(CLASSES)))
        for tr, te in folds:
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
            P[te] = m.predict_proba(X[te])
        acc = float((P.argmax(1) == y).mean())
        log(f"  C={C:<5} CV accuracy {acc:.4f}")
        if best is None or acc > best[1]:
            best = (C, acc, P)
    C, acc, P = best
    p_ai = P[:, 0]
    is_ai, is_reg = (y == 0) & orig, (y == 2) & orig
    not_ai = ~is_ai & orig
    cand = sorted(np.unique(np.round(p_ai[is_reg], 4)))
    suspect = next((float(t) for t in cand if (p_ai[is_reg] >= t).mean() <= max_false_alarm), 0.5)
    ai_t = next((max(float(t), suspect) for t in cand if t >= suspect and (p_ai[is_reg] >= t).mean() <= max_false_alarm / 3),
                max(suspect, 0.6))
    auc = float(roc_auc_score(is_ai[orig], p_ai[orig])) if is_ai.any() and not_ai.any() else None
    cm = confusion_matrix(y[orig], P[orig].argmax(1), labels=range(len(CLASSES)))
    contrib = np.char.startswith(source.astype(str), CONTRIB)
    r4 = lambda v: None if v is None else round(float(v), 4)  # noqa: E731

    def block(mask_ai, mask_reg, mask_rest):
        rec, fa = _rates(suspect, p_ai, mask_ai, mask_reg)
        return {"n_ai": int(mask_ai.sum()), "n_regular": int(mask_reg.sum()), "recall": rec, "false_alarm_regular": fa,
                "false_alarm": _rates(suspect, p_ai, mask_ai, mask_rest)[1]}

    devices = sorted(set(source[is_ai].tolist()))
    cvd = {
        "accuracy": r4(acc), "ai_recall": r4(cm[0, 0] / max(cm[0].sum(), 1)),
        "ai_precision": r4(cm[0, 0] / max(cm[:, 0].sum(), 1)), "confusion": cm.tolist(), "folds": n_folds,
        "ai_auc": r4(auc),
        "at_suspect": {"threshold": r4(suspect), "recall": _rates(suspect, p_ai, is_ai, is_reg)[0],
                       "false_alarm": _rates(suspect, p_ai, is_ai, not_ai)[1], "false_alarm_regular": _rates(suspect, p_ai, is_ai, is_reg)[1]},
        "at_ai": {"threshold": r4(ai_t), "recall": _rates(ai_t, p_ai, is_ai, is_reg)[0],
                  "false_alarm": _rates(ai_t, p_ai, is_ai, not_ai)[1], "false_alarm_regular": _rates(ai_t, p_ai, is_ai, is_reg)[1]},
        "ai_recall_by_device": {d.replace(CONTRIB, ""): _rates(suspect, p_ai, is_ai & (source == d), is_reg)[0] for d in devices},
        "public": block(is_ai & ~contrib, is_reg & ~contrib, not_ai & ~contrib),
    }
    if contrib.any():
        cvd["contributed"] = block(is_ai & contrib, is_reg & contrib, not_ai & contrib)
    log(f"chosen C={C}: CV accuracy {acc:.4f}  AI AUC {auc}  red line {suspect:.4f}: "
        f"AI recall {cvd['at_suspect']['recall']}  regular false alarm {cvd['at_suspect']['false_alarm_regular']}")
    if contrib.any():
        log(f"  on contributed photos: {cvd['contributed']}")

    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, y)
    n_photos = lambda m: {c: int((m & (y == i) & orig).sum()) for i, c in enumerate(CLASSES)}  # noqa: E731
    return {
        "format": "ai-glasses-head/v1", "embedder": EMBEDDER_FILE, "crop": CROP_VERSION, "feature_dim": DIM,
        "classes": CLASSES, "ai_class": "ai_glasses",
        "W": [[round(float(v), 6) for v in row] for row in model.coef_],
        "b": [round(float(v), 6) for v in model.intercept_],
        "thresholds": {"ai": round(ai_t, 4), "suspect": round(suspect, 4)}, "C": C,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_images": n_photos(np.ones_like(orig)), "n_public": n_photos(~contrib), "n_contrib": n_photos(contrib),
        "cv": cvd, "test": None, "base": base, "trainer": trainer, "note": note,
    }
