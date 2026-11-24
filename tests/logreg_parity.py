#!/usr/bin/env python3
"""web/logreg.js (in-browser training) must fit the same model as scikit-learn.

    python tests/logreg_parity.py                      # random embedding-like data, 3 and 2 classes
    python tests/logreg_parity.py --npz data/.feature_cache.npz   # (optional) real cached features
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent


def make_data(rng, n, d, k):
    centres = rng.normal(size=(k, d))
    y = rng.integers(0, k, n)
    y[:k] = np.arange(k)
    X = centres[y] * 0.15 + rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)          # like the L2-normalised embeddings
    return X.astype(np.float32), y


def check(X, y, k, Cs, label):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "x.bin").write_bytes(np.ascontiguousarray(X, np.float32).tobytes())
        (tmp / "m.json").write_text(json.dumps({"n": len(y), "d": X.shape[1], "k": k, "y": y.tolist(), "Cs": Cs}))
        out = subprocess.run(["node", "tests/logreg_parity.mjs", tmp / "x.bin", tmp / "m.json"],
                             cwd=ROOT, capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    worst = 0.0
    for ci, C in enumerate(Cs):
        sk = LogisticRegression(C=C, max_iter=5000, tol=1e-8, class_weight="balanced").fit(X.astype(np.float64), y)
        p_sk = sk.predict_proba(X.astype(np.float64))
        p_js = np.asarray(js[ci]["proba"]).reshape(len(y), k)
        diff = float(np.abs(p_sk - p_js).max())
        agree = float((p_sk.argmax(1) == p_js.argmax(1)).mean())
        worst = max(worst, diff)
        print(f"  {label} C={C:<5} max|Δp|={diff:.5f} argmax agree={agree:.3f} js {js[ci]['ms']} ms")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    worst = 0.0
    X, y = make_data(rng, 400, 2560, 3)
    worst = max(worst, check(X, y, 3, [0.3, 3.0], "3-class"))
    X, y = make_data(rng, 300, 2560, 2)
    worst = max(worst, check(X, y, 2, [1.0], "2-class"))
    if args.npz:
        with np.load(args.npz) as z:
            feats = [z[f] for f in z.files if z[f].size == 2560]
        Xr = np.stack(feats)
        yr = np.arange(len(Xr)) % 3        # labels do not matter for an optimiser check
        worst = max(worst, check(Xr, yr, 3, [1.0], "real-features"))
    assert worst < 2e-3, f"in-browser training differs from scikit-learn: max|Δp|={worst}"
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
