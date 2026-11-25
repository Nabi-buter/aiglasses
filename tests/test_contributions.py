#!/usr/bin/env python3
"""Shared-model pipeline: contribution files -> merge_contributions.py -> merged model + data.

    python tests/test_contributions.py <contribution.json made by the web page>
    python tests/test_contributions.py            # uses a synthetic contribution (no browser needed)

Checks: a valid file is accepted and its rows are added; running again gives the same data
(idempotent); a duplicate and a tampered file are rejected; a failing quality gate leaves the
shared model untouched; the merged base_data is readable by train.py --with-base.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
import basekit as bk  # noqa: E402


def synthetic_contribution(n_people=6, per=4, seed=0) -> dict:
    """Embedding-like rows built from real base rows plus noise (valid norms), labelled ai_glasses."""
    base = bk.load_base_data(ROOT / "web" / "models")
    rng = np.random.default_rng(seed)
    src = base["X"][(base["y"] == 0) & (base["flip"] == 0)]
    rows = []
    for i in range(n_people * per):
        x = src[i % len(src)] + rng.normal(0, 0.01, bk.DIM).astype(np.float32)
        x[: bk.DIM // 2] /= np.linalg.norm(x[: bk.DIM // 2])
        rows.append(x)
    X = np.stack(rows)
    q, scale = bk.quantize(X)
    n = len(X)
    return {"format": "ai-glasses-contribution/v1", "id": f"c_test_{seed}", "created_at": "2026-11-25T00:00:00Z",
            "crop": bk.CROP_VERSION, "dim": bk.DIM, "n": n, "classes": ["ai_glasses"], "counts": {"ai_glasses": n},
            "y": [0] * n, "group": [i // per for i in range(n)], "groups": [f"g{i}" for i in range(n_people)],
            "flip": [0] * n, "device": [0] * n, "devices": ["test-device"], "note": "synthetic", "consent": True,
            "data_b64": base64.b64encode(q.tobytes()).decode(), "scale_b64": base64.b64encode(scale.astype("<f4").tobytes()).decode()}


def merge(contrib: Path, models: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "training/merge_contributions.py", "--contributions", str(contrib),
                           "--models", str(models), "--report", str(models / "report.md"), *extra],
                          cwd=ROOT, capture_output=True, text=True)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="aiglasses-contrib-"))
    try:
        models, contrib = tmp / "models", tmp / "contributions"
        models.mkdir(); contrib.mkdir()
        for f in ("head.json", "base_data.json", "base_data.bin"):
            shutil.copy(ROOT / "web" / "models" / f, models / f)
        n_public = json.loads((models / "base_data.json").read_text())["n"]

        if len(sys.argv) > 1:
            shutil.copy(sys.argv[1], contrib / Path(sys.argv[1]).name)
            print("browser-made contribution:", Path(sys.argv[1]).name)
        else:
            (contrib / "c_test_0.json").write_text(json.dumps(synthetic_contribution()))
        good = sorted(contrib.glob("*.json"))[0]
        c = json.loads(good.read_text())

        # duplicate (same features, new id) and tampered (wrong embedding norms) files
        dup = dict(c, id="c_dup"); (contrib / "z_dup.json").write_text(json.dumps(dup))
        q = np.frombuffer(base64.b64decode(c["data_b64"]), np.int8).copy()
        bad = dict(c, id="c_bad", scale_b64=base64.b64encode((np.ones(c["n"], "<f4") * 0.05).tobytes()).decode(),
                   data_b64=base64.b64encode(q.tobytes()).decode())
        (contrib / "z_bad.json").write_text(json.dumps(bad))
        (contrib / "z_noconsent.json").write_text(json.dumps(dict(c, id="c_nc", consent=False)))

        r = merge(contrib, models)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stderr
        meta = json.loads((models / "base_data.json").read_text())
        head = json.loads((models / "head.json").read_text())
        assert meta["n"] == n_public + c["n"], (meta["n"], n_public, c["n"])
        assert head["n_contributions"] == 1 and sum(head["n_contrib"].values()) > 0, head["n_contrib"]
        assert "contributed" in head["cv"], head["cv"].keys()
        report = (models / "report.md").read_text()
        assert "z_dup.json" in report and "z_bad.json" in report and "z_noconsent.json" in report, report

        # idempotent: merging again over the merged data gives the same rows
        r2 = merge(contrib, models)
        assert r2.returncode == 0, r2.stderr
        assert json.loads((models / "base_data.json").read_text())["n"] == meta["n"]

        # a failing quality gate must leave the shared model untouched
        before = (models / "head.json").read_bytes()
        r3 = merge(contrib, models, "--min-recall", "1.01")
        assert r3.returncode == 2 and (models / "head.json").read_bytes() == before, (r3.returncode, r3.stdout[-400:])

        # train.py --with-base reads the merged data
        sys.path.insert(0, str(ROOT / "training"))
        import importlib
        train = importlib.import_module("train")
        X, y, g = train.load_base(models)
        assert len(y) == meta["n"] and X.shape[1] == bk.DIM
        print(f"OK: public {n_public} rows + contribution {c['n']} rows; duplicate/tampered/no-consent rejected; gate holds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
