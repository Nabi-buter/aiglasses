#!/usr/bin/env python3
"""End-to-end smoke test: synthetic data -> train -> predict (Python) -> same in the browser.

    python tests/test_pipeline.py            # Python only
    python tests/test_pipeline.py --browser  # also checks web/pipeline.js gives the same result,
                                             # and runs the live camera view on a fake webcam
                                             # (needs Node + Playwright with Chromium)
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
import glasses_features as gf  # noqa: E402


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--screenshot", help="save the live-view screenshots here (png)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="aiglasses-test-", dir=ROOT / "tests"))
    try:
        data, head_path = tmp / "data", tmp / "head.json"
        run([sys.executable, "tests/make_synthetic.py", "--out", data, "--n", str(args.n)])
        run([sys.executable, "training/train.py", "--data", data, "--out", head_path, "--target-recall", "0.95"])
        head = json.loads(head_path.read_text())
        assert head["cv"]["accuracy"] >= 0.9, f"synthetic CV accuracy too low: {head['cv']}"
        assert head["cv"]["ai_auc"] >= 0.95 and head["cv"]["at_suspect"]["recall"] >= 0.95, head["cv"]

        samples = [data / c / f"{c}_{i:03d}.jpg" for c in head["classes"] for i in (0, 1)]
        py = {}
        with gf.FeatureExtractor() as fx:
            for s in samples:
                ex = fx.extract(gf.load_rgb(s))
                assert ex.feature is not None, f"no face in {s}"
                probs = gf.head_predict(head, ex.feature)
                py[s.name] = (ex, probs)
                expect = s.parent.name
                print(f"  {s.name:<24} {gf.verdict(head, probs):<11} " +
                      " ".join(f"{k}={v:.2f}" for k, v in probs.items()))
                assert max(probs, key=probs.get) == expect, f"{s.name}: predicted {probs}"

        if args.browser:
            class Quiet(http.server.SimpleHTTPRequestHandler):
                def log_message(self, *a):
                    pass
            handler = functools.partial(Quiet, directory=str(ROOT))
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            rel = lambda p: "/" + p.relative_to(ROOT).as_posix()  # noqa: E731
            env = dict(os.environ)
            node_path = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
            env["NODE_PATH"] = os.pathsep.join(filter(None, [env.get("NODE_PATH"), node_path]))
            out = subprocess.run(["node", "tests/browser_parity.cjs", base, rel(head_path), *map(rel, samples)],
                                 cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
            if out.returncode:
                srv.shutdown()
                sys.exit(out.stderr)
            # Live view: two people side by side on a fake webcam (images the head never saw).
            held = tmp / "held"
            run([sys.executable, "tests/make_synthetic.py", "--out", held, "--n", "2", "--seed", "7"])
            cam = tmp / "cam.y4m"
            run([sys.executable, "tests/make_fake_camera.py", held / "ai_glasses/ai_glasses_000.jpg",
                 held / "regular_glasses/regular_glasses_001.jpg", "--out", cam])
            shot = Path(args.screenshot).resolve() if args.screenshot else tmp / "live.png"
            live = subprocess.run(["node", "tests/live_e2e.cjs", base, str(head_path), str(cam), str(shot)],
                                  cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
            if live.returncode:
                srv.shutdown()
                sys.exit(live.stderr)
            lr = json.loads(live.stdout)
            print("live view (left → right):")
            for t in lr["tracks"]:
                print(f"  #{t['id']} {t['verdict']:<11} ai={t['probs']['ai_glasses']:.2f} ({t['samples']} frames, face {t['facePx']}px)")
            verdicts = [t["verdict"] for t in lr["tracks"]]
            assert len(verdicts) == 2 and lr["ids"] == 2, f"tracking failed: {lr}"
            assert verdicts[0] in ("AI_GLASSES", "SUSPECTED") and verdicts[1] == "REGULAR", verdicts
            assert any(l["verdict"] in ("AI_GLASSES", "SUSPECTED") for l in lr["log"]), "red box was not logged"
            with zipfile.ZipFile(lr["zip"]["path"]) as z:
                assert z.testzip() is None and "labels.csv" in z.namelist()
                assert sum(n.startswith("ai_glasses/") for n in z.namelist()) >= 2
            bad = [e for e in lr["errors"] if not e.startswith("INFO:")]
            assert not bad, bad

            # Before any training: only the built-in presence model -> blue (glasses) / grey (none).
            cam2 = tmp / "cam_nohead.y4m"
            run([sys.executable, "tests/make_fake_camera.py", held / "regular_glasses/regular_glasses_000.jpg",
                 held / "no_glasses/no_glasses_001.jpg", "--out", cam2])
            nh = subprocess.run(["node", "tests/live_e2e.cjs", base, "none", str(cam2), str(shot).replace(".png", "_nohead.png")],
                                cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
            if nh.returncode:
                srv.shutdown()
                sys.exit(nh.stderr)
            nr = json.loads(nh.stdout)
            print("live view without a trained head (left → right):",
                  [(t["verdict"], round(t["probs"]["__glasses"], 2)) for t in nr["tracks"]])
            assert [t["verdict"] for t in nr["tracks"]] == ["GLASSES", "NO_GLASSES"], nr["tracks"]
            assert nr["noModelTag"] and not nr["log"], nr

            # In-browser training page: same dataset as a (deflate) ZIP -> head.json in the browser.
            zpath = tmp / "dataset.zip"
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(data.glob("*/*.jpg")):
                    z.write(f, f.relative_to(data).as_posix())
            tshot = str(shot).replace(".png", "_train.png")
            tr = subprocess.run(["node", "tests/train_e2e.cjs", base, str(zpath), tshot],
                                cwd=ROOT, capture_output=True, text=True, env=env, timeout=1200)
            srv.shutdown()
            if tr.returncode:
                sys.exit(tr.stderr)
            tb = json.loads(tr.stdout)
            hb = tb["head"]
            print(f"browser training: {tb['secs']:.0f}s  C={hb['C']}  cv acc={hb['cv']['accuracy']}  "
                  f"auc={hb['cv']['ai_auc']}  suspect={hb['thresholds']['suspect']}")
            assert hb["cv"]["accuracy"] >= 0.9 and hb["classes"] == head["classes"], hb["cv"]
            if tb.get("merged"):
                m = tb["merged"]
                print(f"  + base data merged: base {m['n_base']} / user {m['n_user']}  "
                      f"user-photo accuracy {m['cv']['user']['accuracy']}  overall {m['cv']['accuracy']}")
                assert m["cv"]["user"]["accuracy"] >= 0.8, m["cv"]["user"]
            if hb["C"] == head["C"]:   # same C -> the final models must agree (features from Python)
                feats = np.stack([ex.feature for ex, _ in py.values()])
                def proba(h):
                    z = feats @ np.asarray(h["W"]).T + np.asarray(h["b"])
                    e = np.exp(z - z.max(1, keepdims=True))
                    return e / e.sum(1, keepdims=True)
                dp = float(np.abs(proba(hb) - proba(head)).max())
                print(f"  browser-trained vs python-trained head: max|Δp|={dp:.4f}")
                assert dp < 0.02, dp
            print("browser vs python:")
            for r in json.loads(out.stdout):
                name = Path(r["file"]).name
                ex, probs = py[name]
                assert r["feature"] is not None, f"browser found no face in {name}"
                fb, fp = np.asarray(r["feature"]), ex.feature
                cos = [float(np.dot(fb[i:i + 1280], fp[i:i + 1280]) /
                             (np.linalg.norm(fb[i:i + 1280]) * np.linalg.norm(fp[i:i + 1280]) + 1e-9)) for i in (0, 1280)]
                dp = max(abs(r["probs"][k] - probs[k]) for k in probs)
                box_diff = max(abs(a - b) for k in r["boxes"] for a, b in zip(r["boxes"][k], ex.boxes[k].as_list()))
                print(f"  {name:<24} cos(band)={cos[0]:.4f} cos(hinge)={cos[1]:.4f} "
                      f"max|Δp|={dp:.3f} max|Δbox|={box_diff}px")
                assert min(cos) > 0.97, f"{name}: embeddings differ too much"
                assert dp < 0.08, f"{name}: probabilities differ too much"
        print("OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
