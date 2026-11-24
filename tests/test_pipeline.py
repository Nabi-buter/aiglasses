#!/usr/bin/env python3
"""End-to-end smoke test: synthetic data -> train -> predict (Python) -> same in the browser.

    python tests/test_pipeline.py            # Python only
    python tests/test_pipeline.py --browser  # also checks web/pipeline.js gives the same result
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
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="aiglasses-test-", dir=ROOT / "tests"))
    try:
        data, head_path = tmp / "data", tmp / "head.json"
        run([sys.executable, "tests/make_synthetic.py", "--out", data, "--n", str(args.n)])
        run([sys.executable, "training/train.py", "--data", data, "--out", head_path])
        head = json.loads(head_path.read_text())
        assert head["cv"]["accuracy"] >= 0.9, f"synthetic CV accuracy too low: {head['cv']}"

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
            srv.shutdown()
            if out.returncode:
                sys.exit(out.stderr)
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
