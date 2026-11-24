// node tests/logreg_parity.mjs <in.bin> <meta.json>  -> JSON probabilities on stdout
// Used by tests/logreg_parity.py to compare web/logreg.js with scikit-learn.
import fs from "node:fs";
import { fitLogReg, predictProba } from "../web/logreg.js";

const [bin, metaPath] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const buf = fs.readFileSync(bin);
const X = new Float32Array(buf.buffer, buf.byteOffset, meta.n * meta.d);
const y = Int32Array.from(meta.y);
const all = [...Array(meta.n).keys()];
const out = [];
for (const C of meta.Cs) {
  const t0 = Date.now();
  const { W, b } = fitLogReg(X, meta.d, y, meta.k, C, all);
  out.push({ proba: Array.from(predictProba(X, meta.d, meta.k, W, b, all)), ms: Date.now() - t0 });
}
process.stdout.write(JSON.stringify(out));
