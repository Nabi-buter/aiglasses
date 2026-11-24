// Runs the heavy part of in-browser training off the main thread.
// The page sends the feature matrix once ("data"), then cross-validation folds
// ("fold": fit every C on the training rows, predict the held-out rows) or a final fit.
import { fitLogReg, predictProba } from "./logreg.js";

let D = null;

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === "data") { D = { X: m.X, d: m.d, y: m.y, k: m.k }; self.postMessage({ type: "ready" }); return; }
    const { X, d, y, k } = D;
    if (m.type === "fold") {
      let init = null;
      const test = Array.from(m.test);
      for (const C of m.Cs) {                           // ascending C, warm-started from the previous fit
        const r = fitLogReg(X, d, y, k, C, Array.from(m.train), { init });
        init = r.theta;
        const p = predictProba(X, d, k, r.W, r.b, test);
        self.postMessage({ type: "foldC", id: m.id, C, proba: p }, [p.buffer]);
      }
      self.postMessage({ type: "done", id: m.id });
    } else if (m.type === "fit") {
      const r = fitLogReg(X, d, y, k, m.C, Array.from(m.train));
      self.postMessage({ type: "fitted", id: m.id, W: r.W, b: r.b });
    }
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, message: String(err?.message || err) });
  }
};
