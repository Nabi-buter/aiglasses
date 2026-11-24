// In-browser training: the same model and evaluation as training/train.py.
//
// Model: multinomial logistic regression, L2 penalty, class_weight="balanced",
// i.e. sklearn's LogisticRegression(C, class_weight="balanced") objective
//      C · Σ_i s_i · CE(softmax(W x_i + b), y_i) + ½‖W‖²     (b not penalised)
// minimised with L-BFGS. With two classes sklearn fits one weight vector v and
// train.py exports [-v/2, v/2]; the multinomial form with C/2 has exactly that optimum.
// Pure math, no DOM: runs in a Web Worker and in Node (tests/logreg_parity.py).

/** X: Float32Array n×d (row-major), y: Int32Array of class indices 0..k-1, idx: rows to use. */
export function fitLogReg(X, d, y, k, C, idx, { init = null, maxIter = 1000, gtol = 1e-6 } = {}) {
  const n = idx.length;
  const count = new Float64Array(k);
  for (const i of idx) count[y[i]]++;
  const present = [...count].filter((c) => c > 0).length;
  const sw = new Float64Array(k);                          // balanced class weights
  for (let c = 0; c < k; c++) sw[c] = count[c] ? n / (present * count[c]) : 0;
  const Ceff = k === 2 ? C / 2 : C;
  const P = k * (d + 1);                                   // [W (k×d), b (k)]
  // Objective scaled by 1/(Ceff·n): mean weighted loss + ‖W‖²/(2·Ceff·n). Same optimum, better conditioned.
  const lam = 1 / (Ceff * n);
  const z = new Float64Array(k);

  function fg(theta, grad) {
    grad.fill(0);
    let f = 0;
    for (let c = 0; c < k; c++) {
      const o = c * d;
      let s = 0;
      for (let j = 0; j < d; j++) { s += theta[o + j] * theta[o + j]; grad[o + j] = lam * theta[o + j]; }
      f += 0.5 * lam * s;
    }
    const bOff = k * d;
    for (const i of idx) {
      const xo = i * d, yi = y[i], wi = sw[yi] / n;
      let m = -Infinity;
      for (let c = 0; c < k; c++) {
        let s = theta[bOff + c];
        const o = c * d;
        for (let j = 0; j < d; j++) s += theta[o + j] * X[xo + j];
        z[c] = s; if (s > m) m = s;
      }
      let se = 0;
      for (let c = 0; c < k; c++) { z[c] = Math.exp(z[c] - m); se += z[c]; }
      f += wi * (Math.log(se) + m - (Math.log(z[yi]) + m));
      for (let c = 0; c < k; c++) {
        const g = wi * (z[c] / se - (c === yi ? 1 : 0));
        if (g === 0) continue;
        const o = c * d;
        for (let j = 0; j < d; j++) grad[o + j] += g * X[xo + j];
        grad[bOff + c] += g;
      }
    }
    return f;
  }

  const theta0 = init ? Float64Array.from(init) : new Float64Array(P);
  const theta = lbfgs(fg, theta0, { maxIter, gtol });
  const W = [], b = [];
  let bm = 0;
  for (let c = 0; c < k; c++) bm += theta[k * d + c] / k;
  for (let c = 0; c < k; c++) { W.push(Array.from(theta.subarray(c * d, (c + 1) * d))); b.push(theta[k * d + c] - bm); }
  return { W, b, theta };
}

function lbfgs(fg, x, { m = 12, maxIter = 1000, gtol = 1e-6 }) {
  const P = x.length;
  let g = new Float64Array(P);
  let f = fg(x, g);
  const S = [], Y = [], R = [];
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const xn = new Float64Array(P), gn = new Float64Array(P), p = new Float64Array(P), alpha = new Float64Array(m);
  for (let it = 0; it < maxIter; it++) {
    let gmax = 0;
    for (let i = 0; i < P; i++) gmax = Math.max(gmax, Math.abs(g[i]));
    if (gmax < gtol) break;
    // two-loop recursion
    p.set(g);
    for (let i = S.length - 1; i >= 0; i--) {
      alpha[i] = R[i] * dot(S[i], p);
      for (let j = 0; j < P; j++) p[j] -= alpha[i] * Y[i][j];
    }
    const gamma = S.length ? dot(S[S.length - 1], Y[S.length - 1]) / dot(Y[S.length - 1], Y[S.length - 1]) : 1 / Math.max(1, Math.sqrt(dot(g, g)));
    for (let j = 0; j < P; j++) p[j] *= gamma;
    for (let i = 0; i < S.length; i++) {
      const beta = R[i] * dot(Y[i], p);
      for (let j = 0; j < P; j++) p[j] += S[i][j] * (alpha[i] - beta);
    }
    for (let j = 0; j < P; j++) p[j] = -p[j];
    let gp = dot(g, p);
    if (!(gp < 0)) { S.length = Y.length = R.length = 0; for (let j = 0; j < P; j++) p[j] = -g[j]; gp = dot(g, p); }
    // backtracking line search (Armijo)
    let t = 1, fnew = Infinity, ok = false;
    for (let ls = 0; ls < 40; ls++) {
      for (let j = 0; j < P; j++) xn[j] = x[j] + t * p[j];
      fnew = fg(xn, gn);
      if (fnew <= f + 1e-4 * t * gp) { ok = true; break; }
      t *= 0.5;
    }
    if (!ok) break;
    const s = new Float64Array(P), yv = new Float64Array(P);
    for (let j = 0; j < P; j++) { s[j] = xn[j] - x[j]; yv[j] = gn[j] - g[j]; }
    const sy = dot(s, yv);
    if (sy > 1e-12) { S.push(s); Y.push(yv); R.push(1 / sy); if (S.length > m) { S.shift(); Y.shift(); R.shift(); } }
    const df = f - fnew;
    x.set(xn); g.set(gn); f = fnew;
    if (df >= 0 && df < 1e-14 * Math.max(1, Math.abs(f))) break;
  }
  return x;
}

/** Class probabilities for rows idx -> Float64Array(len × k). */
export function predictProba(X, d, k, W, b, idx) {
  const out = new Float64Array(idx.length * k), z = new Float64Array(k);
  idx.forEach((i, r) => {
    let m = -Infinity;
    for (let c = 0; c < k; c++) {
      let s = b[c]; const w = W[c];
      for (let j = 0; j < d; j++) s += w[j] * X[i * d + j];
      z[c] = s; if (s > m) m = s;
    }
    let se = 0;
    for (let c = 0; c < k; c++) { z[c] = Math.exp(z[c] - m); se += z[c]; }
    for (let c = 0; c < k; c++) out[r * k + c] = z[c] / se;
  });
  return out;
}

/* ---------------- evaluation helpers (same numbers as train.py) ---------------- */

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/**
 * Stratified, grouped k-fold: every group (person/session) lands in exactly one fold,
 * and each fold gets a similar share of every class. Returns Int32Array fold id per row.
 */
export function groupedFolds(y, groups, k, nFolds, seed = 0) {
  const byGroup = new Map();
  y.forEach((c, i) => {
    let g = byGroup.get(groups[i]);
    if (!g) byGroup.set(groups[i], (g = { rows: [], cnt: new Float64Array(k) }));
    g.rows.push(i); g.cnt[c]++;
  });
  const total = new Float64Array(k);
  for (const c of y) total[c]++;
  const rnd = mulberry32(seed + 1);
  const list = [...byGroup.values()].map((g) => ({ ...g, r: rnd() }));
  list.sort((a, b) => b.rows.length - a.rows.length || a.r - b.r);
  const foldCnt = Array.from({ length: nFolds }, () => new Float64Array(k));
  const fold = new Int32Array(y.length);
  for (const g of list) {
    let best = 0, bestScore = Infinity;
    for (let f = 0; f < nFolds; f++) {
      let score = 0;
      for (let c = 0; c < k; c++) if (total[c]) score += ((foldCnt[f][c] + g.cnt[c]) / total[c]) ** 2;
      if (score < bestScore - 1e-12) { bestScore = score; best = f; }
    }
    for (let c = 0; c < k; c++) foldCnt[best][c] += g.cnt[c];
    for (const i of g.rows) fold[i] = best;
  }
  return fold;
}

/** ROC-AUC of score for positives (Mann–Whitney, ties averaged). */
export function rocAuc(pos, score) {
  const idx = score.map((s, i) => i).sort((a, b) => score[a] - score[b]);
  const rank = new Float64Array(score.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[i]]) j++;
    for (let t = i; t <= j; t++) rank[idx[t]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  let np = 0, sum = 0;
  pos.forEach((p, i) => { if (p) { np++; sum += rank[i]; } });
  const nn = pos.length - np;
  return np && nn ? (sum - (np * (np + 1)) / 2) / (np * nn) : null;
}

export function ratesAt(t, pAi, isAi, isReg) {
  let ai = 0, aiHit = 0, rest = 0, restHit = 0, reg = 0, regHit = 0;
  pAi.forEach((p, i) => {
    if (isAi[i]) { ai++; if (p >= t) aiHit++; } else { rest++; if (p >= t) restHit++; }
    if (isReg && isReg[i]) { reg++; if (p >= t) regHit++; }
  });
  const r4 = (v) => Math.round(v * 1e4) / 1e4;
  const out = { threshold: r4(t), recall: r4(aiHit / Math.max(ai, 1)), false_alarm: r4(restHit / Math.max(rest, 1)) };
  if (reg) out.false_alarm_regular = r4(regHit / reg);
  return out;
}

/** Largest threshold that still flags at least `target` of the AI-glasses rows. */
export function thresholdForRecall(pAi, isAi, target) {
  const s = pAi.filter((_, i) => isAi[i]).sort((a, b) => a - b);
  if (!s.length) return 0.3;
  const i = Math.floor((1 - target) * s.length);
  return s[Math.min(i, s.length - 1)] - 1e-6;
}

export function confusion(yTrue, yPred, k) {
  const m = Array.from({ length: k }, () => new Array(k).fill(0));
  yTrue.forEach((t, i) => m[t][yPred[i]]++);
  return m;
}
