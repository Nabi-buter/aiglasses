// In-browser training page. Produces the same head.json as training/train.py:
// same features (pipeline.js == glasses_features.py), same model (logreg.js == sklearn),
// same grouped cross-validation report and threshold calibration.
import { Detector, blobToCanvas, validateHead, CROP_VERSION } from "./pipeline.js";
import { groupedFolds, rocAuc, ratesAt, thresholdForRecall, confusion } from "./logreg.js";
import { listZip } from "./unzip.js";

const $ = (id) => document.getElementById(id);
const KNOWN = ["ai_glasses", "no_glasses", "regular_glasses"];
const REQUIRED = ["ai_glasses", "regular_glasses"];
const AI = "ai_glasses";
const NAMES = { ai_glasses: "AI 안경", regular_glasses: "일반 안경", no_glasses: "안경 없음" };
const IMG = /\.(jpe?g|png|webp|bmp|heic|heif)$/i;
const C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0];   // == train.py
const HEAD_KEY = "aiGlassesCustomHead.v1";   // == app.js: the live view picks it up
const D = 2560;

const state = { items: new Map(), head: null, running: false, abort: false, workers: [], detector: null, rows: null, base: undefined };

/* ---------------- base data (shipped with the app: features only, no photos) ---------------- */
async function loadBase() {
  if (state.base !== undefined) return state.base;
  try {
    const [m, buf] = await Promise.all([
      fetch("models/base_data.json").then((r) => (r.ok ? r.json() : null)),
      fetch("models/base_data.bin").then((r) => (r.ok ? r.arrayBuffer() : null)),
    ]);
    if (!m || !buf || m.crop !== CROP_VERSION || m.dim !== D || buf.byteLength < m.n * (D + 4)) return (state.base = null);
    const q = new Int8Array(buf, 0, m.n * D), sc = new Float32Array(buf.slice(m.n * D, m.n * D + 4 * m.n));
    const rows = [];
    for (let i = 0; i < m.n; i++) {
      const feat = new Float32Array(D);
      for (let j = 0; j < D; j++) feat[j] = q[i * D + j] * sc[i];
      rows.push({ feat, label: m.classes[m.y[i]], group: "base:" + m.groups[m.group[i]], item: null, flip: !!m.flip[i], base: true, source: m.sources[m.source[i]] });
    }
    const photos = {};
    rows.forEach((r) => { if (!r.flip) photos[r.label] = (photos[r.label] || 0) + 1; });
    return (state.base = { meta: m, rows, photos });
  } catch (e) { console.warn("base data:", e); return (state.base = null); }
}
const useBase = () => $("withBase").checked && !!state.base;

/* ---------------- utils ---------------- */
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
const pct = (p, digits = 0) => (p == null ? "—" : (p * 100).toFixed(digits) + "%");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function fmtDur(s) { s = Math.max(0, Math.round(s)); return s >= 60 ? `${Math.floor(s / 60)}분 ${s % 60}초` : `${s}초`; }
function download(blob, name) {
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
function progress(stage, frac, detail = "") {
  $("progress").hidden = false; $("pStage").textContent = stage;
  $("pBar").style.width = Math.round(Math.min(1, Math.max(0, frac)) * 100) + "%"; $("pDetail").textContent = detail;
}
function notice(box, kind, text) { $(box).replaceChildren(el("div", kind, text)); }

/* ---------------- dataset ---------------- */
function labelOf(path) {
  const segs = path.split("/").slice(0, -1);
  for (let i = segs.length - 1; i >= 0; i--) if (KNOWN.includes(segs[i])) return segs[i];
  return null;
}
/** == train.py group_of(): files named "p07__xxx.jpg" share the group "<class>/p07". */
function groupOf(path, label) {
  const base = path.split("/").pop(), stem = base.replace(/\.[^.]+$/, "");
  return stem.includes("__") ? `${label}/${stem.split("__")[0]}` : path;
}
function addItem(path, getBlob, source) {
  const name = path.split("/").pop();
  if (!IMG.test(name) || name.startsWith("._") || path.includes("__MACOSX/")) return false;
  const label = labelOf(path);
  if (!label) { state.unlabeled = (state.unlabeled || 0) + 1; return false; }
  const key = source + "!" + path;
  if (!state.items.has(key)) state.items.set(key, { key, path, label, group: groupOf(path, label), getBlob });
  return true;
}
async function addZip(file) {
  const entries = await listZip(file);
  let n = 0;
  for (const e of entries) {
    if (addItem(e.name, async () => new Blob([await e.read()]), file.name + ":" + file.size)) n++;
  }
  return n;
}
async function addFiles(files) {
  if (state.running) return;
  let added = 0, errors = [];
  for (const f of files) {
    try {
      if (/\.zip$/i.test(f.name)) added += await addZip(f);
      else if (addItem(f.relPath || f.webkitRelativePath || f.name, async () => f, "fs")) added++;
    } catch (e) { errors.push(`${f.name}: ${e.message || e}`); }
  }
  renderData(errors);
  return added;
}
async function walkEntry(entry, out) {
  if (entry.isFile) {
    const f = await new Promise((res, rej) => entry.file(res, rej));
    f.relPath = entry.fullPath.replace(/^\//, "");
    out.push(f);
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const e of batch) await walkEntry(e, out);
    }
  }
}

function summary() {
  const by = {};
  for (const it of state.items.values()) {
    const s = by[it.label] || (by[it.label] = { n: 0, groups: new Set() });
    s.n++; s.groups.add(it.group);
  }
  return by;
}

function renderData(errors = []) {
  const box = $("dataSummary"); box.replaceChildren();
  const by = summary(), total = state.items.size;
  $("clearData").disabled = !total || state.running;
  $("sharePanel").hidden = !total;
  $("makeShare").disabled = !total || state.running || !$("consent").checked;
  $("trainBtn").disabled = state.running || (useBase() ? total < 5 : !REQUIRED.every((c) => by[c]?.n >= 5));
  if (!total) {
    if (errors.length) box.append(el("div", "err", errors.join("\n")));
    return;
  }
  const t = el("table", "log");
  const hd = el("tr"); ["정답", "사진", "사람/세션 묶음"].forEach((h) => hd.append(el("th", null, h)));
  t.append(el("thead")); t.tHead.append(hd);
  const body = el("tbody");
  for (const c of KNOWN) {
    const s = by[c];
    const tr = el("tr");
    const pill = el("span", "pill " + (c === AI ? "ai" : c === "regular_glasses" ? "ok" : "na"), NAMES[c]);
    const td = el("td"); td.append(pill);
    tr.append(td, el("td", "mono", s ? String(s.n) : "0"), el("td", "mono", s ? String(s.groups.size) : "0"));
    body.append(tr);
  }
  t.append(body); box.append(t);

  const warn = [];
  if (useBase()) { if (total < 5) warn.push("사진이 5장 이상 필요합니다."); }
  else for (const c of REQUIRED) if (!(by[c]?.n >= 5)) warn.push(`${NAMES[c]} 사진이 5장 이상 필요합니다 (또는 ‘기본 공개 데이터도 함께 학습’을 켜세요).`);
  if (!useBase()) for (const c of REQUIRED) if (by[c] && by[c].groups.size < 2) warn.push(`${NAMES[c]}: 서로 다른 사람/세션이 2개 이상 필요합니다 (파일 이름 앞부분 'p01__' 처럼).`);
  const small = Math.min(...REQUIRED.map((c) => by[c]?.n || 0));
  if (small >= 5 && small < 200) warn.push(`사진이 적습니다. 실전에 쓰려면 클래스당 200장 이상, 사람 30명 이상을 권장합니다.`);
  if (!by.no_glasses && !useBase()) warn.push("‘안경 없음’ 사진이 없습니다. 넣지 않으면 안경을 안 쓴 사람도 AI 안경·일반 안경 중 하나로 표시됩니다.");
  if (state.unlabeled) warn.push(`정답 폴더(ai_glasses 등) 밖에 있는 사진 ${state.unlabeled}장은 건너뛰었습니다.`);
  for (const w of warn) box.append(el("div", "warnbox", w));
  if (errors.length) box.append(el("div", "err", errors.join("\n")));
}

/* ---------------- feature cache (IndexedDB) ---------------- */
let dbP = null;
function db() {
  if (!dbP) dbP = new Promise((res) => {
    try {
      const r = indexedDB.open("aiglasses-features", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("f");
      r.onsuccess = () => res(r.result); r.onerror = () => res(null);
    } catch { res(null); }
  });
  return dbP;
}
async function cacheGet(key) {
  const d = await db(); if (!d) return undefined;
  return new Promise((res) => { const q = d.transaction("f").objectStore("f").get(key); q.onsuccess = () => res(q.result); q.onerror = () => res(undefined); });
}
async function cachePut(key, val) {
  const d = await db(); if (!d) return;
  return new Promise((res) => { const t = d.transaction("f", "readwrite"); t.objectStore("f").put(val, key); t.oncomplete = res; t.onerror = res; });
}
async function sha1(buf) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-1", buf));
  return [...h].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function flipped(c) {
  const f = el("canvas"); f.width = c.width; f.height = c.height;
  const x = f.getContext("2d"); x.translate(c.width, 0); x.scale(-1, 1); x.drawImage(c, 0, 0);
  return f;
}

/* ---------------- step 1: features ---------------- */
async function extractAll(flip) {
  if (!state.detector) { progress("판독 엔진 불러오는 중…", 0); state.detector = await Detector.create(".", { video: false }); }
  const items = [...state.items.values()];
  const rows = [];                       // {feat, label, group, item, flip}
  const seen = new Set();
  let done = 0, computed = 0, computeMs = 0, noFace = 0, dup = 0, bad = 0;
  const t0 = performance.now();
  for (const it of items) {
    if (state.abort) throw new Error("중지했습니다.");
    try {
      const blob = await it.getBlob();
      const buf = await blob.arrayBuffer();
      const h = await sha1(buf);
      if (seen.has(h)) { dup++; } else {
        seen.add(h);
        let canvas = null, got = false;
        for (const fl of flip ? [false, true] : [false]) {
          const key = `${h}|${CROP_VERSION}|${fl ? 1 : 0}`;
          let feat = await cacheGet(key);
          if (feat === undefined) {
            const s = performance.now();
            if (!canvas) canvas = await blobToCanvas(new Blob([buf], { type: blob.type || "image/jpeg" }));
            const ex = state.detector.extract(fl ? flipped(canvas) : canvas);
            feat = ex.feature ? Float32Array.from(ex.feature) : null;
            await cachePut(key, feat);
            computeMs += performance.now() - s; computed++;
          }
          if (feat && feat.length === D) { rows.push({ feat, label: it.label, group: it.group, item: it, flip: fl }); got = true; }
        }
        if (!got) noFace++;
        it.hash = h; it.ok = got;
      }
    } catch (e) { console.warn(it.path, e); bad++; }
    done++;
    if (done % 2 === 0 || done === items.length) {
      const left = items.length - done;
      const per = computed ? computeMs / computed : 0;
      progress("① 사진에서 안경 특징 뽑는 중", done / items.length * 0.6,
        `${done} / ${items.length}장${per ? ` · 남은 시간 약 ${fmtDur((left * per * (flip ? 2 : 1)) / 1000)}` : ""}`);
      await sleep(0);
    }
  }
  return { rows, noFace, dup, bad, secs: (performance.now() - t0) / 1000 };
}

/* ---------------- step 2: cross-validation + fit (Web Workers) ---------------- */
function startWorkers(n, X, y, k) {
  const ws = [];
  for (let i = 0; i < n; i++) {
    const w = new Worker(new URL("./train-worker.js", import.meta.url), { type: "module" });
    w.postMessage({ type: "data", X, d: D, y, k });
    ws.push(w);
  }
  state.workers = ws;
  return ws;
}
function stopWorkers() { state.workers.forEach((w) => w.terminate()); state.workers = []; }

function runJob(w, msg, onMsg) {
  return new Promise((res, rej) => {
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") return;
      if (m.type === "error") return rej(new Error(m.message));
      if (onMsg(m)) res(m);
    };
    w.onerror = (e) => rej(new Error(e.message || "worker error"));
    w.postMessage(msg);
  });
}

async function train() {
  const flip = $("flip").checked, target = Number($("target").value), aiT = Number($("aiT").value);
  state.running = true; state.abort = false; state.head = null;
  $("trainBtn").disabled = true; $("stopBtn").hidden = false; $("clearData").disabled = true;
  $("resultPanel").hidden = true; $("applyPanel").hidden = true; $("applyMsg").replaceChildren();
  try {
    const ext = await extractAll(flip);
    const withBase = useBase();
    if (!ext.rows.length) throw new Error("넣은 사진에서 얼굴을 하나도 찾지 못했습니다.");
    const rows = withBase ? ext.rows.concat(state.base.rows) : ext.rows;
    const classes = KNOWN.filter((c) => rows.some((r) => r.label === c));
    for (const c of REQUIRED) if (!classes.includes(c)) throw new Error(`${NAMES[c]} 사진에서 얼굴을 찾지 못했습니다.`);
    const k = classes.length, n = rows.length;
    const userCounts = Object.fromEntries(classes.map((c) => [c, new Set(ext.rows.filter((r) => r.label === c).map((r) => r.item.key)).size]));
    const baseCounts = withBase ? Object.fromEntries(classes.map((c) => [c, state.base.photos[c] || 0])) : null;
    const counts = Object.fromEntries(classes.map((c) => [c, userCounts[c] + (baseCounts?.[c] || 0)]));
    if (Math.min(...Object.values(counts)) < 5) throw new Error("클래스마다 얼굴이 보이는 사진이 최소 5장은 있어야 합니다.");
    const groupsPer = classes.map((c) => new Set(rows.filter((r) => r.label === c).map((r) => r.group)).size);
    const nFolds = Math.min(5, ...groupsPer);
    if (nFolds < 2) throw new Error("교차검증을 하려면 클래스마다 서로 다른 사람/세션 묶음이 2개 이상 필요합니다.");

    const X = new Float32Array(n * D), y = new Int32Array(n), groups = rows.map((r) => r.group);
    rows.forEach((r, i) => { X.set(r.feat, i * D); y[i] = classes.indexOf(r.label); });
    const fold = groupedFolds(y, groups, k, nFolds, 0);
    const aiIdx = classes.indexOf(AI);

    // Cross-validation: each fold fits every C (warm-started), in parallel workers.
    const nW = Math.max(1, Math.min(nFolds, (navigator.hardwareConcurrency || 2) - 1, 4));
    const ws = startWorkers(nW, X, y, k);
    const proba = new Map(C_GRID.map((C) => [C, new Float64Array(n * k)]));
    const queue = [...Array(nFolds).keys()];
    let fits = 0;
    const totalFits = nFolds * C_GRID.length + 1;
    const t0 = performance.now();
    const tick = () => {
      const el = (performance.now() - t0) / 1000, left = fits ? (el / fits) * (totalFits - fits) : 0;
      progress("② 교차검증으로 성능 측정 중 (사람 단위로 나눠 시험)", 0.6 + 0.38 * fits / totalFits,
        `${fits} / ${totalFits - 1}회 학습${fits ? ` · 남은 시간 약 ${fmtDur(left)}` : ""}`);
    };
    tick();
    await Promise.all(ws.map(async (w) => {
      while (queue.length) {
        if (state.abort) throw new Error("중지했습니다.");
        const f = queue.shift();
        const train = Int32Array.from([...fold.keys()].filter((i) => fold[i] !== f));
        const test = Int32Array.from([...fold.keys()].filter((i) => fold[i] === f));
        await runJob(w, { type: "fold", id: f, train, test, Cs: C_GRID }, (m) => {
          if (m.type === "foldC") {
            const P = proba.get(m.C);
            test.forEach((row, r) => { for (let c = 0; c < k; c++) P[row * k + c] = m.proba[r * k + c]; });
            fits++; tick();
          }
          return m.type === "done";
        });
      }
    }));

    // Pick C by accuracy (smallest C on ties, == train.py)
    let best = null;
    for (const C of C_GRID) {
      const P = proba.get(C);
      const pred = Int32Array.from(y, (_, i) => { let b = 0; for (let c = 1; c < k; c++) if (P[i * k + c] > P[i * k + b]) b = c; return b; });
      const acc = pred.reduce((s, p, i) => s + (p === y[i]), 0) / n;
      if (!best || acc > best.acc) best = { C, acc, pred, P };
    }
    const cm = confusion(y, best.pred, k);
    const tp = cm[aiIdx][aiIdx];
    const isAi = Array.from(y, (c) => c === aiIdx), isReg = Array.from(y, (c) => classes[c] === "regular_glasses");
    const pAi = Array.from({ length: n }, (_, i) => best.P[i * k + aiIdx]);
    const r4 = (v) => Math.round(v * 1e4) / 1e4;
    const suspect = Math.min(Math.max(thresholdForRecall(pAi, isAi, target), 0.02), aiT);
    const auc = rocAuc(isAi, pAi);
    const cv = {
      accuracy: r4(best.acc), ai_recall: r4(tp / Math.max(1, cm[aiIdx].reduce((a, b) => a + b, 0))),
      ai_precision: r4(tp / Math.max(1, cm.reduce((s, row) => s + row[aiIdx], 0))),
      confusion: cm, folds: nFolds, ai_auc: auc == null ? null : r4(auc),
      at_suspect: ratesAt(suspect, pAi, isAi, isReg), at_ai: ratesAt(aiT, pAi, isAi, isReg),
    };
    if (withBase) {
      // The numbers that matter most: how the merged model does on YOUR photos (held out by person).
      const u = rows.map((r) => !r.base);
      const uIdx = [...u.keys()].filter((i) => u[i]);
      const sub = (a) => uIdx.map((i) => a[i]);
      cv.user = {
        n: uIdx.length, accuracy: r4(uIdx.filter((i) => best.pred[i] === y[i]).length / Math.max(1, uIdx.length)),
        at_suspect: ratesAt(suspect, sub(pAi), sub(isAi), sub(isReg)),
      };
    }

    // Final model on everything
    progress("③ 전체 사진으로 최종 모델 학습 중", 0.985, "");
    const all = Int32Array.from({ length: n }, (_, i) => i);
    const fitted = await runJob(ws[0], { type: "fit", id: "final", train: all, C: best.C }, (m) => m.type === "fitted");
    stopWorkers();
    const head = validateHead({
      format: "ai-glasses-head/v1", embedder: "mobilenet_v3_large.tflite", crop: CROP_VERSION, feature_dim: D,
      classes, ai_class: AI,
      W: fitted.W.map((row) => row.map((v) => Math.round(v * 1e6) / 1e6)), b: fitted.b.map((v) => Math.round(v * 1e6) / 1e6),
      thresholds: { ai: aiT, suspect: r4(suspect) }, C: best.C,
      trained_at: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"),
      n_images: counts, n_user: userCounts, n_base: baseCounts, cv, test: null, note: $("note").value.trim(), trainer: "browser",
      base_data: withBase ? state.base.meta.trained_head_at : null,
    });
    state.head = head;
    state.rows = { rows, pAi, isAi, classes };
    progress("완료", 1, `특징 추출 ${fmtDur(ext.secs)} · 학습 ${fmtDur((performance.now() - t0) / 1000)} · 얼굴 못 찾음 ${ext.noFace}장 · 중복 ${ext.dup}장${ext.bad ? ` · 읽기 실패 ${ext.bad}장` : ""}`);
    renderResults(head, pAi, isAi, isReg, rows, ext);
  } catch (e) {
    console.error(e);
    stopWorkers();
    progress(state.abort ? "중지했습니다" : "학습하지 못했습니다", 0, String(e.message || e));
  } finally {
    state.running = false; $("stopBtn").hidden = true; renderData();
  }
}

/* ---------------- results ---------------- */
function verdictOf(head) {
  const smallest = Math.min(...Object.values(head.n_images));
  const a = head.cv.at_suspect, fa = a.false_alarm_regular ?? a.false_alarm;
  if (smallest < 50) return ["na", "시험용", "사진이 너무 적어 수치를 믿기 어렵습니다. 동작 확인용으로만 쓰세요."];
  if (smallest < 200) return ["sus", "데이터 부족", "수치는 참고만 하세요. 클래스당 200장 이상, 사람 30명 이상으로 늘리면 믿을 만해집니다."];
  if (a.recall >= 0.95 && fa <= 0.05 && (head.cv.ai_auc ?? 0) >= 0.98) return ["ok", "시범 운영 가능", "목표 수준입니다. 실제 고사장에서 시범 운영하며 틀린 사례를 계속 모으세요."];
  if (a.recall >= 0.9 && fa <= 0.15) return ["sus", "보조용으로 사용 가능", "빨간 박스는 반드시 육안으로 확인하세요. 아래 ‘틀린 사진’과 비슷한 조건의 사진을 더 모으면 좋아집니다."];
  return ["ai", "개선 필요", "아직 오판이 많습니다. 아래 ‘틀린 사진’을 확인하고, 정답 폴더가 맞는지, 비슷한 조건의 사진을 더 모을 수 있는지 보세요."];
}

function renderResults(head, pAi, isAi, isReg, rows, ext) {
  const box = $("results"); box.replaceChildren();
  $("resultPanel").hidden = false; $("applyPanel").hidden = false;
  const [cls, title, text] = verdictOf(head);
  const v = el("div", "verdict " + cls);
  const vh = el("div", "vhead"); vh.append(el("span", "vlabel", title));
  v.append(vh, el("p", "vsum", text)); box.append(v);

  const a = head.cv.at_suspect;
  const cards = el("div", "cards");
  for (const [k, val, sub] of [
    ["AI 안경 탐지율", pct(a.recall, 1), `빨간 박스 기준 ${pct(a.threshold)} 에서`],
    ["일반 안경 오탐률", pct(a.false_alarm_regular ?? a.false_alarm, 1), "일반 안경을 빨간 박스로 잘못 표시"],
    ["전체 정확도", pct(head.cv.accuracy, 1), "세 클래스 중 가장 높은 확률 기준"],
    ["AUC", head.cv.ai_auc == null ? "—" : head.cv.ai_auc.toFixed(3), "1에 가까울수록 AI 안경을 잘 구분"],
  ]) { const c = el("div", "card"); c.append(el("span", "ck", k), el("b", "cv mono", val), el("span", "cs", sub)); cards.append(c); }
  box.append(cards);
  if (head.cv.user) {
    const u = head.cv.user, cu = el("div", "cards");
    for (const [k, val, sub] of [
      ["내 사진: AI 안경 탐지율", u.at_suspect.recall == null ? "—" : pct(u.at_suspect.recall, 1), "직접 넣은 AI 안경 사진 기준"],
      ["내 사진: 일반 안경 오탐률", pct(u.at_suspect.false_alarm_regular ?? u.at_suspect.false_alarm, 1), "직접 넣은 일반 안경 사진 기준"],
      ["내 사진: 정확도", pct(u.accuracy, 1), `${u.n}개 얼굴 (반전 포함)`],
    ]) { const c = el("div", "card mine"); c.append(el("span", "ck", k), el("b", "cv mono", val), el("span", "cs", sub)); cu.append(c); }
    box.append(cu);
    box.append(el("p", "hint", `기본 공개 데이터(${Object.entries(head.n_base).map(([c, n]) => `${NAMES[c]} ${n}`).join(" · ")})와 직접 넣은 사진(${Object.entries(head.n_user).map(([c, n]) => `${NAMES[c]} ${n}`).join(" · ")})을 함께 학습했습니다. 위 카드는 전체, 아래 카드는 직접 넣은 사진만의 성능입니다.`));
  }
  box.append(el("p", "hint", `${head.cv.folds}겹 교차검증, 같은 사람의 사진은 학습과 시험에 나뉘지 않게 묶었습니다. 사진 1장 기준 수치이며, 실시간 화면은 여러 프레임을 평균하므로 보통 더 좋습니다. 학습 사진: ${Object.entries(head.n_images).map(([c, n]) => `${NAMES[c]} ${n}`).join(" · ")}`));

  // confusion matrix
  const sec = el("div", "sec"); sec.append(el("h3", null, "혼동행렬 (행 = 실제, 열 = 모델 판단)"));
  const t = el("table", "log cm"), hr = el("tr"); hr.append(el("th"));
  head.classes.forEach((c) => hr.append(el("th", null, NAMES[c])));
  t.append(hr);
  head.cv.confusion.forEach((row, i) => {
    const tr = el("tr"); tr.append(el("th", null, NAMES[head.classes[i]]));
    row.forEach((v2, j) => tr.append(el("td", "mono" + (i === j ? " diag" : v2 ? " off" : ""), String(v2))));
    t.append(tr);
  });
  sec.append(t); box.append(sec);

  // threshold table
  const s2 = el("div", "sec"); s2.append(el("h3", null, "빨간 박스 기준에 따른 변화"));
  const tt = el("table", "log"), th = el("tr");
  ["기준 확률", "AI 안경 탐지", "일반 안경 오탐", "전체 오탐"].forEach((h) => th.append(el("th", null, h)));
  tt.append(th);
  const ts = [...new Set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, a.threshold])].sort((x, y) => x - y);
  for (const thr of ts) {
    const r = ratesAt(thr, pAi, isAi, isReg), tr = el("tr");
    if (thr === a.threshold) tr.className = "chosen";
    tr.append(el("td", "mono", pct(thr, thr === a.threshold ? 1 : 0) + (thr === a.threshold ? " ← 선택" : "")), el("td", "mono", pct(r.recall, 1)),
      el("td", "mono", pct(r.false_alarm_regular, 1)), el("td", "mono", pct(r.false_alarm, 1)));
    tt.append(tr);
  }
  s2.append(tt); box.append(s2);

  // mistakes
  const susT = a.threshold;
  const orig = rows.map((r, i) => ({ r, p: pAi[i], ai: isAi[i] })).filter((x) => !x.r.flip && x.r.item);
  const missed = orig.filter((x) => x.ai && x.p < susT).sort((p, q) => p.p - q.p).slice(0, 12);
  const falseA = orig.filter((x) => !x.ai && x.p >= susT).sort((p, q) => q.p - p.p).slice(0, 12);
  for (const [title2, list, hint] of [
    ["놓친 AI 안경 사진", missed, "모델이 AI 안경으로 보지 못한 사진입니다. 같은 기종·각도·조명 사진을 더 모으세요. 정답 폴더가 잘못된 사진은 빼세요."],
    ["AI 안경으로 잘못 본 사진", falseA, "AI 안경과 닮은 일반 안경 사진을 더 모으면 줄어듭니다."],
  ]) {
    const s3 = el("div", "sec"); s3.append(el("h3", null, `${title2} (${list.length})`));
    if (!list.length) { s3.append(el("p", "hint", "없습니다.")); box.append(s3); continue; }
    s3.append(el("p", "hint", hint));
    const g = el("div", "gallery");
    for (const x of list) {
      const fig = el("figure"), c = el("canvas"); c.width = 160; c.height = 120;
      fig.append(c, el("figcaption", null, `AI ${pct(x.p)} · ${x.r.item.path.split("/").slice(-2).join("/")}`));
      fig.title = x.r.item.path; g.append(fig);
      x.r.item.getBlob().then((b) => createImageBitmap(b, { imageOrientation: "from-image" })).then((bmp) => {
        const s = Math.min(c.width / bmp.width, c.height / bmp.height), w = bmp.width * s, h = bmp.height * s;
        c.getContext("2d").drawImage(bmp, (c.width - w) / 2, (c.height - h) / 2, w, h);
      }).catch(() => {});
    }
    s3.append(g); box.append(s3);
  }
  if (ext.noFace) box.append(el("p", "hint", `얼굴을 찾지 못한 사진 ${ext.noFace}장은 학습에서 빠졌습니다. 얼굴이 작거나 많이 가려진 사진입니다.`));
}

/* ---------------- apply ---------------- */
$("useHere").onclick = () => {
  if (!state.head) return;
  try { localStorage.setItem(HEAD_KEY, JSON.stringify(state.head)); localStorage.removeItem("aiGlassesHeadPref.v1"); } catch { return notice("applyMsg", "err", "이 브라우저에 저장하지 못했습니다. ‘head.json 내려받기’ 후 판독 화면에서 ‘모델 파일 불러오기’를 쓰세요."); }
  const m = el("div", "setup");
  const a = el("a", "btn primary", "실시간 판독 화면 열기"); a.href = "./";
  m.append(el("p", null, "이 기기(이 브라우저)의 판독 화면이 새 모델을 씁니다. 다른 기기와 다른 사용자에게는 적용되지 않습니다 — 아래 ‘공용 모델에 기여’로 올리면 모두의 공용 모델에 합쳐집니다."), a);
  $("applyMsg").replaceChildren(m);
};
$("dlHead").onclick = () => {
  if (!state.head) return;
  download(new Blob([JSON.stringify(state.head)], { type: "application/json" }), "head.json");
};

/* ---------------- contribution to the shared model ---------------- */
// File format: ai-glasses-contribution/v1 (read by training/merge_contributions.py).
// Features only, int8 per row + float32 scale per row (same encoding as base_data.bin).
const REPO = (() => {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/i), repo = location.pathname.split("/").filter(Boolean)[0];
  return m && repo ? `${m[1]}/${repo}` : "Nabi-buter/aiglasses";
})();
function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
/** Device tag from collected file names: "<person>__<device>_<YYYYMMDD>_..." */
function deviceOf(path) {
  const stem = path.split("/").pop().replace(/\.[^.]+$/, ""), after = stem.split("__")[1] || "";
  const tok = after.split("_")[0];
  return tok && !/^\d/.test(tok) ? tok.toLowerCase().slice(0, 32) : "unknown";
}
async function sha256hex(text) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...h].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function makeContribution() {
  if (state.running) return;
  if (!$("consent").checked) return notice("shareMsg", "warnbox", "동의 확인란을 먼저 체크하세요.");
  state.running = true; $("makeShare").disabled = true;
  try {
    const ext = await extractAll($("flip").checked);
    const rows = ext.rows;
    if (!rows.length) throw new Error("넣은 사진에서 얼굴을 찾지 못했습니다.");
    const n = rows.length, salt = crypto.getRandomValues(new Uint32Array(4)).join("-");
    const q = new Int8Array(n * D), scale = new Float32Array(n);
    rows.forEach((r, i) => {
      let m = 0;
      for (let j = 0; j < D; j++) m = Math.max(m, Math.abs(r.feat[j]));
      const sc = Math.max(m, 1e-8) / 127; scale[i] = sc;
      for (let j = 0; j < D; j++) q[i * D + j] = Math.max(-127, Math.min(127, Math.round(r.feat[j] / sc)));
    });
    const classes = KNOWN.filter((c) => rows.some((r) => r.label === c));
    const gNames = [...new Set(rows.map((r) => r.group))], groups = [];
    for (const g of gNames) groups.push((await sha256hex(salt + "|" + g)).slice(0, 12));   // anonymised person/session
    const devs = [...new Set(rows.map((r) => (r.label === AI ? deviceOf(r.item.path) : "")))];
    const d = new Date(), p2 = (v) => String(v).padStart(2, "0");
    const id = `c_${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}_${salt.split("-")[0].slice(-6)}`;
    const counts = Object.fromEntries(classes.map((c) => [c, rows.filter((r) => r.label === c && !r.flip).length]));
    const file = {
      format: "ai-glasses-contribution/v1", id, created_at: d.toISOString().replace(/\.\d+Z$/, "Z"),
      crop: CROP_VERSION, dim: D, n, classes, counts,
      y: rows.map((r) => classes.indexOf(r.label)), group: rows.map((r) => gNames.indexOf(r.group)), groups,
      flip: rows.map((r) => (r.flip ? 1 : 0)), device: rows.map((r) => devs.indexOf(r.label === AI ? deviceOf(r.item.path) : "")), devices: devs,
      note: $("shareNote").value.trim().slice(0, 120), consent: true, app: "web/train.js",
      data_b64: b64(new Uint8Array(q.buffer)), scale_b64: b64(new Uint8Array(scale.buffer)),
    };
    download(new Blob([JSON.stringify(file)], { type: "application/json" }), `${id}.json`);
    state.contribution = file;
    const a = $("uploadLink"); a.href = `https://github.com/${REPO}/upload/main/contributions`; a.hidden = false;
    notice("shareMsg", "setup", `${id}.json 을 내려받았습니다 (얼굴 ${Object.entries(counts).map(([c, k]) => `${NAMES[c]} ${k}`).join(" · ")}, 좌우 반전 포함 ${n}행, 약 ${Math.round((n * D * 1.37) / 1024)}KB). ‘GitHub에 올리기’를 눌러 이 파일을 올리세요.`);
  } catch (e) {
    notice("shareMsg", "err", String(e.message || e));
  } finally {
    state.running = false; renderData();
  }
}

/* ---------------- wire up ---------------- */
$("zipIn").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
$("dirIn").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
$("clearData").onclick = () => { state.items.clear(); state.unlabeled = 0; renderData(); };
$("trainBtn").onclick = train;
$("stopBtn").onclick = () => { state.abort = true; stopWorkers(); };
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault(); drop.classList.remove("over");
  const files = [];
  const items = [...(e.dataTransfer?.items || [])];
  if (items.length && items[0].webkitGetAsEntry) {
    const entries = items.map((i) => i.webkitGetAsEntry()).filter(Boolean);
    for (const en of entries) await walkEntry(en, files);
  } else files.push(...(e.dataTransfer?.files || []));
  addFiles(files);
});
window.__train = state;
$("withBase").onchange = () => renderData();
$("consent").onchange = () => renderData();
$("makeShare").onclick = makeContribution;
renderData();
loadBase().then((b) => {
  if (!b) { $("withBase").checked = false; $("withBase").disabled = true; $("withBaseText").textContent = "기본 공개 데이터 없음 (이 배포에 포함되지 않음)"; }
  else $("withBaseText").textContent = `기본 공개 데이터도 함께 학습 (권장) — ${Object.entries(b.photos).map(([c, n]) => `${NAMES[c]} ${n.toLocaleString()}`).join(" · ")}`;
  renderData();
});
