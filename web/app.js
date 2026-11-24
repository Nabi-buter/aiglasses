import { Detector, blobToCanvas, validateHead, validatePresence, scoreFace, decide, shownProb, GLASSES_KEY } from "./pipeline.js";
import { Tracker } from "./tracker.js";
import { makeZip } from "./zip.js";

const $ = (id) => document.getElementById(id);
const LOG_KEY = "aiGlassesLog.v2";
const HEAD_KEY = "aiGlassesCustomHead.v1";
const OPT_KEY = "aiGlassesOptions.v1";
const PHOTO_MIN_FACE = 40;          // photos are chosen deliberately: classify smaller faces too
const COLLECT_EVERY_MS = 600;
const COLLECT_MAX = 2000;

const COLOR = { ai: "#e5262f", ok: "#1a9f4b", det: "#2f80ed", na: "#8a96a3" };
const CLASS_NAMES = { ai_glasses: "AI 안경", regular_glasses: "일반 안경", no_glasses: "안경 없음" };
const VERDICTS = {
  AI_GLASSES: { cls: "ai", label: "AI 안경", color: COLOR.ai, prob: "ai_glasses",
    steps: ["응시자에게 안경을 벗어 제출하도록 요청하세요.", "다리 안쪽 충전 단자, 경첩 쪽 카메라 렌즈, 브랜드 각인을 직접 확인하세요.", "고사본부 지침에 따라 보고하세요."] },
  SUSPECTED: { cls: "ai", label: "AI 안경 의심", color: COLOR.ai, prob: "ai_glasses",
    steps: ["가까이 가거나 확대해서 측면이 보이게 비춰 보세요. 여러 프레임이 쌓이면 판정이 안정됩니다.", "경첩 부근 카메라 렌즈, LED, 두꺼운 다리를 육안으로 확인하세요.", "의심이 남으면 안경을 벗어 확인하세요."] },
  REGULAR: { cls: "ok", label: "일반 안경", color: COLOR.ok, prob: "regular_glasses", steps: [] },
  NO_GLASSES: { cls: "na", label: "안경 없음", color: COLOR.na, prob: "no_glasses", steps: [] },
  GLASSES: { cls: "det", label: "안경 착용", color: COLOR.det, prob: GLASSES_KEY,
    steps: ["AI 안경인지 일반 안경인지는 판독 모델(head.json)을 학습해야 표시됩니다. 위 ‘모델 학습’을 보세요.", "그 전까지는 육안 확인 포인트를 참고해 직접 확인하세요."] },
};
const isFlag = (v) => v === "AI_GLASSES" || v === "SUSPECTED";

const DEFAULT_OPTS = { hinge: false, noGlasses: true, sound: false, perf: false, minFace: 88 };

const state = {
  detector: null, head: null, headSource: "", presence: null,
  stream: null, facing: "environment", zoom: 1, hwZoom: null,
  mode: "live", paused: false, work: document.createElement("canvas"), lastVideoTime: -1,
  tracker: new Tracker(), photo: null, selectedId: null,
  log: loadLog(), opts: { ...DEFAULT_OPTS, ...readJson(OPT_KEY, {}) },
  collect: { items: [], auto: false, lastAt: 0, seq: 0 },
  perf: { det: 0, cls: 0, fps: 0, lastAt: 0 }, dirty: true, listAt: 0,
};
window.__aig = state; // for tests and debugging in the console

/* ---------------- utils ---------------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function store(key, val) { try { val == null ? localStorage.removeItem(key) : localStorage.setItem(key, val); return true; } catch { return false; } }
function read(key) { try { return localStorage.getItem(key); } catch { return null; } }
function readJson(key, dflt) { try { return JSON.parse(read(key) || "null") ?? dflt; } catch { return dflt; } }
function loadLog() { const v = readJson(LOG_KEY, []); return Array.isArray(v) ? v : []; }
function saveLog() { store(LOG_KEY, JSON.stringify(state.log.slice(0, 120))); }
const pct = (p) => Math.round((p ?? 0) * 100) + "%";
const p2 = (n) => String(n).padStart(2, "0");
function fmtTime(ts) { const d = new Date(ts); return `${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; }
function stamp(d = new Date()) { return `${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}_${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}`; }
function canvasToBlob(c, q = 0.92) { return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("toBlob"))), "image/jpeg", q)); }
function thumb(canvas, w = 160) {
  const c = document.createElement("canvas");
  c.width = w; c.height = Math.max(1, Math.round((w * canvas.height) / canvas.width));
  c.getContext("2d").drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.7);
}
function download(blob, name) {
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
function setSvc(kind, text) { $("svcDot").className = "dot " + (kind === "ok" ? "on" : kind === "bad" ? "off" : kind === "warn" ? "warn" : ""); $("svcText").textContent = text; }
function stageMsg(text, spin = false) {
  $("stageMsg").hidden = !text; $("stageMsgText").textContent = text || "";
  $("stageMsg").querySelector(".spin").hidden = !spin;
}

/* ---------------- models ---------------- */
async function loadModels() {
  try {
    state.detector = await Detector.create(".");
  } catch (e) {
    console.error(e);
    setSvc("bad", "판독 엔진을 불러오지 못했습니다");
    stageMsg("판독 엔진을 불러오지 못했습니다. 최신 Chrome·Edge·Safari에서 열어 주세요. 파일을 직접 열었다면 웹 서버(예: GitHub Pages)로 열어야 합니다.");
    return;
  }
  try {
    const r = await fetch("models/presence.json", { cache: "no-cache" });
    if (r.ok) state.presence = validatePresence(await r.json());
  } catch (e) { console.warn("presence.json:", e); }
  const custom = read(HEAD_KEY);
  if (custom) {
    try { useHead(validateHead(JSON.parse(custom)), "custom"); } catch { store(HEAD_KEY, null); }
  }
  if (!state.head) await loadBundledHead();
  if (!state.head) useHead(null, "");
  updateButtons();
}
async function loadBundledHead() {
  try {
    const r = await fetch("models/head.json", { cache: "no-cache" });
    if (r.ok) useHead(validateHead(await r.json()), "bundled");
  } catch (e) { console.warn("head.json:", e); }
}

function useHead(head, source) {
  state.head = head; state.headSource = source;
  for (const t of state.tracker.tracks) { t.samples = []; t.probs = null; t.verdict = null; t.stable = false; t.lastClassified = -Infinity; t.flagged = false; }
  if (head) setSvc("ok", "판독 준비 완료 · 기기 안에서 실행");
  else if (state.presence) setSvc("warn", "AI 안경 판독 모델 없음 · 안경 유무만 판독");
  else setSvc("bad", "학습된 판독 모델이 없습니다 · 안경 영역만 표시");
  $("noModelTag").hidden = !!head;
  $("noModelTag").textContent = state.presence ? "AI 안경 판독 모델 없음 · 안경 착용 여부만 표시" : "판독 모델 없음 · 얼굴의 안경 위치만 표시";
  $("resetHead").hidden = source !== "custom";
  renderModelInfo();
  if (state.mode === "photo" && state.photo) analyzePhoto(state.photo.canvas);
  state.dirty = true;
}

function renderModelInfo() {
  const box = $("modelInfo"); box.replaceChildren();
  const h = state.head, pr = state.presence;
  if (pr) {
    const d = el("dl");
    d.append(el("dt", null, "안경 착용 판독"), el("dd", null,
      `기본 포함 · 공개 사진 ${pr.cv.n.toLocaleString()}명 얼굴로 학습 · 정확도 ${pct(pr.cv.accuracy)} (안경 ${pct(pr.cv.glasses_recall)} · 안경 없음 ${pct(pr.cv.no_glasses_recall)})`));
    box.append(d);
  }
  if (!h) {
    const s = el("div", "setup");
    s.append(el("p", null, pr
      ? "AI 안경 판독 모델(head.json)이 아직 없습니다. 지금은 안경을 썼는지(파란 박스)·안 썼는지(회색 박스)만 판독합니다. AI 안경(빨강)과 일반 안경(초록)을 구분하려면 AI 안경 사진으로 모델을 학습해야 합니다."
      : "아직 학습된 모델이 없습니다. 화면의 박스는 얼굴에서 안경이 있을 위치일 뿐, 안경 착용 여부와 종류는 판독하지 않습니다."));
    const a = el("a", "btn primary", "모델 학습 페이지 열기"); a.href = "train.html";
    s.append(el("p", null, "아래 ‘학습 데이터 수집’으로 사진을 모은 뒤, 모델 학습 페이지에 ZIP을 넣으면 이 브라우저에서 바로 학습됩니다."), a);
    box.append(s);
    return;
  }
  const dl = el("dl");
  const add = (k, v) => { if (v == null || v === "") return; dl.append(el("dt", null, k), el("dd", null, String(v))); };
  add("출처", state.headSource === "custom" ? "직접 불러온 파일 (이 브라우저에 저장됨)" : "기본 포함 모델");
  add("클래스", h.classes.map((c) => CLASS_NAMES[c] || c).join(", "));
  if (h.n_images) add("학습 사진", Object.entries(h.n_images).map(([c, n]) => `${CLASS_NAMES[c] || c} ${n}`).join(" · "));
  if (h.cv) add("교차검증", `정확도 ${pct(h.cv.accuracy)} · AI 안경 재현율 ${pct(h.cv.ai_recall)} · 정밀도 ${pct(h.cv.ai_precision)}` + (h.cv.ai_auc != null ? ` · AUC ${h.cv.ai_auc.toFixed(3)}` : ""));
  if (h.test) add("별도 평가", `${h.test.n}장 · 정확도 ${pct(h.test.accuracy)} · AI 안경 재현율 ${pct(h.test.ai_recall)} · 정밀도 ${pct(h.test.ai_precision)}`);
  add("판정 기준", `AI 안경 확률 ${pct(h.thresholds?.ai ?? 0.6)} 이상 → AI 안경, ${pct(h.thresholds?.suspect ?? 0.3)} 이상 → AI 안경 의심 (빨간 박스)`);
  if (h.cv?.at_suspect) {
    const a = h.cv.at_suspect;
    add("빨간 박스 기준 예상", `AI 안경 놓침 ${pct(1 - a.recall)} · ` + (a.false_alarm_regular != null
      ? `일반 안경을 빨간 박스로 잘못 표시 ${pct(a.false_alarm_regular)}` : `다른 사람을 잘못 표시 ${pct(a.false_alarm)}`) + " (사진 1장 기준)");
  }
  add("학습 시각", h.trained_at ? new Date(h.trained_at).toLocaleString("ko-KR") : "");
  add("메모", h.note);
  box.append(dl);
  const small = Math.min(...Object.values(h.n_images || { x: 0 }));
  if (small < 200) box.append(el("div", "warnbox", `클래스당 학습 사진이 ${small}장뿐입니다. 실전에 쓰기 전에 클래스당 200장 이상(사람 30명 이상)으로 늘려 다시 학습하세요.`));
  if (!h.classes.includes("no_glasses")) box.append(el("div", "warnbox", "‘안경 없음’ 클래스 없이 학습한 모델입니다. 안경을 쓰지 않은 사람도 AI 안경·일반 안경 중 하나로 표시됩니다."));
}

/* ---------------- camera ---------------- */
async function startCamera() {
  stopCamera();
  $("retryCam").hidden = true;
  if (state.mode === "live") stageMsg("카메라를 켜는 중입니다…", true);
  if (!navigator.mediaDevices?.getUserMedia) return cameraUnavailable();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
    });
    state.stream = stream;
    const v = $("video"); v.srcObject = stream;
    await v.play().catch(() => {});
    const track = stream.getVideoTracks()[0];
    const caps = track?.getCapabilities?.() || {};
    state.hwZoom = caps.zoom && caps.zoom.max > caps.zoom.min ? caps.zoom : null;
    state.facing = track?.getSettings?.().facingMode || state.facing;
    try { $("flipBtn").hidden = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput").length < 2; } catch {}
    applyZoom();
    state.tracker.reset();
    if (state.mode === "live") stageMsg("");
    updateButtons();
  } catch (e) { console.warn(e); cameraUnavailable(); }
}
function cameraUnavailable() {
  state.stream = null;
  $("retryCam").hidden = false; $("flipBtn").hidden = true;
  if (state.mode === "live") stageMsg("카메라를 쓸 수 없습니다. 브라우저의 카메라 권한을 허용하고 ‘카메라 다시 시도’를 누르세요. 카메라 없이도 ‘사진 판독’으로 사진을 판독할 수 있습니다.");
  updateButtons();
}
function stopCamera() { state.stream?.getTracks().forEach((t) => t.stop()); state.stream = null; }
async function applyZoom() {
  $("zoomVal").textContent = state.zoom.toFixed(1) + "×";
  const track = state.stream?.getVideoTracks()[0];
  if (state.hwZoom && track) {
    // Real (sensor/optical) zoom gives the classifier more pixels than cropping does.
    const z = Math.min(state.hwZoom.max, Math.max(state.hwZoom.min, state.zoom));
    try { await track.applyConstraints({ advanced: [{ zoom: z }] }); state.digital = state.zoom / z; } catch { state.digital = state.zoom; }
  } else state.digital = state.zoom;
  state.tracker.reset();
}
const mirrored = () => state.mode === "live" && state.facing === "user";

/* ---------------- live loop ---------------- */
function grabFrame() {
  const v = $("video"), vw = v.videoWidth, vh = v.videoHeight;
  const z = Math.max(1, state.digital || 1);
  const sw = Math.round(vw / z), sh = Math.round(vh / z);
  const w = state.work;
  if (w.width !== sw || w.height !== sh) { w.width = sw; w.height = sh; }
  w.getContext("2d", { willReadFrequently: true }).drawImage(v, Math.round((vw - sw) / 2), Math.round((vh - sh) / 2), sw, sh, 0, 0, sw, sh);
}

function process(now) {
  grabFrame();
  const t0 = performance.now();
  let faces = [];
  try { faces = state.detector.facesInFrame(state.work, now); } catch (e) { console.error(e); }
  const t1 = performance.now();
  const tracks = state.tracker.update(faces, now);
  if (state.head || state.presence) {
    const t = state.tracker.next(now, (x) => x.facePx >= state.opts.minFace);
    if (t) classifyTrack(t, state.work, now);
  }
  const t2 = performance.now();
  const P = state.perf, a = 0.15;
  P.det += a * (t1 - t0 - P.det);
  if (t2 - t1 > 1) P.cls += a * (t2 - t1 - P.cls);
  if (P.lastAt) P.fps += a * (1000 / Math.max(1, now - P.lastAt) - P.fps);
  P.lastAt = now;
  if (state.collect.auto && now - state.collect.lastAt >= COLLECT_EVERY_MS) {
    const big = tracks.filter((x) => x.visible && x.facePx >= state.opts.minFace).sort((p, q) => q.facePx - p.facePx)[0];
    if (big) { state.collect.lastAt = now; collectFace(big, $("cLabel").value, "auto"); }
  }
  if (state.selectedId != null && !tracks.some((x) => x.id === state.selectedId)) state.selectedId = null;
  state.dirty = true;
}

function classifyTrack(t, canvas, now) {
  const r = state.detector.featureFor(canvas, t.boxes);
  if (!r) return;
  const probs = scoreFace(state.head, state.presence, r.feature);
  state.tracker.addSample(t, probs, now);
  t.crops = r.crops; t.cropAt = now;
  t.verdict = decide(state.head, state.presence, t.probs);
  if (t.stable && state.head) t.maxAi = Math.max(t.maxAi || 0, t.probs[state.head.ai_class] ?? 0);
  if (t.stable && isFlag(t.verdict) && !t.flagged) { t.flagged = true; onFlag(t); }
}

function onFlag(t) {
  addLog(t, "live", "auto");
  const meta = VERDICTS[t.verdict];
  const b = $("alertBanner");
  b.textContent = `${meta.label} 감지 · #${t.id} · AI 확률 ${pct(t.probs[state.head.ai_class])} — 안경을 직접 확인하세요`;
  b.hidden = false;
  clearTimeout(onFlag.timer); onFlag.timer = setTimeout(() => (b.hidden = true), 5000);
  if (state.selectedId == null) state.selectedId = t.id;
  if (state.opts.sound) {
    try { navigator.vibrate?.([120, 80, 120]); } catch {}
    try {
      const ac = onFlag.ac || (onFlag.ac = new AudioContext());
      const o = ac.createOscillator(), g = ac.createGain();
      o.frequency.value = 880; g.gain.value = 0.08; o.connect(g); g.connect(ac.destination);
      o.start(); o.stop(ac.currentTime + 0.18);
    } catch {}
  }
}

function tick(now) {
  requestAnimationFrame(tick);
  const v = $("video");
  if (state.mode === "live" && !state.paused && state.detector && state.stream && v.readyState >= 2 && v.videoWidth) {
    if (v.currentTime !== state.lastVideoTime) { state.lastVideoTime = v.currentTime; process(now); }
  }
  if (state.dirty) { state.dirty = false; render(); }
  if (now - state.listAt > 300) { state.listAt = now; renderPeople(); }
}

/* ---------------- drawing ---------------- */
function currentTracks() {
  if (state.mode === "photo") return state.photo?.tracks || [];
  return state.tracker.tracks.filter((t) => t.visible || state.paused);
}
function sourceCanvas() { return state.mode === "photo" ? state.photo?.canvas : state.work; }

function styleFor(t, minFace) {
  const far = t.facePx < minFace;
  if (!state.head && !state.presence) return { color: COLOR.det, text: "안경 위치", dashed: far };
  if (!t.probs || !t.verdict) return far ? { color: COLOR.det, text: "멀어요 · 확대하세요", dashed: true } : { color: COLOR.det, text: "안경 인식 · 판독 중" };
  const meta = VERDICTS[t.verdict] || VERDICTS.SUSPECTED;
  return { color: meta.color, text: `${meta.label} ${pct(shownProb(state.head, t.verdict, t.probs) ?? 0)}`, dashed: far, faint: !t.stable, hide: t.verdict === "NO_GLASSES" && !state.opts.noGlasses, na: t.verdict === "NO_GLASSES" };
}

/** Draw the frame and the boxes. W×H is the output size in drawing units; scale maps frame px -> units. */
function drawScene(ctx, src, W, H, tracks, { mirror = false, minFace = state.opts.minFace, selected = null, unit = 1 } = {}) {
  // unit: drawing units per CSS pixel (devicePixelRatio on screen, larger for full-resolution snapshots)
  const s = W / src.width;
  ctx.save();
  if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  ctx.drawImage(src, 0, 0, W, H);
  ctx.restore();
  const X = (x) => (mirror ? W - x * s : x * s);
  const cssW = W / unit;
  const fs = Math.round(Math.min(20, Math.max(13, cssW / 42)) * unit);
  const lw = Math.max(2, Math.round(Math.max(2.5, cssW / 300) * unit));
  ctx.textBaseline = "middle";
  ctx.font = `700 ${fs}px "Pretendard","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif`;
  // draw low-priority boxes first so red ones end up on top
  const rank = (t) => (isFlag(t.verdict) ? 3 : t.verdict === "REGULAR" ? 2 : t.verdict === "NO_GLASSES" ? 0 : 1);
  for (const t of [...tracks].sort((a, b) => rank(a) - rank(b))) {
    const st = styleFor(t, minFace);
    if (st.hide) continue;
    const xa = X(t.box.x0), xb = X(t.box.x1);
    const x0 = Math.min(xa, xb), x1 = Math.max(xa, xb), y0 = t.box.y0 * s, y1 = t.box.y1 * s;
    const sel = selected === t.id;
    const w = (st.na ? lw * 0.6 : lw) * (sel ? 1.6 : 1);
    ctx.globalAlpha = st.faint ? 0.75 : 1;
    if (sel) { ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = w + 2 * unit; ctx.setLineDash([]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); }
    ctx.strokeStyle = st.color; ctx.lineWidth = w;
    ctx.setLineDash(st.dashed ? [lw * 4, lw * 3] : []);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);

    if (state.opts.hinge && t.boxes) {
      ctx.lineWidth = Math.max(1, lw / 2); ctx.setLineDash([lw * 2, lw * 2]);
      for (const k of ["hinge_l", "hinge_r"]) {
        const b = t.boxes[k]; if (!b || b.empty) continue;
        const ha = X(b.x0), hb = X(b.x1);
        ctx.strokeRect(Math.min(ha, hb), b.y0 * s, Math.abs(hb - ha), (b.y1 - b.y0) * s);
      }
      ctx.setLineDash([]);
    }

    const text = st.text, idText = `#${t.id}`;
    const f = st.na ? Math.round(fs * 0.8) : fs;
    ctx.font = `700 ${f}px "Pretendard","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif`;
    const tw = ctx.measureText(text).width, iw = ctx.measureText(idText).width * 0.8;
    const padX = f * 0.5, th = Math.round(f * 1.6), tabW = tw + iw + padX * 3;
    let ty = y0 - th; if (ty < 0) ty = y0;               // no room above: put the tab inside the box
    let tx = x0 - w / 2; if (tx + tabW > W) tx = W - tabW;
    ctx.fillStyle = st.color; ctx.fillRect(Math.max(0, tx), ty, tabW, th);
    ctx.fillStyle = "#fff"; ctx.fillText(text, Math.max(0, tx) + padX, ty + th / 2 + 1);
    ctx.globalAlpha *= 0.8;
    ctx.font = `600 ${Math.round(f * 0.8)}px ui-monospace,Menlo,monospace`;
    ctx.fillText(idText, Math.max(0, tx) + padX * 2 + tw, ty + th / 2 + 1);
    ctx.globalAlpha = 1;
  }
}

function render() {
  const src = sourceCanvas(), c = $("view");
  if (!src || !src.width || (state.mode === "live" && !state.stream)) { c.hidden = true; $("stage").style.height = ""; $("modeTag").hidden = true; return; }
  c.hidden = false;
  const stage = $("stage");
  const maxH = Math.max(260, window.innerHeight * 0.72);
  let W = stage.clientWidth, H = W * src.height / src.width;
  if (H > maxH) { H = maxH; W = H * src.width / src.height; }
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const bw = Math.round(W * dpr), bh = Math.round(H * dpr);
  if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
  c.style.width = W + "px"; c.style.height = H + "px";
  stage.style.height = H + "px";
  const ctx = c.getContext("2d");
  drawScene(ctx, src, bw, bh, currentTracks(), {
    mirror: mirrored(), selected: state.selectedId, unit: dpr,
    minFace: state.mode === "photo" ? PHOTO_MIN_FACE : state.opts.minFace,
  });
  const P = state.perf;
  $("perfTag").hidden = !state.opts.perf || state.mode !== "live";
  $("perfTag").textContent = `${P.fps.toFixed(0)} fps · 얼굴 ${P.det.toFixed(0)}ms · 판독 ${P.cls.toFixed(0)}ms`;
  $("modeTag").hidden = false;
  $("modeTag").className = "mode " + (state.mode === "photo" ? "photo" : state.paused ? "paused" : "live");
  $("modeText").textContent = state.mode === "photo" ? "사진 판독" : state.paused ? "멈춤" : "실시간";
}

function hitTest(ev) {
  const c = $("view"), r = c.getBoundingClientRect(), src = sourceCanvas();
  if (!src) return null;
  const s = src.width / r.width;
  let x = (ev.clientX - r.left) * s; const y = (ev.clientY - r.top) * s;
  if (mirrored()) x = src.width - x;
  const hits = currentTracks().filter((t) => x >= t.box.x0 && x <= t.box.x1 && y >= t.box.y0 - 30 * s && y <= t.box.y1);
  return hits.sort((a, b) => (a.box.x1 - a.box.x0) - (b.box.x1 - b.box.x0))[0] || null;
}

/* ---------------- side list ---------------- */
function trackSummary(t) {
  if (!state.head && !state.presence) return { cls: "det", label: "안경 위치 (모델 없음)", p: null };
  if (!t.probs || !t.verdict) return { cls: "det", label: t.facePx < state.opts.minFace && state.mode === "live" ? "멀어서 보류" : "판독 중", p: null };
  const meta = VERDICTS[t.verdict];
  return { cls: meta.cls, label: meta.label, p: shownProb(state.head, t.verdict, t.probs) };
}

function renderPeople() {
  const list = $("people"), tracks = currentTracks();
  $("peopleCount").textContent = tracks.length ? `${tracks.length}명` : "";
  list.replaceChildren();
  if (!tracks.length) {
    list.append(el("p", "empty", state.mode === "photo" ? "사진에서 얼굴을 찾지 못했습니다." :
      state.stream ? "화면에 얼굴이 보이면 여기에 표시됩니다. 응시자의 얼굴과 안경이 보이게 비춰 주세요." : "카메라가 꺼져 있습니다."));
  }
  for (const t of tracks) {
    const s = trackSummary(t);
    const b = el("button", "person " + s.cls + (state.selectedId === t.id ? " sel" : ""));
    b.type = "button";
    b.onclick = () => { state.selectedId = state.selectedId === t.id ? null : t.id; state.dirty = true; renderPeople(); };
    b.append(el("i", "sw " + s.cls), el("span", "pid mono", "#" + t.id), el("span", "plabel", s.label),
      el("span", "pval mono", s.p == null ? "" : pct(s.p)),
      el("span", "pmeta", state.mode === "photo" ? "" : t.samples?.length ? `${t.samples.length}프레임` : ""));
    list.append(b);
  }
  renderDetail(tracks.find((t) => t.id === state.selectedId));
}

function renderDetail(t) {
  const box = $("detail");
  if (!t) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false; box.replaceChildren();
  const s = trackSummary(t);
  const head = el("div", "dhead");
  head.append(el("b", null, `#${t.id} ${s.label}`), el("span", "mono", s.p == null ? "" : pct(s.p)));
  box.append(head);
  if (t.crops) {
    const row = el("div", "crops");
    for (const [k, cap] of [["band", "안경 전체"], ["hinge_l", "경첩 (화면 왼쪽)"], ["hinge_r", "경첩 (화면 오른쪽)"]]) {
      const fig = el("figure");
      if (t.crops[k]) {
        const c = el("canvas"); c.width = t.crops[k].width; c.height = t.crops[k].height;
        c.getContext("2d").drawImage(t.crops[k], 0, 0); fig.append(c);
      } else fig.append(el("div", "slot", "화면 밖"));
      fig.append(el("figcaption", null, cap)); row.append(fig);
    }
    box.append(el("p", "hint", "모델이 실제로 본 영역입니다. 경첩 칸에 안경테 모서리가 들어와야 카메라 렌즈를 볼 수 있습니다."), row);
  }
  if (t.probs) {
    const wrap = el("div", "probs");
    const bars = [];
    if (state.presence && t.probs[GLASSES_KEY] != null) bars.push(["안경 착용 (기본 모델)", t.probs[GLASSES_KEY], "det"]);
    if (state.head) for (const c of [...state.head.classes].sort((a, b) => (a === state.head.ai_class ? -1 : b === state.head.ai_class ? 1 : a.localeCompare(b))))
      bars.push([CLASS_NAMES[c] || c, t.probs[c], c === state.head.ai_class ? "ai" : c === "regular_glasses" ? "reg" : "na"]);
    for (const [name, v, cls] of bars) {
      const row = el("div", "prow " + cls);
      const bar = el("div", "pb"), fill = el("i"); fill.style.width = pct(v); bar.append(fill);
      row.append(el("span", null, name), bar, el("span", "pv", pct(v)));
      wrap.append(row);
    }
    box.append(wrap);
    if (state.mode === "live") box.append(el("p", "hint", `최근 ${t.samples.length}개 프레임 평균${t.stable ? "" : " · 아직 판정이 안정되지 않았습니다"}${t.maxAi ? ` · 지금까지 최고 AI 확률 ${pct(t.maxAi)}` : ""}`));
    const steps = VERDICTS[t.verdict]?.steps || [];
    if (steps.length) { const ol = el("ol", "steps"); steps.forEach((x) => ol.append(el("li", null, x))); box.append(ol); }
  }
  const act = el("div", "controls");
  const logBtn = el("button", "small", "기록에 남기기");
  logBtn.disabled = !t.probs || !state.head;
  logBtn.onclick = () => { addLog(t, state.mode, "manual"); logBtn.textContent = "기록했습니다"; logBtn.disabled = true; };
  act.append(logBtn);
  box.append(act);
  const fb = el("div", "feedback");
  fb.append(el("span", null, "판정이 틀렸나요? 실제 정답을 누르면 이 얼굴이 학습 데이터에 담깁니다:"));
  for (const [lab, name] of [["ai_glasses", "AI 안경"], ["regular_glasses", "일반 안경"], ["no_glasses", "안경 없음"]]) {
    const b = el("button", "small ghost", name);
    b.onclick = async () => { await collectFace(t, lab, "correction"); b.textContent = name + " ✓"; };
    fb.append(b);
  }
  box.append(fb);
}

/* ---------------- photo mode ---------------- */
async function openPhoto(file) {
  let c;
  try { c = await blobToCanvas(file); } catch { return showAlert("이 사진을 열 수 없습니다. JPG나 PNG 사진으로 다시 시도하세요."); }
  const s = Math.min(1, 2400 / Math.max(c.width, c.height));
  if (s < 1) {
    const d = el("canvas"); d.width = Math.round(c.width * s); d.height = Math.round(c.height * s);
    d.getContext("2d").drawImage(c, 0, 0, d.width, d.height); c = d;
  }
  state.mode = "photo"; state.selectedId = null; state.photo = { canvas: c, tracks: [] };
  $("liveBtn").hidden = false; stageMsg(state.detector ? "" : "판독 엔진을 불러오는 중입니다…", !state.detector);
  updateButtons();
  while (!state.detector) await new Promise((r) => setTimeout(r, 200));
  stageMsg("");
  analyzePhoto(c, true);
}

function analyzePhoto(canvas, log = false) {
  const faces = state.detector.facesInImage(canvas);
  const tracks = faces.map((f, i) => ({ id: i + 1, box: f.boxes.band, boxes: f.boxes, facePx: f.facePx, rect: f.rect, samples: [], visible: true }));
  for (const t of tracks) {
    if (!(state.head || state.presence) || t.facePx < PHOTO_MIN_FACE) continue;
    const r = state.detector.featureFor(canvas, t.boxes);
    if (!r) continue;
    t.probs = scoreFace(state.head, state.presence, r.feature); t.verdict = decide(state.head, state.presence, t.probs);
    t.stable = true; t.crops = r.crops;
    if (log && state.head && t.verdict !== "NO_GLASSES") addLog(t, "photo", "auto");
  }
  state.photo.tracks = tracks;
  state.selectedId = tracks.find((t) => isFlag(t.verdict))?.id ?? (tracks.length === 1 ? 1 : null);
  if (!tracks.length) showAlert("사진에서 얼굴을 찾지 못했습니다. 얼굴과 안경이 크게 나온 사진을 쓰세요.");
  state.dirty = true; renderPeople();
}

function backToLive() {
  state.mode = "live"; state.photo = null; state.selectedId = null; state.paused = false;
  $("liveBtn").hidden = true;
  state.tracker.reset();
  if (!state.stream) cameraUnavailable(); else stageMsg("");
  updateButtons(); state.dirty = true;
}

function showAlert(text) {
  const b = $("alertBanner"); b.textContent = text; b.hidden = false; b.classList.add("info");
  clearTimeout(showAlert.t); showAlert.t = setTimeout(() => { b.hidden = true; b.classList.remove("info"); }, 4000);
}

/* ---------------- snapshot ---------------- */
async function snapshot() {
  const src = sourceCanvas(); if (!src?.width) return;
  const c = el("canvas"); c.width = src.width; c.height = src.height;
  const ctx = c.getContext("2d");
  drawScene(ctx, src, c.width, c.height, currentTracks(), { mirror: mirrored(), unit: Math.max(1, c.width / 1000), minFace: state.mode === "photo" ? PHOTO_MIN_FACE : state.opts.minFace });
  const info = [new Date().toLocaleString("ko-KR"), $("room").value.trim(), "AI 안경 판독기 (보조 판독)"].filter(Boolean).join("  ·  ");
  const f = Math.max(12, Math.round(c.width / 70));
  ctx.font = `600 ${f}px system-ui,sans-serif`; ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(0, c.height - f * 1.8, ctx.measureText(info).width + f, f * 1.8);
  ctx.fillStyle = "#fff"; ctx.fillText(info, f / 2, c.height - f * 0.9);
  download(await canvasToBlob(c, 0.9), `aiglasses_snapshot_${stamp()}.jpg`);
}

/* ---------------- log ---------------- */
function addLog(t, src, how) {
  if (!state.head || !t.probs) return;
  state.log.unshift({
    ts: Date.now(), seat: how === "manual" || state.selectedId === t.id ? $("seat").value.trim().slice(0, 40) : "",
    room: $("room").value.trim().slice(0, 40), verdict: t.verdict, pAi: t.probs[state.head.ai_class] ?? null,
    thumb: t.crops?.band ? thumb(t.crops.band, 160) : null, src, id: t.id, n: t.samples?.length || 1,
  });
  state.log = state.log.slice(0, 120);
  saveLog(); renderLog();
}
function renderLog() {
  const body = $("logBody"); body.replaceChildren();
  if (!state.log.length) {
    const tr = el("tr"), td = el("td", "empty", "아직 기록이 없습니다.");
    td.colSpan = 5; tr.append(td); body.append(tr);
  }
  for (const it of state.log) {
    const meta = VERDICTS[it.verdict] || { cls: "na", label: it.verdict };
    const tr = el("tr"), tdImg = el("td");
    if (typeof it.thumb === "string" && it.thumb.startsWith("data:image/")) { const im = el("img"); im.src = it.thumb; im.alt = ""; tdImg.append(im); }
    const tdV = el("td"); tdV.append(el("span", "pill " + meta.cls, meta.label));
    tr.append(tdImg, el("td", "mono", fmtTime(it.ts)), el("td", null, [it.room, it.seat].filter(Boolean).join(" · ") || "—"),
      tdV, el("td", "mono", it.pAi == null ? "—" : pct(it.pAi)));
    body.append(tr);
  }
  const c = $("counts"); c.replaceChildren(el("span", "pill na", `전체 ${state.log.length}건`));
  const tally = new Map();
  for (const it of state.log) { const m = VERDICTS[it.verdict]; if (m) tally.set(m, (tally.get(m) || 0) + 1); }
  for (const [m, n] of tally) c.append(el("span", "pill " + m.cls, `${m.label} ${n}`));
  $("clearLog").disabled = $("exportCsv").disabled = !state.log.length;
}
function askClear() {
  const box = $("confirmBox"); box.hidden = false; box.replaceChildren();
  const w = el("div", "confirm");
  const yes = el("button", "small danger", "기록 모두 지우기");
  const no = el("button", "small ghost", "취소");
  yes.onclick = () => { state.log = []; saveLog(); renderLog(); box.hidden = true; };
  no.onclick = () => { box.hidden = true; };
  w.append(el("span", null, `기록 ${state.log.length}건을 지울까요? 되돌릴 수 없습니다.`), yes, no);
  box.append(w);
}
function exportCsv() {
  const q = (s) => '"' + String(s ?? "").replace(/"/g, '""') + '"';
  const rows = [["시각", "고사실", "좌석", "판정", "AI 안경 확률", "평균 프레임 수", "출처"].map(q).join(",")];
  for (const it of state.log.slice().reverse())
    rows.push([new Date(it.ts).toLocaleString("ko-KR"), it.room, it.seat, (VERDICTS[it.verdict] || {}).label || it.verdict,
      it.pAi == null ? "" : pct(it.pAi), it.n ?? "", it.src === "photo" ? "사진" : "실시간"].map(q).join(","));
  download(new Blob(["\ufeff" + rows.join("\r\n")], { type: "text/csv;charset=utf-8" }), `aiglasses_log_${stamp()}.csv`);
}

/* ---------------- training data collection ---------------- */
const safeId = (s, d) => (s || "").trim().replace(/[^0-9A-Za-z가-힣_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32) || d;

async function collectFace(t, label, source) {
  const C = state.collect, src = sourceCanvas();
  if (!src || !t.rect || C.items.length >= COLLECT_MAX) { if (C.items.length >= COLLECT_MAX) setAuto(false); return; }
  // Face with a generous margin: train.py runs the face finder again on this crop.
  const r = t.rect, fw = r.x1 - r.x0, fh = r.y1 - r.y0, m = 0.6 * Math.max(fw, fh);
  const x0 = Math.max(0, Math.floor(r.x0 - m)), y0 = Math.max(0, Math.floor(r.y0 - m));
  const x1 = Math.min(src.width, Math.ceil(r.x1 + m)), y1 = Math.min(src.height, Math.ceil(r.y1 + m));
  const k = Math.min(1, 1024 / Math.max(x1 - x0, y1 - y0));
  const c = el("canvas"); c.width = Math.round((x1 - x0) * k); c.height = Math.round((y1 - y0) * k);
  c.getContext("2d").drawImage(src, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
  const data = new Uint8Array(await (await canvasToBlob(c, 0.92)).arrayBuffer());
  const session = source === "correction" ? `field-${stamp().slice(0, 8)}-${t.id}` : safeId($("cSession").value, "p01");
  const device = safeId($("cDevice").value, "");
  C.seq += 1;
  const name = `${label}/${session}__${device ? device + "_" : ""}${stamp()}_${String(C.seq).padStart(4, "0")}.jpg`;
  C.items.push({ name, data, label, session, device, source, pAi: t.probs && state.head ? t.probs[state.head.ai_class] : null, verdict: t.verdict || "" });
  renderCollect();
}

function setAuto(on) {
  state.collect.auto = on && !!state.stream && state.mode === "live";
  $("cAuto").textContent = state.collect.auto ? "자동 수집 멈춤" : "자동 수집 시작";
  $("cAuto").classList.toggle("danger", state.collect.auto);
  renderCollect();
}

function renderCollect() {
  const C = state.collect, st = $("cStats"); st.replaceChildren();
  const by = {};
  for (const it of C.items) by[it.label] = (by[it.label] || 0) + 1;
  st.append(el("span", "pill na", `모은 사진 ${C.items.length}장`));
  for (const [l, n] of Object.entries(by)) st.append(el("span", "pill " + (l === "ai_glasses" ? "ai" : l === "regular_glasses" ? "ok" : "na"), `${CLASS_NAMES[l]} ${n}`));
  if (C.auto) st.append(el("span", "pill det", "● 수집 중"));
  $("collectBadge").hidden = !C.items.length; $("collectBadge").textContent = C.items.length + "장";
  $("cZip").disabled = $("cClear").disabled = !C.items.length;
  updateButtons();
}

function exportZip() {
  const C = state.collect; if (!C.items.length) return;
  const q = (s) => '"' + String(s ?? "").replace(/"/g, '""') + '"';
  const csv = ["file,label,session,device,source,ai_prob_at_capture,verdict_at_capture",
    ...C.items.map((it) => [it.name, it.label, it.session, it.device, it.source, it.pAi == null ? "" : it.pAi.toFixed(4), it.verdict].map(q).join(","))].join("\n");
  const entries = [...C.items.map((it) => ({ name: it.name, data: it.data })), { name: "labels.csv", data: new TextEncoder().encode(csv) }];
  download(makeZip(entries), `aiglasses_dataset_${stamp()}.zip`);
  C.exported = C.items.length;
}

/* ---------------- options ---------------- */
function bindOptions() {
  const o = state.opts, save = () => { store(OPT_KEY, JSON.stringify(o)); state.dirty = true; };
  for (const [id, key] of [["optHinge", "hinge"], ["optNoGlasses", "noGlasses"], ["optSound", "sound"], ["optPerf", "perf"]]) {
    $(id).checked = !!o[key];
    $(id).onchange = (e) => { o[key] = e.target.checked; save(); };
  }
  const mf = $("optMinFace");
  mf.value = o.minFace; $("optMinFaceVal").textContent = o.minFace + "px";
  mf.oninput = () => { o.minFace = Number(mf.value); $("optMinFaceVal").textContent = o.minFace + "px"; save(); };
}

function updateButtons() {
  const live = state.mode === "live";
  $("pauseBtn").disabled = !live || !state.stream;
  $("pauseBtn").textContent = state.paused ? "다시 시작" : "멈춤";
  $("snapBtn").disabled = !sourceCanvas()?.width || (live && !state.stream);
  $("zoom").disabled = !live;
  const canCollect = !!state.detector && live && !!state.stream;
  $("cAuto").disabled = !canCollect;
  $("cOne").disabled = !canCollect;
}

async function loadHeadFile(file) {
  try {
    const head = validateHead(JSON.parse(await file.text()));
    if (!store(HEAD_KEY, JSON.stringify(head))) console.warn("head not persisted");
    useHead(head, "custom");
    showAlert("새 판독 모델을 불러왔습니다.");
  } catch (e) {
    showAlert("모델 파일을 쓸 수 없습니다: " + (e?.message || e));
  }
}
async function resetHead() {
  store(HEAD_KEY, null); state.head = null;
  await loadBundledHead();
  if (!state.head) useHead(null, "");
}

/* ---------------- wire up ---------------- */
$("pauseBtn").onclick = () => { state.paused = !state.paused; updateButtons(); state.dirty = true; };
$("snapBtn").onclick = snapshot;
$("photoIn").onchange = (e) => { if (e.target.files[0]) openPhoto(e.target.files[0]); e.target.value = ""; };
$("liveBtn").onclick = backToLive;
$("flipBtn").onclick = () => { state.facing = state.facing === "environment" ? "user" : "environment"; startCamera(); };
$("retryCam").onclick = startCamera;
$("zoom").oninput = (e) => { state.zoom = Number(e.target.value) || 1; applyZoom(); };
$("view").addEventListener("click", (ev) => { const t = hitTest(ev); state.selectedId = t ? t.id : null; state.dirty = true; renderPeople(); });
$("clearLog").onclick = askClear;
$("exportCsv").onclick = exportCsv;
$("headFile").onchange = (e) => { if (e.target.files[0]) loadHeadFile(e.target.files[0]); e.target.value = ""; };
$("resetHead").onclick = resetHead;
$("cAuto").onclick = () => setAuto(!state.collect.auto);
$("cOne").onclick = () => {
  const big = currentTracks().filter((t) => t.visible).sort((a, b) => b.facePx - a.facePx)[0];
  if (big) collectFace(big, $("cLabel").value, "manual"); else showAlert("화면에 얼굴이 없습니다.");
};
$("cNext").onclick = () => {
  const v = $("cSession").value.trim(), m = v.match(/^(.*?)(\d+)$/);
  $("cSession").value = m ? m[1] + String(Number(m[2]) + 1).padStart(m[2].length, "0") : (v || "p") + "02";
};
$("cZip").onclick = exportZip;
$("cClear").onclick = () => { state.collect.items = []; state.collect.exported = 0; renderCollect(); };
window.addEventListener("beforeunload", (e) => {
  if (state.collect.items.length > (state.collect.exported || 0)) { e.preventDefault(); e.returnValue = ""; }
});
window.addEventListener("resize", () => (state.dirty = true));
document.addEventListener("keydown", (e) => {
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "")) return;
  if (e.code === "Space" && state.mode === "live" && state.stream) { e.preventDefault(); $("pauseBtn").click(); }
  if (e.key === "s" || e.key === "S") { if (!$("snapBtn").disabled) snapshot(); }
  if (e.key === "Escape" && state.mode === "photo") backToLive();
});
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/")); if (f) openPhoto(f); });
$("stage").addEventListener("dragover", (e) => e.preventDefault());
$("stage").addEventListener("drop", (e) => { e.preventDefault(); const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/")); if (f) openPhoto(f); });
try { if (matchMedia("(min-width: 901px)").matches) $("checklist").open = false; } catch {}

(() => {
  const a = $("repoLink"), m = location.hostname.match(/^([^.]+)\.github\.io$/i);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  if (m && repo) a.href = `https://github.com/${m[1]}/${repo}`; else a.hidden = true;
})();

if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});

bindOptions(); renderLog(); renderModelInfo(); renderCollect(); renderPeople(); updateButtons();
requestAnimationFrame(tick);
startCamera();
loadModels();
