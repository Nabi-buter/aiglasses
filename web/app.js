import { Detector, blobToCanvas, predict, verdictFor, validateHead } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const MAX_SHOTS = 3;
const LOG_KEY = "aiGlassesLog.v2";
const HEAD_KEY = "aiGlassesCustomHead.v1";

const CLASS_NAMES = { ai_glasses: "AI 안경", regular_glasses: "일반 안경", no_glasses: "안경 없음" };
const VERDICTS = {
  AI_GLASSES: { cls: "ai",  label: "AI 안경",      short: "AI 안경",
    summary: "AI 안경일 가능성이 높습니다.",
    steps: ["응시자에게 안경을 벗어 제출하도록 요청하세요.", "다리 안쪽 충전 단자, 경첩 쪽 카메라 렌즈, 브랜드 각인을 직접 확인하세요.", "고사본부 지침에 따라 보고하고 대체 안경 여부를 확인하세요."] },
  SUSPECTED:  { cls: "sus", label: "AI 안경 의심", short: "의심",
    summary: "AI 안경의 특징이 일부 보입니다. 직접 확인이 필요합니다.",
    steps: ["측면이나 더 가까운 거리에서 한 장 더 찍어 다시 판독하세요.", "경첩 부근 카메라 렌즈, LED, 두꺼운 다리를 육안으로 확인하세요.", "의심이 남으면 안경을 벗어 확인하세요."] },
  REGULAR:    { cls: "ok",  label: "일반 안경",    short: "일반 안경",
    summary: "일반 안경으로 보입니다.",
    steps: ["다른 감독 절차를 그대로 진행하세요.", "두꺼운 뿔테 등 애매한 경우에는 측면 사진으로 한 번 더 확인하세요."] },
  NO_GLASSES: { cls: "na",  label: "안경 없음",    short: "안경 없음",
    summary: "안경을 쓰고 있지 않은 것으로 보입니다.", steps: [] },
  NO_FACE:    { cls: "na",  label: "얼굴을 찾지 못함", short: "얼굴 없음",
    summary: "사진에서 얼굴을 찾지 못했습니다.",
    steps: ["얼굴과 안경이 화면 가운데에 크게 나오도록 다시 찍어 주세요.", "역광이나 흔들림이 없는지 확인하세요."] },
  NO_MODEL:   { cls: "na",  label: "모델 없음",    short: "모델 없음",
    summary: "얼굴과 안경 영역은 찾았지만 학습된 판독 모델이 없어 판정하지 못했습니다.", steps: [] },
};

const state = {
  detector: null, head: null, headSource: "", shots: [], stream: null,
  facing: "environment", zoom: 1, busy: false, log: loadLog(),
};

/* ---------------- utils ---------------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function store(key, val) { try { val == null ? localStorage.removeItem(key) : localStorage.setItem(key, val); return true; } catch { return false; } }
function read(key) { try { return localStorage.getItem(key); } catch { return null; } }
function loadLog() { try { const v = JSON.parse(read(LOG_KEY) || "[]"); return Array.isArray(v) ? v : []; } catch { return []; } }
function saveLog() { store(LOG_KEY, JSON.stringify(state.log.slice(0, 80))); }
const pct = (p) => Math.round((p ?? 0) * 100) + "%";
const p2 = (n) => String(n).padStart(2, "0");
function fmtTime(ts) { const d = new Date(ts); return `${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}`; }
function canvasToBlob(c, q = 0.92) { return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("toBlob"))), "image/jpeg", q)); }
function thumb(canvas, w = 160) {
  const c = document.createElement("canvas");
  c.width = w; c.height = Math.round((w * canvas.height) / canvas.width);
  c.getContext("2d").drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.7);
}
function setSvc(kind, text) { $("svcDot").className = "dot " + (kind === "ok" ? "on" : kind === "bad" ? "off" : ""); $("svcText").textContent = text; }

/* ---------------- models ---------------- */
async function loadModels() {
  try {
    state.detector = await Detector.create(".");
  } catch (e) {
    console.error(e);
    setSvc("bad", "판독 엔진을 불러오지 못했습니다");
    showMessage("err", "판독 엔진을 불러오지 못했습니다. 최신 Chrome·Edge·Safari에서 열어 주세요. 파일을 직접 열었다면 웹 서버(예: GitHub Pages)로 열어야 합니다.");
    return;
  }
  const custom = read(HEAD_KEY);
  if (custom) {
    try { useHead(validateHead(JSON.parse(custom)), "custom"); } catch { store(HEAD_KEY, null); }
  }
  if (!state.head) {
    try {
      const r = await fetch("models/head.json", { cache: "no-cache" });
      if (r.ok) useHead(validateHead(await r.json()), "bundled");
    } catch (e) { console.warn("head.json:", e); }
  }
  if (!state.head) useHead(null, "");
  renderTray();
}

function useHead(head, source) {
  state.head = head; state.headSource = source;
  if (head) setSvc("ok", "판독 준비 완료 · 기기 안에서 실행");
  else setSvc("bad", "학습된 판독 모델이 없습니다");
  $("resetHead").hidden = source !== "custom";
  renderModelInfo();
  if (!$("result").dataset.filled) renderIdle();
}

function renderModelInfo() {
  const box = $("modelInfo"); box.replaceChildren();
  const h = state.head;
  if (!h) {
    const s = el("div", "setup");
    s.append(el("p", null, "아직 학습된 모델(head.json)이 없습니다. 사진으로 얼굴·안경 영역 찾기는 되지만 AI 안경 여부는 판정하지 않습니다."));
    const pre = el("pre");
    pre.textContent = "pip install -r training/requirements.txt\npython training/train.py --data data --out web/models/head.json";
    s.append(el("p", null, "저장소에서 아래처럼 학습한 뒤 다시 배포하거나, 이미 만든 head.json 을 ‘모델 파일 불러오기’로 넣으세요."), pre);
    box.append(s);
    return;
  }
  const dl = el("dl");
  const add = (k, v) => { if (v == null || v === "") return; dl.append(el("dt", null, k), el("dd", null, String(v))); };
  add("출처", state.headSource === "custom" ? "직접 불러온 파일 (이 브라우저에 저장됨)" : "기본 포함 모델");
  add("클래스", h.classes.map((c) => CLASS_NAMES[c] || c).join(", "));
  if (h.n_images) add("학습 사진", Object.entries(h.n_images).map(([c, n]) => `${CLASS_NAMES[c] || c} ${n}`).join(" · "));
  if (h.cv) add("교차검증", `정확도 ${pct(h.cv.accuracy)} · AI 안경 재현율 ${pct(h.cv.ai_recall)} · 정밀도 ${pct(h.cv.ai_precision)}`);
  if (h.test) add("별도 평가", `${h.test.n}장 · 정확도 ${pct(h.test.accuracy)} · AI 안경 재현율 ${pct(h.test.ai_recall)} · 정밀도 ${pct(h.test.ai_precision)}`);
  add("판정 기준", `AI 안경 확률 ${pct(h.thresholds?.ai ?? 0.6)} 이상 → AI 안경, ${pct(h.thresholds?.suspect ?? 0.3)} 이상 → 의심`);
  add("학습 시각", h.trained_at ? new Date(h.trained_at).toLocaleString("ko-KR") : "");
  add("메모", h.note);
  box.append(dl);
  const small = Math.min(...Object.values(h.n_images || { x: 0 }));
  if (small < 100) box.append(el("div", "warnbox", `클래스당 학습 사진이 ${small}장뿐입니다. 결과를 참고용으로만 쓰고 사진을 더 모아 다시 학습하세요.`));
}

/* ---------------- camera ---------------- */
async function startCamera() {
  stopCamera();
  $("vfMsg").hidden = false; $("vfMsgText").textContent = "카메라를 켜는 중입니다…"; $("retryCam").hidden = true;
  if (!navigator.mediaDevices?.getUserMedia) return cameraUnavailable();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: state.facing }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false,
    });
    state.stream = stream;
    const v = $("video"); v.srcObject = stream;
    v.hidden = false; $("guide").hidden = false; $("zoomTag").hidden = false; $("vfMsg").hidden = true; $("liveControls").hidden = false;
    await v.play().catch(() => {});
    $("camName").textContent = (stream.getVideoTracks()[0]?.label || "").slice(0, 32);
    try { $("flip").hidden = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput").length < 2; } catch {}
    applyZoom();
  } catch { cameraUnavailable(); }
}
function cameraUnavailable() {
  $("video").hidden = true; $("guide").hidden = true; $("zoomTag").hidden = true; $("liveControls").hidden = true;
  $("vfMsg").hidden = false; $("retryCam").hidden = false; $("camName").textContent = "";
  $("vfMsgText").textContent = "실시간 카메라를 쓸 수 없습니다. 카메라 권한을 허용하거나, ‘기기 카메라로 촬영’을 눌러 휴대폰·태블릿 카메라로 찍으세요.";
}
function stopCamera() { state.stream?.getTracks().forEach((t) => t.stop()); state.stream = null; }
function applyZoom() { $("video").style.transform = `scale(${state.zoom})`; $("zoomTag").textContent = state.zoom.toFixed(1) + "×"; }

async function captureFromVideo() {
  const v = $("video");
  if (!state.stream || !v.videoWidth) return;
  if (state.shots.length >= MAX_SHOTS) return flashTray();
  const vw = v.videoWidth, vh = v.videoHeight, boxAR = 4 / 3;
  let cw = vw, ch = vh;
  if (vw / vh > boxAR) cw = vh * boxAR; else ch = vw / boxAR;
  cw /= state.zoom; ch /= state.zoom;
  const c = document.createElement("canvas");
  c.width = Math.round(cw); c.height = Math.round(ch);
  c.getContext("2d").drawImage(v, (vw - cw) / 2, (vh - ch) / 2, cw, ch, 0, 0, c.width, c.height);
  const f = $("flash"); f.classList.add("go"); requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove("go")));
  addShot(c);
}

/* ---------------- tray ---------------- */
const SLOT_LABELS = ["정면", "측면", "추가"];
function addShot(canvas) {
  if (state.shots.length >= MAX_SHOTS) return flashTray();
  state.shots.push({ canvas, url: thumb(canvas, 360) });
  renderTray();
}
function flashTray() { $("trayCount").style.color = "var(--ai)"; setTimeout(() => ($("trayCount").style.color = ""), 900); }
function renderTray() {
  const tray = $("tray"); tray.replaceChildren();
  for (let i = 0; i < MAX_SHOTS; i++) {
    const s = state.shots[i], slot = el("div", "slot");
    if (s) {
      const img = el("img"); img.src = s.url; img.alt = SLOT_LABELS[i] + " 사진";
      const rm = el("button", "rm", "×"); rm.setAttribute("aria-label", SLOT_LABELS[i] + " 사진 지우기");
      rm.onclick = () => { state.shots.splice(i, 1); renderTray(); };
      slot.append(img, el("span", "lbl", SLOT_LABELS[i]), rm);
    } else slot.textContent = ["정면 사진", "측면 사진 (권장)", "추가 사진"][i];
    tray.append(slot);
  }
  $("trayCount").textContent = `${state.shots.length} / ${MAX_SHOTS}`;
  $("analyze").disabled = !state.shots.length || state.busy || !state.detector;
}
async function handleFiles(files) {
  for (const f of files) {
    if (state.shots.length >= MAX_SHOTS) { flashTray(); break; }
    try {
      let c = await blobToCanvas(f);
      const s = Math.min(1, 2400 / Math.max(c.width, c.height));   // keep memory sane on phones
      if (s < 1) {
        const d = document.createElement("canvas"); d.width = Math.round(c.width * s); d.height = Math.round(c.height * s);
        d.getContext("2d").drawImage(c, 0, 0, d.width, d.height); c = d;
      }
      addShot(c);
    } catch { showMessage("err", "이 사진을 열 수 없습니다. JPG나 PNG 사진으로 다시 시도하세요."); }
  }
}

/* ---------------- analysis ---------------- */
async function analyze() {
  if (!state.detector || !state.shots.length || state.busy) return;
  state.busy = true; renderTray();
  showStatus("판독하는 중입니다…");
  await new Promise((r) => setTimeout(r, 30)); // let the status paint before heavy work
  try {
    const results = state.shots.map((s, i) => {
      const ex = state.detector.extract(s.canvas);
      const probs = ex.feature && state.head ? predict(state.head, ex.feature) : null;
      return { idx: i, ex, probs };
    });
    const withFace = results.filter((r) => r.ex.feature);
    let verdict, decisive = null;
    if (!withFace.length) verdict = "NO_FACE";
    else if (!state.head) verdict = "NO_MODEL";
    else {
      // Screening rule: the photo that looks MOST like AI glasses decides.
      decisive = withFace.reduce((a, b) => (b.probs[state.head.ai_class] > a.probs[state.head.ai_class] ? b : a));
      verdict = verdictFor(state.head, decisive.probs);
    }
    renderResult({ verdict, results, decisive });
    if (verdict !== "NO_MODEL") addLog(verdict, decisive, state.shots[decisive ? decisive.idx : 0].canvas);
  } catch (e) {
    console.error(e);
    showMessage("err", "판독 중 오류가 났습니다. 사진을 바꿔 다시 시도하세요.");
  } finally {
    state.busy = false; renderTray();
  }
}

/* ---------------- result rendering ---------------- */
function showStatus(msg) {
  const box = $("result"); box.dataset.filled = "1"; box.replaceChildren();
  const s = el("div", "status"); s.append(el("span", "spin"), el("span", null, msg)); box.append(s);
}
function showMessage(kind, msg) { const box = $("result"); box.dataset.filled = "1"; box.replaceChildren(el("div", kind, msg)); }
function renderIdle() {
  const box = $("result"); box.replaceChildren();
  const s = el("div", "setup");
  s.append(el("p", null, state.head
    ? "사진을 찍고 ‘판독하기’를 누르면 여기에 판정, 클래스별 확률, 모델이 살펴본 영역(안경 전체, 양쪽 경첩)이 나옵니다."
    : "학습된 모델이 없어 지금은 얼굴·안경 영역 찾기만 확인할 수 있습니다. 아래 ‘판독 모델’ 안내를 참고하세요."));
  box.append(s);
}

function probRows(probs) {
  const wrap = el("div", "probs");
  const order = Object.keys(probs).sort((a, b) => (a === state.head.ai_class ? -1 : b === state.head.ai_class ? 1 : a.localeCompare(b)));
  for (const c of order) {
    const row = el("div", "prow " + (c === state.head.ai_class ? "ai" : c === "regular_glasses" ? "reg" : "na"));
    const bar = el("div", "pb"); const fill = el("i"); fill.style.width = pct(probs[c]); bar.append(fill);
    row.append(el("span", null, CLASS_NAMES[c] || c), bar, el("span", "pv", pct(probs[c])));
    wrap.append(row);
  }
  return wrap;
}

function renderResult({ verdict, results, decisive }) {
  const box = $("result"); box.dataset.filled = "1"; box.replaceChildren();
  const meta = VERDICTS[verdict];
  const card = el("div", "verdict " + meta.cls);
  const head = el("div", "vhead"); head.append(el("span", "vlabel", meta.label));
  if (decisive && results.length > 1) head.append(el("span", "badge", `${SLOT_LABELS[decisive.idx]} 사진 기준`));
  card.append(head);
  if (decisive) {
    const pAi = decisive.probs[state.head.ai_class];
    const m = el("div", "meter"), bar = el("div", "bar"), fill = el("i");
    fill.style.width = pct(pAi); bar.append(fill);
    const cap = el("div", "cap"); cap.append(el("span", null, "AI 안경 확률"), el("span", null, pct(pAi)));
    m.append(bar, cap); card.append(m);
  }
  card.append(el("p", "vsum", meta.summary));
  box.append(card);

  if (decisive) { const sec = el("div", "sec"); sec.append(el("h3", null, "클래스별 확률"), probRows(decisive.probs)); box.append(sec); }

  const sec = el("div", "sec"); sec.append(el("h3", null, "모델이 살펴본 영역"));
  const shots = el("div", "shots");
  for (const r of results) {
    const row = el("div", "shot");
    const hd = el("div", "shot-head");
    const t = el("span"); t.append(el("b", null, SLOT_LABELS[r.idx] + " 사진"));
    hd.append(t, el("span", null, r.probs ? "AI 확률 " + pct(r.probs[state.head.ai_class]) : r.ex.feature ? "" : "얼굴 못 찾음"));
    row.append(hd);
    if (r.ex.crops) {
      for (const [k, cap] of [["band", "안경 전체"], ["hinge_l", "왼쪽 경첩"], ["hinge_r", "오른쪽 경첩"]]) {
        const fig = el("figure");
        if (r.ex.crops[k]) { const img = el("img"); img.src = thumb(r.ex.crops[k], k === "band" ? 320 : 160); img.alt = cap; fig.append(img); }
        else fig.append(el("div", "slot", "화면 밖"));
        fig.append(el("figcaption", null, cap)); row.append(fig);
      }
    } else row.append(el("div", "noface", "얼굴이 보이지 않아 이 사진은 판독에서 뺐습니다."));
    shots.append(row);
  }
  sec.append(shots); box.append(sec);

  if (meta.steps.length) {
    const s2 = el("div", "sec"); s2.append(el("h3", null, "감독관 조치"));
    const ol = el("ol", "steps"); meta.steps.forEach((t) => ol.append(el("li", null, t)));
    s2.append(ol); box.append(s2);
  }
  if (verdict === "NO_MODEL") box.append(el("div", "warnbox", "판정을 하려면 학습된 모델(head.json)이 필요합니다. 아래 ‘판독 모델’ 안내를 참고하세요."));
}

/* ---------------- log ---------------- */
function addLog(verdict, decisive, canvas) {
  state.log.unshift({
    ts: Date.now(), seat: $("seat").value.trim().slice(0, 40), room: $("room").value.trim().slice(0, 40),
    verdict, pAi: decisive ? decisive.probs[state.head.ai_class] : null, thumb: thumb(canvas),
  });
  state.log = state.log.slice(0, 80);
  saveLog(); renderLog();
}
function renderLog() {
  const body = $("logBody"); body.replaceChildren();
  if (!state.log.length) {
    const tr = el("tr"), td = el("td", "empty", "아직 판독 기록이 없습니다. 사진을 찍고 ‘판독하기’를 누르면 여기에 쌓입니다.");
    td.colSpan = 5; tr.append(td); body.append(tr);
  }
  for (const it of state.log) {
    const meta = VERDICTS[it.verdict] || VERDICTS.NO_FACE;
    const tr = el("tr"), tdImg = el("td");
    if (typeof it.thumb === "string" && it.thumb.startsWith("data:image/")) { const im = el("img"); im.src = it.thumb; im.alt = ""; tdImg.append(im); }
    const tdV = el("td"); tdV.append(el("span", "pill " + meta.cls, meta.short));
    tr.append(tdImg, el("td", "mono", fmtTime(it.ts)), el("td", null, [it.room, it.seat].filter(Boolean).join(" · ") || "—"),
      tdV, el("td", "mono", it.pAi == null ? "—" : pct(it.pAi)));
    body.append(tr);
  }
  const c = $("counts"); c.replaceChildren(el("span", "pill na", `전체 ${state.log.length}건`));
  const tally = new Map();
  for (const it of state.log) { const m = VERDICTS[it.verdict] || VERDICTS.NO_FACE; tally.set(m, (tally.get(m) || 0) + 1); }
  for (const [m, n] of tally) c.append(el("span", "pill " + m.cls, `${m.short} ${n}`));
  $("clearLog").disabled = $("exportCsv").disabled = !state.log.length;
}
function askClear() {
  const box = $("confirmBox"); box.hidden = false; box.replaceChildren();
  const w = el("div", "confirm");
  const yes = el("button", "small", "기록 모두 지우기"); yes.style.borderColor = yes.style.color = "var(--ai)";
  const no = el("button", "small ghost", "취소");
  yes.onclick = () => { state.log = []; saveLog(); renderLog(); box.hidden = true; };
  no.onclick = () => { box.hidden = true; };
  w.append(el("span", null, `기록 ${state.log.length}건을 지울까요? 되돌릴 수 없습니다.`), yes, no);
  box.append(w);
}
function exportCsv() {
  const q = (s) => '"' + String(s ?? "").replace(/"/g, '""') + '"';
  const rows = [["시각", "고사실", "좌석", "판정", "AI 안경 확률"].map(q).join(",")];
  for (const it of state.log.slice().reverse())
    rows.push([new Date(it.ts).toLocaleString("ko-KR"), it.room, it.seat, (VERDICTS[it.verdict] || VERDICTS.NO_FACE).label, it.pAi == null ? "" : pct(it.pAi)].map(q).join(","));
  const d = new Date();
  const a = el("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + rows.join("\r\n")], { type: "text/csv;charset=utf-8" }));
  a.download = `AI안경_감독기록_${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}.csv`;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

/* ---------------- custom model file ---------------- */
async function loadHeadFile(file) {
  try {
    const head = validateHead(JSON.parse(await file.text()));
    if (!store(HEAD_KEY, JSON.stringify(head))) console.warn("head not persisted");
    useHead(head, "custom");
    showMessage("setup", "새 모델을 불러왔습니다. 사진을 판독해 보세요.");
  } catch (e) {
    showMessage("err", "모델 파일을 쓸 수 없습니다: " + (e?.message || e));
  }
}
async function resetHead() {
  store(HEAD_KEY, null); state.head = null;
  try { const r = await fetch("models/head.json", { cache: "no-cache" }); if (r.ok) return useHead(validateHead(await r.json()), "bundled"); } catch {}
  useHead(null, "");
}

/* ---------------- wire up ---------------- */
$("shutter").onclick = captureFromVideo;
$("zoom").oninput = (e) => { state.zoom = Number(e.target.value) || 1; applyZoom(); };
$("flip").onclick = () => { state.facing = state.facing === "environment" ? "user" : "environment"; startCamera(); };
$("retryCam").onclick = startCamera;
$("fileCam").onchange = (e) => { handleFiles([...e.target.files]); e.target.value = ""; };
$("fileUp").onchange = (e) => { handleFiles([...e.target.files]); e.target.value = ""; };
$("analyze").onclick = analyze;
$("clearLog").onclick = askClear;
$("exportCsv").onclick = exportCsv;
$("headFile").onchange = (e) => { if (e.target.files[0]) loadHeadFile(e.target.files[0]); e.target.value = ""; };
$("resetHead").onclick = resetHead;
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && state.stream && !/INPUT|TEXTAREA|BUTTON|SELECT/.test(document.activeElement?.tagName || "")) { e.preventDefault(); captureFromVideo(); }
});
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].filter((x) => x.type.startsWith("image/")); if (f.length) handleFiles(f); });
$("vf").addEventListener("dragover", (e) => e.preventDefault());
$("vf").addEventListener("drop", (e) => { e.preventDefault(); handleFiles([...(e.dataTransfer?.files || [])].filter((x) => x.type.startsWith("image/"))); });
try { if (matchMedia("(min-width: 901px)").matches) $("checklist").open = true; } catch {}

// Footer link: on GitHub Pages (user.github.io/repo) point to the repository itself.
(() => {
  const a = $("repoLink"), m = location.hostname.match(/^([^.]+)\.github\.io$/i);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  if (m && repo) a.href = `https://github.com/${m[1]}/${repo}`; else a.hidden = true;
})();

if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});

renderTray(); renderIdle(); renderLog(); renderModelInfo();
startCamera();
loadModels();
