// On-device AI-glasses classifier. Mirrors training/glasses_features.py exactly:
// same MediaPipe models, same crop math, same feature layout, same verdict rule.
// If you change anything here, change it there too and bump CROP_VERSION.

import { FilesetResolver, FaceLandmarker, ImageEmbedder } from "./vendor/vision_bundle.mjs";

export const CROP_VERSION = "band+hinges/v1";
const EMBED_DIM = 1280;
const FEATURE_DIM = EMBED_DIM * 2;

const FACE_SIDE_A = 234, FACE_SIDE_B = 454;
const BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300];
const EYE_BOTTOM = [145, 374];
const EYE_OUTER = [33, 263];
const BAND_PAD_X = 0.12, BAND_PAD_TOP = 0.12, BAND_PAD_BOTTOM = 0.22;
const HINGE_SIZE = 0.40, HINGE_OUTWARD = 0.6;

const rnd = (v) => Math.floor(v + 0.5); // == glasses_features.rnd

function clampBox(x0, y0, x1, y1, w, h) {
  const c = (v, m) => Math.max(0, Math.min(m, rnd(v)));
  const b = { x0: c(x0, w), y0: c(y0, h), x1: c(x1, w), y1: c(y1, h) };
  b.empty = b.x1 - b.x0 < 2 || b.y1 - b.y0 < 2;
  return b;
}

export function cropBoxes(lm, w, h) {
  const px = (i) => lm[i].x * w, py = (i) => lm[i].y * h;
  const xa = px(FACE_SIDE_A), xb = px(FACE_SIDE_B);
  const left = Math.min(xa, xb), right = Math.max(xa, xb);
  const fw = Math.max(right - left, 1);

  const browTop = Math.min(...BROWS.map(py));
  const eyeBottom = Math.max(...EYE_BOTTOM.map(py));
  const band = clampBox(left - BAND_PAD_X * fw, browTop - BAND_PAD_TOP * fw,
    right + BAND_PAD_X * fw, eyeBottom + BAND_PAD_BOTTOM * fw, w, h);

  const eyeLevel = EYE_OUTER.map(py).reduce((a, b) => a + b, 0) / EYE_OUTER.length;
  const hingeY = (eyeLevel + browTop) / 2;
  const outerL = Math.min(...EYE_OUTER.map(px));
  const outerR = Math.max(...EYE_OUTER.map(px));
  const cxL = outerL + HINGE_OUTWARD * (left - outerL);
  const cxR = outerR + HINGE_OUTWARD * (right - outerR);
  const half = HINGE_SIZE * fw / 2;
  return {
    band,
    hinge_l: clampBox(cxL - half, hingeY - half, cxL + half, hingeY + half, w, h),
    hinge_r: clampBox(cxR - half, hingeY - half, cxR + half, hingeY + half, w, h),
  };
}

const faceWidth = (lm) => Math.abs(lm[FACE_SIDE_A].x - lm[FACE_SIDE_B].x);

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function validateHead(head) {
  if (!head || head.format !== "ai-glasses-head/v1") throw new Error("head.json 형식이 아닙니다.");
  if (head.crop !== CROP_VERSION) throw new Error(`crop 버전이 다릅니다 (${head.crop} ≠ ${CROP_VERSION}). 최신 코드로 다시 학습하세요.`);
  if (!Array.isArray(head.classes) || head.W?.length !== head.classes.length || head.b?.length !== head.classes.length)
    throw new Error("head.json 의 classes / W / b 크기가 맞지 않습니다.");
  if (head.W.some((r) => r.length !== FEATURE_DIM)) throw new Error(`특징 차원이 ${FEATURE_DIM} 이 아닙니다.`);
  if (!head.classes.includes(head.ai_class)) throw new Error("ai_class 가 classes 에 없습니다.");
  return head;
}

export function predict(head, feature) {
  const z = head.W.map((row, k) => {
    let s = head.b[k];
    for (let i = 0; i < row.length; i++) s += row[i] * feature[i];
    return s;
  });
  const p = softmax(z);
  return Object.fromEntries(head.classes.map((c, i) => [c, p[i]]));
}

// Same rule as glasses_features.verdict()
export function verdictFor(head, probs) {
  const t = head.thresholds || {};
  const pAi = probs[head.ai_class] ?? 0;
  if (pAi >= (t.ai ?? 0.6)) return "AI_GLASSES";
  if (pAi >= (t.suspect ?? 0.3)) return "SUSPECTED";
  const top = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
  if (top === "no_glasses") return "NO_GLASSES";
  if (top === "regular_glasses") return "REGULAR";
  return "SUSPECTED";
}

function cropCanvas(src, b) {
  const c = document.createElement("canvas");
  c.width = b.x1 - b.x0; c.height = b.y1 - b.y0;
  c.getContext("2d").drawImage(src, b.x0, b.y0, c.width, c.height, 0, 0, c.width, c.height);
  return c;
}

/** Decode a Blob/File into a canvas with EXIF orientation applied (== load_rgb). */
export async function blobToCanvas(blob) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const c = document.createElement("canvas");
  c.width = bmp.width; c.height = bmp.height;
  c.getContext("2d").drawImage(bmp, 0, 0);
  bmp.close?.();
  return c;
}

export const MAX_FACES = 8;   // == glasses_features.MAX_FACES

/** Face bounding box (pixels) from landmarks, for size gating and data collection. */
export function faceRect(lm, w, h) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of lm) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  return { x0: x0 * w, y0: y0 * h, x1: x1 * w, y1: y1 * h };
}

/** Face width in pixels (same measure the crop geometry is based on). */
export const facePx = (lm, w) => faceWidth(lm) * w;

export class Detector {
  /**
   * Two Face Landmarker instances: IMAGE mode for still photos (identical to the
   * Python side) and VIDEO mode for the live camera (tracks faces between frames,
   * which is much faster). The embedder is shared.
   */
  static async create(base = ".", { video = true } = {}) {
    const fileset = await FilesetResolver.forVisionTasks(`${base}/vendor/wasm`);
    const faceOpts = (mode) => ({
      baseOptions: { modelAssetPath: `${base}/models/face_landmarker.task`, delegate: "CPU" },
      runningMode: mode, numFaces: MAX_FACES, minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    const face = await FaceLandmarker.createFromOptions(fileset, faceOpts("IMAGE"));
    const faceVideo = video ? await FaceLandmarker.createFromOptions(fileset, faceOpts("VIDEO")) : null;
    const embedder = await ImageEmbedder.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${base}/models/mobilenet_v3_large.tflite`, delegate: "CPU" },
      runningMode: "IMAGE", l2Normalize: true, quantize: false,
    });
    return new Detector(face, faceVideo, embedder);
  }

  constructor(face, faceVideo, embedder) {
    this.face = face; this.faceVideo = faceVideo; this.embedder = embedder; this.lastTs = 0;
  }

  embed(canvas) {
    return Float32Array.from(this.embedder.embed(canvas).embeddings[0].floatEmbedding);
  }

  /** All faces in a still image: [{lm, boxes, facePx}] sorted largest first. */
  facesInImage(canvas) {
    return this._faces(this.face.detect(canvas), canvas.width, canvas.height);
  }

  /** All faces in a live frame. `canvas` holds the current frame; ts in ms. */
  facesInFrame(canvas, ts) {
    const t = Math.max(Math.round(ts), this.lastTs + 1);   // VIDEO mode needs strictly increasing timestamps
    this.lastTs = t;
    return this._faces(this.faceVideo.detectForVideo(canvas, t), canvas.width, canvas.height);
  }

  _faces(res, w, h) {
    return (res.faceLandmarks || [])
      .map((lm) => ({ lm, boxes: cropBoxes(lm, w, h), facePx: facePx(lm, w), rect: faceRect(lm, w, h) }))
      .sort((a, b) => b.facePx - a.facePx);
  }

  /** Feature vector for one face, given its crop boxes. -> {feature, crops} or null */
  featureFor(canvas, boxes) {
    if (boxes.band.empty) return null;
    const crops = {};
    for (const k of ["band", "hinge_l", "hinge_r"]) if (!boxes[k].empty) crops[k] = cropCanvas(canvas, boxes[k]);
    const band = this.embed(crops.band);
    const hinges = ["hinge_l", "hinge_r"].filter((k) => crops[k]).map((k) => this.embed(crops[k]));
    const hinge = new Float32Array(EMBED_DIM);
    for (const h of hinges) for (let i = 0; i < EMBED_DIM; i++) hinge[i] += h[i] / hinges.length;
    const feature = new Float32Array(FEATURE_DIM);
    feature.set(band, 0); feature.set(hinge, EMBED_DIM);
    return { feature, crops };
  }

  /** Largest face only (== glasses_features.FeatureExtractor.extract). */
  extract(canvas) {
    const faces = this.facesInImage(canvas);
    if (!faces.length) return { feature: null, boxes: null, crops: null, reason: "no_face" };
    const { boxes } = faces[0];
    const r = this.featureFor(canvas, boxes);
    if (!r) return { feature: null, boxes, crops: null, reason: "crop_empty" };
    return { feature: r.feature, boxes, crops: r.crops, reason: "" };
  }
}
