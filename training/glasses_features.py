"""Feature extraction shared by training and prediction.

The browser (web/pipeline.js) re-implements exactly the same steps, so a head
trained here gives the same answer in the web app:

  1. Decode the image (EXIF orientation applied) as RGB.
  2. Find the largest face with MediaPipe Face Landmarker (478 landmarks).
  3. Cut three crops in pixel space:
       - band:   the whole glasses area (brows to below the eyes, face edge to face edge)
       - hinge L/R: squares around the outer frame corners (between the outer
         eye corner and the face edge, at brow/eye level), where camera-type
         AI glasses put the camera and LED.
  4. Embed each crop with the MediaPipe Image Embedder (MobileNetV3-Large,
     L2-normalised, 1280-d).
  5. Feature = concat(band, mean(hinge L, hinge R))  -> 2560-d.

Keep CROP_VERSION in sync with web/pipeline.js. Change it whenever the crop
math changes, because old heads are no longer valid then.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

CROP_VERSION = "band+hinges/v1"
EMBEDDER_FILE = "mobilenet_v3_large.tflite"
FACE_FILE = "face_landmarker.task"
EMBED_DIM = 1280
FEATURE_DIM = EMBED_DIM * 2
MAX_FACES = 8  # == web/pipeline.js MAX_FACES

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "web" / "models"

# Landmark indices (MediaPipe face mesh, 478-point topology).
FACE_SIDE_A = 234          # face contour at eye level, image-left for a frontal face
FACE_SIDE_B = 454          # face contour at eye level, image-right
BROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
EYE_BOTTOM = (145, 374)
EYE_OUTER = (33, 263)

# Crop geometry, in units of face width W = |x(234) - x(454)|.
BAND_PAD_X = 0.12
BAND_PAD_TOP = 0.12
BAND_PAD_BOTTOM = 0.22
HINGE_SIZE = 0.40
HINGE_OUTWARD = 0.6   # hinge centre: this far from the outer eye corner towards the face edge


def rnd(v: float) -> int:
    """Round half up, identical to Math.floor(v + 0.5) in JS."""
    return int(math.floor(v + 0.5))


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def empty(self) -> bool:
        return self.x1 - self.x0 < 2 or self.y1 - self.y0 < 2

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def clamp_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> Box:
    return Box(
        max(0, min(w, rnd(x0))),
        max(0, min(h, rnd(y0))),
        max(0, min(w, rnd(x1))),
        max(0, min(h, rnd(y1))),
    )


def crop_boxes(landmarks, w: int, h: int) -> dict[str, Box]:
    """Compute the three crop boxes from normalised landmarks."""
    px = lambda i: landmarks[i].x * w  # noqa: E731
    py = lambda i: landmarks[i].y * h  # noqa: E731

    xa, xb = px(FACE_SIDE_A), px(FACE_SIDE_B)
    left, right = min(xa, xb), max(xa, xb)
    fw = max(right - left, 1.0)

    brow_top = min(py(i) for i in BROWS)
    eye_bottom = max(py(i) for i in EYE_BOTTOM)
    band = clamp_box(
        left - BAND_PAD_X * fw,
        brow_top - BAND_PAD_TOP * fw,
        right + BAND_PAD_X * fw,
        eye_bottom + BAND_PAD_BOTTOM * fw,
        w, h,
    )

    eye_level = sum(py(i) for i in EYE_OUTER) / len(EYE_OUTER)
    hinge_y = (eye_level + brow_top) / 2
    outer_l = min(px(i) for i in EYE_OUTER)
    outer_r = max(px(i) for i in EYE_OUTER)
    cx_l = outer_l + HINGE_OUTWARD * (left - outer_l)
    cx_r = outer_r + HINGE_OUTWARD * (right - outer_r)
    half = HINGE_SIZE * fw / 2
    hinge_l = clamp_box(cx_l - half, hinge_y - half, cx_l + half, hinge_y + half, w, h)
    hinge_r = clamp_box(cx_r - half, hinge_y - half, cx_r + half, hinge_y + half, w, h)
    return {"band": band, "hinge_l": hinge_l, "hinge_r": hinge_r}


def face_width(landmarks) -> float:
    return abs(landmarks[FACE_SIDE_A].x - landmarks[FACE_SIDE_B].x)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"))


@dataclass
class Extraction:
    feature: np.ndarray | None      # (FEATURE_DIM,) or None when no face was found
    boxes: dict[str, Box] | None
    reason: str = ""


class FeatureExtractor:
    """Wraps the two MediaPipe tasks. Use as a context manager."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_dir / FACE_FILE)),
                num_faces=MAX_FACES,
                min_face_detection_confidence=0.5,
            )
        )
        self._embedder = vision.ImageEmbedder.create_from_options(
            vision.ImageEmbedderOptions(
                base_options=BaseOptions(model_asset_path=str(model_dir / EMBEDDER_FILE)),
                l2_normalize=True,
                quantize=False,
            )
        )

    def close(self):
        self._face.close()
        self._embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _embed(self, rgb: np.ndarray) -> np.ndarray:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._embedder.embed(img)
        return np.asarray(res.embeddings[0].embedding, dtype=np.float32)

    def extract(self, rgb: np.ndarray) -> Extraction:
        h, w = rgb.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._face.detect(img)
        if not res.face_landmarks:
            return Extraction(None, None, "no_face")
        landmarks = max(res.face_landmarks, key=face_width)
        boxes = crop_boxes(landmarks, w, h)
        if boxes["band"].empty:
            return Extraction(None, boxes, "crop_empty")

        def emb(b: Box) -> np.ndarray | None:
            if b.empty:
                return None
            return self._embed(rgb[b.y0:b.y1, b.x0:b.x1])

        band = emb(boxes["band"])
        hinges = [e for e in (emb(boxes["hinge_l"]), emb(boxes["hinge_r"])) if e is not None]
        hinge = np.mean(hinges, axis=0) if hinges else np.zeros(EMBED_DIM, np.float32)
        return Extraction(np.concatenate([band, hinge]).astype(np.float32), boxes)

    def extract_faces(self, rgb: np.ndarray) -> list[dict]:
        """Every face in the image (== web/pipeline.js facesInImage + featureFor):
        [{"rect": (x0, y0, x1, y1) normalised, "face_px": float, "boxes": {...}, "feature": array | None}]"""
        h, w = rgb.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = []
        for lm in self._face.detect(img).face_landmarks:
            xs, ys = [p.x for p in lm], [p.y for p in lm]
            boxes = crop_boxes(lm, w, h)
            feat = None
            if not boxes["band"].empty:
                crop = lambda b: rgb[b.y0:b.y1, b.x0:b.x1]  # noqa: E731
                band = self._embed(crop(boxes["band"]))
                hinges = [self._embed(crop(boxes[k])) for k in ("hinge_l", "hinge_r") if not boxes[k].empty]
                hinge = np.mean(hinges, axis=0) if hinges else np.zeros(EMBED_DIM, np.float32)
                feat = np.concatenate([band, hinge]).astype(np.float32)
            out.append({"rect": (min(xs), min(ys), max(xs), max(ys)), "face_px": face_width(lm) * w,
                        "boxes": boxes, "feature": feat})
        return out

    def extract_path(self, path: str | Path, flip: bool = False) -> Extraction:
        rgb = load_rgb(path)
        if flip:
            rgb = rgb[:, ::-1]
        return self.extract(rgb)


# ---------------------------------------------------------------- head (classifier)

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def head_predict(head: dict, feature: np.ndarray) -> dict[str, float]:
    W = np.asarray(head["W"], dtype=np.float32)
    b = np.asarray(head["b"], dtype=np.float32)
    p = softmax(W @ feature + b)
    return {c: float(v) for c, v in zip(head["classes"], p)}


def verdict(head: dict, probs: dict[str, float]) -> str:
    """Same rule as web/pipeline.js -> verdictFor()."""
    t = head.get("thresholds", {})
    p_ai = probs.get(head["ai_class"], 0.0)
    if p_ai >= t.get("ai", 0.6):
        return "AI_GLASSES"
    if p_ai >= t.get("suspect", 0.3):
        return "SUSPECTED"
    top = max(probs, key=probs.get)
    if top == "no_glasses":
        return "NO_GLASSES"
    if top == "regular_glasses":
        return "REGULAR"
    return "SUSPECTED"
