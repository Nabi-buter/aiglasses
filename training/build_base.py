#!/usr/bin/env python3
"""Build the BASE model that ships with the web app, and the shareable BASE DATA it was trained on.

Goal: gray / green / red work out of the box, before anyone has collected photos.
Every later training (web/train.html or train.py --with-base) adds the user's photos to the
same base data, so the model only gets better as more data is collected.

Sources (only freely licensed material; no photos are stored in the repository):
  regular_glasses / no_glasses
      Open Images V7 faces with human-drawn eyewear boxes (see build_presence.py).
      Annotations CC BY 4.0 (Google LLC), images listed as CC BY 2.0.
  ai_glasses
      Wikimedia Commons photos of people wearing smart / AI glasses
      (Google Glass, Ray-Ban Stories / Meta, Snap Spectacles, Vuzix, EyeTap, ...).
  regular_glasses (same source as above, so the model cannot learn "Commons photo = AI glasses")
      Wikimedia Commons photos of people wearing ordinary glasses.
  Every Commons file's author and license is written to docs/BASE_DATA_SOURCES.md.

Outputs
  web/models/head.json        base 3-class model (same format as train.py; "base": true)
  web/models/base_data.json   labels, groups, sources of the base rows
  web/models/base_data.bin    their features, int8 per-row scaled (numbers only, no images)
  docs/BASE_DATA_SOURCES.md   attribution list

    python training/build_base.py --cache ~/.cache/aiglasses-base --oi-cache ~/.cache/aiglasses-oi
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basekit  # noqa: E402
import build_presence as bp  # noqa: E402
import glasses_features as gf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UA = "aiglasses-dataset-builder/0.1 (open-source research tool)"
API = "https://commons.wikimedia.org/w/api.php"
CLASSES = basekit.CLASSES

# Commons categories -> (class, device tag). Sub-categories are followed (depth 3) except EXCLUDE.
COMMONS = {
    "Category:Google Glass": ("ai_glasses", "google_glass"),
    "Category:Ray-Ban Meta": ("ai_glasses", "rayban_meta"),
    "Category:Ray-Ban Stories": ("ai_glasses", "rayban_stories"),
    "Category:Snapchat Spectacles": ("ai_glasses", "spectacles"),
    "Category:EyeTap": ("ai_glasses", "other_smart"),
    "Category:Epiphany Eyewear": ("ai_glasses", "other_smart"),
    "Category:Smartglasses": ("ai_glasses", "other_smart"),
    "Category:Vuzix": ("ai_glasses", "other_smart"),
}
EXCLUDE_CAT = re.compile(r"taken with|hololens|forte|virtual reality|headset|logos?$|screenshots", re.I)
EXCLUDE_FILE = re.compile(r"taken with|screenshot|logo|diagram|patent|\.svg$|\.pdf$|\.webm$|\.ogv$", re.I)
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?([- ][0-9.]+)?([- ]\w+)?|public domain|pd.*|attribution|no restrictions)$", re.I)


# ----------------------------------------------------------------- Commons client (polite, cached)

class Commons:
    """The API is strictly rate-limited for anonymous clients, the image server much less.
    So images are fetched by computed URL and the API is only asked for license metadata."""

    def __init__(self, cache: Path, gap: float = 3.0, img_gap: float = 1.0):
        self.cache, self.gap, self.img_gap, self.last = cache, gap, img_gap, {}
        (cache / "api").mkdir(parents=True, exist_ok=True)
        (cache / "img").mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, binary=False):
        host = urllib.parse.urlparse(url).netloc
        gap = self.img_gap if host.startswith("upload.") else self.gap
        for attempt in range(30):
            wait = gap - (time.time() - self.last.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=90) as r:
                    data = r.read()
                return data if binary else json.loads(data)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 410):
                    return None
                ra = int(e.headers.get("Retry-After") or 30)
                if ra > 5:
                    print(f"    (HTTP {e.code} from {host}, waiting {ra}s)", flush=True)
                time.sleep(ra + 1)
            except Exception as e:  # noqa: BLE001
                print(f"    ({type(e).__name__} from {host}: {str(e)[:80]}; retrying)", flush=True)
                time.sleep(5 + 5 * min(attempt, 6))
        raise RuntimeError(f"giving up: {url}")

    def api(self, **params):
        params.update(format="json", formatversion="2")
        url = API + "?" + urllib.parse.urlencode(sorted(params.items()))
        f = self.cache / "api" / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if f.exists():
            return json.loads(f.read_text())
        d = self._get(url)
        f.write_text(json.dumps(d))
        return d

    def members(self, cat: str, kind: str) -> list[str]:
        out, cont = [], {}
        while True:
            d = self.api(action="query", list="categorymembers", cmtitle=cat, cmtype=kind, cmlimit=500, **cont)
            out += [m["title"] for m in d["query"]["categorymembers"]]
            if "continue" not in d:
                return out
            cont = d["continue"]

    def walk(self, root: str, depth: int = 3):
        """-> [(file title, category it was found in)]"""
        seen, out, todo = set(), [], [(root, 0)]
        while todo:
            cat, dep = todo.pop()
            if cat in seen:
                continue
            seen.add(cat)
            out += [(t, cat) for t in self.members(cat, "file") if not EXCLUDE_FILE.search(t)]
            if dep < depth:
                todo += [(c, dep + 1) for c in self.members(cat, "subcat") if not EXCLUDE_CAT.search(c)]
        return out

    def info(self, titles: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(titles), 50):
            d = self.api(action="query", prop="imageinfo", titles="|".join(titles[i:i + 50]),
                         iiprop="url|extmetadata|mime|size", iiurlwidth=1280)
            for p in d["query"]["pages"]:
                ii = (p.get("imageinfo") or [{}])[0]
                md = ii.get("extmetadata", {})
                val = lambda k: html.unescape(re.sub(r"<[^>]+>", "", md.get(k, {}).get("value", ""))).strip()  # noqa: E731
                out[p["title"]] = {
                    "url": ii.get("thumburl") or ii.get("url"), "page": ii.get("descriptionurl"),
                    "mime": ii.get("mime", ""), "license": val("LicenseShortName"), "license_url": val("LicenseUrl"),
                    "artist": val("Artist")[:120] or "unknown",
                }
        return out

    @staticmethod
    def file_urls(title: str) -> list[str]:
        """960px thumbnail (standard width), then the original if it is smaller than that."""
        name = title.split(":", 1)[1].replace(" ", "_")
        h = hashlib.md5(name.encode()).hexdigest()
        q = urllib.parse.quote(name)
        base = "https://upload.wikimedia.org/wikipedia/commons"
        thumb = f"{base}/thumb/{h[0]}/{h[:2]}/{q}/960px-{q}" + (".jpg" if name.lower().endswith((".tif", ".tiff")) else "")
        return [thumb, f"{base}/{h[0]}/{h[:2]}/{q}"]

    def image(self, title: str) -> Path | None:
        f = self.cache / "img" / (hashlib.sha1(title.encode()).hexdigest() + ".img")
        if f.exists():
            return f if f.stat().st_size else None
        for url in self.file_urls(title):
            data = self._get(url, binary=True)
            if data:
                f.write_bytes(data)
                return f
        f.write_bytes(b"")
        return None


def pick_face(faces, min_px=56.0):
    """The one clear subject of a photo, or None (crowds are ambiguous for labels)."""
    faces = sorted((f for f in faces if f["feature"] is not None), key=lambda f: -f["face_px"])
    if not faces or faces[0]["face_px"] < min_px:
        return None
    if len(faces) > 1 and faces[0]["face_px"] < 1.4 * faces[1]["face_px"]:
        return None
    return faces[0]


# ----------------------------------------------------------------- data assembly

def commons_rows(args, presence, exclude: set[str]):
    cm = Commons(args.cache / "commons")
    cache_path = args.cache / f"commons_features_{gf.CROP_VERSION.replace('/', '_')}.npz"
    fc = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            fc = {k: z[k] for k in z.files}
    roots = dict(COMMONS)
    for cat in args.regular_cats:
        roots[cat] = ("regular_glasses", "commons_regular")
    files: dict[str, tuple[str, str, str]] = {}
    rng = np.random.default_rng(1)
    for root, (label, device) in roots.items():
        found = cm.walk(root, depth=3 if label == "ai_glasses" else 1)
        print(f"  {root}: {len(found)} files", flush=True)
        if label == "regular_glasses" and len(found) > args.regular_per_cat:
            found = [found[i] for i in sorted(rng.choice(len(found), args.regular_per_cat, replace=False))]
        for t, cat in found:
            if t not in files:              # first (most specific) category wins
                files[t] = (label, device, cat)
    keyed = {t: hashlib.sha1(t.encode()).hexdigest() for t in files if t not in exclude}
    todo = [t for t in sorted(keyed) if f"{keyed[t]}|0" not in fc]
    print(f"  {len(keyed)} files, {len(todo)} to download and process", flush=True)
    with gf.FeatureExtractor() as fx:
        for n, title in enumerate(todo, 1):
            key = keyed[title]
            fc[f"{key}|0"] = fc[f"{key}|1"] = np.zeros(0, np.float32)
            try:
                path = cm.image(title)
                if path:
                    with Image.open(path) as im:
                        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                    for flip in (0, 1):
                        face = pick_face(fx.extract_faces(rgb[:, ::-1] if flip else rgb))
                        if face:
                            fc[f"{key}|{flip}"] = face["feature"]
                        if face and not flip:
                            b = face["boxes"]["band"]
                            Image.fromarray(rgb[b.y0:b.y1, b.x0:b.x1]).save(args.cache / "commons" / f"crop_{key}.jpg")
            except Exception as e:  # noqa: BLE001
                print(f"    ! {title}: {e}")
            if n % 25 == 0 or n == len(todo):
                np.savez(cache_path, **fc)
                print(f"  processed {n}/{len(todo)}", flush=True)
    usable = [t for t, k in keyed.items() if fc.get(f"{k}|0") is not None and fc[f"{k}|0"].size == gf.FEATURE_DIM]
    print(f"  {len(usable)} files with one clear face; fetching license metadata", flush=True)
    meta = cm.info(sorted(usable))
    rows, sources, review = [], [], []
    for title in sorted(usable):
        label, device, cat = files[title]
        m, key = meta.get(title), keyed[title]
        if not m or not OK_LICENSE.match(m["license"] or ""):
            continue
        f0 = fc[f"{key}|0"]
        p_glasses = gf.head_predict(presence, f0)["glasses"] if presence else 1.0
        if label == "regular_glasses" and p_glasses < 0.5:
            continue          # glasses not actually visible on the main face
        # group: person sub-category if there is one, else the author
        group = f"commons/{cat if cat not in roots else m['artist']}"
        for flip in (0, 1):
            f = fc.get(f"{key}|{flip}")
            if f is not None and f.size == gf.FEATURE_DIM:
                rows.append((f, label, group, device, flip))
        sources.append({"title": title, "label": label, "device": device, **{k: m[k] for k in ("page", "license", "license_url", "artist")}})
        if label == "ai_glasses":
            review.append((title, key))
    np.savez(cache_path, **fc)
    return rows, sources, review


def open_images_rows(args, rng):
    """Faces from the presence build's cache: eyewear -> regular_glasses, none -> no_glasses."""
    path = args.oi_cache / f"features_{gf.CROP_VERSION.replace('/', '_')}.npz"
    if not path.exists():
        sys.exit(f"Open Images feature cache not found: {path}\nRun build_presence.py with --cache {args.oi_cache} first.")
    with np.load(path) as z:
        keys = [k for k in z.files if not k.endswith("/done")]
        labels = {}
        for split in sorted({k.split("/")[0] for k in keys}):
            for img, faces in bp.label_faces(args.oi_cache, split).items():
                labels[(split, img)] = faces
        by_class = {"regular_glasses": [], "no_glasses": []}
        for k in keys:
            split, img, j, _ = k.split("/")
            faces = labels.get((split, img))
            if faces is None or int(j) >= len(faces):
                continue
            by_class["regular_glasses" if faces[int(j)][1] else "no_glasses"].append(k)
        rows = []
        for label, ks in by_class.items():
            ks = sorted(ks)
            rng.shuffle(ks)
            for k in ks[: args.oi_per_class]:
                f = z[k]
                if f.size == gf.FEATURE_DIM:
                    rows.append((f, label, "oi/" + k.split("/")[1], "open_images", 0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--oi-cache", type=Path, required=True, help="cache folder used by build_presence.py")
    ap.add_argument("--oi-per-class", type=int, default=2500)
    ap.add_argument("--regular-cats", nargs="*", default=[
        "Category:Portrait photographs of men with glasses", "Category:Portrait photographs of women wearing glasses",
        "Category:Young women wearing glasses", "Category:People with glasses"],
        help="Commons categories of people wearing ordinary glasses (same source as the AI photos)")
    ap.add_argument("--regular-per-cat", type=int, default=150)
    ap.add_argument("--exclude", type=Path, default=Path(__file__).with_name("base_exclude.json"),
                    help="Commons file titles removed after manual review (wrong label, glasses not worn, ...)")
    ap.add_argument("--max-false-alarm", type=float, default=0.05,
                    help="red-box threshold: at most this share of regular-glasses faces flagged (cross-validated)")
    ap.add_argument("--models", type=Path, default=ROOT / "web" / "models")
    ap.add_argument("--sources-md", type=Path, default=ROOT / "docs" / "BASE_DATA_SOURCES.md")
    ap.add_argument("--review-sheet", type=Path, help="write a contact sheet of the AI-glasses crops here (for manual review)")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    exclude = set(json.loads(args.exclude.read_text())["reasons"]) if args.exclude.exists() else set()
    presence_path = args.models / "presence.json"
    presence = json.loads(presence_path.read_text()) if presence_path.exists() else None

    print("Wikimedia Commons")
    c_rows, sources, review = commons_rows(args, presence, exclude)
    print("Open Images")
    o_rows = open_images_rows(args, rng)
    rows = c_rows + o_rows
    X = np.stack([r[0] for r in rows]).astype(np.float32)
    y = np.array([CLASSES.index(r[1]) for r in rows])
    groups = np.array([r[2] for r in rows])
    device = np.array([r[3] for r in rows])
    flip = np.array([r[4] for r in rows])
    orig = flip == 0
    print("rows (incl. mirrored):", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
          "| photos:", {c: int(((y == i) & orig).sum()) for i, c in enumerate(CLASSES)},
          "| AI devices:", {d: int(((device == d) & orig).sum()) for d in sorted(set(device[y == 0]))})

    if args.review_sheet and review:
        tiles = []
        for i, (title, key) in enumerate(review):
            p = args.cache / "commons" / f"crop_{key}.jpg"
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((220, 110))
                tile = Image.new("RGB", (230, 130), (255, 255, 255))
                tile.paste(im, (5, 5))
                from PIL import ImageDraw
                ImageDraw.Draw(tile).text((5, 116), str(i), fill=(200, 0, 0))
                tiles.append(tile)
        cols = 8
        sheet = Image.new("RGB", (cols * 230, ((len(tiles) + cols - 1) // cols) * 130), (255, 255, 255))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 230, (i // cols) * 130))
        sheet.save(args.review_sheet)
        (args.review_sheet.with_suffix(".txt")).write_text("\n".join(f"{i}\t{t}" for i, (t, _) in enumerate(review)))
        print(f"review sheet: {args.review_sheet}")

    head = basekit.fit_head(X, y, groups, device, flip, max_false_alarm=args.max_false_alarm, base=True,
                            trainer="build_base.py",
                            note="기본 모델 (공개 데이터: Open Images, Wikimedia Commons). 기초 수준이며, 직접 모은 사진을 더해 학습하면 좋아집니다.")
    (args.models / "head.json").write_text(json.dumps(head, ensure_ascii=False, separators=(",", ":")))
    basekit.save_base_data(args.models, X, y, groups, device, flip, {"trained_head_at": head["trained_at"], "n_contributions": 0})

    lines = ["# 기본 데이터 출처", "",
             "`web/models/head.json`(기본 모델)과 `web/models/base_data.*`(기본 데이터의 특징값)는 아래 공개 자료로 만들었습니다.",
             "저장소에는 사진이 들어 있지 않고, 사진에서 계산한 숫자(특징값)와 모델 가중치만 들어 있습니다.", "",
             "- **Open Images V7** (Google LLC): 주석 CC BY 4.0, 이미지 CC BY 2.0. 일반 안경·안경 없음 얼굴.", "",
             "## Wikimedia Commons", "", "| 파일 | 분류 | 작성자 | 라이선스 |", "| --- | --- | --- | --- |"]
    for s in sorted(sources, key=lambda s: (s["label"], s["title"])):
        name = s["title"].replace("|", "/")
        lines.append(f"| [{name}]({s['page']}) | {s['label']} ({s['device']}) | {s['artist'].replace('|', '/')} | {s['license']} |")
    args.sources_md.write_text("\n".join(lines) + "\n")
    size = (args.models / "base_data.bin").stat().st_size // 1024
    print(f"saved head.json, base_data.bin ({size} KB), base_data.json, {args.sources_md}")


if __name__ == "__main__":
    main()
