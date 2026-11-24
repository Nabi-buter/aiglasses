#!/usr/bin/env python3
"""Classify photos with a trained head (same pipeline as the web app).

    python training/predict.py photo1.jpg photo2.jpg --head web/models/head.json
    python training/predict.py data/ai_glasses/*.jpg --save-crops /tmp/crops   # 잘라낸 영역 확인
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glasses_features as gf  # noqa: E402

LABELS = {"AI_GLASSES": "AI 안경", "SUSPECTED": "AI 안경 의심", "REGULAR": "일반 안경",
          "NO_GLASSES": "안경 없음", "NO_FACE": "얼굴 못 찾음"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--head", type=Path, default=Path("web/models/head.json"))
    ap.add_argument("--models", type=Path, default=gf.DEFAULT_MODEL_DIR)
    ap.add_argument("--save-crops", type=Path, help="잘라낸 영역(band, hinge_l, hinge_r)을 저장할 폴더")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 줄로 출력")
    args = ap.parse_args()

    head = json.loads(args.head.read_text()) if args.head.exists() else None
    if head and head.get("crop") != gf.CROP_VERSION:
        sys.exit(f"head.json 의 crop 버전({head.get('crop')})이 코드({gf.CROP_VERSION})와 다릅니다. 다시 학습하세요.")
    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)

    with gf.FeatureExtractor(args.models) as fx:
        for path in args.images:
            rgb = gf.load_rgb(path)
            ex = fx.extract(rgb)
            out = {"file": str(path)}
            if ex.boxes:
                out["boxes"] = {k: b.as_list() for k, b in ex.boxes.items()}
                if args.save_crops:
                    for k, b in ex.boxes.items():
                        if not b.empty:
                            Image.fromarray(rgb[b.y0:b.y1, b.x0:b.x1]).save(args.save_crops / f"{path.stem}_{k}.png")
            if ex.feature is None:
                out["verdict"] = "NO_FACE"
            elif head:
                probs = gf.head_predict(head, ex.feature)
                out["probs"] = {k: round(v, 4) for k, v in probs.items()}
                out["verdict"] = gf.verdict(head, probs)
            if args.json:
                print(json.dumps(out, ensure_ascii=False))
            else:
                probs = "  ".join(f"{k}={v:.2f}" for k, v in out.get("probs", {}).items())
                print(f"{path.name:<32} {LABELS.get(out.get('verdict'), '-'):<10} {probs}")


if __name__ == "__main__":
    main()
