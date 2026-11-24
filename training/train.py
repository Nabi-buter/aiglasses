#!/usr/bin/env python3
"""Train the AI-glasses classifier head and export it for the web app.

Folder layout (one sub-folder per class, any image format Pillow reads):

    data/
      ai_glasses/        <- required: people wearing AI / smart glasses
      regular_glasses/   <- required: people wearing ordinary glasses
      no_glasses/        <- optional: people without glasses

Usage:
    python training/train.py --data data --out web/models/head.json

What it does:
  * extracts features for every image (and its mirror image) with
    glasses_features.FeatureExtractor, caching them in <data>/.feature_cache.npz
  * picks the regularisation strength by grouped, stratified k-fold
    cross-validation (an image and its mirror are always in the same fold)
  * prints a report (accuracy, AI-glasses precision/recall, confusion matrix)
  * fits on everything and writes head.json for web/ and predict.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glasses_features as gf  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
AI_CLASS = "ai_glasses"
REQUIRED = {AI_CLASS, "regular_glasses"}
KNOWN = REQUIRED | {"no_glasses"}
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)


def list_images(data: Path) -> list[tuple[Path, str]]:
    items = []
    for d in sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if d.name not in KNOWN:
            print(f"  ! 알 수 없는 폴더 '{d.name}' 는 건너뜁니다 (허용: {', '.join(sorted(KNOWN))})")
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMAGE_EXT:
                items.append((f, d.name))
    return items


def group_of(p: Path) -> str:
    """Photos of the same person/session must stay in one CV fold.
    Name files like  'person07__front.jpg'  and everything before '__' is the group."""
    stem = p.stem
    return f"{p.parent.name}/{stem.split('__', 1)[0]}" if "__" in stem else str(p)


def file_key(p: Path) -> str:
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{gf.CROP_VERSION}".encode()).hexdigest()


def extract_all(items, data: Path, model_dir: Path, flip: bool):
    cache_path = data / ".feature_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}

    X, y, groups, skipped = [], [], [], []
    variants = (False, True) if flip else (False,)
    with gf.FeatureExtractor(model_dir) as fx:
        for i, (path, label) in enumerate(items, 1):
            key = file_key(path)
            got_any = False
            for fl in variants:
                k = f"{key}_{int(fl)}"
                if k not in cache:
                    try:
                        ex = fx.extract_path(path, flip=fl)
                    except Exception as e:  # unreadable file
                        ex = gf.Extraction(None, None, f"read_error: {e}")
                    cache[k] = ex.feature if ex.feature is not None else np.zeros(0, np.float32)
                feat = cache[k]
                if feat.size == gf.FEATURE_DIM:
                    X.append(feat); y.append(label); groups.append(group_of(path)); got_any = True
            if not got_any:
                skipped.append(path)
            if i % 25 == 0 or i == len(items):
                print(f"  특징 추출 {i}/{len(items)}", end="\r", flush=True)
    print()
    np.savez_compressed(cache_path, **cache)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(groups), skipped


def make_model(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


def cross_validate(X, y, groups, classes, k):
    """Grouped CV over C_GRID. Returns the best C, its metrics and out-of-fold AI probabilities."""
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
    folds = list(cv.split(X, y, groups))
    ai = classes.index(AI_CLASS)
    best = None
    for C in C_GRID:
        pred = np.empty(len(y), dtype=object)
        p_ai = np.zeros(len(y))
        for tr, te in folds:
            m = make_model(C).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
            p_ai[te] = m.predict_proba(X[te])[:, list(m.classes_).index(AI_CLASS)]
        acc = float((pred == y).mean())
        print(f"  C={C:<5} 교차검증 정확도 {acc:.3f}")
        if best is None or acc > best[1]:
            best = (C, acc, pred, p_ai)
    C, acc, pred, p_ai = best
    cm = confusion_matrix(y, pred, labels=classes)
    tp = cm[ai, ai]
    recall = tp / max(cm[ai].sum(), 1)
    precision = tp / max(cm[:, ai].sum(), 1)
    is_ai = y == AI_CLASS
    auc = float(roc_auc_score(is_ai, p_ai)) if 0 < is_ai.sum() < len(y) else None
    return C, {"accuracy": round(acc, 4), "ai_recall": round(float(recall), 4),
               "ai_precision": round(float(precision), 4), "confusion": cm.tolist(), "folds": k,
               "ai_auc": None if auc is None else round(auc, 4)}, p_ai


def rates_at(t: float, p_ai: np.ndarray, y: np.ndarray) -> dict:
    """Screening view of one threshold: AI glasses caught, and false alarms among the rest
    (split into people wearing regular glasses and people without glasses)."""
    is_ai = y == AI_CLASS
    out = {"threshold": round(float(t), 4), "recall": round(float((p_ai[is_ai] >= t).mean()), 4),
           "false_alarm": round(float((p_ai[~is_ai] >= t).mean()), 4)}
    reg = y == "regular_glasses"
    if reg.any():
        out["false_alarm_regular"] = round(float((p_ai[reg] >= t).mean()), 4)
    return out


def threshold_for_recall(p_ai: np.ndarray, y: np.ndarray, target: float) -> float:
    """Largest threshold that still flags at least `target` of the AI-glasses photos."""
    s = np.sort(p_ai[y == AI_CLASS])
    if not len(s):
        return 0.3
    idx = int(np.floor((1 - target) * len(s)))
    return float(s[min(idx, len(s) - 1)]) - 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("web/models/head.json"))
    ap.add_argument("--models", type=Path, default=gf.DEFAULT_MODEL_DIR, help="MediaPipe 모델 폴더")
    ap.add_argument("--no-flip", action="store_true", help="좌우 반전 증강 끄기")
    ap.add_argument("--ai-threshold", type=float, default=0.6, help="이 확률 이상이면 'AI 안경'")
    ap.add_argument("--suspect-threshold", type=float, default=0.3, help="이 확률 이상이면 'AI 안경 의심'")
    ap.add_argument("--target-recall", type=float,
                    help="교차검증에서 AI 안경 사진의 이 비율 이상을 빨간 박스로 잡도록 의심 기준을 자동으로 정합니다 (예: 0.95)")
    ap.add_argument("--note", default="", help="head.json 에 남길 메모 (데이터 출처 등)")
    ap.add_argument("--test", type=Path, help="학습에 쓰지 않은 별도 평가 폴더 (같은 구조). 다른 날·다른 사람 사진 권장")
    args = ap.parse_args()

    if not args.data.is_dir():
        sys.exit(f"데이터 폴더가 없습니다: {args.data}")
    items = list_images(args.data)
    present = sorted({lbl for _, lbl in items})
    missing = REQUIRED - set(present)
    if missing:
        sys.exit(f"필수 클래스 폴더에 사진이 없습니다: {', '.join(sorted(missing))}")

    print(f"사진 {len(items)}장 · 클래스 {present}")
    X, y, groups, skipped = extract_all(items, args.data, args.models, flip=not args.no_flip)
    if skipped:
        print(f"  ! 얼굴을 찾지 못해 {len(skipped)}장을 뺐습니다 (예: {skipped[0]})")

    classes = sorted(set(y.tolist()))
    used = {str(p) for p, _ in items} - {str(p) for p in skipped}
    counts = {c: sum(1 for p, l in items if l == c and str(p) in used) for c in classes}
    print("학습에 쓰는 원본 사진 수:", counts, f"· 묶음(사람/세션) {len(set(groups))}개")
    if min(counts.values()) < 5:
        sys.exit("클래스마다 얼굴이 보이는 사진이 최소 5장은 있어야 합니다. 권장은 클래스당 200장 이상입니다.")
    if min(counts.values()) < 50:
        print("  ! 사진이 적습니다. 결과를 실전에 쓰기 전에 클래스당 200장 이상으로 늘리세요.")

    groups_per_class = {c: len({g for g, l in zip(groups, y) if l == c}) for c in classes}
    k = int(min(5, min(groups_per_class.values())))
    if k < 2:
        sys.exit("교차검증을 하려면 클래스마다 서로 다른 사람/세션 묶음이 2개 이상 필요합니다.")
    print(f"{k}-겹 교차검증 (원본·반전 사진은 같은 묶음)")
    C, metrics, p_ai = cross_validate(X, y, groups, classes, k)
    auc = f"  AUC {metrics['ai_auc']:.3f}" if metrics["ai_auc"] is not None else ""
    print(f"\n선택: C={C}  정확도 {metrics['accuracy']:.3f}  "
          f"AI 안경 재현율 {metrics['ai_recall']:.3f}  정밀도 {metrics['ai_precision']:.3f}{auc}")
    print("혼동행렬 (행=정답, 열=예측):", classes)
    for c, row in zip(classes, metrics["confusion"]):
        print(f"  {c:<16} {row}")

    suspect = args.suspect_threshold
    if args.target_recall:
        suspect = min(max(threshold_for_recall(p_ai, y, args.target_recall), 0.02), args.ai_threshold)
        print(f"\n목표 재현율 {args.target_recall:.0%} → 의심 기준 {suspect:.3f}")
    metrics["at_suspect"] = rates_at(suspect, p_ai, y)
    metrics["at_ai"] = rates_at(args.ai_threshold, p_ai, y)
    print("판정 기준별 예상 성능 (사진 1장 기준, 실시간은 여러 프레임 평균이라 보통 더 좋음):")
    for name, r in (("의심(빨간 박스)", metrics["at_suspect"]), ("AI 안경", metrics["at_ai"])):
        print(f"  {name:<12} 기준 {r['threshold']:.3f}: AI 안경 탐지 {r['recall']:.1%} · "
              f"다른 사람을 잘못 표시 {r['false_alarm']:.1%}")
    print("  기준별 표:  " + "  ".join(f"{t:.1f}→탐지 {rates_at(t, p_ai, y)['recall']:.0%}/오탐 {rates_at(t, p_ai, y)['false_alarm']:.0%}"
                                   for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)))

    model = make_model(C).fit(X, y)
    order = [list(model.classes_).index(c) for c in classes]
    W = model.coef_ if len(classes) > 2 else np.vstack([-model.coef_[0] / 2, model.coef_[0] / 2])
    b = model.intercept_ if len(classes) > 2 else np.array([-model.intercept_[0] / 2, model.intercept_[0] / 2])
    W, b = W[order], b[order]

    test_metrics = None
    if args.test:
        t_items = list_images(args.test)
        Xt, yt, _, t_skipped = extract_all(t_items, args.test, args.models, flip=False)
        keep = np.isin(yt, classes)
        Xt, yt = Xt[keep], yt[keep]
        if len(yt):
            pt = model.predict(Xt)
            cm = confusion_matrix(yt, pt, labels=classes)
            ai = classes.index(AI_CLASS)
            test_metrics = {
                "accuracy": round(float((pt == yt).mean()), 4),
                "ai_recall": round(float(cm[ai, ai] / max(cm[ai].sum(), 1)), 4),
                "ai_precision": round(float(cm[ai, ai] / max(cm[:, ai].sum(), 1)), 4),
                "confusion": cm.tolist(), "n": int(len(yt)),
            }
            print(f"\n별도 평가 ({len(yt)}장, 얼굴 못 찾음 {len(t_skipped)}장 제외): 정확도 {test_metrics['accuracy']:.3f}  "
                  f"AI 안경 재현율 {test_metrics['ai_recall']:.3f}  정밀도 {test_metrics['ai_precision']:.3f}")
            for c, row in zip(classes, test_metrics["confusion"]):
                print(f"  {c:<16} {row}")

    head = {
        "format": "ai-glasses-head/v1",
        "embedder": gf.EMBEDDER_FILE,
        "crop": gf.CROP_VERSION,
        "feature_dim": gf.FEATURE_DIM,
        "classes": classes,
        "ai_class": AI_CLASS,
        "W": [[round(float(v), 6) for v in row] for row in W],
        "b": [round(float(v), 6) for v in b],
        "thresholds": {"ai": args.ai_threshold, "suspect": round(float(suspect), 4)},
        "C": C,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_images": counts,
        "cv": metrics,
        "test": test_metrics,
        "note": args.note,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(head, ensure_ascii=False, separators=(",", ":")))
    print(f"\n저장: {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
