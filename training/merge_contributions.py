#!/usr/bin/env python3
"""Merge every user contribution into the SHARED data and retrain the SHARED model.

    contributions/*.json   files made on the web training page ("공용 모델에 기여")
                           features only (numbers), no photos
    web/models/base_data.*  public base data + all accepted contributions (rewritten)
    web/models/head.json    shared model every device downloads (rewritten if the quality gate passes)

The repository only keeps the public base data and the contribution files. The Pages deploy
(.github/workflows/pages.yml) runs this script on every deploy and publishes the merged model and
data with the site, so nothing large is ever committed. Idempotent: contribution rows already in
base_data are dropped and re-added from contributions/.

Quality gate (the shared model is NOT replaced when it fails, the run exits with an error):
  - cross-validated AI-glasses AUC must not drop more than --max-auc-drop below the current model
  - AI-glasses recall at the red line must stay >= --min-recall
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basekit as bk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS_PER_FILE = 20000


def load_contribution(path: Path) -> tuple[dict | None, str]:
    """-> (rows dict, "") or (None, reason)."""
    try:
        c = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return None, f"JSON을 읽을 수 없음 ({e})"
    if c.get("format") != "ai-glasses-contribution/v1":
        return None, "형식이 ai-glasses-contribution/v1 이 아님"
    if c.get("crop") != bk.CROP_VERSION or c.get("dim") != bk.DIM:
        return None, f"특징 버전이 다름 ({c.get('crop')}, {c.get('dim')})"
    n = c.get("n")
    if not isinstance(n, int) or not 1 <= n <= MAX_ROWS_PER_FILE:
        return None, "행 수가 잘못됨"
    if not c.get("consent"):
        return None, "동의 표시가 없음"
    classes = c.get("classes") or []
    if not set(classes) <= set(bk.CLASSES):
        return None, f"알 수 없는 클래스 {classes}"
    for k in ("y", "group", "flip", "device"):
        if not isinstance(c.get(k), list) or len(c[k]) != n:
            return None, f"'{k}' 길이가 n과 다름"
    try:
        q = np.frombuffer(base64.b64decode(c["data_b64"]), np.int8)
        scale = np.frombuffer(base64.b64decode(c["scale_b64"]), "<f4")
    except Exception:  # noqa: BLE001
        return None, "특징값을 해독할 수 없음"
    if q.size != n * bk.DIM or scale.size != n or not np.isfinite(scale).all() or (scale <= 0).any() or (scale > 1).any():
        return None, "특징값 크기나 배율이 잘못됨"
    q = q.reshape(n, bk.DIM)
    X = bk.dequantize(q, scale)
    band = np.linalg.norm(X[:, :bk.DIM // 2], axis=1)
    if not ((band > 0.85) & (band < 1.15)).all():
        return None, "앱이 만든 특징값이 아님 (임베딩 크기 검사 실패)"
    try:
        y = np.array([bk.CLASSES.index(classes[i]) for i in c["y"]])
        groups, devices = c.get("groups") or [], c.get("devices") or []
        group = np.array([f"contrib/{c['id']}/{groups[g]}" for g in c["group"]])
        source = np.array([bk.CONTRIB + str(devices[d] or "unknown")[:32] for d in c["device"]])
    except Exception:  # noqa: BLE001
        return None, "라벨·묶음·기종 색인이 잘못됨"
    flip = np.array([1 if v else 0 for v in c["flip"]])
    return {"id": str(c.get("id", path.stem))[:64], "X": X, "q": q, "scale": scale, "y": y, "group": group,
            "source": source, "flip": flip, "note": str(c.get("note", ""))[:200], "created_at": str(c.get("created_at", ""))[:40]}, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contributions", type=Path, default=ROOT / "contributions")
    ap.add_argument("--models", type=Path, default=ROOT / "web" / "models")
    ap.add_argument("--report", type=Path, help="write a Markdown report here (CI appends it to the run summary)")
    ap.add_argument("--max-false-alarm", type=float, default=0.05)
    ap.add_argument("--max-auc-drop", type=float, default=0.03)
    ap.add_argument("--min-recall", type=float, default=0.6)
    ap.add_argument("--force", action="store_true", help="write the new model even if the quality gate fails")
    args = ap.parse_args()

    base = bk.load_base_data(args.models)
    keep = ~np.char.startswith(base["source"].astype(str), bk.CONTRIB)
    print(f"public rows: {int(keep.sum())} (dropped {int((~keep).sum())} previously merged contribution rows)")
    parts = {k: [base[k][keep]] for k in ("q", "scale", "y", "group", "source", "flip")}
    seen = {hashlib.sha1(r.tobytes()).digest() for r in base["q"][keep]}

    accepted, rejected = [], []
    for path in sorted(args.contributions.glob("*.json")):
        c, why = load_contribution(path)
        if c is None:
            rejected.append((path.name, why)); print(f"  ✗ {path.name}: {why}"); continue
        new = np.array([hashlib.sha1(r.tobytes()).digest() not in seen for r in c["q"]])
        if not new.any():
            rejected.append((path.name, "이미 들어 있는 특징값만 있음 (중복)")); print(f"  ✗ {path.name}: duplicate"); continue
        for r in c["q"][new]:
            seen.add(hashlib.sha1(r.tobytes()).digest())
        for k in parts:
            parts[k].append(c[k][new])
        counts = {cl: int(((c["y"] == i) & (c["flip"] == 0) & new).sum()) for i, cl in enumerate(bk.CLASSES)}
        accepted.append({"file": path.name, "id": c["id"], "counts": counts, "rows": int(new.sum()),
                         "devices": sorted({s.replace(bk.CONTRIB, "") for s in c["source"][new & (c["y"] == 0)]}),
                         "note": c["note"], "created_at": c["created_at"]})
        print(f"  ✓ {path.name}: {counts} ({int(new.sum())} rows)")

    if not accepted:
        print("no accepted contributions: shared model left as committed")
        if args.report:
            args.report.write_text("# 공용 데이터 기여\n\n받아들인 기여 파일이 없어 공개 데이터 모델을 그대로 배포합니다.\n" +
                                   "".join(f"\n- 거부: {f} — {w}" for f, w in rejected) + "\n")
        return
    q = np.concatenate(parts["q"]); scale = np.concatenate(parts["scale"])
    y = np.concatenate(parts["y"]); group = np.concatenate(parts["group"])
    source = np.concatenate(parts["source"]); flip = np.concatenate(parts["flip"])
    X = bk.dequantize(q, scale)
    print(f"total rows {len(y)}: " + str({c: int(((y == i) & (flip == 0)).sum()) for i, c in enumerate(bk.CLASSES)}))

    head = bk.fit_head(X, y, group, source, flip, max_false_alarm=args.max_false_alarm, base=True,
                       trainer="merge_contributions.py",
                       note=f"공용 모델: 공개 데이터 + 사용자 기여 {len(accepted)}건")
    head["n_contributions"] = len(accepted)

    old_path = args.models / "head.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    old_auc, new_auc = (old.get("cv") or {}).get("ai_auc"), head["cv"]["ai_auc"]
    recall = head["cv"]["at_suspect"]["recall"]
    problems = []
    if old_auc is not None and new_auc is not None and new_auc < old_auc - args.max_auc_drop:
        problems.append(f"AI 안경 AUC가 {old_auc} → {new_auc} 로 {args.max_auc_drop} 넘게 떨어짐")
    if recall is None or recall < args.min_recall:
        problems.append(f"빨간 박스 기준 AI 안경 탐지율 {recall} < {args.min_recall}")

    lines = ["# 공용 데이터 기여 목록", "",
             "`contributions/` 폴더의 파일이 공개 기본 데이터와 합쳐져 공용 모델(`web/models/head.json`)이 됩니다.",
             "파일에는 사진이 아니라 앱이 계산한 특징값(숫자)만 들어 있습니다. 자동 생성 문서입니다 (`training/merge_contributions.py`).", "",
             f"- 마지막 재학습: {head['trained_at']}" + (" (품질 기준 미달로 **반영 안 됨**)" if problems and not args.force else ""),
             f"- 받아들인 기여 {len(accepted)}건 · 거부 {len(rejected)}건",
             "- 공용 데이터 얼굴 수: " + " · ".join(f"{c} {n}" for c, n in head["n_images"].items()) + " (이 중 기여: " +
             " · ".join(f"{c} {n}" for c, n in head["n_contrib"].items()) + ")",
             f"- 교차검증: AI 안경 AUC {new_auc} · 빨간 박스 기준 AI 탐지 {recall} · 일반 안경 오탐 {head['cv']['at_suspect']['false_alarm_regular']}"]
    if "contributed" in head["cv"]:
        cb = head["cv"]["contributed"]
        lines.append(f"- 기여된 사진에서만: AI 탐지 {cb['recall']} ({cb['n_ai']}장) · 일반 안경 오탐 {cb['false_alarm_regular']} ({cb['n_regular']}장)")
    lines += ["", "| 파일 | 만든 시각 | AI 안경 | 일반 안경 | 안경 없음 | AI 기종 | 메모 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in accepted:
        cnt = a["counts"]
        lines.append(f"| {a['file']} | {a['created_at']} | {cnt['ai_glasses']} | {cnt['regular_glasses']} | {cnt['no_glasses']} | "
                     f"{', '.join(a['devices']) or '—'} | {a['note'].replace('|', '/') or '—'} |")
    if rejected:
        lines += ["", "## 거부된 파일", "", "| 파일 | 이유 |", "| --- | --- |"] + [f"| {f} | {w} |" for f, w in rejected]
    if args.report:
        args.report.write_text("\n".join(lines) + "\n")

    if problems and not args.force:
        print("QUALITY GATE FAILED — shared model and data left unchanged:\n  " + "\n  ".join(problems))
        sys.exit(2)
    old_path.write_text(json.dumps(head, ensure_ascii=False, separators=(",", ":")))
    bk.save_base_data(args.models, X, y, group, source, flip,
                      {"trained_head_at": head["trained_at"], "n_contributions": len(accepted)}, q=q, scale=scale)
    print(f"saved {old_path} and base_data ({len(y)} rows), report {args.report}")


if __name__ == "__main__":
    main()
