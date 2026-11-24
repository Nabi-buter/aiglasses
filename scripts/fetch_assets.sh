#!/usr/bin/env bash
# 웹 앱이 오프라인으로 동작하도록 MediaPipe 런타임과 모델 파일을 web/ 아래에 받습니다.
# 저장소에 이미 포함되어 있으므로 버전을 올릴 때만 다시 실행하면 됩니다.
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS_VISION_VERSION="1.2.2"   # training/requirements.txt 의 mediapipe 버전과 맞출 것
MODELS="https://storage.googleapis.com/mediapipe-models"

mkdir -p web/vendor/wasm web/models
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

( cd "$tmp" && npm pack "@mediapipe/tasks-vision@${TASKS_VISION_VERSION}" >/dev/null && tar xzf ./*.tgz )
cp "$tmp/package/vision_bundle.mjs" web/vendor/
cp "$tmp/package/wasm/vision_wasm_internal.js" "$tmp/package/wasm/vision_wasm_internal.wasm" web/vendor/wasm/

curl -fsSL -o web/models/face_landmarker.task \
  "$MODELS/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
curl -fsSL -o web/models/mobilenet_v3_large.tflite \
  "$MODELS/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite"

echo "done:"; ls -la web/vendor web/vendor/wasm web/models
