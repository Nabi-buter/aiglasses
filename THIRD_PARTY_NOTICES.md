# Third-party notices

## MediaPipe Tasks Vision (web/vendor/)
`@mediapipe/tasks-vision` 1.2.2 — Copyright Google LLC — Apache License 2.0
https://github.com/google-ai-edge/mediapipe · https://www.apache.org/licenses/LICENSE-2.0

## MediaPipe models (web/models/)
- `face_landmarker.task` — MediaPipe Face Landmarker
- `mobilenet_v3_large.tflite` — MediaPipe Image Embedder (MobileNetV3-Large)

Copyright Google LLC — Apache License 2.0. Downloaded from
https://storage.googleapis.com/mediapipe-models/ by `scripts/fetch_assets.sh`.
Model cards: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker ,
https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder

## Test image
`tests/make_synthetic.py` uses `skimage.data.astronaut()` (NASA photograph of Eileen Collins, public domain),
which ships with scikit-image. No image is stored in this repository.

## docs/live_view.jpg

The README screenshot shows the astronaut photo bundled with scikit-image (Eileen Collins, NASA, public domain) with cartoon glasses drawn on by `tests/make_synthetic.py`.

## web/models/presence.json

Weights learned by `training/build_presence.py` from Open Images V7 (https://storage.googleapis.com/openimages/web/index.html).
Annotations © Google LLC, licensed under CC BY 4.0. Images are listed as having a CC BY 2.0 license; no images are redistributed here, only model weights.

## web/models/head.json, web/models/base_data.*

The base model and base feature set are computed from Open Images V7 (as above) and from freely licensed
Wikimedia Commons photos (CC0 / CC BY / CC BY-SA / public domain). Author and license of every Commons file
are listed in [docs/BASE_DATA_SOURCES.md](docs/BASE_DATA_SOURCES.md). No images are redistributed here,
only numeric features and model weights.
