# AI 안경 판독기

시험 감독관이 응시자의 안경을 촬영하면, 그 안경이 **AI 안경(스마트 안경)** 인지 **일반 안경** 인지 판독하는 웹 앱입니다.
AI 안경은 카메라·마이크·스피커·디스플레이가 들어간 안경으로, Ray-Ban Meta, Xiaomi, Rokid, Even Realities 등이 있습니다.

- **기기 안에서만 동작합니다.** 사진은 서버로 전송되지 않습니다. 클라우드 API나 사용료가 없습니다.
- **오프라인으로 동작합니다.** 한 번 열면 인터넷이 없는 고사실에서도 동작합니다(서비스 워커 캐시).
- **정적 사이트입니다.** GitHub Pages에 그대로 올리면 됩니다. 휴대폰·태블릿·노트북 브라우저에서 엽니다.

> ⚠️ 저장소에는 **학습된 판독 모델(`web/models/head.json`)이 들어 있지 않습니다.**
> 실제 AI 안경 사진으로 직접 학습해야 판정이 나옵니다. 모델이 없으면 얼굴·안경 영역 찾기까지만 동작합니다.
> 아래 [모델 학습](#모델-학습)과 [docs/DATA.md](docs/DATA.md)를 보세요.

## 동작 방식

```
사진 ─▶ ① 얼굴 랜드마크 ─▶ ② 안경 영역 3곳 잘라내기 ─▶ ③ 이미지 임베딩 ─▶ ④ 분류기 ─▶ 판정
        (MediaPipe          · 안경 전체(band)           (MobileNetV3-Large,    (로지스틱 회귀,
         Face Landmarker)   · 왼쪽 경첩, 오른쪽 경첩      1280차원 × 2)          head.json)
```

1. **얼굴 찾기:** MediaPipe Face Landmarker가 얼굴 랜드마크 478개를 찾습니다. 얼굴이 여러 개면 가장 큰 얼굴을 씁니다.
2. **영역 잘라내기:** 랜드마크로 세 영역을 잘라냅니다.
   - 눈썹부터 광대 위까지, 얼굴 끝에서 끝까지의 **안경 전체**
   - 카메라형 AI 안경이 카메라와 LED를 두는 **양쪽 경첩 부근**. 눈꼬리와 얼굴 가장자리 사이, 눈썹 높이입니다.

   경첩 부근을 따로 크게 보기 때문에 작은 카메라 렌즈도 놓치지 않습니다.
3. **특징 추출:** 각 영역을 MobileNetV3 임베더에 넣습니다. 특징은 `[안경 전체, 경첩 평균]`을 이어 붙인 2560차원입니다.
4. **분류:** 직접 학습한 로지스틱 회귀(`head.json`, 70KB 안팎)가 `ai_glasses / regular_glasses / no_glasses` 확률을 냅니다.
   - 사진을 여러 장 넣으면 **AI 안경 확률이 가장 높은 사진**을 기준으로 판정합니다. 놓치지 않는 쪽을 우선한 규칙입니다.

학습(Python)과 판독(브라우저)은 **같은 모델 파일과 같은 영역 계산**을 씁니다.
`tests/test_pipeline.py --browser`가 두 쪽의 특징값과 확률이 일치하는지 확인합니다. 현재 차이는 0입니다.

## 바로 써 보기

```bash
cd web
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000
```

카메라는 `https://` 또는 `localhost`에서만 켜집니다. 파일을 더블클릭해서 열면 동작하지 않습니다.

## GitHub Pages 배포

1. 이 저장소를 GitHub에 올립니다.
2. **Settings → Pages → Build and deployment → Source** 를 **GitHub Actions** 로 바꿉니다.
3. `main` 브랜치에 push하면 `.github/workflows/pages.yml`이 `web/` 폴더를 배포합니다.
   주소는 `https://<사용자>.github.io/<저장소>/` 입니다.

학습한 `head.json`을 `web/models/`에 넣고 push하면 모든 감독관이 같은 모델을 씁니다.
모델을 공개하고 싶지 않다면 배포에는 빼고, 각 기기에서 **판독 모델 → 모델 파일 불러오기**로 넣어도 됩니다.
불러온 모델은 그 브라우저에만 저장됩니다.

## 모델 학습

### 1. 환경

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
# Linux 서버라면 MediaPipe에 필요한 라이브러리: sudo apt-get install -y libegl1 libgles2
```

### 2. 데이터

```
data/
  ai_glasses/        AI 안경을 쓴 사람 사진        (필수)
  regular_glasses/   일반 안경을 쓴 사람 사진      (필수)
  no_glasses/        안경을 안 쓴 사람 사진        (선택, 권장)
```

얼굴이 보이는 사진이어야 합니다. 얼굴을 찾지 못한 사진은 자동으로 빠집니다.
**클래스당 최소 200장, 가능하면 500장 이상**을 권장합니다. 무엇을 어떻게 모을지는 [docs/DATA.md](docs/DATA.md)에 정리했습니다.
`data/` 폴더는 `.gitignore`에 들어 있습니다. 사람 얼굴 사진을 공개 저장소에 올리지 마세요.

### 3. 학습

```bash
python training/train.py --data data --out web/models/head.json --note "2026-11 1차 수집 / 고사장 촬영"
```

- 사진과 좌우 반전본으로 특징을 뽑습니다. 결과는 `data/.feature_cache.npz`에 캐시되어 다시 학습할 때 빠릅니다.
- 5겹 교차검증으로 정규화 강도를 고르고 결과를 출력합니다. 원본과 반전본은 같은 겹에 묶입니다.
- 출력 항목은 정확도, AI 안경 재현율·정밀도, 혼동행렬입니다.
- 마지막으로 전체 데이터로 학습해 `head.json`을 저장합니다. 교차검증 결과도 파일에 같이 기록되어 웹 앱의 **판독 모델** 칸에 표시됩니다.

판정 기준은 학습할 때 바꿀 수 있습니다.

```bash
python training/train.py ... --ai-threshold 0.6 --suspect-threshold 0.3
```

AI 안경 확률이 `--ai-threshold` 이상이면 **AI 안경**, `--suspect-threshold` 이상이면 **의심**으로 판정합니다.
감독 현장에서는 놓치는 것(미탐)이 더 위험하므로 의심 기준을 낮게 두는 편을 권합니다.

### 4. 확인

```bash
python training/predict.py 새사진1.jpg 새사진2.jpg --head web/models/head.json
python training/predict.py data/ai_glasses/*.jpg --save-crops /tmp/crops   # 잘라낸 영역 눈으로 확인
```

## 테스트

```bash
python tests/test_pipeline.py            # 합성 데이터 → 학습 → 예측
python tests/test_pipeline.py --browser  # + 브라우저(web/pipeline.js)와 결과가 같은지 (Node + Playwright 필요)
```

`tests/make_synthetic.py`는 scikit-image에 포함된 공개 사진에 만화 같은 안경을 그려 **배관 점검용** 데이터를 만듭니다.
이 데이터로 학습한 모델은 실제 판독에 쓸 수 없습니다.

## 폴더 구조

```
web/                      GitHub Pages로 배포되는 정적 앱
  index.html, app.js      화면, 카메라, 촬영함, 기록, CSV 저장
  pipeline.js             판독 파이프라인 (training/glasses_features.py 와 1:1 대응)
  sw.js                   오프라인 캐시
  models/                 face_landmarker.task, mobilenet_v3_large.tflite, (head.json)
  vendor/                 @mediapipe/tasks-vision 1.2.2 런타임 (wasm 포함)
training/
  glasses_features.py     영역 계산·특징 추출 (웹과 동일)
  train.py                학습 + 교차검증 + head.json 내보내기
  predict.py              명령줄 판독
tests/                    합성 데이터 생성, 파이프라인·브라우저 일치 테스트
scripts/fetch_assets.sh   MediaPipe 런타임·모델 다시 받기 (버전 올릴 때)
docs/DATA.md              학습 데이터 수집 가이드
```

## 한계와 주의

- **보조 도구입니다.** 판정은 확률이며 틀릴 수 있습니다. 의심되면 감독관이 안경을 직접 확인하세요.
- **데이터가 성능을 정합니다.** 모은 데이터에 없는 기종, 각도, 조명에서는 성능이 떨어집니다. 새 기종이 나오면 사진을 추가해 다시 학습하세요.
- **카메라가 보이지 않는 기종은 어렵습니다.** 스피커만 있는 오디오 안경(Echo Frames, Bose 등)과 디스플레이만 있는 안경은 외형 차이가 작습니다. 테 두께, 다리 모양, 렌즈 반사 같은 단서에 기대게 되므로 이런 기종의 사진을 특히 많이 모아야 합니다.
- **개인정보:** 응시자 촬영은 각 시험의 규정과 개인정보 보호 지침을 따르세요. 앱은 사진을 저장하거나 전송하지 않습니다. 감독 기록에는 작은 썸네일이 그 브라우저에만 남고, **비우기**로 지울 수 있습니다.

## 라이선스

- 이 저장소의 코드: [MIT](LICENSE)
- `web/vendor/`(MediaPipe Tasks Vision)와 `web/models/`의 MediaPipe 모델: Apache License 2.0, Google. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 보세요.
