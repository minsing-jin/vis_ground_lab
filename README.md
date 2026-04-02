# vis-ground-lab (`myvg`, `vis-ground-lab`)

로컬에서 UI 비전 모델을 빠르게 만들고 운영하기 위한 실험 프레임워크입니다.

- Task 1: `grounding` (기존 Florence2 + LoRA)
- Task 2: `tool_button_detection` (신규, 툴/게임 전용 소형 detector)

## 1. 핵심 아이디어

`tool_button_detection`은 일반화가 목적이 아닙니다.
하나의 툴/게임(Excel, PowerPoint, Civ, StarCraft 등) UI에서 특정 버튼/아이콘을 빠르게 탐지하는 소형 모델을 만드는 파이프라인입니다.

전체 흐름:
`extract -> prelabel -> label -> optimize -> export -> run`

`data-harvest`는 별도 경로입니다.
현재 active scope는 detector/grounding이 아니라 **작은 vision router 학습용 데이터 생산**입니다.
즉 `screen_type`, `situation_id`, `primitive_id`를 안정적으로 수집하고 검수하는 것이 우선입니다.

## 2. 설치

```bash
source .venv/bin/activate
python -m pip install -e .
```

Detector 파이프라인(권장 extras):

```bash
python -m pip install -e ".[detector]"
```

Data harvest 파이프라인(권장 extras):

```bash
python -m pip install -e ".[harvest]"
```

### 2.1 Data Harvest 요구 사양

`data-harvest`는 화면 캡처 + 입력 기록 + Gemini teacher 기반 자동 라벨링을 수행합니다.
현재 active path는 **routing-only**입니다. 기본 라벨 단위는 page-level sample이며 아래 2가지만 다룹니다.

- 현재 페이지가 어떤 `situation_id`인지
- 지금 호출해야 하는 `primitive_id`가 무엇인지

버튼 bbox, semantic 의미, grounding/detection용 상세 라벨은 legacy/future work로 남겨두고 기본 경로에서는 숨깁니다.

- 최소(기능 확인/스모크)
  - CPU: 4코어 이상
  - RAM: 8GB 이상
  - GPU: 없어도 가능 (자동 라벨링 속도 저하)
  - 저장공간: 20GB+ 권장 (세션 이미지 누적)
- 권장(실사용)
  - CPU: 8코어급
  - RAM: 16GB 이상
  - GPU: 8GB+ VRAM 또는 Apple Silicon(MPS)
  - 저장공간: SSD 50GB+
- 쾌적(반복 실험)
  - CPU: 8~12코어
  - RAM: 32GB
  - GPU: 12GB+ VRAM
  - 저장공간: SSD 100GB+

### 2.2 MacBook M2 Pro 권장 설정

M2 Pro 환경에서는 기본값으로도 동작하지만, 장시간 수집 시 아래 설정을 권장합니다.

- `configs/harvest.yaml`
  - `recorder.capture_fps: 5~10`
  - `recorder.enable_hover: false` (기본값 유지)
  - `labeler.vlm.device_map: "auto"` (기본값 유지)
  - `labeler.use_ocr: false` (기본값 유지)
  - `labeler.legacy_weak_signals: false` (기본값 유지, diff/OCR 약지도 off)
- 운영 팁
  - 장시간 `record` 시 충전기 연결
  - 저장공간 부족 방지를 위해 `runs/harvest_session_01/samples` 주기적 정리

CLI 엔트리포인트:
- `myvg`
- `vis-ground-lab`
- `data-harvest`

주요 `data-harvest` export:
- `router_full`: 전체 screenshot 기반 primitive routing dataset
- `router_roi`: 상황별 primary ROI crop 기반 primitive routing dataset
- `unified`: routing-only JSONL (`screen_type`, `situation_id`, `primitive_id`)

legacy export:
- `grounding`
- `yolo` / `coco`
- `roi_state`

### 2.3 Data Harvest Quickstart

현재 `data-harvest`의 active path는 **routing-only**입니다.
샘플마다 아래 3개를 모으고 검수합니다.

- `screen_type`: 큰 화면 종류 (`main_map`, `popup`, `tech_tree` 등)
- `situation_id`: 현재 화면 상태에 대한 coarse label
- `primitive_id`: 지금 호출해야 하는 primitive class

실행 전에 먼저 맞춰야 하는 것:

- macOS 권한
  - `System Settings -> Privacy & Security -> Screen Recording`
  - `System Settings -> Privacy & Security -> Accessibility`
- 게임 실행 상태
  - Civ6를 `fullscreen` 또는 `borderless fullscreen`으로 고정
  - 가능하면 UI scale도 고정
- Python 환경
  - `source .venv/bin/activate`
  - `python -m pip install -e ".[harvest]"`

실제로 준비해야 하는 파일은 2개입니다.

- `configs/harvest.yaml`
  - 실행 설정
  - 주로 `workdir`, `recorder.monitor_index`, `review.server_port`
- `configs/harvest_taxonomy/civ6.yaml`
  - Civ6 routing taxonomy
  - `primitives`, `situations`, `rois`가 source of truth

Gemini 키는 `.env` 또는 shell env에 넣습니다. `.env`를 쓰면 `data-harvest`가 자동으로 읽습니다.

```bash
export GEMINI_API_KEY=YOUR_KEY
```

기본 실행 순서:

```bash
data-harvest record -c configs/harvest.yaml
data-harvest label-auto -c configs/harvest.yaml
data-harvest review -c configs/harvest.yaml
data-harvest export -c configs/harvest.yaml --format all
```

의미:

- `record`
  - 화면과 입력 이벤트를 수집
  - 샘플은 `runs/.../samples/sample_xxxxxx/` 아래에 저장
- `label-auto`
  - Gemini teacher가 `screen_type`, `situation_id`, `primitive_id`, `roi_name` 자동 라벨링
  - 호출 전에 pHash dedup을 수행해, near-duplicate 클러스터는 대표 샘플만 Gemini 호출 후 나머지 샘플에 라벨을 복사
- `review`
  - 사람이 최종 수정
- `export --format all`
  - `router_full`
  - `router_roi`

### 2.4 Review UI 사용법

review 페이지는 routing 검수 전용입니다.

- 왼쪽 패널
  - 어떤 라벨을 수정할지 선택:
    - `screen_type`
    - `situation_id`
    - `primitive_id`
    - `router_roi`
- 오른쪽 패널
  - 왼쪽에서 선택한 라벨의 후보 목록
  - Gemini 후보가 먼저, taxonomy 전체 후보가 뒤에 표시
- 상단 이미지
  - 왼쪽: 전체 screenshot preview
  - 오른쪽: 현재 `router_roi` crop preview
- evidence 패널
  - `must-have`
  - `strong cues`
  - `hard negatives`
  - `conflict`
  - `open screen`
  - `reasoning`

ROI 수정:

- preview 위 ROI 박스가 오버레이로 보임
- 스크린샷을 2번 클릭해서 ROI 두 코너를 지정 가능
- `roi x1/y1/x2/y2` 숫자로 미세조정 가능

버튼:

- `Approve`
- `Save Edit + Next`
- `Reject`
- `Update Preview`

### 2.5 결과물 위치

기본 `workdir`가 `runs/harvest_session_01`이면:

- 원본 샘플: `runs/harvest_session_01/samples/sample_xxxxxx/`
- 자동 라벨: `label.json`
- 검수 결과: `review.json`
- export:
  - `runs/harvest_session_01/export/router_full/`
  - `runs/harvest_session_01/export/router_roi/`

## 3. Task 레이어

`configs/config.example.yaml` (기존 grounding):

```yaml
task:
  name: grounding
```

`configs/config.tool_button_detection.yaml` (신규 detector):

```yaml
task:
  name: tool_button_detection
model:
  backend: yolo_ultralytics
  name: yolov8n.pt
```

`task`가 없으면 기본값은 `grounding`이라 기존 config와 호환됩니다.

## 4. Tool-specific Button Model Quickstart

### 4.1 데이터 추출 (`extract`)

비디오에서 프레임 추출 + 중복 제거 + 샘플링:

```bash
vis-ground-lab extract \
  --video data/raw/excel_session.mp4 \
  --workdir runs/excel_file_btn \
  --fps 2 \
  --every-nth 2 \
  --dedup-threshold 8 \
  --sample-count 400
```

스크린샷 폴더 입력도 가능:

```bash
vis-ground-lab extract \
  --images data/raw/excel_screens \
  --workdir runs/excel_file_btn \
  --sample-count 400
```

출력:
- `runs/.../extract/raw`
- `runs/.../extract/dedup`
- `runs/.../extract/sample`
- `runs/.../extract/manifest.json`

### 4.2 후보 박스 자동생성 (`prelabel`)

Florence2 teacher로 후보 박스 생성:

```bash
vis-ground-lab prelabel \
  --image-dir runs/excel_file_btn/extract/sample \
  --out-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --backend florence2_teacher \
  --model-name microsoft/Florence-2-base \
  --class-name file_button
```

### 4.3 빠른 수동 라벨링 (`label`)

Gradio UI 실행 후 박스 JSON을 수정/삭제하고 클래스명 부여:

```bash
vis-ground-lab label \
  --image-dir runs/excel_file_btn/extract/sample \
  --candidate-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --out-coco runs/excel_file_btn/labels/gt.coco.json \
  --class-names file_button,save_button
```

### 4.4 최적화 + 자동 베스트 선택 (`optimize`)

Optuna 다중 trial + pruning + 가중 objective:

`score = 0.6*mAP50 + 0.4*click_success - latency_penalty`

```bash
vis-ground-lab optimize \
  --data-yaml data/tool_dataset/data.yaml \
  --val-coco runs/excel_file_btn/labels/val.coco.json \
  --image-dir data/tool_dataset/images/val \
  --workdir runs/excel_file_btn \
  --class-names file_button,save_button \
  --n-trials 20 \
  --tool-id excel \
  --tool-version 365
```

출력:
- `runs/.../leaderboard.json`
- 콘솔 leaderboard table
- `runs/.../best_package` (자동 export 결과)

### 4.5 수동 패키징 (`export`)

```bash
vis-ground-lab export \
  --weights runs/excel_file_btn/yolo_runs/trial_000/weights/best.pt \
  --outdir runs/excel_file_btn/package \
  --class-names file_button,save_button \
  --tool-id excel \
  --tool-version 365 \
  --dataset-dir data/tool_dataset/images/val
```

생성 파일:
- `model.pt`, `model.onnx`
- `label_map.json`
- `preprocessing.json`, `postprocessing.json`
- `metrics.json`, `latency.json`
- `tool_metadata.json` (`tool_id`, `tool_version`, `dataset_hash`)

### 4.6 패키지 실행 (`run`)

```bash
vis-ground-lab run \
  --package-dir runs/excel_file_btn/package \
  --image-path data/tool_dataset/images/val/example_001.png
```

## 5. 기존 Florence2 Grounding 경로 (유지)

학습:

```bash
myvg train --config configs/config.example.yaml
```

평가:

```bash
myvg evaluate \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --eval-jsonl data/eval.jsonl \
  --image-root data/images \
  --normalize-mode 0-1000
```

추론:

```bash
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --image-path data/images/eval_000.png \
  --prompt "click the File button"
```

## 6. 모듈 구조

- `src/vis_ground_lab/config/schema.py`
  - `TaskConfig` 추가 (`grounding`, `tool_button_detection`)
- `src/vis_ground_lab/data/`
  - `extract.py` (video->frames)
  - `dedup.py` (pHash dedup)
  - `coco.py` (COCO import/export)
- `src/vis_ground_lab/prelabel/`
  - `Prelabeler` 인터페이스
  - Florence2 teacher prelabel plugin
- `src/vis_ground_lab/labeling/app_gradio.py`
  - 라벨링 보조 UI
- `src/vis_ground_lab/models/yolo_ultralytics.py`
  - 소형 detector backend wrapper
- `src/vis_ground_lab/optimization/optuna_runner.py`
  - HPO + pruning + leaderboard + best export
- `src/vis_ground_lab/export/packager.py`
  - self-contained package 생성

## 7. 다른 detector 백엔드 추가 방법

1. `src/vis_ground_lab/models/<new_backend>.py`에 wrapper 구현
2. 필수 API 제공:
   - `train(dataset, cfg, workdir)`
   - `predict(image)`
   - `export(outdir, formats)`
   - `benchmark_latency(images)`
3. `src/vis_ground_lab/models/factory.py`에 backend 등록

## 8. 운영 권장 전략

- 초반: 작은 데이터(100~300장), `n_trials` 작게(5~10)
- 중반: 실패 케이스만 모아 라벨 보강
- 후반: latency budget 기준으로 objective 조정
- 배포 전: `run`으로 held-out 화면에서 실제 버튼 성공률 확인

## 9. Work 관리

- 로그: `docs/WORK_LOG.md`
- 보드: `docs/WORKBOARD.md`
