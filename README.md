# vis-ground-lab (`myvg`, `vis-ground-lab`)

로컬에서 UI 비전 모델을 빠르게 만들고 운영하기 위한 실험 프레임워크입니다.

- Task 1: `grounding` (기존 Florence2 + LoRA)
- Task 2: `tool_button_detection` (신규, 툴/게임 전용 소형 detector)

## 1. 핵심 아이디어

`tool_button_detection`은 일반화가 목적이 아닙니다.
하나의 툴/게임(Excel, PowerPoint, Civ, StarCraft 등) UI에서 특정 버튼/아이콘을 빠르게 탐지하는 소형 모델을 만드는 파이프라인입니다.

전체 흐름:
`extract -> prelabel -> label -> optimize -> export -> run`

## 2. 설치

```bash
source .venv/bin/activate
python -m pip install -e .
```

Detector 파이프라인(권장 extras):

```bash
python -m pip install -e ".[detector]"
```

CLI 엔트리포인트:
- `myvg`
- `vis-ground-lab`

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
