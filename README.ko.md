# vis-ground-lab (`myvg`, `vis-ground-lab`)

[English](README.md) | [한국어](README.ko.md)

`vis-ground-lab`은 로컬 환경에서 작은 규모의 UI 비전 모델을 빠르게 만들고 실험하기 위한 프레임워크입니다.

현재 저장소는 크게 세 갈래로 사용합니다.

- `data-harvest`: 게임 UI용 routing-first 데이터 수집, 자동 라벨링, 검수, export
- `router_classification`: 수집한 데이터로 소형 vision router 학습
- `grounding` / `tool_button_detection`: 호환성 유지를 위한 기존 경로 또는 보조 경로

## 개요

현재 active focus는 일반적인 grounding이 아닙니다.
핵심 목표는 작은 vision router 학습에 필요한 안정적인 데이터 생산입니다.

주요 타깃 라벨:

- `screen_type`
- `situation_id`
- `primitive_id`

현재 Civ6 기준으로는 dense detection이나 일반 목적 bbox 예측보다, 페이지 단위 routing 데이터의 품질을 높이는 것이 우선입니다.

전체 흐름:

```text
record -> label-auto -> review -> export -> train/evaluate
```

## 설치

기본 설치:

```bash
source .venv/bin/activate
python -m pip install -e .
```

워크플로별 권장 extras:

```bash
python -m pip install -e ".[harvest]"
python -m pip install -e ".[detector]"
```

CLI 엔트리포인트:

- `myvg`
- `vis-ground-lab`
- `data-harvest`
- `ralph`

## Routing-First Data Harvest

`data-harvest`는 화면 캡처와 입력 이벤트를 수집하고, 자동 라벨링과 검수 UI를 거쳐 routing 학습 데이터를 export합니다.

현재 기본 모드는 `routing_only`입니다.
주요 라벨은 아래 세 가지입니다.

- `screen_type`: `main_map`, `popup`, `tech_tree` 같은 큰 화면 종류
- `situation_id`: 현재 UI/게임 상황
- `primitive_id`: 다음에 호출해야 하는 action class

Grounding, COCO, YOLO, ROI-state 같은 legacy export도 남아 있지만, 기본 경로는 아닙니다.

### 하드웨어 가이드

최소:

- CPU: 4코어
- RAM: 8GB
- GPU: 없어도 가능
- 저장공간: 20GB+

권장:

- CPU: 8코어급
- RAM: 16GB 이상
- GPU: 8GB+ VRAM 또는 Apple Silicon MPS
- 저장공간: SSD 50GB+

반복 실험용으로 여유 있게:

- CPU: 8~12코어
- RAM: 32GB
- GPU: 12GB+ VRAM
- 저장공간: SSD 100GB+

### M2 Pro 권장 설정

`configs/harvest.yaml` 기준:

- `recorder.capture_fps: 5~10`
- `recorder.enable_hover: false`
- `labeler.vlm.device_map: "auto"`
- `labeler.use_ocr: false`
- `labeler.legacy_weak_signals: false`

운영 팁:

- 장시간 녹화 시 전원 연결 권장
- 디스크 사용량이 커지면 `runs/harvest_session_01/samples`를 주기적으로 정리

### 사전 준비

녹화 전에 아래를 맞춰야 합니다.

- macOS `Screen Recording` 권한 활성화
- macOS `Accessibility` 권한 활성화
- 가능하면 Civ6를 fullscreen 또는 borderless fullscreen으로 고정
- 일관된 crop을 원하면 UI scale도 고정

주요 설정 파일:

- `configs/harvest.yaml`: 실행 설정
- `configs/harvest_taxonomy/civ6.yaml`: Civ6 routing taxonomy source of truth

Gemini API 키:

```bash
export GEMINI_API_KEY=YOUR_KEY
```

`.env`에 넣어도 되며, 도구가 자동으로 읽습니다.

### 빠른 시작

```bash
data-harvest record -c configs/harvest.yaml
data-harvest label-auto -c configs/harvest.yaml
data-harvest review -c configs/harvest.yaml
data-harvest export -c configs/harvest.yaml --format all
```

각 단계 설명:

- `record`: 화면 프레임과 입력 기반 샘플 수집
- `label-auto`: Gemini 우선 자동 라벨링, 실패 시 로컬 fallback
- `review`: 사람이 최종 수정하는 Gradio UI 실행
- `export`: `router_full`, `router_roi` 등 routing 데이터셋 생성

자주 쓰는 보조 명령:

```bash
data-harvest relabel -c configs/harvest.yaml
data-harvest filter -c configs/harvest.yaml
data-harvest stats -c configs/harvest.yaml
data-harvest profiles
```

### Review UI

검수 UI는 routing 전용으로 설계되어 있습니다.

수정 가능한 항목:

- `screen_type`
- `situation_id`
- `primitive_id`
- `router_roi`

동작 방식:

- Gemini 후보가 먼저 보입니다
- taxonomy 기반 전체 후보도 함께 제공합니다
- 전체 스크린샷과 현재 ROI crop을 같이 보여줍니다
- ROI는 두 번 클릭하거나 좌표 숫자 입력으로 수정할 수 있습니다

주요 버튼:

- `Approve`
- `Save Edit + Next`
- `Reject`
- `Update Preview`

### 출력 구조

`workdir`가 `runs/harvest_session_01`이라면 일반적인 결과물은 아래와 같습니다.

- 원본 샘플: `runs/harvest_session_01/samples/sample_xxxxxx/`
- 자동 라벨: `label.json`
- 검수 결과: `review.json`
- routing export:
  - `runs/harvest_session_01/export/router_full/`
  - `runs/harvest_session_01/export/router_roi/`

주요 export 형식:

- `router_full`: 전체 스크린샷 기반 routing 데이터셋
- `router_roi`: 주요 ROI crop 기반 routing 데이터셋
- `unified`: `screen_type`, `situation_id`, `primitive_id`를 담은 routing JSONL

legacy export:

- `grounding`
- `coco`
- `yolo`
- `roi_state`

## Router 학습

예시 설정:

- `configs/config.router_classification.yaml`

학습:

```bash
myvg train --config configs/config.router_classification.yaml
```

평가:

```bash
myvg evaluate --config configs/config.router_classification.yaml
```

현재 router 경로는 아래를 전제로 합니다.

- `task.name: router_classification`
- `model.backend: timm_router`
- CSV 기반 train/validation split

## Tool-Specific Button Detector

이 경로는 하나의 툴 또는 게임 UI에 특화된 소형 detector가 필요할 때 유효합니다.
여러 앱에 일반화하는 것이 목적은 아닙니다.

Task 설정:

```yaml
task:
  name: tool_button_detection
model:
  backend: yolo_ultralytics
  name: yolov8n.pt
```

빠른 시작:

```bash
vis-ground-lab extract \
  --video data/raw/excel_session.mp4 \
  --workdir runs/excel_file_btn \
  --fps 2 \
  --every-nth 2 \
  --dedup-threshold 8 \
  --sample-count 400

vis-ground-lab prelabel \
  --image-dir runs/excel_file_btn/extract/sample \
  --out-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --backend florence2_teacher \
  --model-name microsoft/Florence-2-base \
  --class-name file_button

vis-ground-lab label \
  --image-dir runs/excel_file_btn/extract/sample \
  --candidate-coco runs/excel_file_btn/prelabel/candidates.coco.json \
  --out-coco runs/excel_file_btn/labels/gt.coco.json \
  --class-names file_button,save_button

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

패키징과 실행:

```bash
vis-ground-lab export \
  --weights runs/excel_file_btn/yolo_runs/trial_000/weights/best.pt \
  --outdir runs/excel_file_btn/package \
  --class-names file_button,save_button \
  --tool-id excel \
  --tool-version 365 \
  --dataset-dir data/tool_dataset/images/val

vis-ground-lab run \
  --package-dir runs/excel_file_btn/package \
  --image-path data/tool_dataset/images/val/example_001.png
```

## Florence-2 Grounding 경로

기존 grounding 경로도 계속 지원합니다.

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

## Ralph Self-Improvement Loop

`ralph`는 harvest 라벨 품질과 fusion weight를 개선하기 위한 RLAIF 스타일 루프입니다.

주요 설정:

- `configs/ralph.yaml`

주요 명령:

```bash
ralph run -c configs/ralph.yaml
ralph judge -c configs/ralph.yaml
ralph tune-weights -c configs/ralph.yaml
ralph report -c configs/ralph.yaml
ralph apply-weights -c configs/ralph.yaml
```

## 프로젝트 구조

- `src/data_harvest/`: routing-first 캡처, 자동 라벨링, 검수, export
- `src/vis_ground_lab/`: 학습, 평가, 패키징, 모델 백엔드
- `src/ralph_self_improvement/`: judgment, scoring, self-improvement loop
- `configs/`: 학습 및 harvest 설정 파일
- `experiments/`: 목적별 실험 프로그램
- `third_party/autoresearch/`: vendored experiment support metadata

## 개발 메모

권장 작업 순서:

- 작은 데이터셋으로 먼저 시작
- 실패 케이스를 초기에 검수
- 데이터 규모를 키우기 전에 router를 먼저 학습
- 큰 반복 전에 `myvg evaluate` 또는 `data-harvest stats`로 상태 확인

프로젝트 추적 문서:

- `docs/WORK_LOG.md`
- `docs/WORKBOARD.md`
