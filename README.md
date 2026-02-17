# vis-ground-lab (`myvg`)

UI 자동화용 Visual Grounding 모델을 로컬에서 학습/평가/배포하기 위한 프레임워크입니다.

- Local-first: 소비자 GPU/로컬 머신 중심
- PEFT-first: LoRA 어댑터 학습
- End-to-end: 데이터 -> 학습 -> 평가 -> HF Hub 배포
- Pip-installable + CLI

## 1. What This Project Solves

예시 태스크:
- "click the File button"
- "click Settings icon"
- "select the username field"

이 프로젝트는 위 자연어 지시와 UI 스크린샷을 입력으로 받아, 클릭할 bbox를 예측하는 모델을 학습/운영하는 흐름을 제공합니다.

## 2. Architecture

핵심 모듈:
- `src/vis_ground_lab/base.py`
  - `BaseVGModel`, `BaseDataset`
- `src/vis_ground_lab/data_manager.py`
  - JSONL 기반 데이터셋 로더
  - bbox 정규화(`none`, `0-1`, `0-1000`)
- `src/vis_ground_lab/models/florence2.py`
  - Florence-2 래퍼
  - LoRA 주입/Hub 업로드/Hub 재로딩
- `src/vis_ground_lab/models/factory.py`
  - `model.backend` 기준 모델 생성 팩토리
- `src/vis_ground_lab/training/trainer_engine.py`
  - HF Trainer 래핑
- `src/vis_ground_lab/evaluation/evaluator.py`
  - `IoU`, `Center Pixel Distance`
- `src/vis_ground_lab/cli.py`
  - `myvg train|evaluate|infer|push`

## 3. Environment (중요)

실사용 가상환경은 `.venv` 하나만 사용하세요.

```bash
source .venv/bin/activate
```

권장:
- Python 3.11

설치:

```bash
python -m pip install -e .
```

네트워크/빌드 제약 환경:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

## 4. Data Manual

### 4.1 JSONL 포맷

한 줄에 샘플 1개:

```json
{"image_path": "images/ui_001.png", "prompt": "click the File button", "bbox": [12, 18, 98, 46]}
```

필드:
- `image_path` (str): 이미지 경로
- `prompt` (str): 클릭 지시 텍스트
- `bbox` (list[4]): `[x1, y1, x2, y2]` (원본 픽셀 좌표)

### 4.2 데이터 추가 절차

1. 이미지 준비
- 예: `data/images/*.png`

2. `train.jsonl`, `eval.jsonl` 작성
- 예: `data/train.jsonl`, `data/eval.jsonl`

3. `config`에서 경로 연결
- `data.train_jsonl`, `data.eval_jsonl`, `data.image_root`

### 4.3 정규화 모드

- `none`: 원본 픽셀
- `0-1`: 비율 정규화
- `0-1000`: Florence 계열 권장

## 5. Model Choice Manual

### 5.1 현재 지원 모델

- `backend: florence2`

### 5.2 설정에서 모델 선택

`configs/config.example.yaml`:

```yaml
model:
  backend: florence2
  name: microsoft/Florence-2-base
```

### 5.3 기존 adapter로 이어 학습

```yaml
model:
  adapter_path_or_repo: checkpoints
```

또는 HF repo 경로:

```yaml
model:
  adapter_path_or_repo: <hf_user>/<adapter_repo>
```

### 5.4 다른 모델 추가 방법 (확장)

1. `src/vis_ground_lab/models/<new_model>.py`에 `BaseVGModel` 구현
2. `src/vis_ground_lab/models/factory.py`에 `backend` 분기 등록
3. config에서 `model.backend: <new_backend>` 사용

## 6. Training Manual

### 6.1 기본 학습

```bash
myvg train --config configs/config.example.yaml
```

### 6.2 훈련 전략 권장 순서

1. Stage A (smoke)
- `epochs=1`, `batch_size=1`, 작은 데이터로 파이프라인 검증

2. Stage B (stabilize)
- `lora_r`, `learning_rate`, `train_image_seq_length` 튜닝

3. Stage C (scale)
- 데이터 확대 + 평가 반복

### 6.3 핵심 파라미터 튜닝 가이드

- `trainer.learning_rate`
  - 시작: `5e-5`
  - 발산 시: `1e-5`~`3e-5`
- `model.lora_r`
  - 시작: `16`
  - 메모리 타이트: `4`~`8`
- `model.train_image_seq_length`
  - 길이 초과/불안정 시 감소 (`256` -> `192`)
- `trainer.batch_size`
  - OOM/속도 기준 조절

## 7. Evaluation + Feedback Loop Manual

### 7.1 평가 실행

```bash
myvg evaluate \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --eval-jsonl data/eval.jsonl \
  --image-root data/images \
  --normalize-mode 0-1000
```

출력:
- `mean_iou`
- `mean_distance_px`

### 7.2 피드백 루프

1. evaluate 실행
2. 오답 샘플 수집(특히 distance 큰 케이스)
3. 데이터 보강
  - hard negative
  - UI theme/해상도 다양화
4. 재학습
5. 지표 비교

## 8. Hugging Face Manual

### 8.1 어댑터 업로드

```bash
myvg push \
  --base-model microsoft/Florence-2-base \
  --adapter-path checkpoints \
  --repo-name <hf_user>/<adapter_repo> \
  --token <hf_token>
```

### 8.2 Hub adapter 즉시 추론

```bash
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo <hf_user>/<adapter_repo> \
  --image-path data/images/eval_000.png \
  --prompt "click the File button"
```

## 9. PyPI/Package 사용자 빠른 사용법

패키지 설치 후, Hub에 올린 내 모델(adapter)을 바로 불러 쓰는 최소 코드:

```python
from PIL import Image
from vis_ground_lab.models.florence2 import Florence2Wrapper

model = Florence2Wrapper.from_pretrained_adapter(
    base_model_name="microsoft/Florence-2-base",
    adapter_path_or_repo="<hf_user>/<adapter_repo>",
)
image = Image.open("ui.png").convert("RGB")
pred = model.predict(image=image, text="click the File button")
print(pred)
```

CLI로 더 빠르게:

```bash
myvg infer --base-model microsoft/Florence-2-base --adapter-repo <hf_user>/<adapter_repo> --image-path ui.png --prompt "click the File button"
```

## 10. Config Reference

샘플 파일:
- `configs/config.example.yaml`
- `configs/config.cpu_smoke.yaml`

주요 필드:
- `model.backend`: 모델 백엔드 선택
- `model.name`: base model id
- `model.adapter_path_or_repo`: 이어학습용 adapter
- `model.train_image_size`: 학습 이미지 리사이즈 크기
- `model.train_image_seq_length`: 이미지 토큰 시퀀스 길이 제어
- `trainer.learning_rate`, `trainer.batch_size`, `trainer.epochs`
- `data.normalize_mode`: `none` / `0-1` / `0-1000`

## 11. Examples

더미 데이터 생성:

```bash
python3 examples/prepare_dummy_data.py
```

전체 데모:

```bash
bash examples/run_cli_demo.sh
```

## 12. Work Management

- 작업 이력: `docs/WORK_LOG.md`
- 지속 관리 보드: `docs/WORKBOARD.md`

## 13. Tests

```bash
python -m pytest -q
```

## 14. Notes

- Florence-2 remote code 특성상 최초 1회 다운로드가 발생할 수 있습니다.
- 오프라인 환경은 사전 캐시 준비가 필요합니다.
- 학습 결과는 기본적으로 `checkpoints/`에 저장됩니다.
