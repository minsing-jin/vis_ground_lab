# myvg

`myvg`는 UI 자동화용 Visual Grounding 모델을 로컬 환경에서 학습/평가/배포하기 위한 Python 프레임워크입니다.

목표:
- Local-first: 소비자 GPU 환경에서 LoRA 기반 미세조정
- Modular: 모델 래퍼 교체로 백본 확장 가능
- End-to-end: 데이터 준비 -> 학습 -> 평가 -> HF Hub 배포
- Pip-installable: 패키지/CLI로 즉시 사용

## Key Features

- 모델 래퍼 구조 (`BaseVGModel`) + 데이터 추상화 (`BaseDataset`)
- JSONL UI grounding 데이터 로더
- bbox 정규화 지원: `none`, `0-1`, `0-1000`
- HF Trainer 기반 `TrainerEngine`
- UI 클릭 성능 지표
  - IoU
  - Center Pixel Distance
- LoRA adapter 학습/저장/업로드/재로딩 지원

## Project Architecture

핵심 구성:

- `src/vis_ground_lab/base.py`
  - `BaseVGModel`, `BaseDataset`, `BoundingBox`, `VGSample`
- `src/vis_ground_lab/models/florence2.py`
  - `Florence2Wrapper`
  - LoRA 주입(`q_proj`, `v_proj`) 및 HF Hub push
- `src/vis_ground_lab/data_manager.py`
  - `JSONLVisualGroundingDataset`
- `src/vis_ground_lab/training/trainer_engine.py`
  - `TrainerEngine` (HF `Trainer` 래핑)
- `src/vis_ground_lab/evaluation/evaluator.py`
  - IoU / Center Distance 계산
- `src/vis_ground_lab/cli.py`
  - `myvg train|evaluate|infer|push`

실행 흐름:
1. `config.yaml` 로드
2. 모델/프로세서 로드 + LoRA 주입
3. JSONL 데이터셋 구성
4. Trainer 학습
5. 체크포인트/어댑터 저장
6. 평가 또는 허깅페이스 업로드

## Data Format

JSONL 한 줄 예시:

```json
{"image_path": "images/ui_001.png", "prompt": "click the File button", "bbox": [12, 18, 98, 46]}
```

필드:
- `image_path`: 이미지 경로 (`image_root` 기준 상대경로 가능)
- `prompt`: grounding 텍스트
- `bbox`: `[x1, y1, x2, y2]`

## Environment

실사용 가상환경은 `.venv` 하나만 사용하세요.

```bash
source .venv/bin/activate
```

권장 Python:
- Python 3.11

## Installation

```bash
python -m pip install -e .
```

네트워크/빌드 제약 환경에서는:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

## Configuration

샘플 설정:
- `configs/config.example.yaml`
- `configs/config.cpu_smoke.yaml`

기본 학습 설정 예시 (`configs/config.example.yaml`):

```yaml
model:
  name: microsoft/Florence-2-base
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

trainer:
  learning_rate: 5.0e-5
  batch_size: 1
  epochs: 1
  checkpoint_dir: checkpoints

data:
  train_jsonl: data/train.jsonl
  eval_jsonl: data/eval.jsonl
  image_root: data/images
  normalize_mode: 0-1000
```

## CLI Usage

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

단일 추론:

```bash
myvg infer \
  --base-model microsoft/Florence-2-base \
  --adapter-repo checkpoints \
  --image-path data/images/eval_000.png \
  --prompt "click the File button"
```

허깅페이스 업로드:

```bash
myvg push \
  --base-model microsoft/Florence-2-base \
  --adapter-path checkpoints \
  --repo-name <hf_user>/<adapter_repo> \
  --token <hf_token>
```

## Python API Usage

```python
from PIL import Image
from vis_ground_lab.models.florence2 import Florence2Wrapper

# 1) base + local/HF adapter 로딩
model = Florence2Wrapper.from_pretrained_adapter(
    base_model_name="microsoft/Florence-2-base",
    adapter_path_or_repo="<hf_user>/<adapter_repo>"
)

# 2) 예측
image = Image.open("data/images/eval_000.png").convert("RGB")
bbox = model.predict(image=image, text="click the File button")
print(bbox)
```

## Examples

더미 데이터 생성 + end-to-end 실행:

```bash
python3 examples/prepare_dummy_data.py
bash examples/run_cli_demo.sh
```


## Work Management

- 작업 이력: `docs/WORK_LOG.md`
- 지속 관리 보드: `docs/WORKBOARD.md`

## Tests

```bash
python -m pytest -q
```

(환경에 따라 `transformers/peft/pytest` 미설치 시 일부 테스트는 skip 또는 실패할 수 있습니다.)

## Notes

- Florence-2 remote code를 사용하므로 최초 1회 모델/코드 다운로드가 발생할 수 있습니다.
- 오프라인 환경에서는 사전에 모델 파일을 로컬 캐시에 준비해야 합니다.
- 학습 산출물은 기본적으로 `checkpoints/`에 저장됩니다.
