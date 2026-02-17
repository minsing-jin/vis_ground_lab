# Work Log

이 파일은 작업 이력을 날짜별로 기록합니다.

## 2026-02-15

### Summary
- 프로젝트 기본 구조(`src/vis_ground_lab`) 초기화
- 추상 인터페이스 추가: `BaseVGModel`, `BaseDataset`
- JSONL 기반 데이터셋/정규화 기능 구현
- Florence-2 래퍼 + LoRA(PEFT) 주입 구현
- TrainerEngine(HF Trainer 래핑) 구현
- Evaluator(IoU, Center Pixel Distance) 구현
- Typer CLI 구현: `train`, `evaluate`, `infer`, `push`
- HF 업로드/재로딩 경로 구현
- `pyproject.toml`, `setup.py` 패키징 구성
- 예제 스크립트 및 설정 파일 추가
- 테스트 코드 추가(`tests/*`)
- `.venv` 단일 환경으로 통합 및 학습 실행 검증

### Technical Notes
- Florence-2 remote code와 라이브러리 버전 이슈 대응
  - processor/tokenizer fallback 처리
  - attention implementation을 `eager`로 고정
  - 이미지 토큰 동기화 및 토큰 임베딩 resize 처리
- CPU/가속기 미사용 환경에서 dtype 충돌 대응
  - 모델 dtype 자동 조정
  - collate 단계 텐서 dtype 정렬
- 학습 시 시퀀스 길이/포지션 인덱스 문제 대응
  - processor 학습용 image size/image sequence length 축소
  - collate에서 text 입력 길이 상한 처리

### Validation
- `myvg train --config configs/config.example.yaml` 실행 완료
- train log 예시
  - `train_runtime`: 약 30s
  - `train_loss`: 약 14.1

### Next
- 모델별 Strategy 분리(`models/strategies`, `data/strategies`) 정식화
- `myvg evaluate` 결과 리포팅(JSON/Markdown) 강화
- CI(테스트/린트) 자동화 추가
- HF Hub push 후 pull/infer e2e 검증 스크립트 추가

## 2026-02-17

### Summary
- README를 실사용 매뉴얼 중심으로 대폭 확장
  - 데이터 추가/구성
  - 모델 선택/교체
  - 학습 실행/전략
  - 평가 및 피드백 루프
  - 하이퍼파라미터 튜닝 가이드
  - Hugging Face 업로드/재사용
- 모델 생성 경로를 팩토리 패턴으로 정리
  - `src/vis_ground_lab/models/factory.py` 추가
  - CLI `train`이 하드코딩 대신 config 기반 백엔드 선택 사용
- config schema 확장
  - `backend`, `adapter_path_or_repo`, `cache_dir`
  - 학습용 이미지/시퀀스 관련 파라미터 추가
- 패키지 export 경로 정리
  - `create_model_wrapper` 외부 사용 가능하도록 `__init__` 업데이트

### Validation
- `python3 -m compileall src` 통과
- `myvg --help` 실행 정상 확인
- 변경사항 커밋/원격 반영
  - `1c733b6 docs+feat: add end-to-end usage manual and model factory`

### Next
- 백엔드 추가 예시(`kosmos2`, `qwen-vl` 등)용 wrapper 템플릿 제공
- 평가 결과 저장(JSON/Markdown) 옵션을 CLI에 노출
- 튜닝 레시피(소규모/중규모/대규모 데이터셋) 표준 실험표 추가
