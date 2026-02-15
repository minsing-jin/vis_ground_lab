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
