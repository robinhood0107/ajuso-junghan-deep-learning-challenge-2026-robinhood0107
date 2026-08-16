# TODOs

## Gate B hybrid completion gate

- [x] `teacher-pilot-v4`는 committed source `68ff7d4`에서 qualified offline replay와 explicit
  2×32 synthetic canary, immutable authorization을 통과한 뒤 동일 128행 initial 4×32·worker 1을
  실행했다. parsed 3/4, local 승인 79/128로 103/128 gate에 미달해 terminal marker를 썼고
  source JSONL/manifest, repair, audit, receipt는 만들지 않았다. raw-free evidence는
  `docs/15_GATE_B_TEACHER_V4_FAILURE_RECORD.md`에만 기록한다.
- [ ] fixed Qwen base folds 0--4와 complete development OOF를 같은 committed source/B0/
  split/config provenance에서 생성하고 `gate-b-base-development-oof-v2`로 qualify한다.
- [ ] contest raw question/answer/rationale를 쓰지 않는 synthetic harness v2를 128행 8×16,
  retry/repair 0으로 qualify한다. v1 harness와 v1--v4 teacher artifact bytes는 유지한다.
- [ ] harness v2 authorization 뒤에만 v4 prompt bytes를 유지한 teacher v5 8×16 pilot을
  실행한다. terminal failure면 v6 없이 base 경로로 전환한다.
- [ ] development OOF만으로 정확히 한 방법을 freeze하고, 별도 사용자 확인 뒤 one-shot
  holdout 1회, evaluation prediction, submission dual validation과 SHA 기록까지 완료한다.
- [ ] Kaggle upload는 final submission path/SHA를 보고한 뒤 사용자의 별도 명시 요청을
  기다린다.
