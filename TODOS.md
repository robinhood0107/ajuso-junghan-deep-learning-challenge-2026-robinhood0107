# TODOs

## Gate B teacher v4 terminal stop gate

- [x] `teacher-pilot-v4`는 committed source `68ff7d4`에서 qualified offline replay와 explicit
  2×32 synthetic canary, immutable authorization을 통과한 뒤 동일 128행 initial 4×32·worker 1을
  실행했다. parsed 3/4, local 승인 79/128로 103/128 gate에 미달해 terminal marker를 썼고
  source JSONL/manifest, repair, audit, receipt는 만들지 않았다. raw-free evidence는
  `docs/15_GATE_B_TEACHER_V4_FAILURE_RECORD.md`에만 기록한다.
- [ ] harness는 contest raw question/answer/rationale를 공개 fixture로 복사하지 않고,
  fixed synthetic prompt/version/config/transport failure를 분리 평가하며 v1/v2/v3 역사
  artifact의 read/verify 호환성을 유지한다. **v4 pilot은 terminal failure이므로 raw-free
  count/hash/category만 보존하고, versioned synthetic live-eval harness의 새 설계·명시 승인
  전에는 v5 config를 allowlist에 추가하거나 어떤 teacher ledger도 재개하지 않는다.**
