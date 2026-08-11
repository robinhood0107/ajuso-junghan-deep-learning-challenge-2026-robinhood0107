# TODOs

## Gate B teacher v4 organizer-pilot gate

- [ ] `teacher-pilot-v3`는 동일 128행 initial에서 52/128 승인으로 103/128 gate를
  통과하지 못해 repair 없이 fail-closed됐다. v4 config는 policy-bound synthetic candidate로만
  committed source에 존재할 수 있다. **committed clean source의 v4-qualified offline replay,
  explicit 2×32 live canary, immutable authorization을 모두 통과하기 전에는 v4 organizer-data
  plan/run을 시작하지 않는다.**
- [ ] harness는 contest raw question/answer/rationale를 공개 fixture로 복사하지 않고,
  fixed synthetic prompt/version/config/transport failure를 분리 평가하며 v1/v2/v3 역사
  artifact의 read/verify 호환성을 유지한다. v4 canary 또는 pilot이 실패하면 raw-free
  count/hash/category만 보존하고, **versioned synthetic live-eval harness의 새 설계·승인 전에는
  v5 config를 allowlist에 추가하지 않는다.**
