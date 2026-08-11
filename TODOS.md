# TODOs

## Gate B teacher v4 allowlist gate

- [ ] `teacher-pilot-v3`는 동일 128행 initial에서 52/128 승인으로 103/128 gate를
  통과하지 못해 repair 없이 fail-closed됐다. **harness v1 implementation이 committed clean
  source의 qualified offline replay, explicit 2×32 live canary, immutable authorization을 모두
  통과하기 전 v4 config를 allowlist에 추가하지 않는다.**
- [ ] harness는 contest raw question/answer/rationale를 공개 fixture로 복사하지 않고,
  fixed synthetic prompt/version/config/transport failure를 분리 평가하며 v1/v2/v3 역사
  artifact의 read/verify 호환성을 유지한다. failed report가 나오면 harness version 설계·승인
  전에는 v4/v5 allowlist를 추가하지 않는다.
