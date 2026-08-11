# TODOs

## Gate B teacher v4 allowlist gate

- [ ] `teacher-pilot-v3`는 동일 128행 initial에서 52/128 승인으로 103/128 gate를
  통과하지 못해 repair 없이 fail-closed됐다. **versioned synthetic live-eval harness
  설계·승인 전 v4 config를 allowlist에 추가하지 않는다.**
- [ ] harness 제안은 contest raw question/answer/rationale를 공개 fixture로 복사하지 않고,
  synthetic prompt/version/config/transport failure를 분리 평가하며 v1/v2/v3 역사 artifact의
  read/verify 호환성을 유지해야 한다.
