# Gate B synthetic teacher harness v2

기준 시각: **2026-08-11 KST**

이 문서는 teacher v5 전에 실행하는 contest-independent live boundary test의 고정 계약이다.
v1의 64행·2×32 schema와 artifact는 역사 호환용으로 그대로 유지하며 재작성하지 않는다.

## 고정 계약

- config: `configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v2.json`
- config semantic SHA-256:
  `70f300a4e44b2426593ff873b87736ecaa2fe9eb2dff99ccbbb27623f27618a5`
- config file SHA-256:
  `de1ac317f8f9df581a93951cc8ad42cb8113b12ca9ea3d39b98473cdd77a9029`
- fixture schema: `gate-b-codex-teacher-harness-fixture-v2`
- fixture SHA-256:
  `dea9b4cc3c3262de831abba2c7ce36bf6ac2612ee215cbccd34c5d4b3d1a3388`
- report schemas: `gate-b-codex-teacher-harness-replay-v2`,
  `gate-b-codex-teacher-harness-live-v2`
- promotion schema: `gate-b-codex-teacher-harness-authorization-v2`
- 128 fixed signed-integer rows, exactly 8 chunks of 16, worker 1, high effort
- exactly 8 live invocations, max attempts 1, retry 0, repair 0, bank output 0

Offline replay covers process/timeout/nonzero failures, malformed event and agent JSON, unsafe/tool
events, bad usage, 15/16/17 cardinality, duplicate plus omission, unknown/missing/reordered IDs,
invalid target policy, and oversize output. Classification remains raw-free and fail-closed.

## 실행 순서

Committed clean source에서 새 unique path와 source manifest를 만든 뒤 다음 순서만 허용한다.
`$TEACHER_CONFIG`는 v5 구현 후 그 allowlisted config를 사용한다.

```bash
uv run deep-challenge gate-b-teacher-harness-replay \
  --harness-config configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v2.json \
  --teacher-config "$TEACHER_CONFIG" \
  --output "$REPLAY_REPORT"

uv run deep-challenge gate-b-teacher-harness-live \
  --harness-config configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v2.json \
  --teacher-config "$TEACHER_CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --plan-dir "$LIVE_PLAN" \
  --report "$LIVE_REPORT" \
  --acknowledge-synthetic-codex-canary
```

Exit code는 qualified `0`, 정상 실행됐지만 profile failure `1`, 계약/입력 오류 `2`다.
두 report가 qualified이고 동일 config/prompt/source/Codex binary·version에 결속될 때만 v2
authorization을 만들 수 있다. live plan은 assessment나 bank를 포함해서는 안 된다.

v2 live canary는 PR 구현 단계에서는 실행하지 않는다. PR #7의 v5 config와 committed clean
source가 모두 착륙한 뒤에만 실제 8회 호출을 수행한다.
