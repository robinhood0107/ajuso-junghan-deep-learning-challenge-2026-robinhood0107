# Gate B teacher v5 runbook

기준 시각: **2026-08-12 KST**

Teacher v5는 v4의 prompt/template/policy bytes를 바꾸지 않고 실행 신뢰성만 비교하는 마지막
teacher profile이다. v5가 terminal failure면 v6를 만들지 않고 fixed-base 경로로 전환한다.

## 고정 config

- config: `configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v5.json`
- semantic SHA-256:
  `5ef82e71418c872b56f47e4f94124a1c25041e853b9267710082e93369839f88`
- file SHA-256:
  `43ee741ab417b191bb5de31c09c48cf93d693cff3021d5e7a6025ae0487514e5`
- label/version: `codex-gpt-5.6-sol-teacher-pilot-v5` / `pilot-v5`
- prompt version/template SHA: v4와 동일한 `gate-b-codex-teacher-prompt-v4` /
  `3029e9297bdda504e0f48e1ce4d57e363e5d3a5342edf18253b11c4f75ecd8a7`
- pilot 128, initial/repair chunk 16, max attempts 3
- initial과 protocol retry high, semantic repair xhigh
- pilot worker 1, full bank만 최대 2

## 실행 전 gate

PR #5--#7이 모두 병합된 committed clean `main`에서 새 run context와 source manifest를 만든다.
먼저 fixed-base complete OOF를 qualify해야 한다. 이어 harness v2 offline replay와 실제 8×16
live canary가 모두 qualified여야 하며, v5 plan은 그 v2 authorization sidecar를 plan/run/status/
finalize마다 재검증한다.

실제 pilot initial은 다음 계약을 고정한다.

```bash
uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$PILOT_PLAN" \
  --teacher-config configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v5.json \
  --acknowledge-codex-teacher \
  --max-invocations 8 \
  --max-workers 1 \
  $HARNESS_V2_EVIDENCE_ARGS
```

구조/프로토콜 분류 실패 chunk만 같은 ID, 같은 prompt, high로 정확히 한 번 다시 보낸다.
retry wave는 실패 chunk 전체를 canonical order로 한 번에 포함해야 한다. nonzero, unsafe/tool,
target-policy failure 또는 두 번째 protocol failure는 semantic repair로 전환하지 않고 terminal이다.

첫 semantic finalize 결과가 103/128 미만이면 raw-free threshold marker를 쓰고 종료한다. 그
이상이면 rejected ID만 canonical order, 최대 16행, wave당 1--2 invocation xhigh로 repair하며
매 wave 뒤 finalize한다. exhaustion 하나라도 생기면 추가 teacher 호출을 전역 거부한다.

성공은 128/128, exhausted/retryable/unassessed 0, 완전한 source JSONL/manifest SHA 재검증,
answer-hidden audit 60/64 이상, receipt 생성까지다. 성공해도 11,794행 full bank 전에 멈추고
최소 initial 738 calls와 관측 latency/token/quota, repair worst case, worker 2 예상 시간을 제시해
사용자의 별도 확인을 받아야 한다.
