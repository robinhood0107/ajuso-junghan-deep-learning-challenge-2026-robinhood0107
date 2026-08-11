# Gate B synthetic teacher harness v1

## 목적과 범위

`teacher-pilot-v1/v2/v3` ledger는 모두 forensic evidence이며 재개하지 않는다. 이 문서는
새 candidate prompt를 organizer data에 보내기 전에, 같은 ChatGPT-login Codex 경계가
cardinality·ID·순서를 안정적으로 지키는지 contest-independent signed-integer fixture로
확인하는 v1 harness의 고정 계약이다.

이 harness는 Qwen base/revision, split, locked holdout, inference, leaderboard 및 Kaggle
submission 정책을 바꾸지 않는다. live canary도 bank, corpus, GPU artifact를 만들지 않는다.

## Immutable contract

- Harness config: `configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v1.json`
  - semantic SHA-256: `628dd255ad33ca1995f409060e7daeb4f9ed8a26805ac9e4458b1cdfffbfe4b1`
  - file SHA-256: `5e31d82fa7dcd5e3b2dcdcaa4cdcf7b0db1efd40009ed5c0ee53aa23cf4f300c`
  - 64 fixed synthetic rows, 32/32 chunks, 2 invocations, worker 1, high effort,
    attempts 1, retry/repair/bank output 0.
- Fixture SHA-256: `bc314a24ec872edf26bae13e296fb8fd500f80bcf6c00023518844d1b407e3b7`.
- Historic v1/v2/v3 ledgers cannot start a fresh organizer-data run. The only
  next operational candidate is policy-bound `teacher-pilot-v4`: its config may
  be committed for synthetic qualification, but its organizer-data plan is
  forbidden until its *own* qualified replay/live evidence is reverified and
  bound into the immutable authorization sidecar. A fresh v3 canary is not an
  approved substitute for v4 evidence.
- Live source must have a fresh `source-manifest` and plan/report paths must not
  exist. Both runtime paths must be below excluded `artifacts/`, so a successful
  run cannot invalidate the frozen source tree.
- No command accepts live chunk, worker, retry, repair, bank, effort, manual-ID,
  or timeout override. The only acknowledgement is
  `--acknowledge-synthetic-codex-canary`.

The versioned, raw-free schemas are:

```text
gate-b-codex-teacher-failure-classifier-v1
gate-b-codex-teacher-diagnostic-v1
gate-b-codex-teacher-harness-replay-v1
gate-b-codex-teacher-harness-live-v1
gate-b-codex-teacher-harness-authorization-v1
```

Reports retain only a fixed stage/code and `requested_count`, `returned_count`,
`duplicate_count`, `missing_count`, `unexpected_count`, `order_mismatch`, plus
cryptographic provenance. They never serialize IDs, questions, answers, rationale,
prompt, provider stderr, or absolute paths.

The live report additionally binds a private-ledger tree SHA without naming the
ledger path. Before a later pilot is authorized, the harness reopens that private
two-attempt plan, re-hashes it, reclassifies both attempts, and requires an exact
match to the raw-free report; an edited report or a changed private attempt is
therefore rejected even if its own payload hash was recomputed.

Classifier precedence is immutable:

```text
process → timeout/spawn → nonzero → event JSON → unsafe/error event
→ terminal/usage → agent JSON → output schema → cardinality
→ ID set/duplicate/order → target policy → success
```

## Required local freeze and replay

Run these only from a committed clean source snapshot. The private checksum ledger
is deliberately local-only; the repository workflow runs Ruff, CPU pytest,
public-repo guard, and PR diff check but does not receive private checksums.

```bash
PROJECT=/mnt/c/Users/pjjpj/Desktop/deepleaning
cd "$PROJECT"
RUN_TAG=YYYYMMDDTHHMMSSKST
HARNESS_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v1.json"
TEACHER_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v4.json"
SOURCE_MANIFEST="$PROJECT/artifacts/analysis/source-manifest-teacher-harness-v1-$RUN_TAG.json"
REPLAY_REPORT="$PROJECT/artifacts/analysis/gate-b-teacher-harness-replay-v1-$RUN_TAG.json"
LIVE_PLAN="$PROJECT/artifacts/gate_b/codex-teacher-harness-v1-$RUN_TAG"
LIVE_REPORT="$PROJECT/artifacts/analysis/gate-b-teacher-harness-live-v1-$RUN_TAG.json"

test ! -e "$SOURCE_MANIFEST"
test ! -e "$REPLAY_REPORT"
test ! -e "$LIVE_PLAN"
test ! -e "$LIVE_REPORT"

uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q
(cd artifacts/analysis && sha256sum -c CHECKSUMS.sha256)
uv run python -m deep_challenge.public_repo_guard --all
git diff --check

uv run deep-challenge source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"

uv run deep-challenge gate-b-teacher-harness-replay \
  --harness-config "$HARNESS_CONFIG" \
  --teacher-config "$TEACHER_CONFIG" \
  --output "$REPLAY_REPORT"
```

Replay is CPU-only and must exit `0`. It executes a finite fault matrix covering
31/32/33 items, duplicate+omission, unknown/missing/reordered IDs, malformed
JSON/event, tool/error, usage, target-policy/oversize, timeout/spawn, nonzero,
and multi-fault precedence. Its report is immutable and no-overwrite.

## Explicit live canary

Only after the preceding source freeze and qualified replay may the following
two fixed synthetic chunks be sent through the already logged-in Codex CLI:

```bash
uv run deep-challenge gate-b-teacher-harness-live \
  --harness-config "$HARNESS_CONFIG" \
  --teacher-config "$TEACHER_CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --plan-dir "$LIVE_PLAN" \
  --report "$LIVE_REPORT" \
  --acknowledge-synthetic-codex-canary
```

- Exit `0`: qualified. Both chunks are 32/32, all 64 targets are locally valid,
  unique, canonical-order, and tool/unsafe/unclassified count is zero.
- Exit `1`: the command completed normally but the profile failed. Preserve only
  its raw-free report/count/hash; do not retry, repair, resume, create an
  organizer-data plan/run, or create v5.
- Exit `2`: input, provenance, or execution contract failure. Correct the
  contract on a new clean snapshot and generate a new manifest/run tag.

The live report binds fixture/config/prompt-policy/template/source-manifest and
Codex binary/version SHA. It is a canary-quality result, not contest performance
or generalization evidence.

## Read-only historic diagnosis

`gate-b-teacher-diagnose` accepts a verified existing ledger and writes a new
raw-free report without modifying any ledger file. It is the only permitted
operation for the historic v3 failure evidence:

```bash
V3_FORENSIC_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v3.json"

uv run deep-challenge gate-b-teacher-diagnose \
  --plan-dir "$V3_PRIVATE_PLAN" \
  --teacher-config "$V3_FORENSIC_CONFIG" \
  --output "$V3_DIAGNOSTIC_REPORT"
```

For the two known v3 structural failures, the diagnostic must contain the
redacted classification `stage=output_structure`, `code=cardinality_mismatch`,
`requested_count=32`, `returned_count=33`, and `duplicate_count=1`. The source
ledger's complete file digest is compared before and after diagnosis.

## PR #4 gate

The v4 organizer-data planner may proceed only if its own replay and live report
are both qualified and reverified with the private live plan into the
harness-authorization sidecar. It rechecks that sidecar before each plan, run,
status, finalization, audit, and receipt operation. A failed v4 pilot preserves
raw-free counts/hash/category only, creates no v5, and locks full bank, corpus,
audit, GPU, holdout, leaderboard, and submission work.
