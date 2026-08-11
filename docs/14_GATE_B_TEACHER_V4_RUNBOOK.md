# Gate B teacher-pilot-v4 실행 런북

기준 시각: **2026-08-11 KST**

`teacher-pilot-v4`의 실행 계약이다. v1/v2/v3 ledger는 forensic evidence이며 plan, run,
resume, repair에 쓰지 않는다. 고정 Qwen base/revision, split, holdout, inference와 Kaggle
submission 정책은 바꾸지 않는다.

v4는 v3 prompt에 아래 한 문장만 더한다. JSON suffix와 output validator는 byte-identical하다.

```text
Before returning the JSON object, compare the completed items against INPUT_JSON: output exactly one item for every supplied problem_id, with the same item count and original order, and with no duplicate or omitted IDs.
```

## 고정 binding

- config semantic/file SHA-256: `7a7f3e117a3f454c21dd9721ae432b9da6c0813e3b806a19af7f1d9a34d8adef` /
  `063881f7d72a96e25202736a8fe729a0d64271376019b281094a5350c92a0d97`
- prompt template/policy SHA-256: `3029e9297bdda504e0f48e1ce4d57e363e5d3a5342edf18253b11c4f75ecd8a7` /
  `8de961862f2cabf245753ee276d4b833d8917934d4ba84fa8f9caa20a64ab924`
- harness config semantic/file SHA-256: `628dd255ad33ca1995f409060e7daeb4f9ed8a26805ac9e4458b1cdfffbfe4b1` /
  `5e31d82fa7dcd5e3b2dcdcaa4cdcf7b0db1efd40009ed5c0ee53aa23cf4f300c`

Committed clean source에서만 시작한다. 코드·config·tests·docs가 바뀌면 source manifest와
`RUN_TAG`를 폐기하고 새 tag로 시작한다. 기존 경로는 절대 덮어쓰지 않는다.

```bash
PROJECT=/mnt/c/Users/pjjpj/Desktop/deepleaning
cd "$PROJECT"
RUN_TAG=replace-with-new-unique-kst-tag

TEACHER_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v4.json"
HARNESS_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-harness-v1.json"
SOURCE_MANIFEST="$PROJECT/artifacts/analysis/source-manifest-gate-b-teacher-pilot-v4-$RUN_TAG.json"
REPLAY_REPORT="$PROJECT/artifacts/analysis/gate-b-teacher-harness-replay-v1-v4-$RUN_TAG.json"
LIVE_PLAN="$PROJECT/artifacts/gate_b/codex-teacher-harness-v1-v4-$RUN_TAG"
LIVE_REPORT="$PROJECT/artifacts/analysis/gate-b-teacher-harness-live-v1-v4-$RUN_TAG.json"
TEACHER_ROOT="$PROJECT/artifacts/gate_b/codex-teacher-pilot-v4-$RUN_TAG"
TEACHER_STATUS="$PROJECT/artifacts/analysis/gate-b-teacher-pilot-v4-$RUN_TAG-status.json"

test ! -e "$SOURCE_MANIFEST"
test ! -e "$REPLAY_REPORT"
test ! -e "$LIVE_PLAN"
test ! -e "$LIVE_REPORT"
test ! -e "$TEACHER_ROOT"
test ! -e "$TEACHER_STATUS"

uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q
(cd artifacts/analysis && sha256sum -c CHECKSUMS.sha256)
uv run python -m deep_challenge.public_repo_guard --all
git diff --check
git status --short
```

`git status --short` must be empty. The expected PyTorch-absent skip is the only
allowed skip; a new skip or xfail is a stop condition.

## Synthetic gate

Freeze source and run the fixed offline fault matrix first. Replay contains no contest
row and makes no Codex/network request.

```bash
uv run deep-challenge source-manifest --root "$PROJECT" --output "$SOURCE_MANIFEST"

uv run deep-challenge gate-b-teacher-harness-replay \
  --harness-config "$HARNESS_CONFIG" \
  --teacher-config "$TEACHER_CONFIG" \
  --output "$REPLAY_REPORT"

uv run deep-challenge gate-b-teacher-harness-live \
  --harness-config "$HARNESS_CONFIG" \
  --teacher-config "$TEACHER_CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --plan-dir "$LIVE_PLAN" \
  --report "$LIVE_REPORT" \
  --acknowledge-synthetic-codex-canary
```

Replay and live must both exit `0`. Live is exactly two contest-independent 32-row
calls: worker 1, high effort, retry/repair/bank output 0. Exit `1` records only its
raw-free report and stops. Do not retry, repair, make the organizer plan, or create v5.
Exit `2` requires a corrected clean snapshot and new tag.

All following v4 commands reverify the same evidence bundle. The planner opens the
private two-attempt live plan, requires its binary/version to match current Codex, and
writes immutable `harness-authorization-v1.json` in the organizer plan directory.

```bash
HARN_ARGS=(
  --harness-config "$HARNESS_CONFIG"
  --harness-replay-report "$REPLAY_REPORT"
  --harness-live-report "$LIVE_REPORT"
  --harness-live-plan-dir "$LIVE_PLAN"
  --harness-source-root "$PROJECT"
  --harness-source-manifest "$SOURCE_MANIFEST"
)
```

## Deterministic 128-row pilot

Use no manual ID selection and no reference-answer re-prompt. These are the existing
canonical data bindings.

```bash
DATA_DIR="$PROJECT/deep-learning-challenge-2026"
TRAIN="$DATA_DIR/deep_chal_math_train.csv"
EXCLUSIONS="$DATA_DIR/train_filtered_ids.csv"
SPLIT="$PROJECT/artifacts/analysis/splits-v4.json"
DEV_SHARD="$PROJECT/artifacts/analysis/development-cv-v4"
DATA_ARGS=(
  --train "$TRAIN" --train-exclusions "$EXCLUSIONS" --split-artifact "$SPLIT" --fold 0
  --expected-train-sha256 e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a
  --expected-exclusions-sha256 67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3
  --expected-exclusion-count 627
  --expected-split-sha256 be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db
  --development-shard "$DEV_SHARD"
  --expected-development-shard-sha256 cc5ea51f155f99d1956864c0097c3ac87ad42b89b4bd3c4e09f4d1a281d2fbb4
)

uv run deep-challenge gate-b-teacher-plan \
  "${DATA_ARGS[@]}" --teacher-config "$TEACHER_CONFIG" --pilot-size 128 \
  --output-dir "$TEACHER_ROOT" "${HARN_ARGS[@]}"

uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$TEACHER_ROOT" --teacher-config "$TEACHER_CONFIG" \
  --acknowledge-codex-teacher --max-invocations 4 --max-workers 1 "${HARN_ARGS[@]}"

PRIVATE_TEACHER_JSONL="$TEACHER_ROOT/source-rationales.jsonl"
PRIVATE_TEACHER_MANIFEST="$TEACHER_ROOT/source-rationales.manifest.json"
uv run deep-challenge gate-b-teacher-finalize \
  "${DATA_ARGS[@]}" --teacher-config "$TEACHER_CONFIG" --plan-dir "$TEACHER_ROOT" \
  --pilot-size 128 --output-jsonl "$PRIVATE_TEACHER_JSONL" \
  --output-manifest "$PRIVATE_TEACHER_MANIFEST" "${HARN_ARGS[@]}"

uv run deep-challenge gate-b-teacher-status \
  --plan-dir "$TEACHER_ROOT" --teacher-config "$TEACHER_CONFIG" \
  --output "$TEACHER_STATUS" "${HARN_ARGS[@]}"
```

Initial execution is exactly 4×32, worker 1, high. If first finalization is below
103/128, it exits `1`, writes only the raw-free
`initial-threshold-failure-v1.json`, produces no source JSONL/manifest, and blocks all
future teacher execution. Status remains available. This is terminal v4 failure: record
the raw-free count/hash/category and do not create v5 before a newly designed and
approved harness version.

If initial approval is at least 103, repair only canonical rejected rows. The ledger
selects xhigh and chunks of at most 16; one wave has at most two invocations and must
be followed by finalization. Any exhaustion blocks all further teacher execution.

```bash
uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$TEACHER_ROOT" --teacher-config "$TEACHER_CONFIG" \
  --acknowledge-codex-teacher --max-invocations 2 --max-workers 1 "${HARN_ARGS[@]}"
```

Success requires 128/128 accepted, exhausted/retryable/unassessed 0, no tool/schema/ID/
order violation, and valid source JSONL/manifest SHA.

## Success-only stop point

Only then run the existing answer-hidden 64-row logical audit and receipt, always with
the same `HARN_ARGS`:

```bash
AUDIT_ROOT="$PROJECT/artifacts/gate_b/codex-teacher-pilot-v4-audit-$RUN_TAG"
PILOT_RECEIPT="$PROJECT/artifacts/analysis/gate-b-teacher-pilot-v4-authorization-$RUN_TAG.json"

uv run deep-challenge gate-b-teacher-logical-audit-plan \
  --teacher-config "$TEACHER_CONFIG" --teacher-plan-dir "$TEACHER_ROOT" \
  --source-jsonl "$PRIVATE_TEACHER_JSONL" --source-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --output-dir "$AUDIT_ROOT" "${HARN_ARGS[@]}"
uv run deep-challenge gate-b-teacher-logical-audit-run \
  --teacher-config "$TEACHER_CONFIG" --teacher-plan-dir "$TEACHER_ROOT" \
  --audit-dir "$AUDIT_ROOT" --acknowledge-codex-teacher "${HARN_ARGS[@]}"
uv run deep-challenge gate-b-teacher-logical-audit-finalize \
  --teacher-config "$TEACHER_CONFIG" --teacher-plan-dir "$TEACHER_ROOT" \
  --audit-dir "$AUDIT_ROOT" "${HARN_ARGS[@]}"
uv run deep-challenge gate-b-teacher-pilot-authorize \
  "${DATA_ARGS[@]}" --teacher-config "$TEACHER_CONFIG" --pilot-plan-dir "$TEACHER_ROOT" \
  --pilot-source-jsonl "$PRIVATE_TEACHER_JSONL" --pilot-source-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --pilot-logical-audit-dir "$AUDIT_ROOT" --output "$PILOT_RECEIPT" "${HARN_ARGS[@]}"
```

Audit must be at least 60/64 and the receipt must verify. Stop there. A full fold-0
bank requires separate user approval after presenting the 11,794-row scope, minimum
369 initial calls, possible repairs, time estimate, and quota. Kaggle upload is always
outside this automation until the user separately requests it.
