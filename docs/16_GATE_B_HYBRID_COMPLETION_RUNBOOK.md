# Gate B hybrid completion runbook

기준 시각: **2026-08-11 KST**

이 문서가 Gate B 이후 최종 제출물까지의 현재 운영 계약이다. teacher v1--v4 문서는
역사적 증거와 호환성 검증용이며 해당 ledger를 재개하는 명령서가 아니다. PR #4는 merge
commit `2b876a89637c2c71392a4ecfe9ce4ea1af9d30ce`로 `main`에 착륙했다. v4는 동일
128행 initial에서 79/128만 승인되어 terminal failure가 됐다.

## 1. 고정 순서

```text
base 5-fold OOF 완성
        ↓
synthetic harness v2 replay + 8×16 live canary
        ↓
teacher v5 8×16 pilot
   ├─ terminal failure → base-only freeze 후보
   └─ 128/128 + audit 60/64 → 승인 뒤 rationale 후보 확장
                                      ↓
                           development OOF로 한 방법 freeze
                                      ↓
                           사용자 확인 뒤 one-shot holdout
                                      ↓
                         evaluation prediction + submission 검증
```

PR은 `base-oof-v1` → `teacher-harness-v2` → `teacher-v5` 순서로만 병합한다. source
manifest를 만든 뒤 tracked source/config/test/docs가 바뀌면 해당 `RUN_TAG`는 폐기한다.

## 2. PR #5: base-only completion contract

다음 공개 schema를 추가한다.

- `gate-b-base-development-oof-v1`: 고정 base의 정확한 5개 fold와 development OOF
  전체를 증명한다.
- `gate-b-base-selection-freeze-v1`: `primary=base`, `fallback=null`,
  `routing_policy=primary_only`인 단일 방법 freeze다.
- `gate-b-run-context-v1`: commit, config, source manifest, split, B0와 fold 경로를
  상대 경로와 SHA로 결속하는 immutable no-overwrite artifact다.
- `gate-b-workflow-status-v1`: stage/state/count/hash/next action만 내보내는 raw-free
  상태 envelope다.

긴 generation과 training은 각각 다음으로 읽는다. `--output`은 monitor가 갱신하는 mutable
snapshot이며 private resume ledger를 수정하지 않는다.

```bash
uv run deep-challenge gate-b-development-status \
  --resume-dir "$RUN_ROOT/fold-0/base-resume" \
  --output "$RUN_ROOT/fold-0/base-status.json"

uv run deep-challenge gate-b-training-status \
  --resume-dir "$CHECKPOINT_ROOT/fold-0/resume" \
  --output "$RUN_ROOT/fold-0/training-status.json"
```

base folds 0--4는 동일 committed source manifest, direct-answer config, split, development
shard와 B0 pair에서 순차 실행한다. 각 fold는 새 records/manifest/resume 경로를 사용한다.
모든 fold가 complete인 뒤에만 다음 형식으로 OOF를 확정한다.

```bash
uv run deep-challenge verify-base-development-oof \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --deployment-fold 0 \
  --base-label base-direct \
  --base-run 0 "$RUN_ROOT/fold-0/base.jsonl" "$RUN_ROOT/fold-0/base.manifest.json" \
  --base-run 1 "$RUN_ROOT/fold-1/base.jsonl" "$RUN_ROOT/fold-1/base.manifest.json" \
  --base-run 2 "$RUN_ROOT/fold-2/base.jsonl" "$RUN_ROOT/fold-2/base.manifest.json" \
  --base-run 3 "$RUN_ROOT/fold-3/base.jsonl" "$RUN_ROOT/fold-3/base.manifest.json" \
  --base-run 4 "$RUN_ROOT/fold-4/base.jsonl" "$RUN_ROOT/fold-4/base.manifest.json" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --output "$RUN_ROOT/base-development-oof.json"
```

이 명령은 fold 누락·중복, validation ID 중복·누락, 다른 split/config/checkpoint/source/B0,
동일 run evidence 재사용을 거부한다. base OOF가 qualified되지 않으면 teacher로 진행하지 않는다.

`freeze-development-base`는 teacher v5가 terminal이거나 승인된 candidate가 최종 OOF gate를
통과하지 못했을 때만 사용한다. teacher 결과가 결정되기 전에는 실행하지 않는다.

## 3. PR #6과 #7

Harness v2는 대회와 무관한 signed-integer 128행을 정확히 8×16으로 실행한다. replay/live/
authorization v2를 새 schema로 추가하되 v1 bytes는 유지한다. live는 worker 1, high effort,
8 invocation, retry/repair/bank 0이다. malformed event/JSON, timeout/nonzero, 15/16/17
cardinality, duplicate/omission/reorder, oversize와 raw/path 누출을 offline fault matrix로
검증한다.

고정 config, schema, SHA와 실행 명령은
`docs/17_SYNTHETIC_TEACHER_HARNESS_V2.md`를 따른다. 실제 live 8회 호출은 v5 config가
착륙한 committed clean source에서만 수행한다.

Teacher v5는 v4 prompt/template/policy bytes를 그대로 두고 chunk만 16으로 바꾼다. initial
8×16, worker 1, high다. output-structure/protocol failure인 chunk에만 동일 ID·동일 prompt로
high protocol retry를 정확히 한 번 허용한다. semantic rejection은 gold 없이 xhigh로 최대
16행씩 repair한다. 행별 총 시도는 3회 이하이고 exhaustion 하나가 생기면 전역 중단한다.

첫 semantic finalize가 103/128 미만이면 terminal이다. 최종 성공은 128/128,
exhausted/retryable/unassessed 0, source JSONL/manifest 재검증, answer-hidden logical audit
60/64 이상과 receipt 생성까지다. 실패하면 raw-free count/hash/category만 남기고 v6를
만들지 않는다.

## 4. candidate 성공 뒤 승인 gate

v5 pilot과 audit가 성공해도 full bank 전에 멈춘다. 사용자에게 fold-0 11,794행,
chunk 16 기준 최소 738 initial calls, 관측 pilot latency/token/quota, 예상 repair와 행별
3회 worst case, worker 최대 2 및 예상 시간을 제시하고 별도 승인을 받아야 한다.

승인 뒤에만 full bank → materialization/corpus audit → pinned-tokenizer SFT preflight → 새
source/B0 → fold-0 rationale QLoRA → fixed base 1,653/2,942와 paired harm screen 순서로
간다. `candidate_full_oof_authorized=true`일 때만 나머지 bank와 folds 1--4를 실행한다.
bank exhaustion, incomplete fold, provenance drift 또는 harm-screen failure는 candidate를
종료하고 base로 돌아간다.

## 5. freeze, holdout, submission

최종 방법은 development OOF만으로 정확히 하나를 freeze한다. candidate는 complete 5-fold
OOF, parser-invalid 비열화 없음, provenance 오류 0과 함께 base 대비 `+1.0%p` 이상 또는
positive delta + bootstrap 95% CI lower bound > 0 + Holm 통과일 때만 primary다. 그 외에는
base다. fallback/routing/ensemble은 없다.

freeze 뒤 새 source manifest와 lockfile을 만들고 backend/freeze SHA를 재검증한다. irreversible
holdout claim 직전에 freeze SHA, B0와 GPU 상태를 제시하고 사용자의 별도 확인을 받는다.
`--acknowledge-one-time-locked-holdout`은 정확히 한 번만 사용하고 자동 retry하지 않는다.
holdout 결과로 모델·prompt·routing을 바꾸지 않는다.

그 뒤 frozen policy로 filtered leaderboard 831행 또는 공식 final test를 예측한다.
`write-submission`은 no-overwrite로 실행하고 기본 validator와 independent validator 모두에서
header `ID,answer`, 행 수, ID 순서와 SHA를 확인한다. 최종 submission path와 SHA를 보고한
뒤 멈춘다. Kaggle upload는 사용자의 별도 명시 요청이 있을 때만 한다.

## 6. 공통 검증과 즉시 중단

각 PR은 아래 검증과 `gstack-review` 뒤에만 병합한다.

```bash
uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q
(cd artifacts/analysis && sha256sum -c CHECKSUMS.sha256)
uv run python -m deep_challenge.public_repo_guard --all
git diff --check
```

기존 missing-PyTorch skip 1개 외 새 skip/xfail은 허용하지 않는다. Qwen base/revision,
inference contract, split/holdout/leaderboard/submission 정책과 v1--v4 ledger는 변경하지 않는다.
새 데이터, self-training, RL, browser/network/API teacher, 수동 ID, gold 재프롬프트, v6와
Kaggle upload는 이 실행 범위 밖이다.
