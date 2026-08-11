# 10. Gate B CPU-ready 실행 런북

기준 시각: **2026-08-11 KST**
대상 호스트: **WSL2 Ubuntu 24.04 + NVIDIA GeForce RTX 4070 SUPER 12GB**
현재 판정: **parser v2 current-source base 완료, answer-only QLoRA 중단, teacher-pilot-v3 CPU READY·live pilot 미실행**

이 문서는 현재 코드의 CLI와 정확히 일치하는 실행 순서다. GPU가 필요한 명령은 맨
뒤의 별도 절에만 둔다. 2026-08-10 이전 source final smoke와 별도로, training cache-off와
eval KV-cache-on을 분리한 당시 source는 run tag `20260810T062500KST`에서 새 tagged smoke를
green으로 닫았다.
이후 실제 모델 실행도 다른 프로세스를 종료하거나 threshold를 완화하지 않고, 새 run마다 physical
VRAM을 다시 검사한다.

## 1. 변경할 수 없는 계약

- 베이스 모델은 `Qwen/Qwen2.5-3B-Instruct` 하나뿐이다.
- revision은 `aa8e72537993ba99e69dfaafa59ed015b17504d1`이다.
- Qwen2.5-Math, DeepSeek, Llama 등 다른 모델의 weight, merge, ensemble, 추론은
  사용하지 않는다.
- organizer train만 첫 baseline과 첫 QLoRA SFT에 사용한다.
- leaderboard/test 문제와 그 파생물은 학습, self-training seed, 외부 API 입력에
  사용하지 않는다.
- split v4를 재생성하지 않는다. soft number-masked template는 hard cluster가 아니다.
- development 단계는 `development-cv-v4`만 읽는다. 원본 train은 내용이 아니라
  SHA-256 확인에만 사용한다.
- primary/fallback을 development 근거로 freeze하기 전 locked holdout을 열지 않는다.
- parser conflict/invalid는 숨기지 않고, submission에 silent `0`을 넣지 않는다.
- 모든 새 출력은 새 version/tag 경로를 쓴다. 기존 artifact를 덮어쓰지 않는다.

## 2. 현재 호스트와 canonical 값

```bash
PROJECT=/absolute/path/to/deepleaning
DATA_DIR="$PROJECT/deep-learning-challenge-2026"
GPU_ENV=/absolute/path/on/ext4/to/deep-challenge-gpu-venv
MODEL_CACHE=/absolute/path/to/huggingface/models--Qwen--Qwen2.5-3B-Instruct

REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
TRAIN_SHA=e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a
EXCLUSIONS_SHA=67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3
SPLIT_SHA=be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db
DEV_SHARD_SHA=cc5ea51f155f99d1956864c0097c3ac87ad42b89b4bd3c4e09f4d1a281d2fbb4
FILTERED_LB_SHA=032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1

TRAIN="$DATA_DIR/deep_chal_math_train.csv"
EXCLUSIONS="$DATA_DIR/train_filtered_ids.csv"
FILTERED_LB="$DATA_DIR/deep_chal_math_leaderboard_filtered.csv"
SPLIT="$PROJECT/artifacts/analysis/splits-v4.json"
DEV_SHARD="$PROJECT/artifacts/analysis/development-cv-v4"
CONFIG="$PROJECT/configs/gate_b/rtx4070-super-12gb-direct-answer-v1.json"
RATIONALE_CONFIG="$PROJECT/configs/gate_b/rtx4070-super-12gb-concise-rationale-v1.json"
TEACHER_CONFIG="$PROJECT/configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v3.json"
RUN_TAG=replace-with-new-unique-tag
SOURCE_MANIFEST="$PROJECT/artifacts/analysis/source-manifest-gate-b-teacher-pilot-v3-$RUN_TAG.json"
```

여기서 `SPLIT_SHA`는 `splits-v4.json` 안의 논리 split SHA다. split JSON 파일 자체의
SHA는 `5b1969e79da08fa8347569c55d6d40b1fbccfcb2e5fc0e0f7a3295386a260520`이다.
`DEV_SHARD_SHA`는 `development-cv-v4/CHECKSUMS.sha256` 파일로 고정한 bundle SHA이고,
그 안의 `development-train.csv` SHA는
`4dcd86cb6a9366c8049c9ec39770e599e2788261cdb1f7c180ce120f958c0306`이다. config
내부 논리 SHA는 `4530c14a4782c439ea3a8325b90d997793eda368b0371d765cb810690bb40028`,
config JSON 파일 SHA는
`703926d84ec6c7a95f7ce50de384fb5dcb1bb35d98cd52cbb6ab846f980d83c3`이다. CLI의
`--expected-split-sha256`와 `--expected-development-shard-sha256`에는 각각 위의 논리
split SHA와 bundle SHA만 넣는다.
`RATIONALE_CONFIG`의 semantic/file SHA는 각각
`75a315b638481a0c8213c413aa3a1253d269776d08bd2252b68654fb38c3f053`와
`66a4c5c145881c92cb4b260ef000bd89bd62119b644f6bd1e49e9894c431064f`다.
fresh `TEACHER_CONFIG` (`teacher-pilot-v3`)의 semantic/file/prompt-template/prompt-policy
SHA는 각각
`deafe380e20079ef5e5fb2917c9f91d7a235d1135a23c64dcbd4ea7dddd38613`,
`54a2e31e716edfdd3d5a5d22a2d5124da14f552b4ceeab31fdf7c5ea11ddba01`,
`cf56fc2c021410337f8be8f5f519912eabf6390aa8892ecd92cac1ced6175c72`,
`953d62e283d5237f29b2145b5ed513246d737acd7ec40879450d7bcc8d08402b`다. historic
`teacher-v1` config의 semantic/file SHA는 각각
`63129b89c5daa33aad5906d15f893371aba2ac1a022172c32399fd9c0ccd29cd`와
`9480d8d083f1f21b6c78bbf8607de70393ff1011e3919b32bce5a4434499fc75`이며, forensic
evidence 전용으로 남긴다. failed `teacher-pilot-v2` config와 ledger도 forensic evidence
전용이며 새 v3 plan/status/finalize 입력으로 쓰지 않는다.

현재 canonical 사실은 다음과 같다.

| 항목 | 값 |
|---|---:|
| train | 17,000행 |
| split v4 hard cluster | 16,992개 |
| locked holdout | 1,700행 |
| development shard | 15,300행 |
| organizer exclusion | 627 ID |
| hard-group 확장 exclusion | 629행 |
| eligible development CV | 14,736행 |
| fold 0 training / validation | 11,794 / 2,942행 |
| filtered leaderboard | 831행 |

development shard의 자체 ledger는
`artifacts/analysis/development-cv-v4/CHECKSUMS.sha256`이며 bundle SHA는
`cc5ea51f155f99d1956864c0097c3ac87ad42b89b4bd3c4e09f4d1a281d2fbb4`다.
그 shard를 한 번 만들 때는 단일 organizer CSV를 읽었지만 holdout 행을 출력하지 않았다.
그 이후의 development/SFT/비교 CLI는 shard만 파싱한다. 과거 Gate A에서 split 통계를
만들며 holdout bytes를 물리적으로 읽은 사실까지 없었다고 주장하지는 않는다.

## 3. 지금 실행 가능한 CPU-only 검증

```bash
cd "$PROJECT"

uv sync --extra model --group dev
uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q

cd artifacts/analysis/development-cv-v4
sha256sum -c CHECKSUMS.sha256
cd "$PROJECT"
cd artifacts/analysis
sha256sum -c CHECKSUMS.sha256
cd "$PROJECT"
```

기본 `.venv`에는 GPU 학습 package를 의도적으로 넣지 않는다. 아래 명령의 exit 1은
현재 blocker를 기록하는 정상 결과다. 모델을 load하거나 GPU kernel을 실행하지 않는다.

```bash
cd "$PROJECT"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run deep-challenge model-preflight \
  --revision "$REVISION" \
  --output "artifacts/analysis/model-preflight-cpu-hidden-$RUN_TAG.json"
```

CPU-only SFT encoding 증거는 이미
`gate-b-sft-encoding-preflight-v3-fold0-20260804.json`에 고정했다. 새로 재현할 때는
기존 파일을 덮어쓰지 말고 새 tag를 넣는다.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run deep-challenge gate-b-sft-preflight \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold 0 \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --revision "$REVISION" \
  --config "$CONFIG" \
  --output "artifacts/analysis/gate-b-sft-encoding-preflight-v3-fold0-$RUN_TAG.json"
```

green 기준은 `torch_or_cuda_used=false`, `model_weights_loaded=false`,
`locked_holdout_accessed=false`, `leaderboard_or_test_used=false`, truncation 0이다. 현재 실제
결과는 max sequence 1,127/2,048, response label 최소 7 token, training 11,794행,
validation 2,942행이다.

### 3.1 ChatGPT 로그인 Codex teacher bank (GPU 없음)

이 경로는 API key나 Kaggle token을 사용하지 않는다. 현재 ChatGPT 로그인 상태의 local
Codex CLI만 사용하고, 최종 student/추론 모델은 여전히 고정 Qwen 하나뿐이다. Codex에는
organizer train의 **question만** 전달한다. local finalizer가 나중에 organizer reference와
exact match를 검증하며, reference answer/locked holdout/leaderboard/test는 prompt, raw
teacher plan, logical-audit 입력 어디에도 넣지 않는다.

CLI는 실행마다 빈 working directory와 별도의 임시 `CODEX_HOME`을 만들고, 기존 ChatGPT
authentication state만 짧게 복사한다. 전역 skills/config/API 환경변수는 전달하지 않으며,
`--ephemeral --json --sandbox read-only --ignore-user-config --ignore-rules`를 고정한다.
따라서 raw event/prompt/question/rationale/ID는 private `artifacts/gate_b/`에만 남고 Git에
올라가지 않는다. auth state의 복사본은 실행 중 별도 temporary tree에만 존재하며 종료 시
정리된다. raw-free status snapshot만 `artifacts/analysis/`에 쓸 수 있다.

먼저 `teacher-pilot-v3` config로 fold 0의 정확한 `training_ids(0)`에서 stable-hash
sign/magnitude stratified 128문제 pilot plan을 만든다. `gate-b-teacher-v2-*`는 이 pilot의
새 이름이 아니라, 이후 positive harm screen 뒤 remaining development-CV bank를 확장하는
별도 명령이다. full 11,794문제 bank v1은 `--pilot-size`를 생략해 만들 수 있지만, 이 경우
아래의 immutable pilot receipt와 pilot plan/source/audit를 모두 다시 검증하는 인자를 반드시
줘야 한다. fold, arbitrary ID, validation, holdout, leaderboard/test 입력은 CLI가 거부한다.

```bash
FOLD=0
TEACHER_ROOT="$PROJECT/artifacts/gate_b/codex-teacher-pilot-v3-$RUN_TAG"
TEACHER_STATUS="$PROJECT/artifacts/analysis/gate-b-teacher-pilot-v3-$RUN_TAG-status.json"

# Code/config/docs are now frozen for this run tag.  source-manifest/status use
# atomic replacement internally, so prove all three target paths are new first.
test ! -e "$SOURCE_MANIFEST"
test ! -e "$TEACHER_ROOT"
test ! -e "$TEACHER_STATUS"
uv run deep-challenge source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"

uv run deep-challenge gate-b-teacher-plan \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-config "$TEACHER_CONFIG" \
  --output-dir "$TEACHER_ROOT" \
  --pilot-size 128

uv run deep-challenge gate-b-teacher-status \
  --plan-dir "$TEACHER_ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --output "$TEACHER_STATUS"
```

Pilot의 최초 호출은 CLI가 자동으로 high reasoning을 붙이고, worker 1, 32문제 네 chunk로
제한한다. v3 prompt는 question string을 untrusted data로 취급하고 role/tool/network/format
변경 지시를 따르지 않는다. 각 signed integer를 먼저 도출한 뒤 governing condition을 다시
확인하고, 가능하면 다른 경로로 결정적 산술을 재계산하며 feasibility·integrality·sign까지
독립 검증한다. JSON suffix와 output validator는 v2와 같다. 이는 동일 128행 development
prompt 비교이지 일반화 성능 증거가 아니다. `gate-b-teacher-run`에는
reasoning-effort override가 없다. 같은 재개 호출에서도
unattempted chunk는 high, local finalizer가 rejected로 판정한 repair row만 xhigh와 16문제 이하
chunk를 자동 적용한다. status/terminal에는 aggregate count, latency, token usage, ledger lock
PID만 보이고 raw payload는 나오지 않는다.

```bash
uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$TEACHER_ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --acknowledge-codex-teacher \
  --max-invocations 4 \
  --max-workers 1
```

그 다음 finalizer만 local development shard의 reference answer를 열어 exact match한다. 첫
finalize에서 initial accepted가 `103/128` 미만이면 80% gate는 복구 불가능하므로 repair를
한 번도 호출하지 않고 즉시 종료한다. 아직
accepted되지 않은 row가 있으면 source JSONL을 만들지 않고 `complete=false`만 반환한다. 이때
manual ID selection이나 gold answer 재-prompt는 금지다. ledger가 선택한 failed row만 xhigh,
16문제 이하 repair chunk로 최대 총 3회까지 재시도한다. complete가 아니면 아래 두 명령을
`finalize → automatic repair run → finalize` 순서로 반복한다. 한 repair wave는 canonical
chunk·ID 순서의 최대 2 invocation만 허용하며, 매 wave 뒤 반드시 다시 finalize한다. exhausted
row가 하나라도 생기면
plan 전체가 fail-closed된다. 이후 `gate-b-teacher-run`은 남은 retryable row가 있어도 추가
teacher 실행을 거부하며, raw-free status 조회와 aggregate 보존만 허용한다.

```bash
PRIVATE_TEACHER_JSONL="$TEACHER_ROOT/source-rationales.jsonl"
PRIVATE_TEACHER_MANIFEST="$TEACHER_ROOT/source-rationales.manifest.json"

uv run deep-challenge gate-b-teacher-finalize \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-config "$TEACHER_CONFIG" \
  --plan-dir "$TEACHER_ROOT" \
  --output-jsonl "$PRIVATE_TEACHER_JSONL" \
  --output-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --pilot-size 128

# local finalizer가 complete=false + retryable rows를 보고한 뒤에만 재개한다.
# pending repair row는 xhigh, 아직 시도하지 않은 row는 high가 자동 적용된다.
uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$TEACHER_ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --acknowledge-codex-teacher \
  --max-invocations 2 \
  --max-workers 1
```

Pilot promotion requires initial accepted at least 103/128, final accepted 128/128, zero
exhausted/retryable/unassessed rows, at most three attempts per row, and zero
tool/error/schema/ID/order/provenance violations. The complete source JSONL and manifest must reverify
their SHA. The separate 64-row answer-hidden logical audit must then reach at least 60 consistent rows.
Only then make the full v1 plan; it may use two workers. A successful fold 0 GPU harm screen is still
required before adding the other 2,942 development-CV IDs to bank v2.

모든 teacher/audit 실행은 시작 직전에 ChatGPT Codex CLI의 resolved executable과 version을
다시 probe해 immutable plan과 정확히 일치해야 한다. 불일치면 어떤 question도 보내지 않는다.
각 worker의 auth-only `CODEX_HOME`은 `-C` model workspace 밖의 별도 ephemeral temp tree에만
생기고, model-generated read-only tool process에는 `shell_environment_policy.inherit="none"`을
고정한다. tool/error/schema/ID violation은 기존처럼 ledger에서 fail-closed다.

현재 `20260811T103224KST` pilot은 first pass 103/128(80.47%)이었지만 final 111/128,
exhausted 17로 fail-closed 됐다. 이 plan에서 logical audit, full v1 bank, corpus, GPU를
이어 실행하지 않는다. 이 historic ledger는 현재의
`shell_environment_policy.inherit="none"` safe-command contract 이전에 만들어졌으므로,
현재 status/finalize loader가 의도적으로 reject한다. raw-free final aggregate는 보존 증거일
뿐 재개 입력이 아니며, 호환성을 위해 prompt/argv 검증을 완화하지 않는다.

`20260811T132301KST` fresh `teacher-pilot-v2` run은 이 런북의 32행×4/worker 1 profile로
실행됐다. first pass는 105/128(82.03%)로 80% gate를 넘었지만, 최대 3회 후 106/128만
승인되고 7개가 exhaustion되어 fail-closed됐다. raw-free 결과는
`artifacts/analysis/gate-b-teacher-pilot-v2-20260811T132301KST-final-v2.json` (SHA-256
`5d50fdb41c0503546e673393d97b24bf7dc5c92e52577738eada35c143ac874e`)에 고정했다. 이 tag의
ledger를 더 실행하거나 partial source를 materialize하지 않는다. 새 v3 prompt/config는 별도
version/tag에서만 실행하며 v1/v2 ledger를 resume하지 않는다. v3 pilot이 위의 원자적 성공
조건을 모두 충족하기 전에는 이 절 아래의 audit/full-bank/GPU 명령을 실행하지 않는다.

complete private bank가 생기면 audit agent에는 problem, candidate rationale와 candidate가 주장한
final integer만 보낸다. organizer reference answer를 다시 열거나 전달하지 않는다. audit의
raw event도 private directory에만 남고 status/receipt에는 aggregate와 SHA만 남는다.
`gate-b-teacher-logical-audit-run`도 effort override가 없으며 first audit은 high, transport/event
failure 뒤의 retry만 xhigh를 ledger 상태에서 자동 선택한다.

```bash
LOGICAL_AUDIT_ROOT="$PROJECT/artifacts/gate_b/codex-teacher-pilot-audit-$RUN_TAG"
PILOT_RECEIPT="$PROJECT/artifacts/analysis/gate-b-teacher-pilot-authorization-$RUN_TAG.json"

uv run deep-challenge gate-b-teacher-logical-audit-plan \
  --teacher-config "$TEACHER_CONFIG" \
  --teacher-plan-dir "$TEACHER_ROOT" \
  --source-jsonl "$PRIVATE_TEACHER_JSONL" \
  --source-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --output-dir "$LOGICAL_AUDIT_ROOT"

uv run deep-challenge gate-b-teacher-logical-audit-run \
  --teacher-config "$TEACHER_CONFIG" \
  --teacher-plan-dir "$TEACHER_ROOT" \
  --audit-dir "$LOGICAL_AUDIT_ROOT" \
  --acknowledge-codex-teacher

uv run deep-challenge gate-b-teacher-logical-audit-finalize \
  --teacher-config "$TEACHER_CONFIG" \
  --teacher-plan-dir "$TEACHER_ROOT" \
  --audit-dir "$LOGICAL_AUDIT_ROOT"

uv run deep-challenge gate-b-teacher-pilot-authorize \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-config "$TEACHER_CONFIG" \
  --pilot-plan-dir "$TEACHER_ROOT" \
  --pilot-source-jsonl "$PRIVATE_TEACHER_JSONL" \
  --pilot-source-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --pilot-logical-audit-dir "$LOGICAL_AUDIT_ROOT" \
  --output "$PILOT_RECEIPT"
```

이 receipt 생성은 first-pass local exact-match 80% 이상, 128행 전부의 최대 3회 내 승인,
완결 source-bank provenance 및 passed 64→60 audit를 매번 다시 계산한다. 이후 full v1 plan을
만들 때도 receipt만 신뢰하지 않고 동일 private evidence를 다시 hash-validate한다.

```bash
FULL_TEACHER_ROOT="$PROJECT/artifacts/gate_b/codex-teacher-pilot-v3-full-$RUN_TAG"
FULL_TEACHER_STATUS="$PROJECT/artifacts/analysis/gate-b-teacher-pilot-v3-full-$RUN_TAG-status.json"
FULL_TEACHER_JSONL="$FULL_TEACHER_ROOT/source-rationales.jsonl"
FULL_TEACHER_MANIFEST="$FULL_TEACHER_ROOT/source-rationales.manifest.json"

test ! -e "$FULL_TEACHER_ROOT"
test ! -e "$FULL_TEACHER_STATUS"

uv run deep-challenge gate-b-teacher-plan \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-config "$TEACHER_CONFIG" \
  --output-dir "$FULL_TEACHER_ROOT" \
  --pilot-authorization "$PILOT_RECEIPT" \
  --pilot-plan-dir "$TEACHER_ROOT" \
  --pilot-source-jsonl "$PRIVATE_TEACHER_JSONL" \
  --pilot-source-manifest "$PRIVATE_TEACHER_MANIFEST" \
  --pilot-logical-audit-dir "$LOGICAL_AUDIT_ROOT"

# This is a separate high-cost gate: 11,794/32 requires at least 369 initial
# calls, before repairs. Show current cost/quota and obtain separate user
# confirmation before running it. The full bank may use at most two workers
# and keeps the same v3 prompt policy.
uv run deep-challenge gate-b-teacher-run \
  --plan-dir "$FULL_TEACHER_ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --acknowledge-codex-teacher \
  --max-workers 2

uv run deep-challenge gate-b-teacher-status \
  --plan-dir "$FULL_TEACHER_ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --output "$FULL_TEACHER_STATUS"

uv run deep-challenge gate-b-teacher-finalize \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-config "$TEACHER_CONFIG" \
  --plan-dir "$FULL_TEACHER_ROOT" \
  --output-jsonl "$FULL_TEACHER_JSONL" \
  --output-manifest "$FULL_TEACHER_MANIFEST"
```

### 3.2 verified concise-rationale CPU gate

이 단계는 GPU를 쓰지 않는다. `PRIVATE_TEACHER_JSONL`은 Git에서 제외한 training-only
입력이어야 하며 leaderboard/test/locked holdout에서 만든 행을 한 개라도 포함하면 안 된다.
현재 production 파일은 아직 생성하지 않았다. 각 row는 exact fold-training ID, organizer
question SHA, canonical `Final answer: <integer>`로 끝나는 concise rationale, target SHA,
teacher provider/model/revision, prompt/generation/raw SHA, seed/sample index,
`reference_answer_in_prompt=false`, `network_scope=training_only`, reference-answer 검증
상태를 갖는다. Python/SymPy tool 사용은 현재 config에서 금지다.

```bash
FOLD=0
PRIVATE_TEACHER_JSONL="$FULL_TEACHER_JSONL"
PRIVATE_TEACHER_MANIFEST="$FULL_TEACHER_MANIFEST"
RATIONALE_ROOT="$PROJECT/artifacts/gate_b/rationale-$RUN_TAG/fold-$FOLD"
test ! -e "$RATIONALE_ROOT"
mkdir "$RATIONALE_ROOT"

# receipt-bound full fold-0 bank만 materialize한다. pilot 128행 bank는 full
# training coverage가 아니므로 이 입력으로 사용할 수 없다.
uv run deep-challenge gate-b-materialize-teacher-bank \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --teacher-bank "$FULL_TEACHER_ROOT" "$PRIVATE_TEACHER_JSONL" "$PRIVATE_TEACHER_MANIFEST" \
  --output-jsonl "$RATIONALE_ROOT/teacher-source.jsonl" \
  --output-manifest "$RATIONALE_ROOT/teacher-source.manifest.json"

uv run deep-challenge build-rationale-corpus \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --source-jsonl "$RATIONALE_ROOT/teacher-source.jsonl" \
  --rationale-config "$RATIONALE_CONFIG" \
  --output-jsonl "$RATIONALE_ROOT/rationales.jsonl" \
  --output-manifest "$RATIONALE_ROOT/manifest.json"

uv run deep-challenge audit-rationale-corpus \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --rationale-corpus "$RATIONALE_ROOT/rationales.jsonl" \
  --rationale-manifest "$RATIONALE_ROOT/manifest.json" \
  --rationale-config "$RATIONALE_CONFIG" \
  --output "$RATIONALE_ROOT/audit.json"

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run deep-challenge gate-b-sft-preflight \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --revision "$REVISION" \
  --config "$CONFIG" \
  --rationale-corpus "$RATIONALE_ROOT/rationales.jsonl" \
  --rationale-manifest "$RATIONALE_ROOT/manifest.json" \
  --rationale-audit "$RATIONALE_ROOT/audit.json" \
  --rationale-config "$RATIONALE_CONFIG" \
  --output "$RATIONALE_ROOT/sft-encoding-preflight-v4.json"
```

세 명령은 모두 exact fold-training coverage와 모든 checksum을 다시 계산하고 기존 출력을
덮어쓰지 않는다. audit에는 raw rationale/ID/question/reference answer/parsed integer/teacher
prompt가 직렬화되지 않으며 `reference_answer_in_prompt_true_count=0`을 aggregate로 남긴다.
마지막 artifact가 schema v4, `status=green`, truncation 0,
`torch_or_cuda_used=false`이고 corpus records/manifest/audit SHA를 모두 결속할 때만 이후
rationale QLoRA의 입력 후보가 된다. production corpus가 없으므로 이 절의 실제 완료 점수는
아직 없다.

## 4. 규칙 상태와 보수적 feature gate

사용자가 제공한 규칙 원문으로 다음은 확인했다.

- 공개·동등 접근 외부 학습 데이터 허용
- 상용 API의 training-data/rationale 생성 허용
- leaderboard/test 답을 API로 직접 만드는 행위 금지
- 추론 중 인터넷/API 금지
- Majority Voting, Self-Consistency 등 test-time 기법 허용
- 제출 논리 schema `ID,answer`, 정수 Exact Match

authenticated Kaggle API로 Rules/Data/Evaluation/Submission contract와 현재 file listing,
submission allowance를 read-only 확인했다. 다음은 아직 확인하지 못했다.

- 실제 sample submission 파일의 row order와 SHA-256
- 일일·전체 제출 횟수
- 최종 제출 창의 정확한 timezone
- 로컬 Python/SymPy tool inference의 명시 허용 범위
- same-base multi-adapter/checkpoint voting·weight soup·selector 허용 범위

따라서 Python/SymPy TIR과 same-base multi-adapter 결합은 off다. organizer-only direct-answer
candidate는 이미 중단했고, 다음 모델 실험도 단일 base와 단일 rationale adapter만 비교한다.
teacher rationale는 training-only로 허용되지만 3.1절 품질 검증 corpus가 없으면 시작하지 않는다.

## 5. B0 GPU 승인 증거와 새 run 직전 재검사

2026-08-10 target-host preflight와 local synthetic smoke는 parser v2 production source의
run tag `20260810T234907KST`로 green이고 같은 tag의 B1 v2까지 완료했다. 이 pair는 그
source의 production B1에만 bind한다. rationale 구현 이후의 다른 run에는 재사용하지 않고
이 절의 새 `RUN_TAG`
preflight와 smoke를 순서대로 다시 만든다.
다른 GPU 프로세스를 종료하거나 선점하지 않으며, 모든 새 GPU run 직전에는 아래 두 조건을
다시 관측한다.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version \
  --format=csv,noheader,nounits
```

- free VRAM이 10,240MiB 이상
- pre-existing used VRAM이 2,048MiB 이하

이전-source 비교 evidence는
`artifacts/analysis/model-preflight-gpu-current-20260810T001500KST.json`와
`artifacts/analysis/gpu-smoke-20260810T001500KST.json`에 있다. 당시-current-source production
binding은 `model-preflight-gpu-ready-20260810T062500KST.json`(physical 912/11,086MiB,
`training_ready=true`)와 `gpu-smoke-20260810T062500KST.json`(`status=green`)이다. GPU 값은
계속 변하므로 snapshot 수치를 현재 실시간 값이라고 주장하지 않고, 실제 실행 여부는 위
`nvidia-smi`의 직전 관측으로 다시 판정한다.

현재 parser v2 production binding은
`model-preflight-gpu-ready-20260810T234907KST.json` (SHA
`c1b29c10ad76f3eb2e94781c4dc271951a5b638af8cdcc2d291d3062b55a349f`)과
`gpu-smoke-20260810T234907KST.json` (SHA
`70689026618f45e468565742a0afb10472a8373329d35c5da840786ead12e30b`)이다. 이 pair는
완료된 base evidence에만 결속돼 있으며 새 rationale 학습을 승인하지 않는다.

GPU 전용 환경은 WSL ext4에 둔다. 재동기화가 필요할 때만 다음을 실행한다.

```bash
cd "$PROJECT"
UV_PROJECT_ENVIRONMENT="$GPU_ENV" \
  uv sync --frozen --extra model --extra gpu --group dev
```

그 뒤 인터넷을 끄고 새 출력 경로로 실제 preflight와 final smoke를 순서대로 실행한다.
`--acknowledge-gpu-use`가 없으면 smoke와 이후 모든 CUDA 명령은 시작 전에 실패한다.

```bash
GPU_CLI="$GPU_ENV/bin/deep-challenge"
PREFLIGHT="artifacts/analysis/model-preflight-gpu-ready-$RUN_TAG.json"
SMOKE="artifacts/analysis/gpu-smoke-$RUN_TAG.json"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$GPU_CLI" model-preflight \
  --revision "$REVISION" \
  --output "$PREFLIGHT"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$GPU_CLI" gpu-smoke \
  --preflight-report "$PREFLIGHT" \
  --output "$SMOKE" \
  --acknowledge-gpu-use
```

preflight와 smoke가 모두 exit 0이고 smoke artifact `status=green`일 때만 다음 절로 간다.
smoke는 로컬 synthetic `2+3`만 사용하여 pinned NF4 load, 실제
`paged_adamw_8bit` backward/step, training cache-off/eval KV-cache-on generation, parser
exact match, peak VRAM과 latency를 검증한다. organizer/leaderboard/test 문제는 smoke 입력으로 받지 않는다.

smoke는 CUDA context를 만들기 **전** read-only `nvidia-smi`로 external used/free를 다시
확인하고, context 생성 **후** CUDA free VRAM으로 실제 load 가능 여유를 확인한다. 현재
12GiB WDDM display host의 정상 desktop baseline은 CUDA compute process 없이 약 1.4GiB로
측정됐다. 따라서 versioned smoke contract는 used 2,048MiB 상한과 free 10,240MiB 하한을
동시에 요구한다. free 하한이 실제 capacity gate이고, context 자체의 driver/WSL overhead를
external occupancy로 바꾸어 해석하지 않는다. 두 측정치는 green artifact의 runtime
evidence에 함께 남으며, 이후 training/inference gate도 이를 요구한다.

새 B0 pair를 만들기 직전 source/config/tests/docs tree를 snapshot한다. 이 단계는 GPU를
사용하지 않으며, 이후 B1 v2 manifest와 QLoRA adapter manifest가 이 file byte와 tree SHA를
기록한다. source가 바뀌면 새 `RUN_TAG`와 새 output path를 사용해 다시 만든다.

```bash
"$GPU_CLI" source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"
```

## 6. Gate B1 — base direct-answer development 기준선

parser v2 production tag `20260810T234907KST`는 이 절을 완료했다. records 2,942행과 v2
manifest가 atomic publish됐고 records/manifest SHA-256은 각각
`d26196283ef0a9f350d252703f40797eb9cc1eafffa676e6b5a961404a1126b4`와
`5a7c97f070fc7f9861a5c7bd92739c43cad0761714c373c046f485e4d300ea92`다. exact match는
1,653/2,942(56.1863%), parser `ok/conflict/invalid=2705/3/234`, finish
`stop/length=2134/808`이었다. parser audit
`parser-golden-20260810T234907KST-fold0-v6.json`도 raw-free 20 classes로 통과했다. 이
bundle은 stored/current parser가 일치하는 selection evidence다. 이후 rationale 코드·문서·
테스트가 source tree를 바꾸므로 새 학습 전에는 새 tag/source manifest/B0 pair를 만든다.

GPU green 뒤 첫 모델 실행은 fold 0 base greedy 한 개뿐이다. 아래 `FOLD=0`을 그대로
두고 먼저 실행한다. 2026-08-10에 attention mask와 eval KV-cache 보강 전 시작한 동일 목적의 diagnostic
run은 parser golden 관찰용으로만 취급한다. 그 run은 method selection, QLoRA authorization,
OOF comparison에 재사용하지 않으며, 아래 명령은 보강된 source에서 새 `RUN_TAG`로 다시
실행한다.

이 diagnostic은 이후 정상 종료되어 2,942행 중 1,210 exact match(41.1285%)와 redacted
parser audit 19개 outcome class를 남겼다. public regression은 safe synthetic structural
case만 추가했고 raw output은 private artifact에 남겼다. 하지만 run source가
attention-mask/eval-cache 보강 전이므로 위 aggregate는 production score도 selection evidence도
아니다. 아래 새 v2 current-source 명령의 B0 binding과 output target을 대체하지 않는다.

```bash
FOLD=0
RUN_ROOT="$PROJECT/artifacts/gate_b/$RUN_TAG"
CHECKPOINT_ROOT="/absolute/path/on/ext4/to/deep-challenge-checkpoints/$RUN_TAG"
RUN_DIR="$RUN_ROOT/fold-$FOLD"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/fold-$FOLD"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

"$GPU_CLI" gate-b-development \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --acknowledge-gpu-use \
  --output-jsonl "$RUN_DIR/base-direct-predictions.jsonl" \
  --output-manifest "$RUN_DIR/base-direct-manifest.json"
```

저장되는 각 row에는 raw generation, parser 결과, seed, prompt/config/checkpoint SHA,
token 수, latency, peak VRAM, fold/group이 들어간다. v2 run manifest는 별도로 source
manifest/tree SHA, config-file byte SHA, preflight/smoke byte SHA, GPU device name, seed/prompt
sequence digest, latency summary를 결속한다. 하나라도 나중에 재검증되지 않으면 QLoRA 입력으로
거부한다. 실제 generation을 얻은 직후 parser golden regression corpus를 추가하고 전체
Ruff/pytest를 다시 통과시킨다.

두 output이 atomic publish된 뒤에는 GPU를 다시 쓰지 않고 먼저 private parser audit을
실행한다. 이 명령은 raw completion이나 train-derived ID/answer를 stdout이나 새 artifact에
복사하지 않는다. stale checksum, non-development partition, stored parser/result mismatch는
fail-closed이며 output을 만들지 않는다.

```bash
PARSER_GOLDEN="artifacts/analysis/parser-golden-$RUN_TAG-fold$FOLD.json"
uv run deep-challenge audit-parser-golden \
  --records "$RUN_DIR/base-direct-predictions.jsonl" \
  --manifest "$RUN_DIR/base-direct-manifest.json" \
  --output "$PARSER_GOLDEN"
```

parser source를 바꾼 뒤 기존 immutable bundle의 영향만 확인할 때는 다음 diagnostic을 쓴다.
이 명령은 raw/ID/answer를 새 artifact에 쓰지 않고 stored/current aggregate만 비교하며,
결과 자체에 `selection_eligible=false`를 기록한다.

```bash
uv run deep-challenge audit-parser-rescore \
  --records "$RUN_DIR/base-direct-predictions.jsonl" \
  --manifest "$RUN_DIR/base-direct-manifest.json" \
  --output "artifacts/analysis/parser-rescore-$RUN_TAG-fold$FOLD-base-v1.json"
```

parser v2의 기존 base rescore 1,653/2,942(56.1863%), invalid 234, conflict 3 자체는
GPU generation을 재사용한 구현 진단이다. 다만 위 `20260810T234907KST` 새 generation이
같은 aggregate를 별도 atomic bundle로 재현했으므로 current base 재실행 조건은 완료됐다.
selection에는 rescore 파일이 아니라 새 v2 bundle만 사용한다.

aggregate의 source/status/reason code에 새 구조가 있으면 공개 test에는 그 구조를
재현하는 안전한 synthetic completion만 추가한다. 실제 question, ID, answer, raw completion,
completion hash는 `artifacts/` 밖으로 옮기거나 Git에 stage하지 않는다. 이 regression과
full CPU test가 green이 되기 전에는 QLoRA를 시작하지 않는다.

새 source의 CLI는 25 generation마다 question/answer/raw completion을 출력하지 않는
`gate_b_development_progress` JSON status line도 보낸다. 이 status는 장시간 run의 liveness
확인용일 뿐 selection evidence가 아니며, 최종 JSONL과 manifest가 함께 atomic publish되기
전에는 run을 완료로 판정하지 않는다.

## 7. Gate B2 — organizer-only direct-answer QLoRA

같은 fold의 base manifest가 성공하기 전에는 학습 명령이 거부된다. fold 0 학습은
정확히 11,794개 training ID만 사용하고 holdout, validation 2,942행, hard-expanded
exclusion을 제외한다.

`20260810T192204KST` 첫 시도는 738/738 optimizer step, train runtime 5,375.9894초,
train loss 0.4807161241까지 완료했지만 post-training tokenizer byte gate에서
`saved tokenizer.json differs from the pinned snapshot`으로 exit 2가 났다. Transformers
4.57.6의 `save_pretrained()`가 BPE JSON에 `ignore_merges=false`를 추가하고 chat template를
별도 파일로 외부화한 것이 원인이었다. incomplete adapter와 temporary training directory는
publish되지 않고 정리됐다.

현재 runtime은 tokenizer를 재직렬화하지 않는다. pinned revision cache에서
`tokenizer.json` SHA `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`와
`tokenizer_config.json` SHA
`5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`를 확인한 뒤 exact
bytes만 export한다. 실제 cache copy/reload에서 vocab, encode, chat-template token sequence
동일성을 확인했고 full CPU suite도 통과했다. 실제 재시도 `20260810T210605KST`는
738/738 step, runtime 5,171.3711초, loss 0.4800419로 성공했고 adapter bundle의 504 tensor,
shape/dtype, exact tokenizer와 모든 provenance를 검증했다.

같은 fold generation은 627/2,942(21.3120%)로 base 1,210/2,942(41.1285%)보다
-19.8165%p 낮았다. paired cluster-bootstrap 95% CI는 [-21.9240, -17.7831]%p이고
Holm-adjusted exact McNemar p는 `3.3053653375511645e-72`다. 따라서 아래 명령은 재현용으로
보존하되 이 exact answer-only candidate를 fold 1--4에서 다시 실행하지 않는다.

```bash
ADAPTER_DIR="$CHECKPOINT_DIR/adapter-direct"

"$GPU_CLI" gate-b-train-fold \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --base-baseline-manifest "$RUN_DIR/base-direct-manifest.json" \
  --output-dir "$ADAPTER_DIR" \
  --acknowledge-gpu-use

"$GPU_CLI" gate-b-development \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --adapter "$ADAPTER_DIR" \
  --base-baseline-manifest "$RUN_DIR/base-direct-manifest.json" \
  --acknowledge-gpu-use \
  --output-jsonl "$RUN_DIR/adapter-direct-predictions.jsonl" \
  --output-manifest "$RUN_DIR/adapter-direct-manifest.json"
```

adapter는 exact 36-layer Qwen LoRA tensor 504개, shape/dtype, tokenizer bytes와 semantic
contract, split/fold/exclusion/training·validation payload SHA, preflight/smoke SHA와 source
manifest/tree SHA를 검증한 뒤 directory rename으로 publish한다. 불완전 shard나 다른 tokenizer는
거부한다.

concise rationale는 3.1절의 corpus/audit/preflight v4가 green이고, 새 source manifest/B0
pair가 green일 때만 direct-answer와 독립 실험으로 추가한다. 현재 direct-answer artifact를
rationale 결과로 덮어쓰지 않는다. production teacher corpus가 아직 없으므로 아래 명령은
**다음 실행 계약**이며 실행 완료 기록이 아니다.

```bash
BASE_RUN_DIR="$PROJECT/artifacts/gate_b/20260810T234907KST/fold-0"
RATIONALE_ADAPTER_DIR="$CHECKPOINT_DIR/adapter-concise-rationale-v1"

"$GPU_CLI" gate-b-train-fold \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --base-baseline-manifest "$BASE_RUN_DIR/base-direct-manifest.json" \
  --rationale-corpus "$RATIONALE_ROOT/rationales.jsonl" \
  --rationale-manifest "$RATIONALE_ROOT/manifest.json" \
  --rationale-audit "$RATIONALE_ROOT/audit.json" \
  --rationale-config "$RATIONALE_CONFIG" \
  --output-dir "$RATIONALE_ADAPTER_DIR" \
  --acknowledge-gpu-use

"$GPU_CLI" gate-b-development \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --source-root "$PROJECT" \
  --source-manifest "$SOURCE_MANIFEST" \
  --adapter "$RATIONALE_ADAPTER_DIR" \
  --base-baseline-manifest "$BASE_RUN_DIR/base-direct-manifest.json" \
  --acknowledge-gpu-use \
  --output-jsonl "$RUN_DIR/adapter-rationale-predictions.jsonl" \
  --output-manifest "$RUN_DIR/adapter-rationale-manifest.json"

uv run deep-challenge compare-development \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold "$FOLD" \
  --reference base-current \
    "$BASE_RUN_DIR/base-direct-predictions.jsonl" \
    "$BASE_RUN_DIR/base-direct-manifest.json" \
  --candidate qlora-concise-rationale-v1 \
    "$RUN_DIR/adapter-rationale-predictions.jsonl" \
    "$RUN_DIR/adapter-rationale-manifest.json" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260731 \
  --confidence 0.95 \
  --alpha 0.05 \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --output "$RUN_DIR/base-vs-rationale-probe.json"

uv run deep-challenge decide-candidate-probe \
  --comparison-artifact "$RUN_DIR/base-vs-rationale-probe.json" \
  --candidate-label qlora-concise-rationale-v1 \
  --output "artifacts/analysis/gate-b-candidate-probe-decision-$RUN_TAG-fold$FOLD-rationale-v1.json"
```

adapter manifest schema v4가 candidate config byte/semantic SHA와 corpus records/manifest/audit
SHA를 모두 다시 결속해야 한다. generation은 기존 direct-answer inference route를 그대로
사용한다. `candidate_full_oof_authorized=true`일 때만 folds 1--4에 fold별로 독립 teacher
corpus/audit/adapter/run을 만들고 complete OOF로 간다.

아래 harm-screen 명령은 이미 중단된 answer-only candidate의 역사적 재현 형식이다. 새
rationale candidate에는 바로 위의 별도 label/output 명령을 사용한다.

```bash
uv run deep-challenge decide-candidate-probe \
  --comparison-artifact "$RUN_DIR/base-vs-adapter-probe.json" \
  --candidate-label qlora-direct \
  --output "artifacts/analysis/gate-b-candidate-probe-decision-$RUN_TAG-fold$FOLD-v1.json"
```

각 candidate의 `candidate_full_oof_authorized=true`와 전체 CPU 회귀가 모두 green일 때만
`FOLD=1`, `2`, `3`, `4`로 바꾸어 fold별 base/corpus/training/generation을 반복한다.
`false`이면 그 exact candidate는 중단하고 새 versioned candidate부터 다시 시작한다. 이
artifact는 비용 통제 전용이며
`final_selection_eligible=false`, `complete_oof_required_before_freeze=true`다. 각 fold에서도
반드시 base run을 먼저 만들고 그 fold의 base manifest로 학습을 승인한다. fold별
`RUN_DIR`, `CHECKPOINT_DIR`, `ADAPTER_DIR`를 새 fold 값으로 다시 계산하며 공유하거나
덮어쓰지 않는다. JSONL과 작은 evidence는 project의 `artifacts/gate_b`에 두지만, Trainer
work directory와 adapter/checkpoint는 여유 공간이 큰 WSL ext4의 `CHECKPOINT_ROOT`에
둔다.

## 8. 비교, Holm 보정, freeze

```bash
COMPARE="$RUN_ROOT/development-oof-comparison.json"
SOURCE_MANIFEST="$RUN_ROOT/source-manifest.json"
FREEZE="$RUN_ROOT/frozen-selection.json"

uv run deep-challenge compare-development-oof \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --deployment-fold 0 \
  --reference-label base-direct \
  --candidate-label qlora-concise-rationale-v1 \
  --base-run 0 \
    "$BASE_RUN_DIR/base-direct-predictions.jsonl" \
    "$BASE_RUN_DIR/base-direct-manifest.json" \
  --adapter-run 0 qlora-concise-rationale-v1 \
    "$RUN_ROOT/fold-0/adapter-rationale-predictions.jsonl" \
    "$RUN_ROOT/fold-0/adapter-rationale-manifest.json" \
    "$CHECKPOINT_ROOT/fold-0/adapter-concise-rationale-v1" \
  --base-run 1 \
    "$RUN_ROOT/fold-1/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-1/base-direct-manifest.json" \
  --adapter-run 1 qlora-concise-rationale-v1 \
    "$RUN_ROOT/fold-1/adapter-rationale-predictions.jsonl" \
    "$RUN_ROOT/fold-1/adapter-rationale-manifest.json" \
    "$CHECKPOINT_ROOT/fold-1/adapter-concise-rationale-v1" \
  --base-run 2 \
    "$RUN_ROOT/fold-2/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-2/base-direct-manifest.json" \
  --adapter-run 2 qlora-concise-rationale-v1 \
    "$RUN_ROOT/fold-2/adapter-rationale-predictions.jsonl" \
    "$RUN_ROOT/fold-2/adapter-rationale-manifest.json" \
    "$CHECKPOINT_ROOT/fold-2/adapter-concise-rationale-v1" \
  --base-run 3 \
    "$RUN_ROOT/fold-3/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-3/base-direct-manifest.json" \
  --adapter-run 3 qlora-concise-rationale-v1 \
    "$RUN_ROOT/fold-3/adapter-rationale-predictions.jsonl" \
    "$RUN_ROOT/fold-3/adapter-rationale-manifest.json" \
    "$CHECKPOINT_ROOT/fold-3/adapter-concise-rationale-v1" \
  --base-run 4 \
    "$RUN_ROOT/fold-4/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-4/base-direct-manifest.json" \
  --adapter-run 4 qlora-concise-rationale-v1 \
    "$RUN_ROOT/fold-4/adapter-rationale-predictions.jsonl" \
    "$RUN_ROOT/fold-4/adapter-rationale-manifest.json" \
    "$CHECKPOINT_ROOT/fold-4/adapter-concise-rationale-v1" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260731 \
  --confidence 0.95 \
  --alpha 0.05 \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --development-shard "$DEV_SHARD" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --output "$COMPARE"

# OOF artifact의 accuracy, paired CI, exact McNemar, Holm 결과를 먼저 확인한다.
jq '{comparisons, statistics, runs: (.runs | with_entries(.value |= {oof_exact_match_accuracy, deployment_fold, checkpoint_sha256}))}' \
  "$COMPARE"

# 위 evidence와 사전 승격 기준으로 아래 두 값 중 하나만 명시적으로 선택한다.
# qlora-concise-rationale-v1 또는 base-direct 외의 값은 이후 명령을 시작하지 못한다.
PRIMARY_LABEL=replace-after-reading-oof-evidence
case "$PRIMARY_LABEL" in
  qlora-concise-rationale-v1)
    PRIMARY_KIND=adapter
    PRIMARY_ADAPTER="$CHECKPOINT_ROOT/fold-0/adapter-concise-rationale-v1"
    ;;
  base-direct)
    PRIMARY_KIND=base
    PRIMARY_ADAPTER=
    ;;
  *)
    echo "PRIMARY_LABEL must be qlora-concise-rationale-v1 or base-direct" >&2
    exit 2
    ;;
esac

PRIMARY_BACKEND_ARGS=(--primary-kind "$PRIMARY_KIND")
if [[ "$PRIMARY_KIND" == adapter ]]; then
  PRIMARY_BACKEND_ARGS+=(--primary-adapter "$PRIMARY_ADAPTER")
fi

uv run deep-challenge source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"

uv run deep-challenge freeze-development-selection \
  --comparison-artifact "$COMPARE" \
  --primary-label "$PRIMARY_LABEL" \
  --decision-note "Selected $PRIMARY_LABEL from complete OOF paired evidence; deployment fold 0 was predeclared; no leaderboard score used" \
  --source-manifest "$SOURCE_MANIFEST" \
  --lockfile "$PROJECT/uv.lock" \
  --output "$FREEZE" \
  --confirm-no-leaderboard-selection
```

`PRIMARY_LABEL`은 반드시 위 OOF evidence와 사전 승격 기준에 맞게 고른다. 비교기는 5개
fold의 모든 OOF 예측이
정확히 한 번씩 덮였는지 검증하고, split group mapping을 사용한 paired cluster bootstrap,
exact McNemar, 모든 candidate/policy 비교에 대한 Holm 보정을 한 family로 기록한다.
각 `--base-run`은 pinned base checkpoint만 허용하고, 각 `--adapter-run`은 지정한 실제
adapter bundle의 manifest/checksum, fold, train·validation ID와 example digest, 데이터
provenance를 다시 검증한다. 같은 run/adapter의 label 재사용이나 fold 사이 training-method
fingerprint 혼합도 실패한다. 단일-fold `compare-development` artifact는 probe 전용이며
최종 freeze, holdout, leaderboard/test 경로에 사용할 수 없다.
`deployment-fold 0`은 fold별 checkpoint를 ensemble하지 않고 fold 0의 단일 checkpoint만
holdout/test에 쓰도록 사전 지정한다. full-development refit은 현재 필수 실행 경로가 아니며
그 결과인 것처럼 주장하지 않는다.

same-base multi-checkpoint fallback의 규칙 답변 전 기본은 primary-only다. 서면 허용을
받고 OOF routing evidence까지 승격된 경우에만 freeze에 `--fallback-label base-direct`를
추가한다. leaderboard 점수로 label이나 routing을 바꾸지 않는다.

## 9. locked holdout 1회

freeze 뒤에만 실행한다. 이 명령은
`artifacts/analysis/locked-holdout-access-v1`에 durable claim을 먼저 만든 다음에만 원본
train의 holdout 질문·정답을 load한다. 실패해도 claim은 소비된다. ledger를 삭제·복사하여
재실행하지 않는다.

```bash
RUN_DIR="$RUN_ROOT/fold-0"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/fold-0"

"$GPU_CLI" gate-b-locked-holdout-evaluate \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold 0 \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --freeze-artifact "$FREEZE" \
  "${PRIMARY_BACKEND_ARGS[@]}" \
  --fallback-kind none \
  --output "$RUN_DIR/locked-holdout-evaluation.json" \
  --acknowledge-gpu-use \
  --acknowledge-one-time-locked-holdout
```

primary kind와 adapter 인자는 freeze의 deployment-fold checkpoint와 정확히 맞아야 한다.
서면 허용 후 fallback을 freeze한 경우에만 `--fallback-kind base` 같은 정확한 backend를
준다. holdout 결과를 본 뒤 모델·prompt·routing을 다시 선택하는 것은 금지한다.

## 10. filtered leaderboard 또는 final test 추론

아래 명령은 freeze 이후에만 실행하며 입력 CSV SHA를 명시적으로 잠근다. leaderboard에는
현재 831행 filtered 파일만 사용한다.

```bash
"$GPU_CLI" gate-b-predict-evaluation \
  --train "$TRAIN" \
  --train-exclusions "$EXCLUSIONS" \
  --split-artifact "$SPLIT" \
  --fold 0 \
  --expected-train-sha256 "$TRAIN_SHA" \
  --expected-exclusions-sha256 "$EXCLUSIONS_SHA" \
  --expected-exclusion-count 627 \
  --expected-split-sha256 "$SPLIT_SHA" \
  --expected-development-shard-sha256 "$DEV_SHARD_SHA" \
  --preflight-report "$PREFLIGHT" \
  --gpu-smoke-report "$SMOKE" \
  --config "$CONFIG" \
  --evaluation "$FILTERED_LB" \
  --dataset-role leaderboard \
  --expected-evaluation-sha256 "$FILTERED_LB_SHA" \
  --freeze-artifact "$FREEZE" \
  "${PRIMARY_BACKEND_ARGS[@]}" \
  --fallback-kind none \
  --output-predictions "$RUN_DIR/leaderboard-predictions.json" \
  --output-artifact "$RUN_DIR/leaderboard-inference-manifest.json" \
  --acknowledge-gpu-use
```

primary 전체를 끝내고 close한 뒤 parser-invalid ID에만 fallback을 load하는
single-model residency를 사용한다. prediction JSON을 먼저, manifest를 마지막 commit
marker로 publish한다. 강제 종료로 prediction만 남고 manifest가 없으면 그 쌍은
**미완료 orphan**이다. 자동 삭제·덮어쓰기를 하지 말고 새 versioned 경로로 재실행한다.

invalid answer가 하나라도 남으면 prediction 명령은 non-zero로 끝나며 submission을 쓰지
않는다. 모두 valid일 때만 다음을 실행한다.

```bash
(
  set -euo pipefail
  set -o noclobber

  uv run deep-challenge write-submission \
    --predictions "$RUN_DIR/leaderboard-predictions.json" \
    --expected "$FILTERED_LB" \
    --output "$RUN_DIR/submission.csv" \
    > "$RUN_DIR/submission-write-report.json"

  uv run deep-challenge validate-submission \
    --submission "$RUN_DIR/submission.csv" \
    --expected "$FILTERED_LB" \
    > "$RUN_DIR/submission-validation-report.json"

  uv run deep-challenge verify-submission-independent \
    --submission "$RUN_DIR/submission.csv" \
    --expected "$FILTERED_LB" \
    --expected-sha256 "$FILTERED_LB_SHA" \
    > "$RUN_DIR/submission-independent-report.json"

  sha256sum "$RUN_DIR/submission.csv" > "$RUN_DIR/submission.sha256"

  cat "$RUN_DIR/submission-write-report.json"
  cat "$RUN_DIR/submission-validation-report.json"
  cat "$RUN_DIR/submission-independent-report.json"
  cat "$RUN_DIR/submission.sha256"
)
```

writer 기본 header는 `ID,answer`이고 exact ID/order/row count/canonical integer를
round-trip 검증한다. `verify-submission-independent`는 primary loader/submission 구현을
import하지 않는 독립 CSV parser로 같은 불변식을 교차검증한다. 각 CLI의 JSON stdout과
최종 checksum도 새 evidence 파일로 보존한다. subshell의 `noclobber`와 각 writer의 atomic
no-overwrite가 기존 결과를 보호하며 `--overwrite`는 대회 운영 artifact에 사용하지 않는다.

final test가 공개되면 `FILTERED_LB`, `FILTERED_LB_SHA`, `--dataset-role leaderboard`만
공식 test path, 사전 검증한 SHA, `--dataset-role test`로 바꾼다. test를 외부 API나
학습 입력으로 전달하지 않는다.

## 11. 즉시 중단 조건

- 모델/revision/config/data/split/development-shard SHA 불일치
- GPU free VRAM 10,240MiB 미만 또는 pre-existing used VRAM 2,048MiB 초과
- preflight/smoke가 green이 아님
- tokenizer/model shard/adapter shard가 하나라도 누락되거나 checksum 불일치
- fold training/validation ID 또는 response-only label digest 불일치
- rationale corpus/audit가 exact fold-training coverage, config/teacher/target SHA,
  reference-answer/parser match, training-only/no-tool/no-test/no-holdout 중 하나라도 불충족
- parser conflict/invalid, prediction 누락, submission ID/order/header 오류
- freeze 전에 holdout 접근을 요구하거나, leaderboard 점수로 freeze를 바꾸려는 경우
- 인터넷 차단 상태에서 cache miss/download 시도

이 경우 blocker artifact만 남기고 GPU 학습·평가·제출로 진행하지 않는다.
