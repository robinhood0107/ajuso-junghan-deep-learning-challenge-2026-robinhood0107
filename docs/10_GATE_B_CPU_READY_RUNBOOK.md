# 10. Gate B CPU-ready 실행 런북

기준 시각: **2026-08-10 KST**
대상 호스트: **WSL2 Ubuntu 24.04 + NVIDIA GeForce RTX 4070 SUPER 12GB**
현재 판정: **CPU 준비 완료, 이전 source의 GPU preflight/final synthetic smoke green, eval KV-cache source의 새 B0 refresh 대기**

이 문서는 현재 코드의 CLI와 정확히 일치하는 실행 순서다. GPU가 필요한 명령은 맨
뒤의 별도 절에만 둔다. 2026-08-10 이전 source final smoke는 green artifact를 남겼다.
training cache-off와 eval KV-cache-on을 분리한 현재 source는 새 tagged smoke를 요구한다.
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
RUN_TAG=replace-with-new-unique-tag
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

따라서 Python/SymPy TIR과 same-base multi-adapter 결합은 off다. 첫 실행은 organizer-only,
단일 base와 단일 direct-answer adapter만 비교한다. teacher rationale의 규칙은 허용으로
읽히지만 품질 검증 corpus가 아직 없으므로 첫 baseline에는 넣지 않는다.

## 5. B0 GPU 승인 증거와 새 run 직전 재검사

2026-08-10 target-host preflight와 local synthetic smoke는 이전 source에서 green이다.
현재 eval KV-cache source의 production B1에는 그 artifact를 재사용하지 않고, 이 절의
새 `RUN_TAG` preflight와 smoke를 순서대로 다시 만든다. 다른 GPU 프로세스를 종료하거나
선점하지 않으며, 모든 새 GPU run 직전에는 아래 두 조건을 다시 관측한다.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version \
  --format=csv,noheader,nounits
```

- free VRAM이 10,240MiB 이상
- pre-existing used VRAM이 1,024MiB 이하

2026-08-10 evidence는
`artifacts/analysis/model-preflight-gpu-current-20260810T001500KST.json`와
`artifacts/analysis/gpu-smoke-20260810T001500KST.json`에 있다. GPU 값은 계속 변하므로
snapshot 수치를 현재 실시간 값이라고 주장하지 않고, 실제 실행 여부는 위 `nvidia-smi`의
직전 관측으로 다시 판정한다.

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
확인하고, context 생성 **후** CUDA free VRAM으로 실제 load 가능 여유를 확인한다. context
자체의 driver/WSL overhead를 external occupancy로 바꾸어 해석하지 않지만, 어느 쪽의
immutable threshold도 완화하지 않는다. 두 측정치는 green artifact의 runtime evidence에
함께 남으며, 이후 training/inference gate도 이를 요구한다.

## 6. Gate B1 — base direct-answer development 기준선

GPU green 뒤 첫 모델 실행은 fold 0 base greedy 한 개뿐이다. 아래 `FOLD=0`을 그대로
두고 먼저 실행한다. 2026-08-10에 attention mask와 eval KV-cache 보강 전 시작한 동일 목적의 diagnostic
run은 parser golden 관찰용으로만 취급한다. 그 run은 method selection, QLoRA authorization,
OOF comparison에 재사용하지 않으며, 아래 명령은 보강된 source에서 새 `RUN_TAG`로 다시
실행한다.

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
  --acknowledge-gpu-use \
  --output-jsonl "$RUN_DIR/base-direct-predictions.jsonl" \
  --output-manifest "$RUN_DIR/base-direct-manifest.json"
```

저장되는 각 row에는 raw generation, parser 결과, seed, prompt/config/checkpoint SHA,
token 수, latency, peak VRAM, fold/group이 들어간다. 실제 generation을 얻은 직후 parser
golden regression corpus를 추가하고 전체 Ruff/pytest를 다시 통과시킨다.

새 source의 CLI는 25 generation마다 question/answer/raw completion을 출력하지 않는
`gate_b_development_progress` JSON status line도 보낸다. 이 status는 장시간 run의 liveness
확인용일 뿐 selection evidence가 아니며, 최종 JSONL과 manifest가 함께 atomic publish되기
전에는 run을 완료로 판정하지 않는다.

## 7. Gate B2 — organizer-only direct-answer QLoRA

같은 fold의 base manifest가 성공하기 전에는 학습 명령이 거부된다. fold 0 학습은
정확히 11,794개 training ID만 사용하고 holdout, validation 2,942행, hard-expanded
exclusion을 제외한다.

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
  --adapter "$ADAPTER_DIR" \
  --base-baseline-manifest "$RUN_DIR/base-direct-manifest.json" \
  --acknowledge-gpu-use \
  --output-jsonl "$RUN_DIR/adapter-direct-predictions.jsonl" \
  --output-manifest "$RUN_DIR/adapter-direct-manifest.json"
```

adapter는 exact 36-layer Qwen LoRA tensor 504개, shape/dtype, tokenizer bytes와 semantic
contract, split/fold/exclusion/training·validation payload SHA, preflight/smoke SHA를 검증한
뒤 directory rename으로 publish한다. 불완전 shard나 다른 tokenizer는 거부한다.

concise rationale는 별도 검증 corpus와 별도 immutable config를 만든 뒤 direct-answer와
독립 실험으로만 추가한다. 현재 direct-answer artifact를 rationale 결과로 덮어쓰지 않는다.

fold 0 base generation으로 parser golden corpus/tests를 추가하고 전체 회귀가 다시
green이 된 뒤에만 `FOLD=1`, `2`, `3`, `4`로 바꾸어 6절과 7절을 반복한다. 각 fold에서도
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
  --candidate-label qlora-direct \
  --base-run 0 \
    "$RUN_ROOT/fold-0/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-0/base-direct-manifest.json" \
  --adapter-run 0 qlora-direct \
    "$RUN_ROOT/fold-0/adapter-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-0/adapter-direct-manifest.json" \
    "$CHECKPOINT_ROOT/fold-0/adapter-direct" \
  --base-run 1 \
    "$RUN_ROOT/fold-1/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-1/base-direct-manifest.json" \
  --adapter-run 1 qlora-direct \
    "$RUN_ROOT/fold-1/adapter-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-1/adapter-direct-manifest.json" \
    "$CHECKPOINT_ROOT/fold-1/adapter-direct" \
  --base-run 2 \
    "$RUN_ROOT/fold-2/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-2/base-direct-manifest.json" \
  --adapter-run 2 qlora-direct \
    "$RUN_ROOT/fold-2/adapter-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-2/adapter-direct-manifest.json" \
    "$CHECKPOINT_ROOT/fold-2/adapter-direct" \
  --base-run 3 \
    "$RUN_ROOT/fold-3/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-3/base-direct-manifest.json" \
  --adapter-run 3 qlora-direct \
    "$RUN_ROOT/fold-3/adapter-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-3/adapter-direct-manifest.json" \
    "$CHECKPOINT_ROOT/fold-3/adapter-direct" \
  --base-run 4 \
    "$RUN_ROOT/fold-4/base-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-4/base-direct-manifest.json" \
  --adapter-run 4 qlora-direct \
    "$RUN_ROOT/fold-4/adapter-direct-predictions.jsonl" \
    "$RUN_ROOT/fold-4/adapter-direct-manifest.json" \
    "$CHECKPOINT_ROOT/fold-4/adapter-direct" \
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
# qlora-direct 또는 base-direct 외의 값은 이후 명령을 시작하지 못한다.
PRIMARY_LABEL=replace-after-reading-oof-evidence
case "$PRIMARY_LABEL" in
  qlora-direct)
    PRIMARY_KIND=adapter
    PRIMARY_ADAPTER="$CHECKPOINT_ROOT/fold-0/adapter-direct"
    ;;
  base-direct)
    PRIMARY_KIND=base
    PRIMARY_ADAPTER=
    ;;
  *)
    echo "PRIMARY_LABEL must be qlora-direct or base-direct" >&2
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
- GPU free VRAM 10,240MiB 미만 또는 pre-existing used VRAM 1,024MiB 초과
- preflight/smoke가 green이 아님
- tokenizer/model shard/adapter shard가 하나라도 누락되거나 checksum 불일치
- fold training/validation ID 또는 response-only label digest 불일치
- parser conflict/invalid, prediction 누락, submission ID/order/header 오류
- freeze 전에 holdout 접근을 요구하거나, leaderboard 점수로 freeze를 바꾸려는 경우
- 인터넷 차단 상태에서 cache miss/download 시도

이 경우 blocker artifact만 남기고 GPU 학습·평가·제출로 진행하지 않는다.
