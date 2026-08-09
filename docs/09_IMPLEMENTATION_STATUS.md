# 09. 선행 구현 및 검증 상태

기준 시각: **2026-08-10 KST**
구현 루트: local `$PROJECT` (public repository에서는 환경 변수로 설정)
공식 데이터 루트: `$PROJECT/deep-learning-challenge-2026` (Git 제외)

## 2026-08-10 권위 상태

이 절이 현재 실행 상태의 권위 원본이다. 아래의 “2026-07-31 역사적 snapshot”은
정의 drift와 과거 근거를 보존하기 위한 부록이며 현재 명령·데이터·blocker를 대체하지
않는다.

### 판정

| Gate | 상태 | 근거 |
|---|---|---|
| Gate A model-free | **READY** | strict loader, filtered audit, split v4, group-safe exclusion, parser/evaluator/voting/submission/provenance 구현과 전체 회귀 |
| Gate B0 규칙·데이터 | **READY (non-submission)** | authenticated Kaggle API로 참가 slug, Rules/Data/Evaluation, 4개 data file, `ID,answer`, 현재 allowance 5를 hash와 함께 확인; sample CSV는 현재 file listing에 없음 |
| Gate B0 모델 파일 | **READY** | pinned tokenizer, index, 두 shard의 exact size/SHA/commit 확인 |
| Gate B0 runtime packages | **CPU-hidden READY** | ext4 전용 환경에서 Torch/Transformers/Accelerate/PEFT/bitsandbytes/Triton 격리 import 성공 |
| Gate B0 CPU/code readiness | **READY** | sealed development shard, real tokenizer SFT preflight, GPU runtime/training/inference/OOF selection/one-shot holdout/submission 경로 구현·회귀검증 |
| Gate B0 GPU preflight | **READY (pre-smoke)** | 2026-08-09 pinned revision, full weights, CUDA/BF16/NF4 runtime, physical snapshot `used 1,001MiB/free 10,997MiB` 모두 green |
| Gate B0 final GPU smoke | **READY** | `gpu-smoke-20260810T001500KST.json` green: local `2+3` only, pinned NF4 load, LoRA backward/`paged_adamw_8bit` 1 step, generation/parser exact 5; competition data 0행 |
| Gate B1 base direct-answer | **미실행** | 실제 GPU generation 0건, 모델 점수 없음 |
| Gate B2 QLoRA SFT | **미실행** | GPU smoke green 전 학습 금지 |
| leaderboard prediction/submission | **미실행** | 모델 prediction 0건; leaderboard를 학습/API 입력에 쓰지 않음 |

### 실제 환경

| 항목 | 현재 값 |
|---|---|
| WSL | Ubuntu 24.04.4, WSL 2.7.11.0 |
| 기본 개발 환경 | CPython 3.12.3, uv 0.11.3, `.venv` |
| GPU 전용 환경 | local ext4 `$GPU_ENV` (public repository에서 경로 제외) |
| GPU runtime | Torch 2.13.0, Transformers 4.57.6, Accelerate 1.14.0, PEFT 0.20.0, bitsandbytes 0.50.0, Triton 3.7.1 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282MiB, compute capability 8.9 |
| current GPU evidence | smoke 직전 physical used 864MiB/free 11,134MiB; smoke peak reserved 4,661,968,896 bytes, cleanup 후 physical free 8,305,770,496 bytes. 값은 변동 가능하며 다른 프로세스는 중단하지 않음 |
| WSL memory | RAM 18,857,226,240 bytes, available 15,645,212,672; swap free 6,354,501,632 bytes |
| 저장공간 | Windows C: 58,220,236,800 bytes free, WSL ext4 972,765,863,936 bytes free |
| model cache | local `$MODEL_CACHE/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1` (Git 제외) |

weights/checkpoint/전용 venv는 C:가 아니라 WSL ext4에 둔다. GPU 전용 환경 설치와
import 검사는 `CUDA_VISIBLE_DEVICES=''`에서 수행했다. 2026-08-09 첫 smoke attempt는
CUDA context 뒤 model load 전에 fail-closed로 중단됐지만, guard 보강 뒤 2026-08-10
새 preflight와 local synthetic smoke는 green이다. 이 smoke 이외에 organizer train,
leaderboard/test 입력 또는 실제 development generation은 아직 사용하지 않았다.

### 최신 데이터와 split overlay

| 입력 | 행/ID | SHA-256 |
|---|---:|---|
| train | 17,000 | `e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a` |
| filtered leaderboard | 831 | `032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1` |
| train exclusions | 627 | `67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3` |
| downloaded ZIP | 4 files | `b51d46a2f8a6e5b81344cf884d5c239159b163a3093d984e4b505427fa1316b3` |

- split v4 SHA는 `be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db`이며 재생성하지 않았다.
- organizer 627 exclusions를 기존 hard group으로 확장하면 629행이다. soft
  number-masked template는 hard group으로 쓰지 않았다.
- 전체 eligible 16,371행, development CV 14,736행이다.
- fold training은 11,778~11,794행, fold validation은 2,942~2,958행이다.
- 새 audit/profile은 `cv-only`이고 locked holdout의 질문·정답 통계를 산출하지 않는다.
  과거 split v4가 holdout 분포를 포함한 점은 역사적 artifact 한계로 남기고 모델 선택에
  재사용하지 않는다.

### 새로 구현한 Gate B CPU 계층

- `data.py`: filtered leaderboard `id,question`, organizer exclusion strict loader
- `splits.py`: strict split deserialize/tamper 검증, hard-group exclusion,
  `eligible_training_ids`/`eligible_validation_ids`
- `development_shard.py`: split v4의 CV 15,300행만 atomic no-overwrite bundle로 publish;
  이후 development 명령은 원본 train 내용 대신 이 shard만 load
- `cli.py`: current-contract overlay, CV-only audit/profile, uppercase submission default,
  explicit final `gpu-smoke`, complete OOF 비교와 independent submission 교차검증
- `gate_b.py`: split-bound SFT/dev records, response-only masking, full config SHA,
  structured generation result, parser conflict 보존, atomic no-overwrite JSONL/manifest
- `model_preflight.py`: exact pinned shard contract, isolated import/ABI probe, physical VRAM
  gate, prerequisite-ready와 execution-ready 분리
- `gpu_smoke.py`: 고정 synthetic 2+3 prompt만 쓰는 lazy NF4 load + LoRA backward +
  generation final gate. external occupancy는 CUDA context 전 read-only `nvidia-smi`로,
  usable capacity는 context 후 CUDA free-VRAM으로 각각 검증·artifact화한다.
- `gate_b_sft_preflight.py`: pinned tokenizer로 실제 fold payload를 response-only encode하고
  truncation/holdout·leaderboard 미사용을 증명
- `gate_b_runtime.py`: base/adapter lazy NF4 generation, exact fold QLoRA training,
  adapter manifest와 36-layer×7-projection×A/B=504 safetensors shape/dtype 검증
- `gate_b_selection.py`: single-fold probe와 complete 5-fold OOF union 비교, paired
  duplicate-cluster bootstrap, exact McNemar, 한 family Holm; final freeze는 complete OOF만
  허용하고 base/adapter bundle·fold/data/example SHA·공통 method fingerprint를 결속
- `gate_b_holdout.py`: durable claim/receipt 뒤에만 원 train Q/A를 여는 one-shot frozen-policy
  평가; claim 이후 실패도 소비
- `gate_b_prediction.py`: evaluation CSV 내부 strict load/SHA binding, primary 전체 처리 후
  close하고 invalid ID에만 fallback을 lazy load, manifest-last commit marker
- `independent_submission.py`: production data/submission 모듈과 분리된 두 번째 CSV parser
- submission 기본은 `ID,answer`; 누락/invalid는 실패하고 silent zero fallback이 없다.

현재 12GB direct-answer config는
`configs/gate_b/rtx4070-super-12gb-direct-answer-v1.json`이다. runtime이 결속하는
semantic config SHA는 `4530c14a4782c439ea3a8325b90d997793eda368b0371d765cb810690bb40028`,
파일 바이트 SHA는 `703926d84ec6c7a95f7ce50de384fb5dcb1bb35d98cd52cbb6ab846f980d83c3`다.

### 현재 canonical 분석 artifact

- `artifacts/analysis/data-contract-20260803-v1.json`
- `artifacts/analysis/data-audit-v4-filtered.json`
- `artifacts/analysis/splits-v4.json`
- `artifacts/analysis/train-eligibility-v1-20260804.json`
- `artifacts/analysis/tokenizer-profile-v4-filtered.json`
- `artifacts/analysis/development-cv-v4/`
- `artifacts/analysis/gate-b-sft-encoding-preflight-v3-fold0-20260804.json`
- `artifacts/analysis/model-preflight-current.json`
- `artifacts/analysis/model-preflight-gpu-runtime-cpu-hidden-v2-20260804.json`
- `artifacts/analysis/model-preflight-gpu-current-20260809.json`
- `artifacts/analysis/gpu-smoke-attempt-20260809.json`
- `artifacts/analysis/host-readiness-cpu-only-20260804.json`
- `artifacts/analysis/kaggle-b0-20260804/browser-access-evidence-v1.json`
- `artifacts/analysis/kaggle-b0-20260804/rules-decision-manifest-v1.json`
- `artifacts/analysis/kaggle-b0-20260809/browser-access-evidence-v2.json`
- `artifacts/analysis/kaggle-b0-20260810/authenticated-api-snapshot-v1.json`
- `artifacts/analysis/model-preflight-gpu-current-20260810T001500KST.json`
- `artifacts/analysis/gpu-smoke-20260810T001500KST.json`
- `artifacts/analysis/source-manifest-final-v4.json` (2026-08-09 source edits 뒤 재생성 예정)
- `artifacts/analysis/CHECKSUMS.sha256`

`model-preflight-current.json`은 기본 개발 환경의 의도적 package 부재까지 기록한다.
`model-preflight-gpu-runtime-cpu-hidden-v2-20260804.json`은 pinned files와 모든 GPU package import가
green이지만 CUDA를 숨겼고 물리 free VRAM이 부족해 `training_ready=false`,
`execution_ready=false`다.

`model-preflight-gpu-current-20260810T001500KST.json`은 실제 target GPU에서
`training_ready=true`와 blockers 없음까지 확인했고, 바로 뒤의
`gpu-smoke-20260810T001500KST.json`이 이를 실제 NF4 kernel/model-load와 bounded
LoRA optimizer step으로 확증했다. smoke raw generation은 local prompt의
`Final answer: 5`이며 parser exact match가 통과했다. 이 결과는 B1의 organizer-only
development baseline을 허용하지만 leaderboard/test prediction을 허용하는 근거는 아니다.

CPU-only SFT preflight v3는 fold 0 training 11,794행, validation 2,942행, union
14,736행을 실제 pinned tokenizer로 encode했다. 최대 sequence는 1,127/2,048,
초과 0, 최소 response label 7 token이며 `torch_or_cuda_used=false`,
`model_weights_loaded=false`, `locked_holdout_accessed=false`,
`leaderboard_or_test_used=false`다.

### 최신 전체 회귀

- `uv run ruff check .`: **pass**
- `CUDA_VISIBLE_DEVICES='' ... uv run pytest -s -q`: **330 passed, 1 skipped**
- branch coverage: **79%** (6,518 statements, 2,284 branches)
- skip 1건은 기본 CPU `.venv`에 PyTorch가 없어 실제 PEFT 0.20 serialization
  compatibility test를 건너뛴 것이다. 같은 구조의 real safetensors 파일에 대한
  extra tensor/wrong shape/wrong dtype/incomplete index 음성 회귀는 CPU에서 통과했다.
- CPU 회귀는 CUDA tensor/model load를 만들지 않았다. 별도의 2026-08-09 bounded smoke
  attempt는 CUDA context까지만 도달했고 model load, generation, backward, optimizer step은
  guard 전에 실행되지 않았다.

### 현재 재현 명령

```bash
PROJECT=/absolute/path/to/deepleaning
cd "$PROJECT"

DATA_DIR="$PROJECT/deep-learning-challenge-2026"
REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1

uv sync --extra model --group dev
uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q

uv run deep-challenge build-eligibility-overlay \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --train-exclusions "$DATA_DIR/train_filtered_ids.csv" \
  --split-artifact artifacts/analysis/splits-v4.json \
  --expected-train-sha256 e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a \
  --expected-exclusions-sha256 67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3 \
  --expected-exclusion-count 627 \
  --expected-split-sha256 be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db \
  --output artifacts/analysis/train-eligibility-v1-20260804.json

uv run deep-challenge audit-data \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --train-exclusions "$DATA_DIR/train_filtered_ids.csv" \
  --split-artifact artifacts/analysis/splits-v4.json \
  --train-scope cv-only \
  --leaderboard "$DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
  --source-root . \
  --output artifacts/analysis/data-audit-v4-filtered.json

uv run deep-challenge profile-tokenizer \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --train-exclusions "$DATA_DIR/train_filtered_ids.csv" \
  --split-artifact artifacts/analysis/splits-v4.json \
  --train-scope cv-only \
  --leaderboard "$DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
  --revision "$REVISION" \
  --output artifacts/analysis/tokenizer-profile-v4-filtered.json

# 현재는 blocker artifact와 exit 1이 정상이다. GPU workload는 만들지 않는다.
CUDA_VISIBLE_DEVICES='' uv run deep-challenge model-preflight \
  --revision "$REVISION" \
  --output artifacts/analysis/model-preflight-current.json

cd artifacts/analysis
sha256sum -c CHECKSUMS.sha256
```

GPU를 실제 쓰는 `gpu-smoke`, base generation, QLoRA, leaderboard prediction은 이 명령
묶음에 포함하지 않는다. 실행 순서와 승인 조건은
`docs/10_GATE_B_CPU_READY_RUNBOOK.md`에 고정한다.

## 부록: 2026-07-31 역사적 snapshot

아래 내용의 1,000행 raw leaderboard, GPU/weights 부재, 과거 local path,
소문자 submission 예시는 당시 상태를 설명한다. 현재 실행에는 위 권위 상태만 사용한다.

## 1. 결론

계획만 작성한 상태를 넘어, 모델 가중치 없이 안전하게 완성할 수 있는 대회 기반을 실제로 구현하고 원본 17,000/1,000행에서 실행했다.

- strict CSV loader와 content manifest
- 품질 flag와 중복/오염 audit v3
- 누수 방지 split v4와 locked holdout API
- pinned Qwen tokenizer profiler
- fixed-base/model-cache/GPU/QLoRA preflight
- 정수 answer parser
- exact-match 평가, grouped bootstrap, exact McNemar, Holm correction
- deterministic candidate seed, adaptive budget, provenance-aware voting
- 원자적 submission writer와 strict validator
- source-tree manifest와 CLI

현재 상태는 **model-free Gate A 완료, 실제 QLoRA 학습 Gate B 차단**이다. 차단 사유는 코드 실패가 아니라 전체 pinned weights, PyTorch, CUDA BF16 GPU, `accelerate`/`peft`/`bitsandbytes`, 최신 대회 규칙 확정이 없기 때문이다. 따라서 모델 학습이나 leaderboard 답 생성을 했다고 주장하지 않는다.

## 2. 실행 환경

| 항목 | 실제 상태 |
|---|---|
| Python | uv-managed CPython 3.12.13 |
| uv | 0.11.26 |
| Transformers | 4.57.6 |
| tokenizers | 0.22.2 |
| NumPy | 2.5.1 |
| PyTorch | 미설치 |
| CUDA GPU | 없음 |
| Qwen tokenizer/config | pinned local cache 있음 |
| Qwen weight shards | local cache 없음 |
| 작업 트리 | Git 저장소가 아니므로 source-tree SHA manifest 사용 |

설치는 다음으로 재현한다.

```bash
cd /absolute/path/to/deepleaning
uv sync --extra model --group dev
```

`uv.lock`과 `.python-version`을 함께 보존한다. CUDA용 PyTorch와 QLoRA 패키지는 실제 GPU host/CUDA 조합을 확정하기 전에 현재 CPU workspace에 임의 설치하지 않았다.

## 3. 구현 지도

| 모듈 | 구현 계약 |
|---|---|
| `data.py` | RFC4180 strict parsing, UTF-8, 원문 보존, train canonical integer, 실제 malformed leaderboard header/2-field row의 제한적 수용, SHA-256 |
| `quality.py` | 비파괴 quality flag, math-aware/source-format fingerprint, sign-preserving number-masked soft template |
| `audit.py` | 분포, answer 통계, quality flag, train 내부/교차 중복, 방법 version과 source manifest |
| `splits.py` | transitive exact cluster, stable hash holdout, balanced whole-group K-fold, `training_ids(fold)`의 holdout 제외, actual counts |
| `answers.py` | marker precedence, balanced box, exact integer/fraction/integral decimal, conflict/invalid 분리, `eval` 금지 |
| `evaluation.py` | canonical exact accuracy, exact McNemar, duplicate-cluster bootstrap, Holm family-wise correction |
| `inference.py` | ID 기반 deterministic seed, adaptive budget, exact rational vote weight, tie 보존, generation provenance integrity |
| `submission.py` | exact schema/ID/order/width/canonical integer, no silent fallback, atomic no-replace publish, directory fsync, output digest |
| `tokenizer_profile.py` | 공식 model/immutable commit 강제, tokenizer 1회 load, train/leaderboard 동일 provenance, mixed snapshot 거부 |
| `model_preflight.py` | 공식 base 강제, index의 모든 shard 확인, snapshot consistency, NF4 QLoRA/BF16 readiness gate |
| `provenance.py` | deterministic source manifest, self-output 제외, symlink·swap 방어, atomic JSON |
| `cli.py` | audit/split/tokenizer/preflight/parse/submission/source manifest 명령 |

## 4. 실제 데이터 audit v3

### 4.1 원본 잠금

| 파일 | 행 | bytes | SHA-256 |
|---|---:|---:|---|
| `deep_chal_math_train.csv` | 17,000 | 4,335,431 | `e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a` |
| `deep_chal_math_leaderboard.csv` | 1,000 | 254,899 | `18cae340803dd649ce162a575dcda01bb75bbcf0759f80392d692603951ccd32` |

train은 정확히 `id,question,answer`다. leaderboard 원본 header는 `id,question, answer`이고 data row는 실제 두 field다. Loader는 이 확인된 결함만 좁게 보정하며 임의의 열 손실은 허용하지 않는다.

### 4.2 answer와 중복

- unique answer 2,023개
- 최소 `-5,765,435`, 최대 `3,431,577,212,128,939`
- 음수 502, 0은 222, 양수 16,276
- 최빈 answer는 `2`, 646회
- math-aware train 중복: 4그룹/8행
- source-format train 중복: 8그룹/16행
- sign-preserving number-masked template: 141그룹/322행, 최대 10행 — soft 후보 전용
- source-format train↔leaderboard 직접 중복: 정확히 3 fingerprint/leaderboard 3행
- number-masked train↔leaderboard overlap: 18행 — 의미 동일성은 보장하지 않음

직접 중복으로 확인된 세 쌍의 개별 ID·원문은 대회 데이터 파생물이므로 public repository에
넣지 않는다. leaderboard 답을 생성하거나 이 쌍을 학습 데이터로 되먹이지 않았으며,
private contamination artifact의 aggregate slice로만 사용한다.

### 4.3 정의 drift를 숨기지 않는 정책

초기 exploratory 분석의 공격적 정규화는 13개 후보 그룹을 보고했고, 영구 audit v1/v2의 중간 값도 달랐다. 최종 v3에서는 다음 버그를 고쳤다.

- 단순 `replace("\\left", "")`가 `\\leftarrow`, `\\leftrightarrow` 등을 손상시키던 문제
- number masking이 `-1`과 `1`의 부호까지 지우던 문제
- soft number-masked template를 hard split cluster로 사용하던 문제

실데이터에서 TeX boundary 수정은 48행, sign-preserving template은 1,075행의 fingerprint를 바꿨다. 과거 수치를 맞추려고 새 함수를 왜곡하지 않고 audit version과 차이를 남겼다.

## 5. split v4

### 5.1 hard/soft 신호

hard union:

- 숫자·연산자를 보존하는 math-aware exact fingerprint
- 검증된 좁은 source-format exact fingerprint

soft candidate only:

- number-masked template
- fuzzy/TF-IDF/embedding similarity

soft pair는 수동 또는 symbolic adjudication 없이 cluster를 합치지 않는다.

### 5.2 실제 manifest

| 항목 | 값 |
|---|---|
| version | `v4` |
| algorithm | `locked-holdout-balanced-group-kfold-v2` |
| seed | `20260731` |
| cluster | 16,992 |
| locked holdout | 1,700행 / 1,699그룹 |
| CV | 15,300행 / 15,293그룹 |
| fold 0..4 | 각각 3,060행 |
| source-group SHA | `b965b7ef53e704fe00fbd3f3ff6fc7b9f6696ff89a866391a432901c9392e5af` |
| split SHA | `be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db` |

v3의 size-first holdout selector가 8개 multi-row group 전부를 holdout으로 몰아넣는 현상을 독립 리뷰에서 발견했다. v4는 stable hash 순서로 holdout을 먼저 선택하고 row target에 맞추므로 multi-row group이 holdout/CV에 분산된다. CV fold 할당은 큰 whole group부터 가장 작은 fold로 보내 row 수를 균형화한다.

`training_ids(fold)`는 해당 validation fold뿐 아니라 final locked holdout도 반드시 제외한다. `all_ids - fold_ids`처럼 holdout을 실수로 학습시키는 사용법을 API 차원에서 막는다.

### 5.3 분포 진단

split artifact는 overall/holdout/fold별로 다음 derived 진단을 저장한다.

- answer sign
- 절댓값 bucket
- quality flag
- top-10 answer
- record/group count

예를 들어 holdout은 음수 43, 0은 23, 양수 1,634다. 이 진단은 현재 hash-based assignment를 바꾸지 않으며 split SHA에도 포함하지 않는다. topic/quality stratified allocator를 새로 도입하려면 v4를 폐기하지 말고 새 version에서 group 안전성과 편차 개선을 비교해야 한다.

## 6. pinned tokenizer profile v3

고정 모델과 revision:

```text
Qwen/Qwen2.5-3B-Instruct
aa8e72537993ba99e69dfaafa59ed015b17504d1
```

필수 cache hash:

| 파일 | SHA-256 |
|---|---|
| `config.json` | `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |

정확한 system prompt:

```text
Solve the math problem carefully. Return the final signed integer as `Final answer: <integer>`.
```

| chat input tokens | Train | Leaderboard |
|---|---:|---:|
| p50 | 91 | 92 |
| p90 | 144 | 147.1 |
| p95 | 176 | 177.05 |
| p99 | 284.01 | 321.03 |
| max | 1,733 | 1,276 |
| >1,024 | 4 | 1 |
| >2,048 | 0 | 0 |

tokenizer는 한 번만 load해 두 dataset을 profile하고, 필수 파일이 동일 snapshot commit인지 확인한다. `main` 같은 mutable revision과 mixed snapshot은 거부한다.

## 7. 모델/학습 preflight v4

현재 결과:

| gate | 결과 |
|---|---|
| official model ID | pass |
| immutable 40-hex revision | pass |
| tokenizer/config complete and same snapshot | pass |
| all model weight shards | fail |
| PyTorch runtime | fail |
| CUDA GPU | fail |
| CUDA BF16 | 실행 불가 |
| `accelerate` | fail |
| `peft` | fail |
| `bitsandbytes` | fail |
| `model_runtime_ready` | `false` |
| `training_ready` for `nf4_qlora_bf16` | `false` |

index JSON 하나만 있거나 shard 일부가 빠진 상태를 ready로 보지 않는다. 모든 shard를 열어 size, SHA와 snapshot을 기록한다. 공식 모델이 아니거나 mutable revision이면 report를 만들기 전에 실패한다.

## 8. parser·voting·submission 안전성

### 8.1 parser

marker 우선순위는 `Final answer` → balanced `boxed/fbox` → `####` → `Answer` → 마지막 numeric fallback이다. 선택된 marker class 안의 모든 값이 유효하고 일치해야 한다.

지원:

- signed integer, Unicode minus, valid thousands comma
- integral decimal
- exact integer fraction와 제한된 LaTeX fraction
- `\\boxed{x=42}`

거부:

- non-integral decimal/fraction
- scientific notation, NaN/Infinity
- 여러 값, marker conflict, 깨진 box
- 일반 식/code 실행
- `Answering ...`을 `Answer:`로 오인하는 false marker

### 8.2 voting

- route weight는 binary float로 직접 비교하지 않고 제한된 exact rational로 바꾼다.
- 동점이면 숫자 크기로 몰래 고르지 않고 `tie`를 반환한다.
- 동일 generation identity의 중복 trace는 한 표만 센다.
- 동일 identity인데 completion, verifier 또는 greedy metadata가 다르면 integrity error다.
- candidate는 seed, sample index, prompt/checkpoint/generation-config SHA를 기록하며 final run은 complete provenance를 강제한다.

### 8.3 submission

- 누락/invalid answer는 기본 실패다.
- emergency fallback은 호출자가 명시해야 하며 행별 provenance를 남긴다.
- `overwrite=False`는 temp file을 atomic hard-link publish해 경쟁 writer가 만든 파일을 덮지 않는다.
- parent directory까지 fsync하고 재독·round-trip validation 뒤 size/SHA를 반환한다.
- 현재 leaderboard의 malformed header를 제출 schema로 복사하지 않는다. 공식 sample submission에서 대소문자를 확정해 CLI option으로 고정한다.

## 9. 통계 구현과 성능

두 모델 비교는 같은 ID의 correctness를 사용한다.

- exact McNemar p-value
- duplicate/source cluster를 통째로 재표집하는 paired bootstrap
- 여러 실험 비교용 Holm step-down correction

17,000행, 16,992그룹, bootstrap 10,000회 synthetic benchmark:

```text
elapsed=1.51 sec
max_rss=73,388 KB
bootstrap_unit=duplicate_cluster
```

row-IID bootstrap은 `group_by_id`를 생략했을 때만 명시적으로 사용한다. 실제 모델 선택에서는 split manifest의 group mapping을 전달한다.

## 10. 테스트와 독립 반증

최종 명령:

```bash
uv run ruff check src tests
uv run pytest -s -q
uv run pytest -s --cov=deep_challenge --cov-report=term-missing -q
```

결과:

- Ruff: pass
- pytest: **172 passed**
- branch coverage 포함 total: **89%**
- 주요 모듈 coverage: quality 99%, audit 96%, submission/evaluation 93%, inference 92%, data 91%

Codex desktop의 임시 pytest capture 경로가 정리되는 환경 문제가 간헐적으로 있어 최종 증거 명령은 `-s`를 사용했다. 이는 application test 실패가 아니며 두 번의 전체 run이 통과했다.

독립 리뷰에서 실제로 발견하고 수정한 핵심 항목:

1. index JSON만으로 model-ready 오판
2. tokenizer mixed snapshot 허용
3. `Answering` marker false positive
4. float rounding이 vote tie를 깨는 문제
5. locked holdout이 training set에 섞일 수 있는 API
6. TeX command 손상과 sign을 지우는 template
7. soft template의 잘못된 hard union
8. submission no-overwrite 경쟁 조건
9. source manifest self-reference/symlink swap
10. row-IID bootstrap과 성능 문제
11. generation hash case·metadata·trace provenance 충돌
12. holdout size-first 편향

## 11. 재현 명령

```bash
PROJECT=/absolute/path/to/deepleaning
cd "$PROJECT"

DATA_DIR="$PROJECT/deep-learning-challenge-2026"
REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1

uv run deep-challenge audit-data \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --leaderboard "$DATA_DIR/deep_chal_math_leaderboard.csv" \
  --source-root "$PROJECT" \
  --output artifacts/analysis/data-audit-v3.json

uv run deep-challenge build-splits \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --version v4 \
  --output artifacts/analysis/splits-v4.json

uv run deep-challenge profile-tokenizer \
  --train "$DATA_DIR/deep_chal_math_train.csv" \
  --leaderboard "$DATA_DIR/deep_chal_math_leaderboard.csv" \
  --revision "$REVISION" \
  --output artifacts/analysis/tokenizer-profile-v3.json

# 현재 host에서는 blockers를 기록하고 exit 1이 정상이다.
uv run deep-challenge model-preflight \
  --revision "$REVISION" \
  --output artifacts/analysis/model-preflight-v4.json

uv run deep-challenge source-manifest \
  --root "$PROJECT" \
  --output artifacts/analysis/source-manifest-final.json
```

prediction mapping이 생긴 뒤 제출은 다음처럼 생성한다. 기본 fallback은 없다.

```bash
uv run deep-challenge write-submission \
  --predictions predictions.json \
  --expected "$DATA_DIR/deep_chal_math_leaderboard.csv" \
  --output submission.csv \
  --id-column id \
  --answer-column answer

uv run deep-challenge validate-submission \
  --submission submission.csv \
  --expected "$DATA_DIR/deep_chal_math_leaderboard.csv"
```

공식 sample submission이 `ID` 대문자를 요구하면 두 명령 모두 `--id-column ID`로 고정한다.

## 12. canonical artifact

현재 사용할 artifact는 다음 네 개다.

- `artifacts/analysis/data-audit-v3.json`
- `artifacts/analysis/splits-v4.json`
- `artifacts/analysis/tokenizer-profile-v3.json`
- `artifacts/analysis/model-preflight-v4.json`

`source-manifest-final.json`과 checksum 목록은 문서 수정이 끝난 후 다시 생성한다. 같은 폴더의 이전 v1–v3 중간 artifact는 정의 drift를 추적하기 위한 superseded trace이며 새 실험 입력으로 사용하지 않는다.

## 13. 다음 구현 게이트

### Gate B0 — 규칙·환경

- 로그인 상태 Kaggle Rules/Data/Evaluation/Submission snapshot
- sample submission exact schema
- 외부 공개 데이터, teacher-generated rationale, Python/SymPy TIR, multi-adapter 허용 여부의 운영진 답변
- GPU host와 CUDA/PyTorch 조합 확정
- official weights 전체 cache와 preflight green

### Gate B1 — base baseline

- pinned base greedy/direct/concise-CoT를 development probe에서 실행
- 실제 raw generation golden parser corpus 생성
- latency/VRAM/token budget 기록
- locked holdout 접근 0회 확인

### Gate B2 — QLoRA SFT

- organizer-only clean subset으로 먼저 실행
- response-only loss, 2K/4K, direct answer/verified concise rationale 비교
- fold의 `training_ids(fold)`만 사용
- checkpoint/data/config SHA와 비용 저장

### Gate B3 — self-training과 preference/RL

- rejection self-training을 우선
- KTO/ORPO/DPO/GRPO는 paired grouped evidence와 Holm correction을 통과할 때만 유지
- leaderboard 상승만으로 방법을 채택하지 않음

### Gate B4 — final

- primary/fallback freeze
- locked holdout 1회
- 사전 선언 실패 조건일 때만 이미 freeze한 fallback 선택
- 동일 config all-data refit
- network-disabled inference와 submission 재생성

## 14. 남은 외부 blocker

- 영상에는 자막 stream/동봉 `.srt`/`.vtt`가 없어 음성 전체 전사는 검증하지 못했다. 화면 13개 slide 구간과 전체 frame decode는 완료했다.
- Kaggle 비로그인 요청은 login으로 전환됐고 Chrome bridge는 현재 WSL local-URI 제약으로 연결되지 않았다. 사용자 제공 본문과 영상은 분석했지만 최신 authenticated Rules를 확인했다고 말하지 않는다.
- final test CSV와 sample submission은 로컬에 없다.
- 실제 model weight/GPU가 없어 base score, SFT gain, leaderboard score는 아직 0개의 관측값이다.

이 blocker들은 계획·model-free 기반의 실패가 아니라 Gate B에서 새 authority/환경이 필요한 항목이다. 기존 사용자 파일 `NUL`은 수정·삭제하지 않았다.
