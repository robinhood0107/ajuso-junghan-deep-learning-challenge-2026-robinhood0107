# 11. 지속 실행 체크리스트와 공개 저장소 경계

기준 시각: **2026-08-10 KST**

이 문서는 작업을 중단했다가 다시 시작하더라도 같은 순서·동일한 안전 조건으로
이어가기 위한 실행 원장이다. 상태 값이 바뀌는 GPU 사용량이나 Kaggle 제출 한도는 이
문서의 숫자를 재사용하지 않고 해당 명령 직전 다시 관측한다.

## 1. 공개 저장소 경계

공개 GitHub 저장소에는 다음만 넣는다.

- `src/`, `tests/`, `configs/`, `docs/`, `README.md`, `pyproject.toml`, `uv.lock`
- 재현 명령, 규칙 해석, SHA-256 값, 모델·데이터를 다시 받는 방법
- GNU GPL v3.0 라이선스 전문

현재 canonical public origin은
`https://github.com/robinhood0107/ajuso-junghan-deep-learning-challenge-2026-robinhood0107`이다.
요청한 한글 대회명은 repository description에 보존하고, URL slug는 GitHub API가 허용한
ASCII 표기로 사용한다.

다음은 절대로 stage하거나 push하지 않는다.

- Kaggle 원본 CSV·ZIP와 그 복사본
- `artifacts/` 전체: development shard, split ID, raw generation, adapter manifest,
  screenshot, submission, checksum은 대회 데이터 또는 그 파생물일 수 있다.
- `data/`, `datasets/`, `downloads/`, `cache/`, generated report 및 raw
  CSV/Parquet/JSONL/weight 파일처럼 별도 전달해야 하는 데이터·model 산출물
- `env`, `.env*`, `.kaggle/`, access token, Hugging Face token 등 모든 인증 정보
- local model weight·adapter·checkpoint·prediction·submission과 experiment tracker
- 기존 사용자 파일 `NU_`, 개인 작업 프롬프트, local browser/tool cache

`.gitignore`은 위 경계를 강제한다. 공개 저장소의 문서는 데이터 내용을 포함하지 않고,
로컬 artifact는 수상 검증이 필요한 경우 별도 비공개 전달물로만 사용한다.

현재 checkout은 `.githooks/pre-commit`을 `core.hooksPath=.githooks`로 연결한다.
이 hook은 staged Git blob만 읽어 `.gitignore`를 우회하는 force-add와 recognizable token
pattern을 fail-closed로 막는다. 새 clone에서도 다음을 먼저 실행하고, 전체 tracked tree는
그 아래 명령으로 별도 점검한다.

```bash
git config core.hooksPath .githooks
PYTHONPATH=src python3 -m deep_challenge.public_repo_guard --all
```

## 2. 현재 확인된 운영 계약

- 유일한 베이스: `Qwen/Qwen2.5-3B-Instruct`, 내부 재현 pin:
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- 허용: Full FT/LoRA/QLoRA, SFT, RL, augmentation, quantization, offline
  Majority Voting/Self-Consistency/Best-of-N.
- 금지: 다른 모델 weight/merge/최종 추론 ensemble, pre-training, leaderboard/test를
  학습·self-training seed·외부 API/검색 입력으로 사용하는 행위, 추론 중 인터넷.
- 현재 Kaggle API 인증 snapshot은
  `artifacts/analysis/kaggle-b0-20260810/authenticated-api-snapshot-v1.json`에 있다.
  authenticated 파일 목록에는 sample submission이 없고, Rules/Evaluation은
  `submission.csv`의 `ID,answer`를 요구한다. 현재 API가 보인 제출 가능 횟수는 5이고
  과거 제출은 0건이다.
- local Python/SymPy TIR 및 같은 베이스의 multi-adapter/checkpoint 결합은 운영진의
  명시 답변 전 **off**다. 첫 모델 경로는 single base 또는 single adapter다.

## 3. 재개 순서와 완료 조건

| 순서 | 작업 | 시작 조건 | 완료 증거 | 중단 조건 |
|---:|---|---|---|---|
| P0 | 공개 repo 초기화 | `.gitignore` audit 완료 | GPLv3, clean staged file list, public repo | 민감 파일이 stage됨 |
| B0.1 | Kaggle rules/data snapshot | API token 존재 | page/file/limit hash, sample 부재 기록 | slug/auth 실패 |
| B0.2 | GPU preflight | model/runtime/VRAM 충족 | 새 no-overwrite preflight JSON, `training_ready=true` | GPU used >2,048MiB 또는 free <10,240MiB |
| B0.3 | final synthetic smoke | B0.2 green | `status=green`; only local `2+3` prompt | parser/load/backward/VRAM failure |
| B1.0 | fold 0 base direct-answer | B0.3 green | JSONL + **v2 provenance manifest** + raw generation | invalid parser/artifact or any source/B0/config hash mismatch |
| B1.1 | parser golden corpus | B1.0 real generations | added regression tests + full CPU suite | conflict is hidden or test fails |
| B2.0 | fold 0 answer-only QLoRA | same-fold base manifest | exact adapter bundle/checksum/manifest | train IDs or provenance mismatch |
| B1/B2.1 | folds 1–4 repeat | fold 0 harm screen authorizes exact candidate | five base + five candidate OOF runs | significant fold 0 regression or any fold incomplete |
| B2.2 | complete OOF comparison | all five folds | grouped paired bootstrap, exact McNemar, Holm | single fold/reused run/mixed method |
| B3 | freeze and locked holdout | B2.2 evidence, primary decided | durable claim + one receipt | no freeze, already-consumed claim |
| B4.1 | filtered leaderboard prediction | frozen policy, B3 complete | strict prediction manifest, no invalid answers | data SHA/schema/adapter mismatch |
| B4.2 | submission build/verify | B4.1 complete | writer + two independent validators + checksum | any missing/invalid ID |
| B4.3 | Kaggle upload | explicit user request only | Kaggle submission receipt | no explicit authorization |

### 3.1 완료된 진단과 안전한 종료 처리

2026-08-10에 시작한 아래 fold 0 run은 **진단 전용**이다. attention mask와 eval
KV-cache 보강 전 source에서 시작했으므로, 성공하더라도 model selection, QLoRA 시작,
OOF 비교, primary/fallback freeze의 근거로 사용하지 않는다.

| 항목 | 값 |
|---|---|
| run tag | `20260810T002000KST` |
| 대상 | organizer train의 eligible fold 0 validation 2,942문항, greedy base direct-answer 1회 |
| 출력 | `artifacts/gate_b/20260810T002000KST/fold-0/base-direct-predictions.jsonl`, `base-direct-manifest.json` |
| 공개 경계 | 두 파일 모두 `artifacts/` 아래의 private evidence이며 Git에 올리지 않음 |
| 상태 | 정상 완료: atomic JSONL/manifest 쌍과 checksum이 일치, 2,942 records / 1,210 exact match (41.1285%) |
| 실패 처리 | atomic pair가 없거나 checksum/record count가 다르면 failure evidence만 남기고 재시도 조건을 기록; partial output을 승격하지 않음 |

종료 뒤에는 raw question, answer, completion을 터미널이나 공개 문서에 출력하지 않는다.
아래처럼 aggregate와 checksum만 확인한다.

```bash
RUN_DIR=artifacts/gate_b/20260810T002000KST/fold-0
jq '{schema_version,record_count,problem_count,samples_per_problem,model_id,revision,route,checkpoint_sha256,config_sha256,fold,parser_status_counts,finish_reason_counts,exact_match_count,exact_match_accuracy,input_token_count_total,output_token_count_total,peak_vram_allocated_bytes_max}' \
  "$RUN_DIR/base-direct-manifest.json"
wc -l "$RUN_DIR/base-direct-predictions.jsonl"
sha256sum "$RUN_DIR/base-direct-predictions.jsonl" "$RUN_DIR/base-direct-manifest.json"
```

그 다음 GPU 없이 다음 private audit을 실행한다. stale checksum, non-development partition,
stored parser mismatch 중 하나라도 있으면 output을 만들지 않는다.

```bash
uv run deep-challenge audit-parser-golden \
  --records "$RUN_DIR/base-direct-predictions.jsonl" \
  --manifest "$RUN_DIR/base-direct-manifest.json" \
  --output "artifacts/analysis/parser-golden-20260810T002000KST-fold0-v2.json"
```

이 진단에서 실제 completion 구조를 관찰한 뒤에도 public test에는 question, ID,
reference answer, raw completion을 넣지 않는다. marker/source/status/reason만
익명화한 구조적 regression case를 별도로 재현하고, parser conflict가 보이면 반드시
`conflict` 그대로 검증한다.

R1과 R2는 완료됐다. v2 audit은 19개 outcome class만 기록하고 raw completion/ID/answer를
직렬화하지 않았으며, 그 class를 safe synthetic boxed/final/hash/fallback regression으로
고정했다. 이 완료는 production model-score gate가 아니라 parser behavior gate다.

R4도 당시 source에서 완료됐다. 그 B0 pair는 run tag `20260810T062500KST`의
`model-preflight-gpu-ready-20260810T062500KST.json` (SHA-256
`1c88007e3c714c036907a4dc4c0592b31d0f52d931cd1e412e212f814bc5f603`)와
`gpu-smoke-20260810T062500KST.json` (SHA-256
`98b35fe8a471cab16438b20ac9055d7835f641cd9d8923f4f901916bec2613f0`)이다. preflight의
`training_ready=true`와 smoke의 `status=green`은 당시 R5 시작을 허용했다. 그러나 그때의
development manifest는 v1이라 B0/source/config byte binding이 없어 selection evidence가 될 수
없다. 아래 R5.1의 v2 guard 뒤에는 새 source manifest와 새 B0 pair로 다시 시작하며, 이 pair를
다른 tag나 leaderboard/test prediction에 재사용하지 않는다.

### 3.1.1 20260810T062500KST 당시-current-source B1의 제한된 진단 결과

`artifacts/gate_b/20260810T062500KST/fold-0/`의 fixed-base direct-answer run은 원자적
JSONL/manifest 쌍으로 정상 종료했고, 2,942개 중 1,210 exact match
(`0.4112848402447315`)였다. parser 상태는 `ok=2143`, `conflict=3`, `invalid=796`이고,
최대 allocated VRAM은 2,193,992,192 bytes였다. 새 redacted audit
`artifacts/analysis/parser-golden-20260810T062500KST-fold0-v3.json`
(SHA-256 `78576ddccc8e2b500b2a63ad8f5f6b26a9485ca1fd704c5dc47aaa2490247e1b`)도 19개
structural outcome class를 raw-free로 검증했다.

하지만 이 run의 manifest schema는 `gate-b1-development-run-v1`이다. 실행 당시에는
raw completion, parser/seed/prompt/checkpoint/config semantic SHA, VRAM, latency가 JSONL에
있었지만 B0 report byte SHA, source-tree manifest, config-file byte SHA, run-level seed/prompt/
latency digest를 하나의 fail-closed manifest에 묶지 않았다. 따라서 **실제 관측 점수이지만
QLoRA authorization, OOF comparison, method selection, freeze에는 사용 금지**다. v2 artifact
reader는 이를 의도적으로 거부한다.

### 3.1.2 20260810T131821KST production B1 v2 완료 근거

같은 organizer-only fold 0 base direct-answer를 strict v2 source에서 no-overwrite로 다시
실행했다. `base-direct-predictions.jsonl` 2,942행과 manifest가 함께 atomic publish됐고,
records/manifest SHA-256은 각각
`e25f9468fe4bb3fd2851c4cd69bb340619c2962b851e10f707bb998e18b022e7`와
`e52cc656ff3a17f6b0794fdd39b81190005a43d6c92b8ac6b8c83ecd67771fa6`다.

- exact match: 1,210/2,942 (`0.4112848402447315`)
- parser: `ok=2143`, `conflict=3`, `invalid=796`
- finish: `stop=2134`, `length=808`
- output tokens: 989,549; max allocated VRAM: 2,193,992,192 bytes
- generation latency: total 21,273,884.481795ms, mean 7,231.0960169ms
- source manifest: file SHA
  `20fe5b69fd450381a358e998415f5997ac6e3e9fe974c0a209d65e6e636b013c`, tree SHA
  `0fd2e438eb7184d4d86dc943274c889acff6f8a489362c4f95b440810d560871`
- B0 preflight/smoke SHA:
  `32a09a3667a1fcb7cadfd9929ddd0161727a2966136dad61f60b42a58d8fc11e`,
  `ded4c52db1ce18c533daabd86df890c9320980ddc82d5b4d1a9d812a8ff05be1`
- parser audit v4 SHA:
  `5954dc2ba7b668938fafd0853810034bd41c28456b752ed613b9b3fb44b75b5c`

parser audit v4는 기존과 같은 19개 structural outcome class를 확인했고 raw completion,
question, ID, answer, completion hash, parsed integer를 직렬화하지 않았다. leaderboard/test와
locked holdout 접근은 모두 false다. 따라서 새 공개 fixture를 복사하지 않고 기존 synthetic
regression으로 충분함을 확인했다. 이 v2 base bundle은 같은 fold QLoRA의 authorization
evidence다. 다만 이 문서 반영이 source tree를 바꾸므로 실제 R6 직전에는 새 source manifest와
새 B0 pair로 source를 재결속한다.

### 3.1.3 R6 첫 시도 fail-closed와 tokenizer export 수정

run tag `20260810T192204KST`의 새 source/B0 pair는 green이었고 fold 0 QLoRA는 정확히
738/738 optimizer step을 끝냈다. 관측 train runtime은 5,375.9894초, train loss는
`0.48071612413659653`였다. 그러나 adapter publish 전 exact tokenizer 검증에서
`saved tokenizer.json differs from the pinned snapshot` 오류가 발생해 CLI는 exit 2로
종료했다. target adapter directory와 temporary training directory는 존재하지 않으며 이
시도를 checkpoint, adapter, OOF evidence로 사용하지 않는다.

GPU 없이 재현한 결과 Transformers/tokenizers의 `save_pretrained()`가 pinned
`tokenizer.json`에 `ignore_merges=false`를 추가하고 `tokenizer_config.json`의 chat template를
별도 Jinja 파일로 외부화했다. 원본/saved tokenizer SHA는 각각
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`와
`9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa`였다. runtime을 pinned
cache 두 파일의 exact byte copy + known SHA 검증으로 바꾸고, 실제 cache copy/reload에서
vocab size, encode 결과, chat-template token sequence가
모두 동일함을 확인했다. 두 cache SHA는
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`와
`5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`이다.
regression 2건을 추가한 전체 결과는 Ruff pass, `354 passed, 1 skipped`다.

R6 재시도는 수정 commit 뒤 새 source manifest, 새 B0 preflight/smoke, 새 checkpoint tag를
사용했다. 이전 실패 tag의 경로를 재사용하거나 validation gate를 완화하지 않았다.

### 3.1.4 R6/R7 성공과 answer-only candidate 중단

`20260810T210605KST` source/B0 pair에서 재시도는 738/738 step, runtime 5,171.3711초,
loss `0.4800419177466292`로 완료됐다. adapter artifact SHA는
`e6eb813a7fd36449759df38617576b7e5af2bd3d3b727d4895a16563619f27f4`이며 exact tokenizer,
504 LoRA tensor, split/data/source/B0 provenance와 `CHECKSUMS.sha256`를 모두 검증했다.

adapter generation은 627/2,942(21.3120%), base는 1,210/2,942(41.1285%)였다. paired delta
-19.8165%p의 95% cluster-bootstrap CI가 [-21.9240, -17.7831]%p이고 Holm-adjusted exact
McNemar p가 3.31e-72이므로 cost-control gate는 fold 1--4를 중단했다. 이후 parser v2의
immutable rescore는 base만 1,653/2,942(56.1863%)로 개선했고 adapter는 변하지 않았다.
rescore는 selection-ineligible이며 다음 source-bound B1 전에는 freeze할 수 없다.

### 3.2 전체 백로그와 의존 관계

아래 목록은 재개 시 순서를 바꾸지 않는 실행 단위다. `CPU`는 CUDA workload 없이 가능한
작업이고, `GPU`는 직전 gate의 immutable evidence가 있을 때만 시작한다.

1. **R1 (완료, GPU):** diagnostic run이 atomic 종료했고 aggregate-only checksum 검증을
   통과했다. selection evidence로는 봉인한다.
2. **R2 (완료, CPU):** private parser audit v2와 raw-free structural regression을 추가했다.
   conflict는 여전히 fail-visible이다.
3. **R3 (완료, CPU):** Ruff, full `pytest -s -q` (372 passed, 1 skipped), branch coverage 78%, public-repo guard,
   canonical checksum을 통과했고 docs 04/05/07/09/10/11을 현재 evidence로 갱신했다.
4. **R4 (완료, GPU):** GPU `used/free=912/11,086MiB`의 새 관측값에서 당시 source
   preflight와 cache-on synthetic smoke를 `20260810T062500KST`로 실행했고 green pair를 만들었다.
5. **R5 (완료, CPU → GPU):** strict v2 provenance guard, run tag `20260810T131821KST`의
   source manifest/B0 pair, no-overwrite fold 0 fixed-base output과 raw-free parser audit v4를
   모두 완료했다. selection-eligible base score는 1,210/2,942(41.1285%)다. v1 output은
   진단 보존만 한다.
6. **R6 (완료, GPU):** 첫 학습은 738/738 뒤 tokenizer snapshot drift를 publish gate가
   fail-closed로 거부했다. exact cache-byte export 수정 뒤 `20260810T210605KST` 재시도는
   738/738, runtime 5,171.3711초, loss 0.4800419로 성공했고 adapter 504 tensor와 모든
   checksum/provenance를 검증했다.
7. **R7 (완료, GPU→CPU):** fold 0 adapter generation은 627/2,942(21.3120%)였다. base 대비
   delta -19.8165%p, 95% cluster-bootstrap CI [-21.9240, -17.7831]%p, Holm-adjusted exact
   McNemar p=3.31e-72다. raw-free parser audit v5와 comparison을 완료했다.
8. **R8-A (완료, CPU cost gate):**
   `gate-b-candidate-probe-decision-20260810T210605KST-fold0-v1.json`이
   `stop_before_remaining_folds`와 `candidate_full_oof_authorized=false`를 기록했다. 따라서 이
   exact answer-only QLoRA의 fold 1--4는 실행하지 않는다.
9. **R8-B (완료, CPU parser v2):** `Final answer is:`/nested same-marker/균형 Markdown·LaTeX
   parser를 보강했고 conflict 3건은 유지했다. immutable base raw rescore는
   1,653/2,942(56.1863%), adapter는 627/2,942 그대로다. 둘 다 rescore 자체는
   selection-ineligible이다.
10. **R8-C (다음, CPU→GPU):** 전체 회귀·Git 기록 뒤 새 source manifest/B0 pair를 만들고
    fold 0 current-source base를 다시 atomic publish한다. stored/current parser가 일치한
    selection evidence를 얻기 전에는 새 candidate, freeze, holdout으로 가지 않는다.
11. **R9 (대기):** current-source base 뒤 verified concise rationale 또는 별도 versioned
    candidate를 정의하고 fold 0 harm screen을 먼저 통과시킨다. 통과한 method만 folds 1--4와
    complete OOF grouped paired bootstrap, exact McNemar, Holm을 수행한다.
12. **R10 (CPU):** development evidence만으로 primary/fallback을 freeze하고 one-shot
    holdout claim 전의 immutable method/route/config checkpoint binding을 검증한다.
13. **R11 (GPU):** R10 이후에만 locked holdout을 정확히 한 번 평가한다. claim을 만든 뒤
    실패해도 접근권은 소비되므로 재시도 판단은 별도 기록한다.
14. **R12 (GPU):** holdout 후 고정된 policy로 filtered leaderboard만 예측한다. 이 데이터는
    학습, self-training seed, prompt 개선, 외부 API 입력에 절대 쓰지 않는다.
15. **R13 (CPU):** strict submission writer와 independent validator를 모두 통과시키고
    `ID,answer`, row order, checksum, invalid/missing=0인지 확인한다.
16. **R14 (외부 권한):** Kaggle upload는 사용자의 명시 요청이 있을 때만 수행한다. upload
    전에는 final run manifest와 local submission checksum을 다시 대조한다.
17. **R15 (CPU):** 각 완료 phase마다 source manifest와 canonical `CHECKSUMS.sha256`를
    갱신하고, raw artifact 없이 public code/docs만 guard 검증·commit·push한다.

R4--R8은 R3의 parser/test gate를 우회할 수 없고, R11--R14는 R9--R10의 complete OOF
freeze를 우회할 수 없다. 외부 규칙 확인이 아직 없는 Python/SymPy, teacher rationale,
multi-adapter/checkpoint combination은 위 첫 pass에 포함하지 않는다.

## 4. Immediate exact commands

Set local paths without committing them:

```bash
PROJECT=/absolute/path/to/deepleaning
DATA_DIR="$PROJECT/deep-learning-challenge-2026"
GPU_ENV=/absolute/path/to/deep-challenge-gpu-venv
REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
RUN_TAG=replace-with-new-unique-tag
SOURCE_MANIFEST="artifacts/analysis/source-manifest-gate-b-$RUN_TAG.json"

cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version \
  --format=csv,noheader,nounits
```

Only when the just-observed values satisfy `used <= 2048MiB` and
`free >= 10240MiB`, use a new tag and execute the B0.2/B0.3 commands in
[`10_GATE_B_CPU_READY_RUNBOOK.md`](10_GATE_B_CPU_READY_RUNBOOK.md). Never reuse
an existing preflight or smoke output path. The smoke artifact is the only
permitted first CUDA workload and contains no organizer/leaderboard/test prompt.

Before B0, create and retain the source snapshot that the later B1 v2 manifest will
re-hash. This command is CPU-only and the output is excluded from the source tree
itself.

```bash
"$GPU_ENV/bin/deep-challenge" source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"
```

## 5. Per-step Git record

Commit only public code/document changes after each completed phase:

1. `chore: initialize public reproducibility repository`
2. `docs: record authenticated Kaggle contract`
3. `fix: <runtime guard or parser regression>` when a regression test accompanies it
4. `docs: record Gate B <phase> evidence` with only hashes and non-sensitive summaries

Do not commit raw artifacts merely to make a commit. Before every commit, run
`git diff --cached --check`, `git status --short`, and a staged-secret/data audit.
Push only the intended public files. A Kaggle submission is not a Git action and
is intentionally outside automatic execution.

## 6. Deferred experiments

The first pass ends after single-adapter direct-answer OOF comparison and the
one permitted holdout evaluation. Concise rationale, public external data,
self-training, preference/RL, local tool use, multi-checkpoint voting, and
full-development refit are separate versioned experiments. They may begin only
after their rule, license, contamination, compute, and development-evidence
gates are recorded; none is silently folded into the first-pass result.
