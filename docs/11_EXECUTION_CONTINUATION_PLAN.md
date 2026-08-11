# 11. 지속 실행 체크리스트와 공개 저장소 경계

기준 시각: **2026-08-11 KST**

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
- 기존 사용자 파일 `NUL`/`NU_`, 개인 작업 프롬프트, local browser/tool cache

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
| B1.0 | fold 0 base direct-answer | B0.3 green | `20260810T234907KST` JSONL + **v2 provenance manifest**, 1,653/2,942 | invalid parser/artifact or any source/B0/config hash mismatch |
| B1.1 | parser golden corpus | B1.0 real generations | added regression tests + full CPU suite | conflict is hidden or test fails |
| B2.0 | fold 0 answer-only QLoRA | same-fold base manifest | 627/2,942; significant harm로 candidate 중단 | train IDs or provenance mismatch |
| B2.1 | concise-rationale CPU gate | **harness v1 CPU-ready / organizer-data teacher BLOCKED:** v3 initial 52/128 | qualified replay/live authorization 뒤 새 후보만 별도 검토 | v1/v2/v3 resume, v4 allowlist 조기 추가, initial <103/128 |
| B2.2 | fold 0 rationale QLoRA probe | B2.1와 새 source/B0 green | adapter v4 + generation + paired harm screen | corpus/adapter binding mismatch 또는 significant harm |
| B1/B2.3 | folds 1–4 repeat | fold 0 harm screen authorizes exact candidate | five base + five candidate OOF runs | any fold incomplete or method fingerprint drift |
| B2.4 | complete OOF comparison | all five folds | grouped paired bootstrap, exact McNemar, Holm | single fold/reused run/mixed method |
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

### 3.1.5 R8-C parser v2 current-source B1 완료

`20260810T234907KST`의 source manifest/B0 pair로 fold 0 base를 새로 생성했다. source
manifest file/tree SHA는
`eb5d2e175e26af488040aded858fcf70d2273336aee105f81c3f2613c308b727`와
`ab266cd353b8e87ec2719630723949fe76ab7dce14e0103879b8f38b3d1814dc`, preflight/smoke SHA는
`c1b29c10ad76f3eb2e94781c4dc271951a5b638af8cdcc2d291d3062b55a349f`와
`70689026618f45e468565742a0afb10472a8373329d35c5da840786ead12e30b`다.

records/manifest SHA는
`d26196283ef0a9f350d252703f40797eb9cc1eafffa676e6b5a961404a1126b4`와
`5a7c97f070fc7f9861a5c7bd92739c43cad0761714c373c046f485e4d300ea92`다. 실제 결과는
1,653/2,942(56.1863%), parser `ok/conflict/invalid=2705/3/234`, finish
`stop/length=2134/808`이며 raw-free parser audit v6 SHA는
`ba4e50772a6404b25468f25f31e67787f6819f832e76c34c947bef78aba74da0`다. 이는 과거 raw
rescore를 승격한 것이 아니라 parser v2 source에서 별도로 얻은 selection-eligible v2
bundle이다. holdout/leaderboard/test는 접근하지 않았다.

### 3.1.6 R9-A concise-rationale CPU 계약 완료

새 candidate config
`configs/gate_b/rtx4070-super-12gb-concise-rationale-v1.json`의 semantic/file SHA는
`75a315b638481a0c8213c413aa3a1253d269776d08bd2252b68654fb38c3f053`와
`66a4c5c145881c92cb4b260ef000bd89bd62119b644f6bd1e49e9894c431064f`다. 구현은 exact
fold-training coverage, organizer question/reference, parser result, canonical final marker,
teacher/prompt/generation/raw SHA, `reference_answer_in_prompt=false`,
training-only/no-tool/no-test/no-holdout를 검증하고 canonical
corpus+manifest를 atomic no-overwrite로 publish한다. 별도 audit는 raw rationale, ID,
question, answer, parsed value, teacher prompt를 포함하지 않고
`reference_answer_in_prompt_true_count=0`을 다시 계산한다. pair publish 중 두 번째 파일의
I/O 실패는 첫 번째 파일까지 rollback하고, 기존 direct-answer OOF fingerprint v1은 그대로
보존하면서 rationale에만 fingerprint v2 provenance를 추가했다.

SFT preflight와 QLoRA runtime은 이 corpus/audit/config SHA를 target provenance로 묶는 v4
schema를 사용한다. synthetic training-only fixtures로 corpus→audit→pinned tokenizer
preflight→fake-runtime adapter publish 회귀를 통과했지만, production teacher JSONL, 실제
rationale adapter, GPU generation과 모델 점수는 아직 없다. 다음 GPU workload 전에는 현재
변경을 포함한 새 source manifest/preflight/smoke가 다시 필요하다.

### 3.2 전체 백로그와 의존 관계

아래 목록은 재개 시 순서를 바꾸지 않는 실행 단위다. `CPU`는 CUDA workload 없이 가능한
작업이고, `GPU`는 직전 gate의 immutable evidence가 있을 때만 시작한다.

1. **R1 (완료, GPU):** diagnostic run이 atomic 종료했고 aggregate-only checksum 검증을
   통과했다. selection evidence로는 봉인한다.
2. **R2 (완료, CPU):** private parser audit v2와 raw-free structural regression을 추가했다.
   conflict는 여전히 fail-visible이다.
3. **R3 (완료, CPU):** Ruff, full `pytest -s -q` (398 passed, 1 skipped), branch coverage 79%, public-repo guard,
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
10. **R8-C (완료, CPU→GPU):** `20260810T234907KST` source/B0 pair와 fold 0 current-source
    base를 atomic publish했다. stored/current parser가 일치한 1,653/2,942 selection
    evidence와 raw-free audit v6를 확보했다.
11. **R9-A (완료, CPU):** concise-rationale v1 config, exact fold-training corpus
    canonicalizer, raw-free audit, pinned-tokenizer SFT preflight v4, adapter v4 provenance와
    selection/freeze compatibility를 구현·회귀검증했다.
12. **R9-B (완료, historic fail-closed CPU pilot):** ChatGPT 로그인 Codex `gpt-5.6-sol` question-only
    ledger를 구현하고 fold 0 training ID의 deterministic 128문제 pilot을 실행했다. first
    pass는 103/128(80.47%)로 80% 기준을 통과했지만, initial high/worker 1 64문제 chunk 두
    개 → local reference exact-match finalizer → failed row만 xhigh/16문제 이하로 총 3회
    cap을 마친 뒤 111/128 승인·17 exhaustion으로 끝났다. gold-answer re-prompt, tool,
    leaderboard/test/holdout은 사용하지 않았다. 이 plan은 source bank·64문제 answer-hidden
    logical audit·full 11,794 ID bank v1·corpus/SFT preflight v4로 승격하지 않는다. full
    v1 plan에는 향후 별도 versioned pilot의 immutable receipt가 필요하며, receipt는 80%,
    128/128, passed 64→60 audit와 live provenance를 모두 재검증한다. 이 historic ledger는
    `shell_environment_policy.inherit="none"` safe-command contract 이전의 것이므로 current
    status/finalize가 fail-closed로 거부하며, raw-free aggregate만 보존 증거로 남긴다.
13. **R9-B2 (완료, fresh pilot-v2 fail-closed CPU):** historic v1 ledger를 재개하지 않고
    separate `codex-gpt-5.6-sol-teacher-pilot-v2.json` profile로 32행×4, worker 1 fresh
    128-row plan을 실행했다. first pass 105/128(82.03%)는 80% gate를 넘었지만 최대 3회 후
    106/128 승인·7 exhaustion으로 종료됐다. prompt/template-policy SHA와 input-as-untrusted-
    data instruction은 유지됐고 tool/error/schema transport failure는 없었다. raw-free result
    artifact만 남기며 logical audit, full bank, corpus/SFT preflight, GPU로 승격하지 않는다.
14. **R9-B3 (완료, CPU):** v1 prompt bytes와 역사 schema를 유지한 policy-bound 호환
    refactor를 먼저 분리 커밋했다. v3는 derive-then-independent-verify prompt와 별도
    config/schema/label/version을 추가하고 기존 generic plan/run/status/finalize를 재사용한다.
    전체 CPU suite 457 passed/1 skipped, checksum/public guard와 Ruff가 green이다. 이 변경을
    committed source로 고정한 뒤에만 새 128행 live pilot을 실행한다.
15. **R9-B4 (완료, live CPU fail-closed):** 같은 seed와 deterministic fold-0 training
    128행을 32×4, worker 1로 실행했다. initial 4호출 중 parsed 2/failed 2, local 승인
    52/128·거부 12·pending 76이었다. 103/128 미만이므로 repair invocation 0으로 즉시
    종료했고 source JSONL/manifest도 만들지 않았다. raw-free final SHA는
    `6b5014b3da16fb31a1334ba101ffa1e6031a1aaac8db0369ac7b9ae81790f5e7`이다.
16. **R9-C (blocked, CPU→GPU):** R9-B4가 green이 아니므로 logical audit, authorization,
    full 11,794행 bank, corpus/SFT preflight, source manifest/B0 pair, fold 0 rationale QLoRA,
    adapter generation, paired harm screen을 실행하지 않는다. synthetic harness v1 CPU path는
    구현됐지만 qualified replay/live authorization 전에는 v4를 allowlist에 추가하지 않는다.
17. **R9-D (조건부 GPU):** R9-C가 `candidate_full_oof_authorized=true`일 때만 folds 1--4의
    fold별 corpus/adapter/base/generation을 완성하고 complete OOF grouped paired bootstrap,
    exact McNemar, Holm을 수행한다.
18. **R10 (CPU):** development evidence만으로 primary/fallback을 freeze하고 one-shot
    holdout claim 전의 immutable method/route/config checkpoint binding을 검증한다.
19. **R11 (GPU):** R10 이후에만 locked holdout을 정확히 한 번 평가한다. claim을 만든 뒤
    실패해도 접근권은 소비되므로 재시도 판단은 별도 기록한다.
20. **R12 (GPU):** holdout 후 고정된 policy로 filtered leaderboard만 예측한다. 이 데이터는
    학습, self-training seed, prompt 개선, 외부 API 입력에 절대 쓰지 않는다.
21. **R13 (CPU):** strict submission writer와 independent validator를 모두 통과시키고
    `ID,answer`, row order, checksum, invalid/missing=0인지 확인한다.
22. **R14 (외부 권한):** Kaggle upload는 사용자의 명시 요청이 있을 때만 수행한다. upload
    전에는 final run manifest와 local submission checksum을 다시 대조한다.
23. **R15 (CPU):** 각 완료 phase마다 source manifest와 canonical `CHECKSUMS.sha256`를
    갱신하고, raw artifact 없이 public code/docs만 guard 검증·commit·push한다.

R4--R9는 R3의 parser/test gate를 우회할 수 없고, R11--R14는 R9-D--R10의 complete OOF
freeze를 우회할 수 없다. Python/SymPy와 multi-adapter/checkpoint combination은 운영진
서면 답변 전 포함하지 않는다. teacher rationale는 training-only로 허용되지만 R9-B의
private corpus와 품질/provenance gate 없이 학습에 넣지 않는다.

## 4. Immediate exact commands

Set local paths without committing them:

```bash
PROJECT=/absolute/path/to/deepleaning
DATA_DIR="$PROJECT/deep-learning-challenge-2026"
GPU_ENV=/absolute/path/to/deep-challenge-gpu-venv
REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
RUN_TAG=replace-with-new-unique-tag
SOURCE_MANIFEST="$PROJECT/artifacts/analysis/source-manifest-gate-b-teacher-pilot-v3-$RUN_TAG.json"

cd "$PROJECT"
uv sync --extra model --group dev
uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q
cd artifacts/analysis && sha256sum -c CHECKSUMS.sha256 && cd "$PROJECT"
```

v1/v2에 이어 v3 `20260811T153322KST`도 initial 52/128로 fail-closed됐다. v3 source tree SHA는
`7b55a352902230325bbf25e6a5bcd81e32b8d488fd23af9f5619b229ad196963`, plan SHA는
`efed9c4163a673e03ada9862b16e545e05abd0d04b057ec67fed130c2838265b`다. repair, production
teacher JSONL, logical audit, bank/corpus/preflight/GPU는 시작하지 않았다. v1/v2/v3
ledger/tag를 재개하지 않으며 Kaggle token은 대회 metadata 확인용이지 teacher credential이
아니다.

다음 safe task는 `docs/13_SYNTHETIC_TEACHER_HARNESS_V1.md`의 harness v1을 committed clean
source에서 CPU 검증하고, fresh manifest를 만든 뒤 offline replay와 explicit 2×32 synthetic
live canary를 qualified로 닫는 것이다. replay/live authorization 전에는 v4 config를
allowlist에 추가하거나 organizer-data teacher를 호출하지 않는다. 아래 source-manifest 명령은
그 no-overwrite pattern을 따른다.

```bash
test ! -e "$SOURCE_MANIFEST"
uv run deep-challenge source-manifest \
  --root "$PROJECT" \
  --output "$SOURCE_MANIFEST"
```

향후 별도 승인된 candidate가 128/128과 64→60 audit/authorization을 통과해 rationale QLoRA를
실제 시작할 때만 먼저 다음 read-only 관측을 한다.

```bash
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

The answer-only candidate ended at its fold 0 harm screen. The next single-adapter
experiment is the separately versioned concise-rationale v1 path; only its CPU
contract is complete. Public external data, self-training, preference/RL, local
tool use, multi-checkpoint voting, and full-development refit remain deferred.
They may begin only after their rule, license, contamination, compute, and
development-evidence gates are recorded; none is silently folded into the
concise-rationale result.
