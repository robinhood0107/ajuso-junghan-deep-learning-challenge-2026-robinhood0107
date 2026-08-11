# 12. 새 세션 시작용 상세 handoff prompt

기준 snapshot: **2026-08-11 KST**
목적: 현재 Codex 작업을 새 세션에서 그대로 이어가되, 과거 evidence를 잘못 승격하거나
GPU·holdout·leaderboard gate를 건너뛰지 않도록 시작 문맥과 실행 계약을 한 번에 전달한다.

## 사용 방법

1. 새 Codex 세션을 이 저장소 workspace에서 연다.
2. 아래 `새 세션에 그대로 붙여넣을 prompt` 전체를 첫 메시지로 붙여넣는다.
3. 새 세션은 snapshot 값을 곧바로 사실로 단정하지 않고, prompt에 적힌 read-only 명령으로
   Git·파일·artifact·환경 상태를 다시 확인한다.
4. 이 문서는 공개 저장소에 들어갈 수 있는 aggregate와 hash만 포함한다. raw question,
   answer, rationale, generation, teacher event, credential, model/checkpoint, submission은 포함하지
   않는다.

## 새 세션에 그대로 붙여넣을 prompt

````text
목표
====

Kaggle “Deep Learning Challenge 2026”에서 고정 student/inference base
`Qwen/Qwen2.5-3B-Instruct`의 정수 수학 문제 Exact Match를 높이는 Gate B를 이어서 진행한다.
model-free Gate A는 이미 READY이므로 다시 만들지 않는다. 현재 가장 가까운 후보는
training-only question을 ChatGPT 로그인 Codex `gpt-5.6-sol`에 전달해 검증된 concise
rationale bank를 만든 뒤, 고정 Qwen base에 QLoRA SFT하는 경로다.

중요한 현재 결론은 다음과 같다.

- CPU 구현은 상당 부분 끝났지만 첫 128문제 teacher pilot은 fail-closed됐다.
- 이 실패 pilot을 재개하거나 고쳐 쓰지 않는다. current safe-command contract에 맞는 새
  versioned prompt/config/ledger에서 다시 시작해야 한다.
- 새 pilot, source bank, logical audit, corpus/SFT preflight가 모두 green이 되기 전에는
  GPU rationale 학습을 시작하지 않는다.
- leaderboard/test/locked holdout은 teacher, 학습, prompt 개선, self-training seed에 절대
  쓰지 않는다.
- 사용자가 명시적으로 요청하기 전에는 Kaggle submission을 업로드하지 않는다.

작업 태도
=========

- 불필요한 확인 질문으로 멈추지 말고, read-only 조사와 CPU-only 구현·테스트·문서화는
  안전한 범위에서 끝까지 진행한다.
- 권한 확대, 고비용 external quota 소비, GPU 학습, one-shot holdout, Kaggle upload처럼
  새로운 gate가 필요한 지점에서는 evidence와 blocker를 먼저 제시한다.
- 진행 중 도구를 쓰면 60초를 넘기지 않게 짧은 한국어 상태 업데이트를 보낸다.
- raw question, answer, rationale, individual ID, raw completion/event를 terminal이나 공개
  문서에 출력하지 않는다. aggregate count, hash, latency, token usage, status만 출력한다.
- 기존 사용자 파일과 unrelated 변경을 보존한다. 특히 NUL/`NU_` 파일을 수정·삭제하지 않는다.
- 원본 data와 private artifact를 덮어쓰지 않는다. 모든 새 결과는 새 unique tag와 atomic
  no-overwrite 경로를 사용한다.
- 파일 수정은 최소 범위로 하고, line ending과 기존 의미를 보존한다. unrelated cleanup이나
  대규모 재구성은 하지 않는다.

0. 먼저 workspace와 Git 상태를 확인한다
========================================

현재 Codex workspace에서 다음을 read-only로 실행한다. 개인 absolute path는 공개 문서나
Git에 기록하지 말고 `PROJECT`로 추상화한다.

```bash
git rev-parse --show-toplevel
git status -sb
git branch --show-current
git log --oneline -10
git remote -v
git rev-parse HEAD
git rev-parse origin/main
gh pr view 1 --json url,state,isDraft,baseRefName,headRefName,mergeable,reviewDecision,statusCheckRollup
```

2026-08-11 snapshot은 다음과 같지만 반드시 다시 확인한다.

- branch: `agent/gate-b-no-api-teacher`
- commit: `4ca3c25ab6a83ad4fa49ef91dbe3d4abea416e00`
- origin branch와 local branch가 동일하고 working tree가 clean이었다.
- draft PR #1:
  `https://github.com/robinhood0107/ajuso-junghan-deep-learning-challenge-2026-robinhood0107/pull/1`
- PR은 `main` 대상, OPEN/DRAFT/MERGEABLE이었다.
- repository license는 GNU General Public License v3.0이다.

PR 상태가 달라졌다면 최신 상태를 문서화한다. PR merge는 read-only 확인이 아니므로 현재
사용자 요청에 merge 권한이 포함됐는지 확인하기 전 자동 merge하지 않는다. branch 이동,
reset, discard, clean, force-push도 임의로 하지 않는다.

1. 저장소 전체를 inventory하고 권위 문서를 읽는다
================================================

첫 실행에서 아래 inventory를 만든다. `artifacts/`, 대회 data, model cache의 raw 내용을
무차별 출력하지 않는다.

```bash
rg --files \
  -g '!artifacts/**' \
  -g '!deep-learning-challenge-2026/**' \
  -g '!.venv/**' \
  -g '!.git/**' \
  | sort
```

다음 파일을 반드시 끝까지 읽는다. 일부만 읽고 과거 대화 요약으로 대체하지 않는다.

우선순위 1, 현재 상태와 정확한 재개 순서:

1. `README.md`
2. `docs/09_IMPLEMENTATION_STATUS.md`
3. `docs/11_EXECUTION_CONTINUATION_PLAN.md`
4. `docs/10_GATE_B_CPU_READY_RUNBOOK.md`
5. `docs/04_EXPERIMENT_AND_TRAINING_PLAN.md`
6. `docs/05_INFERENCE_SUBMISSION_REPRODUCIBILITY.md`
7. `docs/07_RULE_CLARIFICATION.md`
8. 이 파일 `docs/12_NEW_SESSION_START_PROMPT.md`

우선순위 2, 배경 evidence와 발표/운영 문맥:

9. `docs/01_EVIDENCE_AND_COMPETITION_AUDIT.md`
10. `docs/02_DATASET_FORENSICS.md`
11. `docs/03_METHODS_AND_LITERATURE.md`
12. `docs/06_OPERATIONS_SCHEDULE_PRESENTATION.md`
13. `docs/08_SOURCES.md`

구현·환경 계약:

14. `.gitignore`
15. `LICENSE`
16. `pyproject.toml`
17. `configs/gate_b/codex-gpt-5.6-sol-teacher-pilot-v2.json`
18. `configs/gate_b/codex-gpt-5.6-sol-teacher-v1.json` (historic forensic evidence only)
19. `configs/gate_b/rtx4070-super-12gb-direct-answer-v1.json`
20. `configs/gate_b/rtx4070-super-12gb-concise-rationale-v1.json`

그 다음 현재 작업과 직접 관련된 source/test를 읽는다.

- `src/deep_challenge/teacher_rationale.py`
- `src/deep_challenge/teacher_pilot_authorization.py`
- `src/deep_challenge/rationale_materialization.py`
- `src/deep_challenge/rationale_corpus.py`
- `src/deep_challenge/gate_b.py`
- `src/deep_challenge/gate_b_runtime.py`
- `src/deep_challenge/gate_b_selection.py`
- `src/deep_challenge/gate_b_holdout.py`
- `src/deep_challenge/gate_b_prediction.py`
- `src/deep_challenge/parser_golden.py`
- `src/deep_challenge/submission.py`
- `src/deep_challenge/independent_submission.py`
- `src/deep_challenge/public_repo_guard.py`
- `src/deep_challenge/cli.py`
- 위 모듈과 이름이 대응되는 모든 `tests/test_*.py`

기존 API와 safety guard를 먼저 이해한 뒤 수정한다. parser conflict visibility, silent-zero
금지, atomic pair/no-overwrite, source/B0 provenance, tokenizer snapshot, shard completeness,
holdout claim, submission independent validation을 우회하지 않는다.

2. 고정 모델과 금지 모델 계약
===========================

student, adapter base, 최종 inference에서 허용되는 모델은 오직 다음 하나다.

- model: `Qwen/Qwen2.5-3B-Instruct`
- internal pinned revision:
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`

이 revision은 프로젝트 재현성 pin이며 운영진이 직접 지정한 revision이라고 표현하지 않는다.

금지:

- Qwen2.5-Math, DeepSeek-R1/DeepSeek 계열, Llama 계열 또는 다른 모델 weight 사용
- 금지 모델 weight merge, adapter merge, weight soup, inference ensemble
- 다른 외부 모델로 leaderboard/test 답 생성
- leaderboard/test를 검색, API, teacher, prompt 개선, synthetic seed, self-training 입력으로 사용
- inference 중 network/API/model download
- 운영진 서면 확인 전 Python/SymPy TIR
- 운영진 서면 확인 전 same-base multi-adapter/checkpoint voting 또는 selector/verifier adapter

허용된 teacher는 final model이 아니다. ChatGPT 로그인 Codex `gpt-5.6-sol`은 organizer
training question만 보고 training rationale candidate를 만들 수 있다. organizer reference
answer는 prompt에 포함하지 않고 local finalizer에서만 exact match 검증에 사용한다.

3. 데이터 계약
=============

`PROJECT`는 `git rev-parse --show-toplevel` 결과로 잡고, data는 기본적으로
`$PROJECT/deep-learning-challenge-2026` 아래에서 찾는다. 실제 경로가 다르면 read-only로
확인하되 개인 absolute path를 Git에 기록하지 않는다.

canonical 값:

- train: `deep_chal_math_train.csv`
- train SHA-256:
  `e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a`
- train rows: 17,000
- organizer exclusions: `train_filtered_ids.csv`
- exclusions SHA-256:
  `67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3`
- exclusion IDs: 627 unique
- current leaderboard: `deep_chal_math_leaderboard_filtered.csv`
- current leaderboard SHA-256:
  `032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1`
- current leaderboard rows: 831
- current leaderboard header: `id,question`
- submission header: exact uppercase `ID,answer`

과거 1,000행 leaderboard와 malformed `id,question, answer` header는 역사적 감사 근거일
뿐 현재 submission ID source가 아니다. old leaderboard를 현재 inference, teacher, training,
self-training, prompt 개선에 사용하지 않는다.

split/shard 계약:

- split: `artifacts/analysis/splits-v4.json`
- split logical SHA:
  `be7368175f8fd4d472f9c6dfb39f05361c8175359d02960962665c049e3940db`
- split file SHA:
  `5b1969e79da08fa8347569c55d6d40b1fbccfcb2e5fc0e0f7a3295386a260520`
- hard clusters: 16,992
- locked holdout: 1,700 rows
- development shard: `artifacts/analysis/development-cv-v4`
- development shard bundle SHA:
  `cc5ea51f155f99d1956864c0097c3ac87ad42b89b4bd3c4e09f4d1a281d2fbb4`
- development shard rows: 15,300
- hard-group-expanded exclusion rows: 629
- eligible development CV: 14,736
- fold 0 training/validation: 11,794 / 2,942

split v4를 재생성하지 않는다. soft number-masked template는 hard cluster로 쓰지 않는다.
각 fold corpus와 training은 반드시 `training_ids(fold)` 의미를 지켜 holdout과 validation
fold를 제외한다. development 단계는 sealed shard만 사용한다.

4. 현재 실제 성능과 미실행 항목
================================

확인된 실제 model evidence:

- parser v2 current-source fixed-base fold 0:
  - 1,653 / 2,942 exact match
  - 56.1863%
  - parser ok/conflict/invalid = 2705/3/234
  - selection-eligible v2 bundle
- answer-only QLoRA fold 0:
  - 627 / 2,942 exact match
  - 21.3120%
  - base보다 -19.8165%p
  - 유의하게 악화되어 folds 1-4 중단

존재하지 않는 결과:

- production concise-rationale source bank
- passed 64-row logical audit
- canonical concise-rationale training corpus
- concise-rationale QLoRA adapter
- concise-rationale development score
- complete 5-fold candidate OOF
- frozen primary/fallback
- locked holdout score
- filtered leaderboard prediction/score/submission
- final test prediction/score/submission

없는 결과를 추정값이나 계획값으로 실제 점수처럼 보고하지 않는다.

5. teacher pilot 현재 상태
=========================

historic pilot snapshot:

- run tag: `20260811T103224KST`
- plan SHA:
  `2431c9a173ed775c55dc332c4639d7777cb228146377dbbe2381e79e49fc75fa`
- total: 128 training questions
- first pass accepted: 103/128 = 80.47%
- final accepted after retry cap: 111/128
- exhausted: 17
- max attempts per problem: 3
- outcome: `failed_closed_retry_exhausted`
- logical audit started: false
- full bank started: false
- GPU workload started: false
- leaderboard/test used: false
- locked holdout accessed: false

raw-free final artifact:

- `artifacts/analysis/gate-b-teacher-pilot-v1-20260811T103224KST-final-v1.json`
- SHA-256:
  `75f835b2d4159c934c0ab8762d413115fb3eead028dcb4727e87770eaf77f1f0`

actual usage aggregate:

- input tokens: 92,381
- output tokens: 102,055
- reasoning output telemetry: 90,074; output과 중복될 수 있으므로 비용에 이중 합산하지 않음
- cached input tokens: 0
- total attempts: 6
- total latency: 3,199,760 ms

fresh pilot-v2 snapshot:

- run tag: `20260811T132301KST`
- plan SHA:
  `b4624eac74f3a2b0238debda96bbc40edfc2e9b37ea27825bf62ed8d4726d4af`
- config semantic/file/prompt-policy SHA:
  `6794dba97ba8b172b07e1b2f942d00d38e64a5f405d1144771ae037c25625de4` /
  `de9abbabe8f88bda17637b0070bc87d0ec694e37f953625ad8c1cbe4fb4b261e` /
  `5ed785c9a02bc84298ed8186681b2b21a80da50d9af4591da0c1586a28e387b3`
- total: 128 training questions; initial chunks: 32×4; worker: 1
- first pass accepted: 105/128 = 82.03%
- final accepted after retry cap: 106/128
- exhausted: 7; retryable at stop: 15; max attempts per problem: 3
- outcome: `failed_closed_retry_exhausted`
- logical audit/full bank/GPU started: false
- leaderboard/test used: false; locked holdout accessed: false
- raw-free final artifact:
  `artifacts/analysis/gate-b-teacher-pilot-v2-20260811T132301KST-final-v2.json`
- final artifact SHA-256:
  `5d50fdb41c0503546e673393d97b24bf7dc5c92e52577738eada35c143ac874e`
- aggregate usage: input 117,241; output 84,014; reasoning telemetry 73,127;
  cached input 0; total attempts 8; total latency 1,809,811 ms

이 historic ledger는 current safe command의
`shell_environment_policy.inherit="none"` 이전에 생성됐다. current loader가 stored argv를
재구성한 safe argv와 비교할 때 의도적으로 reject한다. 따라서 다음을 금지한다.

- historic plan/status/finalize를 current source에서 억지로 통과시키기
- loader 호환성 예외 추가
- stored command/prompt self-hash만 믿기
- old raw attempt를 새 safe attempt로 복사
- 17개 실패 row에 organizer answer를 알려 주고 rationale을 다시 쓰게 하기
- 승인된 111개만 partial bank로 승격하기

historic v1와 fresh v2 artifact 모두 실패 evidence로만 보존한다. 다음 실험은 새
prompt/config/version/tag와 새 immutable ledger를 사용한다.

6. 다음 실제 개발 목표: teacher candidate v3 설계
=============================================

GPU 없이 다음을 수행한다.

1. current v1/v2 prompt builder, schema, event validator, finalizer, authorization test를 읽는다.
2. raw ledger를 재해석하지 말고, 공개-safe aggregate 실패 사실과 synthetic cases만
   이용해 general prompt 개선안을 설계한다.
3. v1/v2 config나 artifact를 덮어쓰지 않는다. 새 config가 필요하면 별도 v3 filename/schema로
   추가하고 semantic/file SHA를 새로 계산한다.
4. 아래 safety fields와 실행 정책은 완화하지 않는다.
   - provider: ChatGPT-login Codex CLI
   - model: `gpt-5.6-sol`
   - organizer question only
   - `reference_answer_in_prompt=false`
   - `allow_tool_use=false`
   - network scope training-only
   - initial reasoning high
   - rejected-row repair reasoning xhigh
   - max attempts 3
   - pilot worker 1
   - full bank worker maximum 2
   - ephemeral, read-only, ignore-user-config, ignore-rules
   - shell environment inheritance none
   - auth-only temporary `CODEX_HOME` outside model `-C` workspace
   - execution-time Codex binary/version re-probe and exact match
5. prompt에는 reference answer, solution, holdout, leaderboard, test, external API token을 넣지 않는다.
6. prompt/config 변경에 synthetic tests를 먼저 추가한다. actual contest question/answer/rationale를
   public fixture로 복사하지 않는다.
7. output은 2-6 concise reasoning lines를 권장하고 최대 1,500 characters/12 lines,
   `Final answer: <signed integer>` 하나를 요구한다. parser marker conflict를 숨기지 않는다.
8. event error, command/MCP/web/tool call, unexpected item, schema violation, missing/duplicate/extra
   ID, truncation, non-integer final, marker conflict는 chunk failure다.

새 candidate의 변경 이유와 expected failure mode를 docs에 기록하고, 전체 CPU verification을
통과하기 전 actual teacher를 호출하지 않는다.

7. 매 구현 묶음의 mandatory CPU 검증
====================================

먼저 `PROJECT`를 현재 repo root로 지정한다. 아래 명령을 그대로 실행하고 실제 exit/result를
기록한다.

```bash
cd "$PROJECT"
uv sync --extra model --group dev
uv run ruff check .
CUDA_VISIBLE_DEVICES='' uv run pytest -s -q

cd "$PROJECT/artifacts/analysis/development-cv-v4"
sha256sum -c CHECKSUMS.sha256

cd "$PROJECT/artifacts/analysis"
sha256sum -c CHECKSUMS.sha256

cd "$PROJECT"
PYTHONPATH=src python3 -m deep_challenge.public_repo_guard --all
git diff --check
```

2026-08-11 latest verified baseline:

- dependency sync success
- Ruff pass
- `454 passed, 1 skipped`
- skip 1은 default CPU `.venv`에 PyTorch가 없기 때문
- canonical checksum pass
- public repository guard pass

새 세션 결과가 다르면 원인을 조사한다. 단순히 expected count를 낮추거나 test를 삭제하지 않는다.
기본 `.venv`에 GPU package를 억지로 섞지 않는다. GPU runtime은 ext4의 별도 `$GPU_ENV`를
사용한다.

8. 다음 128문제 pilot gate
=========================

fresh pilot-v2는 initial 80% gate는 넘었지만 7 exhaustion으로 fail-closed됐다. 따라서 그
ledger/tag를 재개하지 않고, 새로운 v3 prompt/config/tests가 green일 때만
`docs/10_GATE_B_CPU_READY_RUNBOOK.md`의 teacher plan/run/status/finalize 명령을 current CLI
help와 대조한 뒤 새 unique `RUN_TAG`로 실행한다.

pilot scope:

- fold: 0 only
- exact source: fold 0 `training_ids(0)`에서 deterministic stable-hash stratified 128 rows
- initial chunks: locked config 기준
- worker: exactly 1
- unattempted work: high reasoning
- only locally rejected repair rows: xhigh reasoning, bounded repair chunks
- no arbitrary ID input
- no holdout or validation rows
- no leaderboard/test rows

합격 기준:

- reference answer in prompt: 0
- tool invocation: 0
- event/schema/ID/order/provenance violation: 0
- initial exact-match rate: at least 80%
- all 128 accepted within at most 3 attempts per problem
- source bank finalizer complete
- no exhausted row

한 조건이라도 실패하면 full v1 bank와 GPU candidate를 시작하지 않는다. partial source를 쓰지
않고 raw-free failure artifact와 blocker만 남긴다. prompt를 수정하면 같은 output path를
덮어쓰지 말고 새 version/tag에서 처음부터 다시 pilot한다.

9. logical audit와 pilot authorization
=====================================

pilot이 128/128 complete일 때만 logical audit를 시작한다.

- finalized source bank와 teacher plan/manifest/ledger를 재검증한다.
- plan SHA에서 deterministic 64 rows를 선택한다.
- auditor input은 question, candidate rationale, candidate가 주장한 integer만 포함한다.
- organizer reference answer, official solution, local tool, search, external API를 주지 않는다.
- each item은 verified bank의 exact question/candidate binding과 일치해야 한다.
- schema, ID/order, event/command provenance를 fail-closed 검증한다.
- at least 60/64 internally consistent가 합격 기준이다.

audit가 통과하면 immutable pilot authorization receipt를 발행한다. receipt 생성과 full v1 plan
생성 때 다음을 다시 live-validate한다.

- exact deterministic 128 pilot IDs and config/data/split/shard binding
- first-attempt local exact rate at least 80%
- every row accepted within attempt cap
- source/manifest/ledger provenance
- deterministic 64 audit selection and item binding
- audit result at least 60/64
- post-receipt source/config/split/audit tamper 없음

receipt만 존재하고 underlying evidence가 변했으면 full plan을 거부한다.

10. Pro quota와 full bank 비용 gate
=================================

historic pilot의 no-cache retry-heavy profile을 단순 비례하면:

- fold 0 full v1 11,794 rows: 약 8.51M input, 9.40M output, 약 8,117 credits
- later v2 2,942 rows: 약 2.12M input, 2.35M output, 약 2,025 credits
- development-CV total 14,736 rows: 약 10.64M input, 11.75M output, 약 10,141 credits

이는 예산 확정값이 아니라 실패 pilot을 그대로 비례한 경고용 추정이다. full v1을 시작하기 전:

1. 로그인된 Codex usage dashboard 또는 interactive `/status`에서 실제 잔량을 확인한다.
2. credential/token 값은 출력하거나 artifact에 쓰지 않는다.
3. account quota가 확인되지 않거나 11,794-row run을 감당할 수 없으면 정확한 blocker로 기록한다.
4. quota 회피를 위해 다른 모델로 자동 전환하거나 API key 경로를 만들지 않는다.
5. 30초 → 2분 → 5분 → 15분 backoff를 쓰고, 24시간 회복되지 않으면 외부 blocker로 남긴다.

고비용 full run 승인 여부는 현재 사용자의 최신 지시와 quota evidence를 따른다. quota 확인이
막혀도 code/test/docs와 작은 synthetic 검증은 계속 진행한다.

11. teacher bank v1과 rationale corpus
====================================

pilot receipt가 green일 때만 fold 0 `training_ids(0)` 11,794 rows의 full v1 plan을 만든다.

- question당 teacher generation은 한 번만 승인 bank에 추가한다.
- complete verified chunk만 resume 때 건너뛴다.
- partial/corrupt/mismatched chunk는 덮어쓰지 않고 새 attempt로 처리한다.
- append-only ledger, plan SHA, input-ID SHA, event/raw SHA, parsed SHA를 유지한다.
- simultaneous worker/lock/stale PID/tamper를 감지한다.
- max workers 2를 넘지 않는다.
- 3회 cap 뒤 unresolved row가 있으면 source publish를 중단한다.

complete bank 뒤 `gate-b-materialize-teacher-bank`로 exact fold training rows만 materialize한다.
v1 full bank는 valid pilot sidecar 없이는 materializer가 거부해야 한다. 그 다음 순서:

1. `build-rationale-corpus`
2. `audit-rationale-corpus`
3. `gate-b-sft-preflight` rationale v4

필수 검증:

- exact training ID coverage
- validation/holdout/leaderboard/test exclusion
- organizer question/target consistency
- teacher/model/prompt/generation/raw/config SHA
- reference answer hidden from teacher
- local exact-match and parser verification
- required `Final answer:` marker
- rationale length/line constraints
- no duplicate/conflicting/missing rows
- atomic no-overwrite records/manifest
- raw-free audit
- tokenizer snapshot consistency
- development shard completeness

이 중 하나라도 실패하면 GPU로 넘어가지 않는다.

12. GPU는 모든 CPU gate 뒤 마지막에 실행한다
==========================================

GPU workload 직전에만 실제 `$GPU_ENV`와 model cache를 read-only로 찾아 확인한다. 개인 path는
public docs/config에 넣지 않는다. 다른 GPU process를 종료하지 않는다.

먼저 physical state:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version \
  --format=csv,noheader,nounits
```

진입 기준:

- used <= 2,048 MiB
- free >= 10,240 MiB
- NVIDIA GeForce RTX 4070 SUPER 12GB target

조건이 안 맞으면 기다리거나 blocker로 기록한다. process kill, WSL shutdown, threshold 완화는
하지 않는다.

현재 source/docs/tests를 포함한 새 no-overwrite source manifest를 만든다. 예전 manifest,
preflight, smoke를 재사용하지 않는다.

```bash
"$GPU_ENV/bin/deep-challenge" source-manifest \
  --root "$PROJECT" \
  --output "$PROJECT/artifacts/analysis/source-manifest-gate-b-<NEW_TAG>.json"

"$GPU_ENV/bin/deep-challenge" model-preflight \
  --revision "aa8e72537993ba99e69dfaafa59ed015b17504d1" \
  --output "$PROJECT/artifacts/analysis/model-preflight-gpu-ready-<NEW_TAG>.json"
```

그 다음 runbook의 exact synthetic GPU smoke를 실행한다. smoke만 첫 CUDA workload다. smoke는
organizer/leaderboard/test prompt를 쓰지 않는다. source manifest, B0 preflight, smoke,
config byte SHA를 이후 training/generation artifact에 결속한다.

GPU config의 핵심:

- NF4 + double quantization
- BF16 compute/training
- max sequence length 2,048
- microbatch 1
- gradient accumulation 16
- LoRA rank 16, alpha 32, dropout 0.05
- all-linear targets
- gradient checkpointing on
- training `use_cache=false`
- eval/generation KV cache on
- one training epoch
- paged AdamW 8-bit
- deterministic direct-answer inference route

13. fold 0 rationale QLoRA와 harm screen
=======================================

CPU corpus/preflight와 new B0 pair가 모두 green이면:

1. current source의 fold 0 fixed-base generation을 resumable v2 artifact로 다시 확보한다.
2. concise-rationale fold 0 QLoRA를 `--resume-dir` contract로 학습한다.
3. checkpoint를 삭제하지 않는다.
4. source/B0/split/shard/corpus/config/tokenizer SHA가 모두 같을 때만 latest complete checkpoint를
   resume한다.
5. partial/corrupt checkpoint는 forensic attempt로 보존하고 latest prior complete checkpoint를
   선택한다.
6. contract mismatch/tamper checkpoint는 runtime factory 전 fail-closed한다.
7. final adapter는 atomic no-overwrite로 publish하고 exact tokenizer bytes와 LoRA tensor contract를
   검증한다.
8. adapter development generation을 resumable chunk ledger로 완료한다.
9. new real generation 구조를 보고 raw text를 공개하지 않는 synthetic parser golden regression을
   추가한다.
10. base와 candidate를 paired duplicate-cluster bootstrap + exact McNemar로 비교한다.

후보가 유의하게 악화되거나 operationally invalid면 folds 1-4를 중단하고 base fallback을 유지한다.
leaderboard 점수로 후보를 고르지 않는다.

14. candidate 통과 뒤 bank v2와 5-fold OOF
=========================================

fold 0 harm screen이 positive decision artifact로
`candidate_full_oof_authorized=true`를 기록할 때만 다음을 한다.

- development-CV에서 fold 0 training IDs를 뺀 나머지 2,942 rows만 teacher bank v2로 추가
- v2 plan/run/status/finalize는 positive candidate decision과 scope SHA에 결속
- v2 bank는 positive sidecar 없이는 materialize 금지
- v1+v2 union에서 각 fold `training_ids(fold)`만 정확히 materialize
- folds 1-4의 base/candidate generation과 candidate training 완료
- complete five-fold OOF union 생성
- paired duplicate-cluster bootstrap
- exact McNemar
- multiple confirmatory comparisons에 Holm correction
- common method fingerprint, checkpoint/config/data/fold provenance 검증

single fold, incomplete fold union, mixed parser/config/runtime, reused leaderboard evidence는 freeze
입력으로 받지 않는다.

15. freeze, one-shot holdout, leaderboard, submission
==================================================

complete development evidence만으로 primary/fallback을 freeze한다.

- default freeze는 primary-only다.
- same-base fallback은 운영진 허용과 OOF routing evidence가 모두 있을 때만 켠다.
- freeze 뒤 method/route/config/checkpoint binding을 immutable artifact로 만든다.
- 이 뒤에만 locked holdout claim을 만들고 정확히 한 번 평가한다.
- holdout claim을 만든 뒤 실패해도 접근권을 소비한 것으로 기록한다.
- holdout 결과를 보고 method를 다시 선택하지 않는다.

그 다음 filtered leaderboard 831 rows를 offline inference한다.

- internet/API/search/model download off
- exact expected IDs and order
- all answers canonical signed integers
- parser invalid/conflict/missing은 silent zero로 대체하지 않음
- primary submission writer와 independent validator 모두 통과
- output header exact `ID,answer`
- atomic no-overwrite file and checksum

Kaggle upload는 사용자의 최신 메시지가 명시적으로 요청할 때만 한다. upload 직전 final run
manifest, submission checksum, current Kaggle allowance, target competition을 다시 read-only 확인한다.
Kaggle token 값은 절대 출력하지 않는다.

16. 공개 Git 경계와 기록 방식
============================

public Git에 허용:

- source code
- synthetic tests
- docs
- secret-free public configs
- aggregate count/hash/status only
- GPLv3 license

public Git에 금지:

- original train/leaderboard/test CSV and zip
- any data-derived raw artifact
- question, answer, rationale, individual ID
- raw teacher event/prompt/output/assessment ledger
- raw model generation
- model weights, tokenizer cache, adapter, checkpoint
- Kaggle/Codex credential, auth copy, environment file
- personal absolute paths
- prediction, submission, private score evidence
- generated CSV/JSONL/parquet/archive

`.gitignore`와 `public_repo_guard`를 우회하지 않는다. commit 전 반드시:

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --name-only
PYTHONPATH=src python3 -m deep_challenge.public_repo_guard --all
```

`git add -A` 대신 intended public paths를 명시해 stage한다. unrelated user changes가 있으면
포함하지 않는다. force-push, reset --hard, clean, broad deletion을 하지 않는다. push/PR/merge는
사용자의 현재 권한과 repository workflow를 따른다.

17. 문서와 artifact 업데이트
==========================

각 phase 뒤 최소 다음을 갱신한다.

- `docs/09_IMPLEMENTATION_STATUS.md`: current authoritative state
- `docs/10_GATE_B_CPU_READY_RUNBOOK.md`: exact current CLI commands and gates
- `docs/11_EXECUTION_CONTINUATION_PLAN.md`: completed/pending sequence
- 필요 시 `docs/04_EXPERIMENT_AND_TRAINING_PLAN.md`
- 필요 시 `docs/05_INFERENCE_SUBMISSION_REPRODUCIBILITY.md`
- 필요 시 `docs/07_RULE_CLARIFICATION.md`
- 새 versioned artifact under `artifacts/analysis/`
- canonical checksum ledger는 policy에 맞는 artifact만 갱신

문서에는 raw content를 넣지 않고 aggregate, schema, SHA, count, status, blocker만 넣는다.
artifact가 current code를 capture하도록 source manifest 생성 시점을 정확히 기록한다.

2026-08-11 CPU hardening source snapshot:

- file:
  `artifacts/analysis/source-manifest-gate-b-codex-teacher-20260811T122209KST.json`
- file count: 77
- tree SHA-256:
  `3652a31fd91e73f4807a44fd61d477a97aa3f95f72ce2aff1c0b81af2882a107`

새 tracked 문서나 source가 생기면 이 snapshot은 역사적 evidence가 되며 다음 GPU authorization에
재사용하지 않는다.

18. 실패 처리와 중단 기준
=======================

다음 경우 fail-closed한다.

- hash/provenance mismatch
- incomplete or corrupt shard/chunk/checkpoint
- teacher prompt에 reference answer 포함
- tool/web/MCP/command event
- schema/ID/order/marker conflict
- missing/duplicate/extra IDs
- pilot initial <80%
- pilot not 128/128 within attempt cap
- logical audit <60/64
- unqualified v1/v2 bank sidecar
- rationale corpus incomplete or answer mismatch
- B0 preflight/smoke not green
- GPU VRAM threshold not satisfied
- candidate significantly harmful
- incomplete five-fold OOF
- attempted holdout before freeze
- submission invalid/missing row
- network dependency during final inference

실패 artifact를 성공으로 승격하지 않는다. 기존 file을 삭제하거나 덮어써 재시도하지 않는다.
정확한 blocker, attempted action, next safe condition을 raw-free로 기록한다.

19. 최종 보고 형식
================

매 큰 phase의 보고는 반드시 다음을 분리한다.

1. 확인된 사실
2. 실행한 작업
3. 실행한 명령과 실제 exit/result
4. 새/변경 artifact path와 SHA
5. Gate A 상태
6. Gate B0 상태
7. teacher pilot/bank/audit 상태
8. GPU training/inference 상태
9. 외부 blocker와 quota/rule blocker
10. 실제 모델 점수
11. 미실행 항목
12. Git branch/commit/PR 상태
13. 다음 한 단계와 그 진입 조건

실제 generation이나 model score가 없으면 “없음/미실행”이라고 명시한다. 계획 수치, 비용 추정,
historical diagnostic, parser-only rescore를 current selection score로 표현하지 않는다.

20. 새 세션에서 바로 실행할 첫 순서
=================================

1. Git/PR/worktree read-only 확인
2. mandatory docs/config/source/tests 완독
3. mandatory CPU validation 재실행
4. current v1 teacher prompt/safety implementation review
5. historic failed ledger를 건드리지 않는 teacher v2 설계
6. synthetic tests 추가
7. Ruff/full pytest/checksum/public guard
8. docs 09/10/11 업데이트와 새 source manifest
9. fresh 128-row question-only pilot
10. pilot 결과에 따라 fail-closed 또는 logical audit로 진행

이 순서를 바꾸지 말고, 새 pilot이 green이 되기 전에는 full bank나 GPU 작업을 시작하지 않는다.
````

## 이 handoff의 핵심

새 세션이 가장 먼저 해결할 문제는 GPU 학습이 아니다. 안전 계약을 유지한 새 teacher
candidate가 fresh 128-row pilot에서 **128/128 within three attempts**와 별도 **60/64 logical
audit**을 통과하게 만드는 것이다. 그 뒤에만 full fold-0 bank, rationale corpus, GPU harm
screen 순서로 진행한다.
