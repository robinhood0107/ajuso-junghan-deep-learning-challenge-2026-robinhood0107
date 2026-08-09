# 05. 오프라인 추론·제출·재현성

Accuracy가 높아도 정수 추출·CSV schema·offline cache 중 하나가 틀리면 점수는 0이거나 재현 검증에서 탈락할 수 있다. 이 문서는 모델 출력에서 최종 제출까지를 하나의 검증 가능한 제품으로 정의한다.

## 0. 현재 고정 계약

- 실시간 leaderboard 예상 ID source는 반드시
  `deep_chal_math_leaderboard_filtered.csv`의 831개 ID다. 과거 1,000행 raw 파일은
  현재 제출에 사용하지 않는다.
- 규칙 원문이 확정한 제출 파일/열은 `submission.csv`와 `ID,answer`다. 실제 sample
  artifact의 행 순서와 SHA는 아직 확보하지 못했으므로 확보 즉시 현재 validator와
  교차검증한다.
- 추론 중 인터넷/API/검색은 금지다. Majority Voting, Self-Consistency, Best-of-N은
  허용된다. 다른 모델 weight의 merge·추론 ensemble은 금지다.
- Python/SymPy tool 실행은 명시 답변 전 기본 off다. parser 실패와 marker conflict는
  그대로 실패로 남기며 silent zero fallback은 없다.
- final method selection은 5-fold OOF 예측을 모두 모은 paired cluster bootstrap, exact
  McNemar, Holm family로 freeze한다. deployment fold 0의 단일 checkpoint만 실제 추론에
  사용하며 fold checkpoint ensemble은 하지 않는다. 최종 freeze는 complete OOF
  artifact만 받고, base/adapter method kind, pinned checkpoint 또는 실제 adapter
  artifact digest, fold 간 공통 training-method fingerprint를 재검증한다.
- `predictions.jsonl`/inference manifest는 raw completion, finish reason, input/output token 수, parser 결과,
  seed, prompt/checkpoint/config SHA, split/fold/group, latency, peak VRAM을 보존한다.
- production validator와 별개로 `verify-submission-independent`가 primary data/submission
  모듈을 import하지 않는 최소 CSV parser로 header·ID·순서·정수를 교차검증한다.

## 1. 추론 시스템 계약

입력:

- official test CSV
- test SHA-256
- model/tokenizer/adapter artifacts
- prompt, decoding, routing, parser, verifier config

출력:

- `submission.csv`
- `run_manifest.json`
- `predictions.jsonl`
- `validation_report.json`
- 모든 artifact checksum

금지:

- 추론 중 network/API
- test 질문을 외부 모델·검색·상용 서비스에 전달
- 런타임 model/data download
- 실패 시 임의 random answer나 조용한 0 대입
- 사람이 CSV를 직접 편집해 provenance를 끊는 행위

## 2. 한 문제의 추론 레코드

```json
{
  "id": "<evaluation-id>",
  "question_sha256": "...",
  "quality_flags": [],
  "route": "cot",
  "prompt_sha256": "...",
  "candidate_seeds": [0, 1, 2, 3],
  "candidate_sample_indices": [0, 1, 2, 3],
  "checkpoint_sha256": "...",
  "generation_config_sha256": "...",
  "raw_candidates": ["..."],
  "parsed_candidates": [
    {
      "status": "valid",
      "answer_str": "42",
      "source_marker": "final_answer",
      "verifier": {"status": "pass", "checks": []}
    }
  ],
  "vote_table": {"42": 3, "41": 1},
  "selected_answer_str": "42",
  "confidence": 0.75,
  "latency_ms": 0,
  "generated_tokens": 0,
  "model_artifact_sha256": "..."
}
```

원 제출에는 정수만 쓰지만 위 provenance를 별도 보존한다.

## 3. 정수 answer parser

### 3.1 우선순위

1. `Final answer:` marker class
2. balanced `\boxed{...}` marker class
3. GSM8K형 `####`
4. 명시적 `Answer:`
5. 마지막 독립 signed integer — 다른 marker가 없을 때만 fallback

가장 높은 우선순위 marker class가 하나라도 있으면 낮은 class는 무시한다. 선택된 class 안의 marker는 **모두** 형식상 유효하고 같은 정수여야 한다. 하나라도 잘못됐으면 `invalid`, 서로 다른 정수면 `conflict`이며 마지막 값을 맹목적으로 쓰지 않는다.

### 3.2 정규화

- Unicode minus `−`/en dash 문맥 → ASCII `-`
- thousands comma와 LaTeX spacing 제거
- surrounding `$`, `\\(`, `\\)` 제거
- `+42` → `42`
- `-0` → `0`
- `42.0`은 exact integral일 때만 `42`
- `84/2`는 exact rational 계산 후 denominator=1일 때만 허용
- scientific notation은 모델 출력에서는 원칙적으로 invalid; 허용 여부를 사전 test로 고정
- Python arbitrary-precision `int`
- 최종 canonical regex는 `^-?(0|[1-9]\\d*)$`

모델이 만든 일반 수식을 무제한 `eval`하지 않는다. fraction/decimal의 매우 제한된 grammar만 deterministic parser로 계산한다.

### 3.3 전행 완성 실패 정책

현재 Gate B 실행 계약은 fail-closed다.

1. development에서 freeze한 primary를 실행한다.
2. same-base multi-checkpoint가 서면 허용되고 OOF에서 routing까지 승격된 경우에만,
   primary parser-invalid ID에 freeze한 fallback을 한 번 실행한다.
3. 그래도 `invalid`/`conflict`이면 prediction map에서 그 ID를 누락하고 non-zero로 종료한다.
4. strict submission writer가 누락을 발견해 CSV 생성을 거부한다.

`emergency_prior`, 임의 최빈값, silent `0`, test를 본 뒤 만든 수동 답은 현재 CLI 경로에
없다. 전행 정수가 확보되지 않으면 제출을 만들지 않는다. bounded retry나 constrained
decoding을 나중에 추가하려면 먼저 development에서 별도 방법으로 평가·freeze해야 한다.

### 3.4 `boxed` 처리

- 중첩 brace balance
- `\boxed{\frac{84}{2}}` 같은 제한된 exact rational
- `\boxed{x=42}`에서 우변
- `\boxed{42, 7}`처럼 다중 값이면 invalid
- interval, ordered pair, units가 있으면 single-integer 계약과 일치하는지 확인

### 3.5 반드시 포함할 단위 테스트

유효:

```text
Final answer: 42
Final answer: -5765435
\boxed{3,431,577,212,128,939}
#### 0
\boxed{\frac{84}{2}}
Answer: 42.0
```

무효 또는 충돌:

```text
Final answer: 41 ... Final answer: 42
\boxed{3, 4}
The choices are 1, 2, 3.
I cannot determine it.
Final answer: 42.5
NaN
1e6
```

정답 reference 없이 parser가 반환하는 값만 먼저 테스트하는 fixture suite는 구현했다.
실제 base/SFT 모델을 실행한 뒤에는 raw generation과 사람이 확정한 기대 parse를
익명화해 golden regression corpus로 추가한다. 2026-08-10 old-source fold 0 diagnostic은
atomic publish된 2,942행 bundle을 남겼고, private v2 audit은 19개
status/source/reason-code outcome class를 확인했다. 여기서 선택한 boxed/final/hash/fallback
구조만 안전한 synthetic fixture로 재현했으며 실제 completion, ID, question, reference
answer, parsed value는 public code에 넣지 않았다. full CPU regression은 이 fixture를 포함해
349 passed, 1 skipped였다. 이 diagnostic은 parser gate를 닫지만 attention-mask/cache 보강 전
source이므로 다음 GPU 단계의 model selection 또는 QLoRA authorization은 여전히 허용하지
않는다.

`audit-parser-golden`은 이 작업의 private CPU-only 입력 검증 단계다. manifest checksum과
JSONL line count, fold-validation/cross-validation partition, 각 row의 completion hash와
stored parse result를 다시 대조한 뒤, raw completion/ID/question/reference answer/parsed
integer를 전혀 직렬화하지 않은 status/source/reason-code count만 새 no-overwrite artifact에
쓴다. 따라서 terminal 출력과 public Git fixture는 이 aggregate를 보고 만든 안전한
synthetic structure만 사용한다.

```bash
RUN_DIR=artifacts/gate_b/<RUN_TAG>/fold-0
uv run deep-challenge audit-parser-golden \
  --records "$RUN_DIR/base-direct-predictions.jsonl" \
  --manifest "$RUN_DIR/base-direct-manifest.json" \
  --output "artifacts/analysis/parser-golden-<RUN_TAG>-fold0.json"
```

이 명령이 holdout/non-development partition, stale checksum, parser mismatch를 발견하면
output을 publish하지 않는다. actual-output-derived regression은 structural synthetic form만
public code에 고정했고, raw evidence는 `artifacts/`에만 남는다.

## 4. 문제 본문의 답 누출과 parser 분리

일부 question 자체에 `Answer` 또는 `boxed` 값이 있다. Parser는 **assistant completion 영역만** 읽어야 하며 prompt를 이어 붙인 전체 transcript에서 숫자를 찾으면 안 된다.

API/serving framework가 prompt와 completion을 함께 반환하면:

1. token index로 prompt length를 기록
2. 생성 token만 decode
3. parser input hash 저장

를 수행한다.

## 5. 복수 subpart·불완전 문제

단일 정수로 표현하기 어려운 문제에 대해 임의 결합 규칙을 만들지 않는다.

Routing:

- `single_integer_clear`: 정상 생성
- `multi_part_but_train_convention_known`: 같은 source의 label convention을 근거와 함께 적용
- `multi_part_ambiguous`: 여러 prompt 후보 후 low confidence
- `missing_visual_or_fragment`: 일반 모델 출력은 생성하되 별도 flag, 과도한 test-time budget을 쓰지 않음

불완전 문제를 외부 검색으로 복원하지 않는다. 최종 답은 시스템이 낸 best effort로 제출하되, 발표 분석에서는 데이터 품질 한계로 분리한다.

## 6. 후보 생성과 voting

### 6.1 후보 다양성

같은 seed/route/sample index/prompt/checkpoint/generation-config identity를 N번 세지 않는다. 동일 identity인데 trace, greedy flag 또는 verifier 결과가 다르면 순서대로 하나를 고르지 않고 provenance integrity error로 전체 aggregation을 중단한다. SHA-256 표기는 lowercase로 정규화한다.

- seed
- temperature
- prompt variant
- CoT/TIR route
- duplicate trace hash

를 기록한다.

### 6.2 답 집계

기본 score:

\[
S(a)=\sum_i \mathbf{1}[a_i=a]\cdot
w_{\text{route}}\cdot
w_{\text{verify}}\cdot
w_{\text{diversity}}
\]

초기에는 단순 majority를 기준선으로 둔다. learned weight는 locked validation에서만 정하고 public leaderboard에 맞추지 않는다.

### 6.3 Tie-break

사전 고정 순서:

1. deterministic verifier 통과 표가 더 많은 답
2. 독립 route(CoT와 TIR) 모두 지지
3. 같은-base verifier score
4. 더 많은 고유 reasoning trace
5. concise greedy candidate
6. 끝까지 동률이면 canonical numeric sort 같은 임의 규칙이 아니라 validation에서 고정한 fallback model 1개

Tie-break 변경도 실험으로 취급하고 version을 기록한다.

### 6.4 Adaptive budget

| 단계 | 후보 | 종료 조건 |
|---|---:|---|
| A | greedy 1 | deterministic solution이 강하게 검증될 때 |
| B | sample 총 4 | top share ≥0.75 + check pass |
| C | 총 8 | top share ≥0.75 또는 두 route 합의 |
| D | 총 16 | high-confidence selector |
| E | 총 32 | final unresolved, global budget 내 |

confidence threshold는 group validation에서 calibration한다. 단순 softmax probability를 정확도 확률로 간주하지 않는다.

## 7. 로컬 tool sandbox

운영진이 허용한 경우에만 활성화한다.

### Allowlist

- integer arithmetic
- `fractions.Fraction`
- 제한된 `decimal.Decimal`
- 제한된 SymPy expressions/solvers
- pure functions

### Denylist

- network/socket
- subprocess/shell
- arbitrary import
- filesystem read/write
- environment variable 접근
- dynamic code loading
- pickle
- reflection/introspection

### 실행 제한

- AST parse 후 허용 node만
- 문제당 wall/CPU/memory limit
- stdout/stderr cap
- deterministic seed
- timeout은 candidate invalid로 기록
- generated code와 exact output hash 저장

`eval`, `exec`, shell command로 raw model code를 바로 실행하지 않는다.

## 8. 재현성 수준

GPU kernel, batching, library에 따라 bitwise identical generation이 보장되지 않을 수 있다. 목표를 세 단계로 나눈다.

| 수준 | 목표 |
|---|---|
| R1 artifact | 동일 model/data/config/checksum |
| R2 supported-platform | 명시한 GPU/CUDA/PyTorch에서 동일 candidate와 CSV |
| R3 answer-level | 허용된 동등 플랫폼에서 최종 answer가 동일 |

R2를 최대한 달성하기 위한 정책:

- model/tokenizer revision pin
- dependency hash lock
- CUDA/cuDNN/FlashAttention/vLLM 버전 pin
- per-problem seed를 ID hash에서 독립 생성
- batch order가 seed에 영향을 주지 않게 함
- `torch.use_deterministic_algorithms` 가능 범위 명시
- TF32 여부 고정
- sampling implementation 고정
- dynamic early batching이 candidate 순서를 바꾸지 않게 함
- target hardware에서 두 번 full rehearsal

성능을 위해 nondeterministic kernel을 선택하면 그 사실과 허용 오차를 보고하고, 동일 raw outputs를 보존한다.

## 9. Submission 생성

제출 CSV는 leaderboard/test 파일의 header를 복사하지 않는다. 규칙 원문으로 확인된
대소문자와 열 순서 `ID,answer`를 기본 schema로 고정한다. 실제 sample 파일은 아직
없으므로 sample 고유의 row order/hash는 미확정이다.

예상 논리 schema:

```csv
ID,answer
<evaluation-id>,<signed-integer>
```

현재 filtered leaderboard에서는 그 파일 자체의 831개 ID 순서를 expected order로
사용한다. sample artifact가 제공되면 ID set/order가 일치하는지 확인하고 별도 SHA를
잠근다.

### Validator

1. expected row count와 일치
2. expected ID set과 정확히 일치
3. ID 중복 없음
4. 현재 filtered ID 순서와 일치하고, sample 확보 후 sample 순서와 교차검증
5. answer null 없음
6. canonical integer regex
7. decimal/scientific notation/설명 없음
8. CSV를 다시 읽어 값이 동일
9. UTF-8, BOM 정책 일치
10. index column 없음
11. 파일 크기와 SHA-256 기록

Validator report가 green이 아니면 upload하지 않는다.

## 10. 최종 test day 절차

사용자 제공 일정의 2026-08-31 00:00~23:59는 최신 공식 Rules에서 timezone까지 재확인해야 한다.

### T-72h

- final artifacts freeze
- base/model/data cache offline mirror
- clean machine rehearsal
- 두 후보의 예상 runtime 측정
- 디스크 여유·전원·GPU reservation
- Kaggle login/2FA/upload 권한 확인

### T-24h

- 코드 변경 freeze
- test ID가 없어도 결정되는 candidate seed derivation algorithm과 salt를 freeze
- 모든 checksum 출력·보존
- 규칙 원문의 `ID,answer` validator 최종 실행; sample artifact가 있으면 추가 교차검증
- fallback simple model 준비

### Test 공개 시 — 네트워크 구간

1. 공식 test 다운로드
2. 파일명·크기·SHA-256·다운로드 시각 기록
3. schema와 ID만 검사; 질문을 외부로 전송하지 않음
4. freeze한 알고리즘으로 per-ID candidate seed map 생성·hash 기록
5. 네트워크 차단

### 오프라인 추론 구간

1. 환경·artifact preflight
2. F-accuracy dry sample 5행
3. full generation
4. failure row retry
5. parser·vote·submission
6. validator
7. second independent validator
8. 결과 directory를 read-only snapshot

### 업로드 구간

1. 추론 process 종료 확인
2. submission SHA-256 대조
3. 네트워크 재연결
4. 파일 업로드
5. Kaggle receipt/시간/score 보존
6. 업로드한 파일을 다시 다운로드할 수 있으면 hash 대조

추론과 업로드의 네트워크 상태를 로그로 증명한다.

## 11. 운영진 재실행 패키지

최종 패키지:

- base model locator와 허용되는 로컬 artifact
- adapter/checkpoint와 hashes
- tokenizer files
- `requirements.lock` 또는 container recipe
- Python/PyTorch/CUDA/GPU matrix
- training configs
- training entrypoint
- data manifests와 public source locators
- inference entrypoint
- parser/verifier tests
- sample smoke data
- final run manifest
- README
- experiment/ablation report
- license/NOTICE/attribution

Secrets, Kaggle token, W&B token, local absolute user path는 넣지 않는다.

## 12. Failure recovery

| 실패 | 대응 |
|---|---|
| GPU OOM | batch 축소, candidate를 chunk, F-efficient fallback |
| 특정 행 parser invalid | freeze된 fallback이 있으면 그 ID만 재시도; 여전히 invalid면 submission 생성 중단 |
| model cache 누락 | inference 중 download 금지; 사전 preflight 실패 처리 |
| tool timeout | pure CoT fallback |
| verifier crash | unweighted majority fallback |
| 총 runtime 초과 예상 | adaptive max N 하향, easy 종료 강화 |
| CSV validator fail | 원인을 고치고 전체 재생성; 수동 셀 편집 금지 |
| F-accuracy artifact 손상 | checksum으로 감지, F-simple 사용 |

Fallback도 사전에 full rehearsal해야 한다.

## 13. 완료 기준

다음이 모두 true일 때만 “제출 준비 완료”라고 판정한다.

- [x] 규칙 원문에서 `submission.csv`, `ID,answer` 확인
- [ ] 실제 sample submission artifact와 SHA 확보
- [ ] test와 model artifact checksum
- [ ] network-off full run 2회
- [x] parser golden tests (actual fold-0 diagnostic의 redacted outcome class 기반)
- [x] production validator와 독립 최소 CSV validator 교차검증 구현
- [ ] 모든 ID의 raw generation과 selected answer provenance
- [ ] runtime이 window의 50% 이하인 여유 계획
- [ ] fallback model full run
- [ ] organizer rerun README dry-run
- [ ] rule/license checklist green
