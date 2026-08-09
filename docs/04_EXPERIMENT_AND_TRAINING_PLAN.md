# 04. 실험·학습 마스터 플랜

이 문서는 구현자가 순서대로 실행할 수 있는 대회 운영 계획이다. 각 단계는 산출물, 측정값, 승격 조건, 중단 조건을 갖는다. 이전 단계의 gate를 통과하지 않고 다음 고비용 단계로 넘어가지 않는다.

## 0. 2026-08-10 실행 잠금

- 현재 데이터는 filtered leaderboard 831행과 `train_filtered_ids.csv` 627 ID를
  사용한다. raw 1,000행 leaderboard는 현재 제출/추론 입력이 아니다.
- split v4(17,000행, 16,992 hard clusters)는 재생성하지 않는다. organizer exclusion을
  기존 hard group 전체로 확장한 `train-eligibility-v1-20260804.json`을 적용한다.
  직접 627행이 629행으로 확장되며, 전체 eligible 16,371행, development CV 14,736행이다.
- 각 fold 학습은 `eligible_training_ids(manifest, fold, exclusions)`만 사용한다. 이는
  validation fold, locked holdout, organizer exclusion이 속한 hard group을 모두 제외한다.
- post-split development 입력은 canonical `development-cv-v4` shard만 사용한다. shard는
  exclusion overlay 전 CV 15,300행만 포함하고 holdout row를 emit하지 않는다. 현재
  organizer exclusion 적용 후 5-fold OOF union은 정확히 14,736행이다.
- locked holdout은 primary/fallback을 development evidence로 freeze하기 전 문제·정답
  통계, tokenizer profile, 학습 예제, 생성, 모델 선택에 쓰지 않는다. 과거 split v4
  artifact에 holdout 분포가 들어간 사실은 역사적 한계로 보존하되 재사용하지 않는다.
- `ID,answer` 논리 schema는 규칙 원문으로 확인했다. 실제 sample CSV와 그 SHA는 아직
  확보하지 못했다.
- 공개·동등 접근 외부 데이터와 상용 API의 training-data/rationale 생성은 허용된다.
  leaderboard/test 입력은 금지다. Python/SymPy TIR과 same-base multi-adapter/checkpoint
  결합은 명시 답변 전 off다.
- RTX 4070 SUPER 12GB의 이전 source B0 preflight와 local synthetic NF4 smoke는 green이다.
  training cache-off와 eval KV-cache-on을 분리한 source는 새 tagged preflight/smoke를
  다시 만든 뒤에만 production B1을 허용한다. 각 새 GPU run 직전에도 free VRAM 10GiB 이상과
  pre-existing used VRAM 1GiB 이하를 다시 관측한다. leaderboard/test prediction은 여전히
  freeze 이후에만 허용한다.
- 보강 전 source의 fold 0 fixed-base diagnostic은 2,942 validation generation 중
  1,210 exact match(41.1285%)를 기록하고 redacted parser audit 19개 outcome class를
  만들었다. 이 run은 attention-mask/cache 보강 전의 parser·latency 관찰용이라 QLoRA,
  OOF, method selection, holdout으로 승격하지 않는다. 실제 output에서 고른 safe
  structural parser regression만 public code에 추가했고 production baseline은 새 B0 뒤에
  별도 tag로 다시 실행한다.
- fold 0 base/direct-answer를 첫 probe로 실행하고 실제 generation parser golden을 고정한
  뒤, 같은 base→QLoRA 순서를 fold 1~4에 반복한다. `compare-development-oof`는 모든
  label×fold run을 강제하고 전체 OOF union에서 paired cluster bootstrap, exact McNemar,
  Holm 보정을 다시 계산한다. base run은 pinned base checkpoint, adapter run은 실제
  adapter bundle의 fold/data/example provenance와 결속하며 run/adapter 재사용과 fold 간
  training-method fingerprint 혼합을 거부한다. 단일-fold 비교는 probe 전용이고 최종
  freeze·holdout·leaderboard/test 경로로 승격할 수 없다.
- 최종 실행 checkpoint는 OOF 전에 사전 지정한 deployment fold 0의 단일 checkpoint다.
  fold별 checkpoint를 ensemble하지 않는다. full-development refit은 별도 실험이며 현재
  Gate B first-pass 완료 조건으로 가장하지 않는다.
- same-base multi-checkpoint fallback은 운영진 서면 확인 전 off다. 기본 freeze는
  primary-only이고, 허용 확인과 OOF routing evidence가 모두 있을 때만 fallback을 켠다.

## 1. 최적화 목표

### Primary

- private/final test의 integer exact-match accuracy 최대화

### Co-primary

- 운영진이 오프라인에서 동일 submission을 재생성
- fixed-base와 외부 데이터 규칙 준수
- 발표 평가에서 방법의 우수성을 ablation으로 설명

### Guardrail

- public leaderboard 과적합 방지
- corrupted/missing-visual 문제와 정상 문제를 구분
- teacher/source/license/contamination provenance 유지
- local tool이나 다중 adapter의 규칙 상태를 feature flag로 분리

## 2. 전체 파이프라인

```text
raw evidence lock
  → strict CSV ingest
  → quality flags + problem-only view
  → duplicate/template/source clusters
  → immutable group folds
  → base inference baselines
  → QLoRA/BF16 LoRA SFT
  → verified rejection self-training
  → optional KTO/ORPO/DPO
  → optional GRPO/RLVR
  → CoT/TIR + verifier ablations
  → adaptive self-consistency
  → offline freeze rehearsal
  → final test inference
```

## 3. 제안 프로젝트 구조

아직 구현하지 않았으며, 구현 단계에서 만들 구조다.

```text
configs/
  data/
  train/
  infer/
data/
  raw_manifest/
  processed/
  folds/
  external_manifest/
reports/
  data_quality/
  contamination/
  experiments/
  final/
src/
  data/
  evaluation/
  generation/
  training/
  verification/
tests/
  data/
  parser/
  submission/
  sandbox/
artifacts/
  adapters/
  checkpoints/
  predictions/
  submissions/
```

원본 Desktop 파일은 복사본 해시 확인 후 read-only로만 참조하고, 전처리 결과를 원본 위에 쓰지 않는다.

### 규칙·구현 gate 연결

| 증거 감사 gate | 구현 gate | 의미 |
|---|---|---|
| R0 규칙 snapshot | G0의 rule manifest | 공개 외부/teacher training data는 provenance gate 후 on 가능; tool/same-base multi-checkpoint는 답변 전 off |
| R1 데이터 snapshot | G0 | 원본·품질·cluster·development/final split 잠금 |
| R2 compute·offline | G1 전 preflight | base artifact, GPU, cache, network-off smoke |
| — | G1 | base 병목 진단 |
| — | G2 | clean SFT 승격 |
| — | G3 | rejection self-training 승격 |
| — | G4 | preference 선택 또는 폐기 |
| — | G5 | RLVR 선택 또는 폐기 |
| — | G6 | final candidate·offline freeze |

## 4. 실험 원장 규격

모든 run은 다음을 갖는다.

```yaml
experiment_id:
hypothesis:
parent_run:
git_sha: # Git 저장소일 때
source_tree_sha256: # Git이 아닐 때 필수
rules_snapshot:
data_manifest_sha256:
fold_manifest_sha256:
base_model_revision:
tokenizer_revision:
adapter_config:
training_config:
inference_config:
prompt_hash:
parser_version:
verifier_version:
seeds:
hardware:
wall_time:
gpu_hours:
estimated_cost:
metrics:
artifact_checksums:
decision: promote | hold | reject
decision_reason:
```

W&B/TensorBoard를 사용해도 로컬 JSONL/CSV 원장이 권위 원본이어야 한다. 최종 재현은 외부 서비스 없이 가능해야 한다.

현재 workspace는 공개 Git 저장소이며, run에는 Git commit SHA와 content-addressed
`source_tree_manifest.json`을 함께 기록한다. 전자는 공개 코드 revision을, 후자는
uncommitted 상태까지 포함한 source/config/test/document tree를 식별한다. raw data,
artifact, credential, model/checkpoint는 어느 쪽에도 포함하지 않는다.

## 5. Phase 0 — 규칙·데이터·평가 잠금

### 목표

모델 없이도 데이터와 제출을 틀리지 않게 다루고, 이후 점수가 비교 가능한 기반을 만든다.

### 작업

1. 규칙 원문과 브라우저 접근 근거 보존; sample submission 파일/hash는 확보 시 추가
2. 원본 hash manifest
3. strict loader와 malformed leaderboard padding
4. `question_raw` / `problem_only` / `question_canonical` 분리
5. quality flags와 quarantine
6. duplicate/template/source cluster
7. immutable development group folds와 별도 sealed final holdout
8. integer parser와 evaluator
9. most-frequent answer dummy prediction으로 end-to-end smoke test

### 탐색 비용 프로토콜

1. 넓은 탐색: development probe 한 split, 1 seed, 짧은 budget
2. Successive halving: 명백한 loser를 25%/50% budget에서 제거
3. Finalist 2~3개: development 5-fold 각 1 seed와 complete OOF union 비교
4. 현재 first-pass deployment: 사전 지정 fold 0 단일 checkpoint; full-development refit과
   3-seed 안정성은 별도 승격 실험
5. Primary와 허용된 경우에만 fallback을 freeze한 후 sealed final holdout 1회

따라서 표의 실험 ID 수에 무조건 ×5×3을 곱하지 않는다.

### 필수 테스트

- embedded newline CSV roundtrip
- answer column leading-space header
- missing third leaderboard field
- negative/zero/16-digit answer
- duplicate/missing/extra ID
- Unicode minus, comma, decimal integer, fraction
- multiple conflicting final markers
- no answer
- question에 이미 `boxed` answer가 있는 경우

### Gate G0

- 원본과 fold manifest hash 고정
- known duplicate가 다른 fold에 없음
- final locked holdout ID·label 접근을 막는 sealed manifest와 접근 로그
- answer-leaked/contradictory-rationale 행이 clean fold에서 제외
- parser/submission tests 100% 통과
- baseline 결과가 반복 실행에서 동일

G0 전에는 GPU 학습을 하지 않는다.

## 6. Phase 1 — Base model 기준선

### 실험

| ID | 변화 | 목적 |
|---|---|---|
| B00 | 최빈값 dummy | 평가 harness smoke test |
| B01 | base direct greedy | 순수 instruction 능력 |
| B02 | base concise-CoT greedy | reasoning prompt 효과 |
| B03 | base structured-CoT greedy | 형식 안정성 |
| B04 | base concise-CoT sampling N=4/8/16 | pass/majority curve |
| B05 | 유형별 few-shot | in-context exemplar 효과 |
| B06 | max new tokens 256/512/1024 | 장황함과 truncation |
| B07 | context 2K vs 4K | 긴 문제와 처리량 |
| B08 | local char n-gram/BM25 dynamic few-shot | 반복 구조 retrieval 효과 |
| B09 | retrieved answer 표시 vs 가림 | 답 복사 shortcut 진단 |

### 공통 조건

- base revision 고정
- Qwen chat template 그대로
- development GroupKFold; final locked holdout은 이 단계에서 열지 않음
- 3 sampling seeds
- raw text와 normalized view 양쪽에서 corruption regression 검사

### Gate G1

다음 병목을 수치로 판정한다.

- prompt bottleneck: B02/B03이 B01보다 크게 개선
- sampling bottleneck: pass@16 높고 majority@16 낮음
- generator bottleneck: pass@16 자체가 낮음
- parser bottleneck: 사람이 맞는 응답인데 extraction 실패
- data bottleneck: corrupted/answer-leak slice와 clean 성능 차이 큼

이 진단 없이 SFT 데이터 형식을 결정하지 않는다.

## 7. Phase 2 — Clean SFT

### 7.1 학습 데이터 만들기

Organizer train에는 정답만 있고 순수 문제, 답 노출, worked solution, 틀린 solution이 섞여 있다.

각 행을:

- `problem_only`
- `answer_leaked`
- `worked_solution`
- `contradictory_rationale`
- `fragment/unanswerable`

로 분류한다.

Clean SFT의 우선 입력:

- Tier A problem-only
- 독립 검증한 기존 풀이
- 같은 Qwen base가 answer를 보지 않고 생성하고 정답·계산 검증을 통과한 concise rationale
- answer-only completion

Gold answer를 prompt에 보여 준 뒤 생성한 rationale는 별도 ablation 데이터로만 쓴다.

### 7.2 QLoRA 초기 config

시작점이며 고정 답이 아니다.

| 항목 | 시작값 |
|---|---|
| base | Qwen2.5-3B-Instruct pinned revision |
| quantization | NF4 4-bit, double quant, BF16 compute |
| LoRA target | all-linear |
| rank/alpha | **현재 12GB 시작값 16/32**; 32/64는 VRAM smoke 뒤 ablation |
| dropout | 0.05 |
| sequence length | **현재 12GB 시작값 2,048**; 4,096는 별도 VRAM smoke 뒤 비교 |
| microbatch / grad accumulation | 1 / 16 |
| loss | assistant completion only |
| optimizer | paged AdamW 8-bit 또는 검증된 고정 버전 |
| learning rate | 1e-4 시작 |
| scheduler | cosine + 3~5% warmup |
| epochs | 1~3, step 기준 early stop |
| gradient clipping | 1.0 |
| checkpoint | validation interval과 동일 |
| seed | 최소 3개 |

BF16 LoRA가 가능한 환경에서는 동일 data/order/step으로 대조한다.

### 7.3 SFT 실험표

| ID | 데이터 | 형식 | 목적 |
|---|---|---|---|
| S00 | organizer clean | answer-only | 형식/암기 하한 |
| S01 | organizer clean | concise CoT | 주력 |
| S02 | organizer clean | CoT 75% + answer-only 25% | concise 출력 균형 |
| S03 | S01 | attention-only LoRA | target ablation |
| S04 | S01 | all-linear LoRA | target ablation |
| S05 | S01 | rank 16/32/64 | capacity |
| S06 | S01 | QLoRA vs BF16 LoRA | quantization 영향 |
| S07 | S01 | 2K vs 4K | throughput/coverage |
| S08 | organizer clean 5K/10K/all ladder | concise CoT | 제공 데이터 내 한계효용 |

### Gate G2 — SFT 승격 규칙

Base concise-CoT 대비:

- clean overall +1.0%p 이상 또는 신뢰구간상 명확한 개선
- unseen-template 비악화, 허용 회귀 최대 0.5%p
- parser invalid 비증가
- rare-answer/negative slice 비악화
- 3 seed 중 한 run의 우연한 peak가 아니라 평균 개선

점수 차가 작으면 더 단순하고 빠른 설정을 선택한다.

## 8. Phase 3 — Verified rejection self-training

### 8.1 후보 생성

- SFT best checkpoint
- 해당 fold의 organizer training clusters와 규칙상 허용된 procedural/external training clusters만
- development validation과 final locked holdout의 문제·변형·retrieval 결과는 생성·필터·재학습에서 제외
- 문제당 N=8부터
- temperature 0.7/1.0 혼합
- prompt 2종 이내
- raw generations 영구 보존

### 8.2 필터

1. final integer exact match
2. canonical marker와 parser consistency
3. arithmetic/substitution verifier
4. repeated text, self-contradiction, answer copying 제거
5. 너무 긴 trace cap
6. 동일 문제의 near-duplicate rationale dedup
7. problem/template별 최대 accepted 수

### 8.3 난이도 균형

| pass rate | 예산 | 채택 cap |
|---|---:|---:|
| >0.75 easy | 4~8 | 1 |
| 0.25~0.75 medium | 8~16 | 2 |
| 0~0.25 hard | 16~32 | 2 |
| 0 unsolved | 제한적 32 | 검증 성공 때만 1 |

### 실험

- R00: uniform rejection
- R01: per-problem uniform accepted count
- R02: difficulty-proportional budget
- R03: outcome-only vs process-filtered
- R04: iteration 1/2/3

### Gate G3

- clean +0.5%p 이상
- hard/unseen-template 개선
- easy 성능 손실 ≤0.3%p
- output diversity 유지
- 반복 2회 이후 한계효용이 작으면 종료

## 9. Phase 4 — 외부 데이터

외부 데이터는 한 번에 섞지 않고 source별로 추가한다.

### 소스별 절차

1. 공식 URL·revision·license 저장
2. source-level rights와 생성 teacher 기록
3. integer-answer subset
4. challenge train/leaderboard와 contamination audit
5. format·언어·길이·유형 매핑
6. 5K/25K부터 작은 dose
7. source-only validation과 challenge validation 모두 측정

### 실험 순서

- X00: DeepMind procedural integer only
- X01: GSM8K train clean
- X02: MATH train integer subset
- X03: source-balanced mixture
- X04: 같은 Qwen base self-generated verified traces
- X05: 규칙상 허용된 training-only 외부 teacher dataset — 생성 모델/prompt/license/오염 provenance와 품질 검증 후만

### 혼합 sampling

처음에는 organizer data 50% 이상을 유지하고, source별 cap으로 14M 대형 데이터가 지배하지 않게 한다.

예시:

- organizer 50%
- verified self-generated 20%
- procedural 15%
- GSM8K/MATH human 15%

실제 비율은 ablation한다. 대회 유형과 다른 다국어·증명·비정수 문제를 규모만 보고 넣지 않는다.

## 10. Phase 5 — Preference

SFT/rejection best 하나에서만 시작한다.

| ID | 방법 | 데이터 |
|---|---|---|
| P00 | KTO | candidate-level correct/incorrect |
| P01 | KTO | process-filtered labels |
| P02 | ORPO | correct vs model-own near miss |
| P03 | DPO | hard-negative pairs |
| P04 | DPO | parser/format conflicts |

공정 비교:

- 동일 prompt set
- 동일 accepted/rejected candidate pool
- 비슷한 update tokens
- hyperparameter sweep budget 동일

### Gate G4 — Preference 선택 또는 폐기

중단:

- SFT보다 clean validation 낮음
- pass@N 크게 감소
- 같은 final answer만 반복하며 rationale diversity 붕괴
- format은 좋아졌지만 exact accuracy 이득 없음

Preference는 필수 단계가 아니다. 최고 SFT/rejection보다 못하면 최종 시스템에서 제거한다.

## 11. Phase 6 — GRPO/RLVR

### 사전 gate

- medium bucket이 group 내 성공/실패를 모두 생성
- reward parser가 adversarial unit tests 통과
- rollout GPU budget 확정
- fixed validation과 fresh procedural set 준비

### 실험

| ID | 변화 |
|---|---|
| G00 | group size 4, answer-only reward |
| G01 | group size 8 |
| G02 | tiny format reward 추가 |
| G03 | deterministic process bonus |
| G04 | uniform vs uncertain-problem curriculum |
| G05 | standard GRPO vs length-bias 완화 설정 |

### 매 checkpoint 평가

- exact accuracy와 reward
- clean/unseen-template/fresh-procedural
- number/condition perturbation
- greedy/pass@16/majority@16
- response length, repetition, entropy
- answer distribution collapse
- parser exploit 사례

### Gate G5 — RLVR 승격

- best pre-RL checkpoint 대비 clean +0.5%p 이상
- fresh procedural/unseen-template 동시 비악화
- pass@large-N 급락 없음
- output 길이와 latency가 최종 예산 안
- reward hacking audit 통과

RL이 이 gate를 통과하지 못하면 과감히 pre-RL checkpoint를 최종 후보로 유지한다.

## 12. Phase 7 — Verifier·TIR·추론 최적화

### 12.1 Deterministic verifier

문제 유형별 가능한 검사를 library로 만든다.

- arithmetic expression exact evaluation
- equation solution substitution
- system of equations substitution
- divisibility/modular constraints
- combinatorics exact integer
- probability rational denominator check
- geometry formula dimension·range sanity
- final answer와 generated code result 일치

Verifier 실패는 곧 오답 확정이 아니라 reranking signal이다. 자연어 해석이 틀렸으면 계산만 정확할 수 있다.

### 12.2 TIR

운영진이 허용한 경우:

- T00 pure CoT
- T01 Python-only tool trace
- T02 SymPy route
- T03 CoT/TIR mixed sampling
- T04 답 voting

모든 tool은 offline sandbox에서 실행한다.

### 12.3 Local retrieval

- IR00: closed-book vs number-masked char n-gram top-2/top-4
- IR01: BM25 vs char n-gram
- IR02: problem-only vs verified problem+solution example
- IR03: retrieved answer masked ablation
- IR04: exact-template candidate 제외/포함과 answer-copy 오류율

Corpus는 항상 해당 fold의 training clusters만 사용하고 query cluster를 제외한다. 별도 embedding 모델은 운영진 확인 전 사용하지 않는다.

### 12.4 Selector

`pass@16 - majority@16` 간극이 충분히 클 때만:

- same-base Qwen verifier adapter
- candidate 순서 shuffle
- 정답 빈도, verifier, deterministic check feature
- held-out problem과 held-out candidate generator로 평가

Selector가 candidate text에서 정답 cue를 외우지 않는지 number perturbation을 한다.

### 12.5 Adaptive budget

검증 세트에서 confidence→needed-N mapping을 학습하되 test answer를 사용하지 않는다.

```text
N=1 greedy
  if deterministic and high confidence: stop
N=4
  if top vote ≥3/4 and check passed: stop
N=8 or 16
  if unresolved: mix prompt/CoT/TIR
N=32
  only hard/unresolved within global budget
```

전체 평가 세트에서 per-problem max가 아니라 총 token/time budget을 지킨다. 현재
filtered leaderboard rehearsal은 831행 기준이고 final test 행 수는 공개 시 잠근다.

## 13. Phase 8 — 후보 freeze

최소 세 후보를 보존한다.

| 후보 | 목적 |
|---|---|
| F-accuracy | 제약 내 최고 validation accuracy |
| F-efficient | ≤0.5%p 손실에서 낮은 latency |
| F-simple | tool/verifier 없이 가장 재현 쉬움 |

Primary와 규칙상 허용된 경우의 fallback routing은 final holdout을 열기 전에 다음
development evidence로 선택한다.

- complete 5-fold OOF union의 paired duplicate-cluster 통계
- unseen-template
- fresh procedural
- public leaderboard는 sanity signal일 뿐 selection metric에서 제외
- offline runtime
- 규칙 상태

그 뒤 sealed final holdout을 한 번만 열어 사전 선언한 **단일 frozen policy**의 일반화와
catastrophic regression을 확인한다. holdout artifact는 primary와 fallback을 별도 후보처럼
채점하지 않고, 개발에서 freeze한 parser-invalid routing을 포함한 policy 점수 하나만
기록한다. 결과를 본 뒤 checkpoint·prompt·data·threshold·routing을 바꾸지 않는다.

### Gate G6 — Final freeze

- complete OOF development evidence로 primary와 하나의 deployment fold를 사전 선언
- sealed final holdout 1회 접근 로그와 결과
- holdout 결과를 이용한 모델 선택·재튜닝 금지
- model/tokenizer/adapter/source-tree/dependency checksum
- target hardware network-off full run 2회
- 규칙 원문의 `ID,answer` 기반 submission validator green; 실제 sample artifact/hash는 확보 시 교차검증
- final-day 전체 runtime이 공식 window의 50% 이하
- 규칙·라이선스 manifest green

### 현재 최종 학습 데이터 정책

현재 구현된 first-pass는 5-fold OOF로 방법 family를 검증하되, 사전 지정 deployment fold
0의 단일 adapter를 holdout/test에 사용한다. fold ensemble은 하지 않으며 same-base
multi-checkpoint fallback도 규칙 확인 전에는 사용하지 않는다.

development 전체를 다시 학습하는 `F-evidence` refit과 former holdout까지 포함하는
`F-all-data` refit은 이 문서의 장기 실험 후보일 뿐 현재 CLI에 구현됐거나 검증됐다고
간주하지 않는다. 특히 holdout을 연 뒤 `F-all-data`를 새로 만들어 제출 checkpoint를
바꾸는 경로는 현재 frozen-policy 계약에서 제외한다. 나중에 추가하려면 holdout 접근 전
별도 schema, step 산정식, adapter provenance, simulated refit 안정성, G6b를 먼저 구현하고
회귀검증해야 한다.

## 14. Compute별 트랙

### 현재 WSL/RTX 4070 SUPER 12GB

WSL RAM은 약 17.56GiB, swap은 8GiB다. NVIDIA RTX 4070 SUPER 12,282MiB와
WSL CUDA bridge는 존재한다. 2026-08-09 첫 smoke attempt는 model load 전 guard에서
중단됐지만, external occupancy와 CUDA-context overhead를 분리하도록 guard를 보강했다.
2026-08-10 새 target-host preflight와 local synthetic smoke는 pinned NF4 base load,
LoRA backward/optimizer step, generation/parser까지 green으로 확인했다. 이 evidence는
그 source의 organizer-only B1을 허용했지만, 이후 발견한 training/eval cache 분리 보강은
새 smoke가 필요하다. leaderboard/test prediction은 여전히 허용하지 않으며, 실제
development score는 artifact가 atomic publish되기 전에는 기록하지 않는다.

가능:

- CSV 분석, quality annotation
- parser/evaluator/submission tests
- contamination indexing
- 작은 tokenizer/prompt 검사
- result 분석과 문서

현재 점유 상태에서 부적합:

- 3B QLoRA 본학습
- 대규모 N-sampling
- GRPO rollout

### 12GB GPU — 현재 장비의 첫 경로

- NF4 double quant, BF16 compute QLoRA
- sequence 2,048, microbatch 1, gradient accumulation 16
- all-linear rank 16/alpha 32, gradient checkpointing, training `use_cache=false` / eval KV cache on
- base direct-answer B01 → answer-only SFT S00 순서
- self-consistency는 작은 순차 batch와 global token/time budget
- rank 32, sequence 4K, BF16 LoRA는 각각 별도 VRAM smoke 없이는 실행 금지

### 24GB GPU

- 4-bit QLoRA
- 2K/4K, gradient checkpointing
- moderate batch accumulation
- self-training generation batch
- group 4의 제한적 GRPO

### 48GB GPU

- BF16 LoRA 기준선
- 더 긴 sequence·batch
- same-base verifier
- group 8 GRPO

### 80GB 또는 multi-GPU

- full FT ablation
- 큰 rollout/PRM
- 1M data experiment

비용이 늘어도 동일한 validation 질문에 답하지 못하는 실험은 하지 않는다.

### 저장공간

2026-08-04 최신 검사 시 Windows C: 여유는 약 54.2GiB, WSL ext4 여유는 약
906.0GiB였다. full weights, 가상환경, checkpoint는 WSL ext4에 두고, Windows
workspace에는 source와 작은 분석 artifact만 둔다.

| 트랙 | 권장 여유공간 | 주요 소비 |
|---|---:|---|
| MVP | 100GB | base, QLoRA adapters, predictions, 2~3 checkpoints |
| Competitive | 300GB | BF16 variants, rejection rollouts, external subsets |
| Stretch | 1TB | full FT, 대형 rollout, PRM/ensemble |

각 run 종료 후 바로 삭제하지 말고 manifest에서 reject된 intermediate의 보존기한을 정한다. 삭제는 final artifact hash와 dependency graph를 확인한 뒤 별도 승인된 정리 단계에서만 한다.

## 15. 오류 분석 루프

각 promoted model마다 최소 100개의 paired 변화 사례를 검토한다.

### 분류

- problem misunderstood
- missing condition/visual
- wrong formula
- algebra/arithmetic slip
- case split omitted
- copied template answer
- correct reasoning, parser failed
- lucky correct answer
- answer leak followed
- generation truncated/repeated
- tool execution/error
- verifier selected wrong candidate

### 행동

| 오류 | 우선 해결 |
|---|---|
| 이해/공식 | data·SFT·retrieval example |
| 계산 | TIR/deterministic verifier |
| case split | verified rationale와 hard negative |
| parser | unit tests/postprocess |
| truncation | max tokens/prompt concision |
| missing visual | 별도 slice, 규칙상 가능한 정보만 |
| selector | candidate diversity/verification |

오류 분석으로 새 학습 예제를 만들 때 해당 evaluation row 자체나 변형을 train에 넣지 않는다.

## 16. 최종 의사결정 규칙

### Promote

- 사전 정의 primary metric 개선
- 주요 safety slice 회귀 없음
- 규칙 status green
- offline 재현 가능
- 비용 대비 이득 설명 가능

### Hold

- 결과가 confidence interval 내
- 일부 category 이득·일부 손실
- seed variance 큼

추가 1회 확인만 허용하고 끝없는 sweep을 하지 않는다.

### Reject

- public leaderboard만 개선
- group/unseen-template 성능 하락
- 규칙 또는 라이선스 불명
- 재현 실패
- reward hacking·answer copying
- 비용 증가 대비 미미한 이득

## 17. 권장 최종 주 경로

현재 first-pass의 현실적인 순서는 다음으로 고정한다.

1. G0 데이터/평가·규칙 잠금
2. B01 base direct-answer greedy를 모든 development fold에서 실행
3. 실제 raw completion으로 parser golden regression을 추가
4. S00 answer-only QLoRA를 동일 fold·동일 direct-answer 계약에서 실행
5. complete OOF paired cluster bootstrap, exact McNemar, Holm으로 primary/fallback freeze
6. frozen policy만 locked holdout에서 정확히 한 번 평가
7. strict offline prediction과 두 validator로 submission을 만들되, Kaggle upload는 사용자 명시 요청 때만 수행

concise rationale, 외부 공개 데이터, self-training, preference/RL, deterministic tool,
same-base multi-checkpoint fallback, adaptive self-consistency, full-development refit은 이
first-pass 이후의 **별도 versioned 실험**이다. rationale 품질·도구/결합 규칙·오염
provenance·개발 OOF 근거가 모두 green이 되기 전에는 이 경로에 섞지 않는다. PRM, MCTS,
full FT, model soup는 현재 고정 계약 밖이다.
