# 06. 일정·리스크·발표 계획

기준일: 2026-07-31 KST
사용자 제공 일정: 챌린지 2026-07-31~08-30, final test 공개/제출 08-31 00:00~23:59. 최신 공식 timezone과 window는 별도 확인한다.

## 1. 운영 원칙

- public leaderboard가 아니라 immutable group validation으로 모델을 고른다.
- 한 실험은 주 가설 하나만 바꾼다.
- 모든 run에 비용·seed·config·artifact를 남긴다.
- 실험을 시작하기 전에 성공/실패 기준을 적는다.
- 고비용 실험은 저비용 실험의 명확한 병목에서만 파생한다.
- 매주 “더 할 것”뿐 아니라 “영구히 버릴 것”을 결정한다.
- 마지막 4일은 새 기법이 아니라 freeze와 재현성에 쓴다.

## 2. 31일 일정

### Day 0 — 07/31: 증거와 계획

완료 목표:

- 원본 CSV·영상 manifest
- 데이터/영상/규칙 불일치 목록
- 연구 방법론과 end-to-end 계획
- 규칙 질문서

이 문서 세트가 해당 산출물이다.

### Week 1 — 08/01~08/04: 평가 기반

#### 08/01

- 로그인된 Kaggle Rules/Data/Evaluation/Submission snapshot
- sample submission 확보
- 규칙 질문 게시
- 원격 GPU 후보와 storage 결정

#### 08/02

- strict loader
- raw/normalized/problem-only schema
- quality rule 1차
- answer parser tests

#### 08/03

- duplicate/template/source clustering
- development 5-fold와 sealed final holdout
- contamination report v1

#### 08/04

- Qwen base download/revision lock
- greedy/direct/concise-CoT baselines
- extraction failure와 첫 오류 taxonomy

Week 1 gate:

- G0와 G1 통과
- baseline report
- public leaderboard를 아직 사용하지 않아도 됨

### Week 2 — 08/05~08/11: Clean SFT

#### 08/05~08/06

- clean problem-only dataset
- organizer train의 answer-only/concise-CoT target
- QLoRA smoke run, loss/throughput/OOM 확인

#### 08/07~08/08

- S00/S01/S02
- rank·target module 최소 grid
- 2K/4K 비교

#### 08/09

- BF16 LoRA 가능한 경우 QLoRA 대조
- seed 3회

#### 08/10

- category/quality/template slice 분석
- 첫 100개 paired error review

#### 08/11

- SFT best freeze
- public leaderboard upload는 수행하지 않음; 현재 사용자의 명시 요청과 freeze/holdout gate 뒤에만 가능
- 이미 합법적으로 보존된 public result가 있을 때만 내부 결과와 방향을 비교하며, 이를 위해 새 upload를 만들지 않음

Week 2 gate:

- base 대비 clean group validation 개선
- answer-leak shortcut 없음
- QLoRA/BF16 선택 근거

### Week 3 — 08/12~08/18: Rejection·외부 데이터

#### 08/12~08/13

- SFT model의 N=8/16 generation
- empirical difficulty
- answer/process verifier

#### 08/14

- uniform rejection과 DART식 균형 비교
- self-training iteration 1

#### 08/15

- procedural integer dataset pilot
- source/license/contamination manifest

#### 08/16

- GSM8K/MATH 등 허용 source의 작은 dose
- organizer-only 대비 ablation

#### 08/17

- self-training iteration 2
- diversity/length/shortcut 검사

#### 08/18

- best rejection model freeze
- public leaderboard upload는 명확한 내부 개선 모델 하나와 현재 사용자의 명시 요청이 모두 있을 때만

Week 3 gate:

- clean과 hard/unseen-template 동시 개선
- 외부 데이터의 기여를 source별로 설명
- self-training 한계효용 판정

### Week 4 — 08/19~08/25: Preference·RL·추론

#### 08/19

- KTO/ORPO/DPO용 candidate pool
- model-own hard negative

#### 08/20

- preference 2개 이내 비교
- 승자가 없으면 해당 branch 종료

#### 08/21~08/22

- GRPO group 4/8 pilot
- reward hacking, length, entropy 감사
- pre-RL model을 항상 fallback으로 보존

#### 08/23

- deterministic verifier와 self-consistency N curve
- pass@N vs majority@N 병목
- closed-book 대비 local BM25/char n-gram dynamic few-shot

#### 08/24

- 운영진 허용 시 CoT/TIR
- same-base selector는 필요한 경우만

#### 08/25

- adaptive test-time budget
- F-accuracy/F-efficient/F-simple 후보

Week 4 gate:

- preference/RL이 실제로 이기지 않으면 제거
- runtime과 accuracy Pareto frontier
- 최종 inference architecture 선택

### Week 5 — 08/26~08/30: Freeze·재현·발표

#### 08/26

- 데이터·코드·모델 freeze candidate 1
- dependency/container lock

#### 08/27

- network-off full leaderboard rehearsal
- 다른 clean environment에서 organizer-style rerun

#### 08/28

- fallback rehearsal
- parser/submission fault injection
- 발표 ablation table freeze

#### 08/29

- final code freeze
- model/tokenizer/adapter checksum
- 외부 데이터·license appendix
- 발표 초안

#### 08/30

- 최종 두 후보를 처음부터 끝까지 한 번씩 실행
- GPU 예약, 전원·디스크·login preflight
- 새 학습 금지

### 08/31 — 잠정 final test day

- test 다운로드·hash 기록
- network-off 추론
- 독립 validator 두 개
- submission 업로드
- receipt와 artifact 보존

일정이 공식 공지와 다르면 absolute date를 바꾸되 Week 5의 4-day freeze buffer는 유지한다.

### 09/01~09/20 — 운영진 평가·재현 대응

- 업로드 submission, raw predictions, model, source tree, environment를 immutable archive로 보존
- 운영진이 요구한 checkpoint·코드·외부 데이터 manifest를 공식 채널로만 제출
- 재현 문의는 수신 24시간 안에 acknowledgement, 72시간 안에 근거 있는 답변을 목표로 함
- organizer rerun과 원 제출이 다르면 hardware/library/seed/parser 단계별로 diff
- 결과를 맞추기 위해 사후 checkpoint나 CSV를 수정하지 않음
- 공개 금지·검증 중 embargo가 있으면 model/data/score를 외부에 배포하지 않음
- 모든 전달 파일의 SHA-256과 receipt를 보존

### 09/21~09/27 — Top-12 발표 대비

- 운영진의 실제 발표 대상·시간·rubric 확인
- 12장 deck와 5분/10분/15분 버전
- network-off demo 영상과 live fallback screenshots
- 예상 질문: fixed-base 준수, 데이터 오염, 외부 데이터 권리, RL 기여, 재현 비용
- 최소 3회 timed rehearsal와 제3자 반박 리뷰
- 수치 표를 final organizer-verified 결과로 교체하되 실험 기록을 다시 쓰지 않음

### 09/28 — 잠정 수상자 발표 이후

- 공식 결과·검증 상태·공개 가능 범위 확인
- 공개가 허용될 때만 model card, 코드, 보고서 release 후보 작성
- Qwen license와 외부 데이터 attribution 재검토
- 재현 archive를 read-only 장기 보존

## 3. 최소·경쟁·확장 트랙

### Minimum viable

- clean group split
- QLoRA concise-CoT
- rejection self-training 1회
- self-consistency N=8
- deterministic parser
- offline reproducibility

이 트랙은 반드시 완성한다.

### Competitive

- BF16 LoRA 대조
- DART식 balanced self-training
- KTO 또는 DPO
- GRPO가 gate를 통과할 때만
- deterministic verifier
- adaptive N=4~32

### Stretch

- TIR/SymPy
- same-base selector/PRM
- checkpoint ensemble/soup
- full FT
- tree search

Minimum viable가 end-to-end로 실행되기 전에 stretch를 시작하지 않는다.

## 4. 실험 예산

정확한 GPU 단가는 제공사 선택 시 현재 가격을 다시 확인한다. 운영상 실험 수를 다음처럼 제한한다.

아래 숫자는 실행 의무가 아니라 상한이다. Week 1에 500-step SFT pilot과 N=4 generation pilot을 실행해 다음 식으로 다시 계산한다.

\[
H_{train}=\frac{\text{seen tokens}}{\text{measured train tokens/s}\times3600}\times\text{restart overhead}
\]

\[
H_{gen}=\frac{\text{problems}\times N\times\text{mean generated tokens}}{\text{measured generation tokens/s}\times3600}
\]

함께 측정할 값:

- peak/reserved VRAM
- checkpoint GB/run
- raw generation bytes/candidate
- validation wall time
- cache cold/warm start
- 20% failure/retry margin

확정한 총 GPU-hour·비용·저장공간의 70%를 예상 실험, 20%를 finalist 반복, 10%를 장애 여유로 나눈다. 상한을 넘으면 stretch → preference → RL 순으로 자르고 MVP를 보존한다. 개인 참가에서는 모든 grid 조합을 돌리지 않는다. 고정 development probe에서 1 seed successive halving → 상위 2~3개만 5-fold 각 1 seed → 최종 config의 대표 조건만 3 seed로 확인한다. 표의 본실험 상한을 fold/seed 수만큼 다시 곱하지 않는다.

| 단계 | 최대 본실험 | seed | 중단 규칙 |
|---|---:|---:|---|
| baseline | 8 | sampling 3 | 병목 확인 시 종료 |
| SFT config | 12 | finalist 3 | 두 연속 무개선이면 축 종료 |
| data scale/source | 8 | finalist 3 | source가 clean을 악화하면 제거 |
| self-training | 6 | finalist 3 | iteration 이득 <0.3%p |
| preference | 4 | 2~3 | SFT보다 못하면 branch 폐기 |
| GRPO | 5 | pilot 1, finalist 3 | fresh eval 악화/해킹 시 즉시 중단 |
| inference | 12 | 3 | latency budget 밖이면 제거 |

전체 run을 동일한 public leaderboard에 제출하지 않는다. public 제출은 주당 최대 1~2개의 내부 승자만, 실제 Kaggle limit보다 여유 있게 운용한다.

## 5. One-person 운영 역할

개인 참가라면 시간을 나눠 다음 모자를 분리한다.

| 역할 | 책임 | 산출물 |
|---|---|---|
| Data steward | 원본·품질·license·split | manifests, contamination report |
| Modeling lead | SFT/RL 실험 | configs, checkpoints, metrics |
| Evaluation lead | frozen metrics·오류 분석 | scorecards, slices |
| Repro lead | offline package·submission | lockfiles, validators |
| Presentation lead | 주장의 근거와 시각화 | ablations, deck |

모델을 만든 직후 같은 시선으로 평가하지 않도록, 적어도 하루 간격을 두고 Evaluation lead 관점의 blind review를 한다.

팀 참가가 허용된다면 역할은 사람에게 배정하되, data manifest와 experiment ledger는 하나만 유지한다.

## 6. 주간 의사결정 회의

매주 고정 질문:

1. 이번 주 primary hypothesis는 무엇이었나?
2. clean group validation에서 몇 %p, confidence interval은?
3. 어떤 category가 좋아지고 나빠졌나?
4. public score 없이도 이 결정을 했을까?
5. 증가한 GPU-hour와 latency는 얼마인가?
6. 규칙·license·contamination 상태가 바뀌었나?
7. 다음 주에 버릴 branch는 무엇인가?
8. organizer가 지금 재실행할 수 있는가?

결론은 `promote/hold/reject` 셋 중 하나여야 한다.

### 통계 정책

- 반복 선택은 development GroupKFold만 사용
- 같은 문항의 두 모델은 paired correctness bootstrap 95% CI와 McNemar exact test를 참고
- 유형별 slice는 표본 수와 CI를 함께 표기
- 수십 개 탐색 run은 exploratory로 표시하고 p-value를 확정 주장에 사용하지 않음
- finalist 2~3개의 사전 정의 비교만 confirmatory로 두고 Holm 보정 적용
- +0.3/+0.5%p 같은 고정 threshold는 CI·seed variance·비용과 함께 판단
- sealed final holdout 결과를 본 뒤 hypothesis나 threshold를 바꾸지 않음

## 7. 리스크 레지스터

확률과 영향은 High/Medium/Low의 상대평가다.

| 리스크 | 확률 | 영향 | 조기 신호 | 완화 |
|---|---|---|---|---|
| 최신 Rules 오해 | M | H | 설명·영상 충돌 | 공식 snapshot, 서면 질문, feature flag |
| Qwen Research License 범위 | M | H | checkpoint 제출 요구 | 운영진/Qwen 확인, NOTICE·attribution |
| external teacher 금지 해석 | H | H | “고정 모델” 표현 | 승인 전 사용 금지 |
| train↔public 직접 중복 | 확인 | H | 최소 3개 고신뢰 | public 저가중치, group split |
| label conflict/오답 풀이 | 확인 | H | 10201/40401, boxed49 vs label51 | quarantine, independent solve |
| answer-leak shortcut | 확인 | H | boxed label 일치 20행 | problem-only, leak slice |
| missing visual/fragment | 확인 | M | 10개 visual 후보, `8.` | quality route, 별도 보고 |
| multi-part 단일답 모호 | 확인 | M | 두 remainder 등 | convention audit, low confidence |
| public leaderboard 과적합 | H | H | 내부/공개 방향 불일치 | 제출 제한, frozen CV |
| GPU 미확보 | M | H | 로컬 GPU 없음 | Week 1 예약, MVP QLoRA |
| OOM/느린 rollout | M | M | 4K/GRPO pilot 실패 | 2K, batch, QLoRA, early stop |
| RL reward hacking | M | H | format/length 상승, fresh 하락 | fresh procedural, reward ablation |
| self-training 오류 증폭 | M | H | diversity 하락 | process filter, iteration cap |
| 외부 데이터 권리 불명 | M | H | scraped AoPS/PDF | source manifest, 제외 |
| parser 오답 | M | H | human-correct extraction fail | golden tests, two validators |
| nondeterministic rerun | M | H | 반복 CSV 차이 | version/hardware pin, ID seed |
| internet-off cache 누락 | M | H | runtime download 시도 | clean offline rehearsal |
| final test window 오해 | L~M | H | timezone 불명 | 공식 일정 확인, T-72h check |
| 새 기법 늦은 도입 | H | M | 08/27 이후 변경 | freeze policy |
| 발표 근거 부족 | M | H | 최고 score만 있음 | run ledger, ablations, failures |

High-impact 리스크는 owner와 확인 일자를 experiment dashboard에 둔다.

## 8. 공개 리더보드 사용 정책

### 허용

- end-to-end submission이 정상인지 확인
- 내부 승자의 방향성이 완전히 반대인지 경고 신호로 사용
- final inference runtime/schema 검증

### 금지

- 매 prompt/hyperparameter를 제출해 선택
- train과 직접 중복된 공개 문제의 답을 수동 반영
- leaderboard 질문으로 synthetic sibling 생성
- public 점수만 보고 group validation loser를 승격

기록:

- submission artifact hash
- 제출 전 선택 이유
- 내부 예상
- public 결과
- 이후 판단

결과를 본 뒤 선택 이유를 다시 쓰지 않는다.

## 9. 발표 평가 대응

사용자 제공 정보상 최종 평가는 모델 성능 50% + 모델 우수성 발표 50%다. “모델 우수성” 세부 rubric이 아직 없으므로 다음 네 축으로 준비한다.

### 9.1 과학적 엄밀성

- random split의 문제를 실제 중복 수치로 증명
- immutable group validation
- confidence interval과 seed
- one-variable ablation
- 실패 실험 공개

### 9.2 방법의 우수성

- 왜 3B에 QLoRA/concise CoT가 맞는지
- 왜 rejection을 GRPO보다 먼저 했는지
- pass@N과 selector 병목에 따른 adaptive system
- deterministic verification과 exact-integer objective의 결합

### 9.3 효율성

- accuracy vs GPU-hour
- accuracy vs inference tokens/latency
- adaptive N이 uniform N보다 절약한 비율
- F-efficient/F-simple 비교

### 9.4 재현성·책임성

- source/license manifest
- contamination/answer-leak audit
- offline full run
- artifact checksums
- fixed-base compliance

## 10. 12장 발표 구성

1. **문제와 제약** — 3B 범용 모델, 정수 exact match, offline
2. **데이터를 믿기 전에** — label conflict, answer leak, direct duplicate
3. **검증 설계** — template/source group split와 slices
4. **Base 진단** — greedy/pass@N/majority@N
5. **주력 SFT** — concise verified CoT와 QLoRA/BF16
6. **Self-improvement** — balanced rejection
7. **Preference/RLVR** — 채택 또는 폐기 근거
8. **Inference system** — parser, verifier, adaptive voting
9. **Ablation** — 각 단계의 paired delta와 confidence interval
10. **효율** — GPU-hour·latency·tokens Pareto
11. **재현·규칙·라이선스** — organizer rerun
12. **결론과 한계** — missing visual, noisy labels, 다음 개선

Demo는 인터넷을 끊고 작은 sample에서 model→raw trace→integer→CSV를 보여 주는 방식이 가장 설득력 있다.

## 11. 필수 발표 표

### Ablation

| 모델 | Data | Training | Inference N | Verify | Clean CV | Unseen template | Public | GPU-h |
|---|---|---|---:|---|---:|---:|---:|---:|

### 오류 전이

| 오류 | Base | SFT | Rejection | Final |
|---|---:|---:|---:|---:|

### 효율

| 후보 | Accuracy | sec/problem | tokens/problem | VRAM | Repro level |
|---|---:|---:|---:|---:|---|

빈 칸을 추정값으로 채우지 않는다.

## 12. 중단·fallback 기준

### 08/11까지 SFT 개선 없음

- prompt/parser/data quality를 재검토
- 외부 데이터·RL로 도피하지 않음
- clean target과 loss mask 확인

### 08/18까지 self-training 개선 없음

- best SFT + self-consistency를 MVP로 고정
- preference 한 가지만 시도

### 08/23까지 RL 불안정

- RL branch 폐기
- pre-RL model에 inference budget 투자

### 08/27 offline rerun 실패

- F-simple로 즉시 축소
- tool/verifier/ensemble을 제거
- 재현이 될 때까지 새 모델 실험 중지

### Final day GPU 문제

- F-efficient 또는 F-simple artifact
- N budget 축소
- 이미 검증한 submission pipeline 유지

## 13. 완료 정의

대회 계획이 성공적으로 실행됐다는 뜻은 단순히 submission 하나를 올린 것이 아니다.

- clean group validation과 public/private 성능의 관계 설명
- 모든 모델 단계의 ablation
- 규칙·license·data provenance
- offline rerun
- parser/CSV 오류 0
- compute/time budget 준수
- 발표 자료에서 성공과 실패 모두 방어 가능

이 여섯 조건이 모델 점수와 함께 충족되어야 한다.
