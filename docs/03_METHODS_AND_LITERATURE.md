# 03. 방법론·논문 지도

목표는 “유명한 기법을 최대한 많이 붙이는 것”이 아니라, 고정된 3B 범용 모델에서 정수 exact match를 가장 싸고 재현 가능하게 올리는 것이다. 각 방법은 **효과 가설, 비용, 규칙 위험, 반증 가능한 실험**을 함께 갖는다.

## 1. 모델 출발점

[Qwen2.5-3B-Instruct 공식 카드](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct):

- 3.09B parameters
- 36 transformer layers
- GQA 16 query heads / 2 KV heads
- 32,768-token context, 최대 8,192 generation
- BF16 계열 배포 크기 약 6.18GB

[Qwen2.5 기술 보고서](https://arxiv.org/abs/2412.15115)는 계열 전체가 약 18T token pretraining, 100만 개 이상의 SFT 예제, 다단계 RL을 거쳤다고 설명한다. 즉 범용 지식과 instruction-following 능력은 이미 있으므로, 대회 학습은 새 모델을 처음부터 만드는 일이 아니라:

1. 수학 문제를 정확히 구조화하고,
2. 불필요한 장황함과 형식 오류를 줄이고,
3. 정답이 검증되는 추론 경로의 확률을 높이고,
4. 여러 후보 중 정확한 정수를 안정적으로 고르는 일이다.

Qwen2.5-Math의 [기술 보고서](https://arxiv.org/abs/2409.12122)는 synthesis, rejection SFT, GRPO, reward model, CoT/TIR 등의 참고 방법을 제공한다. 그러나 **그 가중치·adapter·reward model을 사용하지 않고 방법만 참고**한다.

## 2. 방법 우선순위

| 방법 | 기대효과 | 비용 | 규칙 위험 | 우선순위 |
|---|---|---|---|---|
| 재현 가능한 base baseline | 모든 개선의 기준 | 낮음 | 낮음 | P0 |
| group split·quality filtering | 허위 상승 방지 | 낮음~중간 | 낮음 | P0 |
| QLoRA/BF16 LoRA SFT | 수학 풀이·형식 적응 | 중간 | 낮음 | P0 |
| verified concise CoT | answer-only보다 추론 강화 | 중간 | 낮음 | P0 |
| rejection/STaR self-training | 같은 베이스의 성공 경로 증폭 | 중간 | 낮음 | P0 |
| self-consistency | 즉시 가능한 정확도 상승 | 추론비 N배 | 낮음 | P0 |
| deterministic parser/verifier | 추출·계산 오류 감소 | 낮음 | 낮음 | P0 |
| DART식 난이도 균형 | 쉬운 문제 과대표집 방지 | 중간 | 낮음 | P1 |
| 로컬 구조 retrieval/few-shot | 반복 템플릿의 풀이 구조 제공 | 낮음~중간 | 확인 필요 | P1 |
| KTO/ORPO/DPO | hard negative에서 선호 학습 | 중간 | 낮음 | P1 |
| GRPO/RLVR | exact reward 직접 최적화 | 높음 | 낮음 | P1 |
| 로컬 Python/SymPy TIR | 계산 정확도 상승 | 중간 | 확인 필요 | P1 조건부 |
| 같은 베이스 verifier/selector | pass@N과 select@N 간극 축소 | 중간~높음 | 확인 필요 | P1 조건부 |
| checkpoint/prediction ensemble | 오류 비상관 시 상승 | 중간 | 확인 필요 | P2 |
| PRM/process reward | 과정 선택 개선 | 높음 | 외부 PRM은 높음 | P2 |
| full fine-tuning | 최대 적응 용량 | 매우 높음 | 낮음 | P3 |
| MCTS/대형 tree search | 고난도 탐색 | 매우 높음 | 중간 | P3 |
| 외부 teacher-generated CoT | 데이터 규모 확대 | 높음 | **높음** | 운영진 승인 후 |

## 3. 먼저 확립할 baseline

모든 반복 실험은 동일한 immutable **development GroupKFold**에서 비교한다. 별도의 final locked holdout은 primary/fallback을 freeze한 뒤 한 번만 연다.

### 3.1 Prompt 축

- direct answer: 풀이 없이 최종 정수만
- concise CoT: 핵심 식과 검산 후 `Final answer: <integer>`
- structured CoT: Given/Plan/Calculation/Check/Final
- zero-shot vs train에서 만든 2~4개 유형별 few-shot
- 영어 system prompt 2종 이내

few-shot 예제는 validation과 template cluster가 겹치지 않아야 한다. 특정 유형 예제가 전체 성능을 올려도 다른 유형을 악화할 수 있으므로 category별로 본다.

### 3.2 Decoding 축

- greedy
- temperature 0.4/0.7/1.0
- top-p 0.9/0.95
- max new tokens 256/512/1,024
- sampling N=4/8/16/32

측정값:

- `greedy@1`
- `pass@N`: N개 중 정답이 하나라도 있는 oracle upper bound
- `majority@N`
- `verified@N`
- `selector@N`
- invalid extraction rate, 평균 tokens, wall time

`pass@N`이 높고 `majority@N`이 낮으면 selector 병목이다. `pass@N` 자체가 낮으면 verifier보다 학습·prompt·tool routing을 먼저 고친다.

## 4. SFT와 parameter-efficient tuning

### 4.1 LoRA

[LoRA](https://arxiv.org/abs/2106.09685)는 base weight를 동결하고 저랭크 delta만 학습한다. 같은 원본에서 여러 데이터·seed·rank 실험을 싸게 비교하고 adapter provenance를 분리하기 좋다.

초기 grid:

- rank 16/32/64
- alpha = rank 또는 2×rank
- dropout 0/0.05
- attention-only vs all-linear target
- learning rate 5e-5/1e-4/2e-4
- effective batch tokens 고정
- epoch보다 optimizer step과 seen tokens를 기준으로 비교

### 4.2 QLoRA

[QLoRA](https://arxiv.org/abs/2305.14314)는 frozen base를 4-bit NF4로 로드하고 LoRA를 학습한다. 24GB급 GPU에서 가장 현실적인 주력이다.

기본안:

- 4-bit NF4, double quantization
- BF16 compute
- all-linear LoRA
- gradient checkpointing
- packing은 문제 경계·loss mask가 정확할 때만
- assistant completion-only loss
- Qwen tokenizer와 chat template 변경 금지

반드시 BF16 LoRA 소규모와 비교한다. 4-bit base가 긴 산술 추론에 미치는 영향을 가정하지 않는다.

### 4.3 Full fine-tuning

3.09B 모델의 mixed-precision AdamW는 parameter, gradient, FP32 master, 두 moment만 대략 `16 bytes × 3.09B ≈ 49GB`이고 activation과 runtime 여유분은 별도다. 단일 80GB 또는 ZeRO/FSDP multi-GPU가 필요하다.

제공 train만으로 full FT하면 과적합과 catastrophic forgetting 위험이 크므로 다음 조건에서만 한다.

- LoRA/QLoRA가 clean validation에서 포화
- 100K 이상의 검증된 다양 데이터 확보
- 80GB 또는 multi-GPU 예산 확보
- held-out general benchmarks와 category regression을 함께 측정

### 4.4 SFT target 형식

다음 세 가지를 독립 ablation한다.

1. answer-only
2. concise verified CoT
3. CoT와 answer-only mixture

권장 completion:

```text
Reasoning:
<short, checkable derivation>
Final answer: <signed integer>
```

새 special token을 추가하지 않는다. embedding/lm_head 처리와 tokenizer 배포 위험 없이 plain-text marker를 쓴다.

문제에 이미 solution이나 boxed answer가 섞인 데이터가 있으므로 question field 전체를 무조건 “순수 문제”로 보지 않는다. problem-only를 추출할 수 없으면 quality flag를 유지하고 clean SFT에서 제외한다.

## 5. 풀이 데이터와 커리큘럼

### 5.1 고품질의 정의

풀이 한 건은 최소한 다음을 만족한다.

- final answer가 gold integer와 exact match
- 식·대입·코드로 중간 계산을 다시 확인
- 문제 조건을 전부 사용하거나 불필요 조건을 명시
- answer를 먼저 보고 꾸민 사후 합리화가 아님
- 불필요한 반복과 장황한 메타 발언 없음
- 원천/생성 모델/prompt/seed/verifier version 추적 가능

[OpenMathInstruct-2](https://arxiv.org/abs/2410.01560)는 강한 teacher, 질문 다양성, solution format의 중요성과 지나치게 장황한 풀이의 역효과 가능성을 보고한다. 3B 모델에서는 긴 풀이를 무조건 고품질로 간주하면 안 된다.

### 5.2 데이터 규모 사다리

동일 validation에서 누적 subset을 비교한다.

- 5K clean
- 25K
- 100K
- 300K
- 1M 이하

[LIMA](https://arxiv.org/abs/2305.11206)는 1,000개 정제 예제로 response behavior를 크게 조정할 수 있음을 보였다. 수학 지식 습득을 그대로 증명하는 결과는 아니지만, 첫 단계에서 대규모 저품질 dump보다 작은 고품질 집합을 우선할 근거가 된다.

### 5.3 난이도

현재 모델로 문제당 8~16개를 생성해 empirical pass rate로 나눈다.

- easy: >0.75
- medium: 0.25~0.75
- hard: 0~0.25
- unsolved: 샘플 내 0

초기에는 easy/medium으로 형식과 기본 계산을 안정화하고, 이후 medium과 현재 약한 유형을 늘린다. hard만 과도하게 넣으면 sparse signal과 noisy rationale가 지배한다.

커리큘럼의 순서 자체도 ablation한다. easy→hard가 보편적 정답이라고 가정하지 않는다.

## 6. 합성·self-training

### 6.1 절차적 데이터

가장 규칙·품질 위험이 낮은 외부 증강은 Python/SymPy 기반 procedural generation이다.

1. 문제 family와 parameter range 정의
2. seed 고정
3. forward solver로 integer answer 생성
4. 다른 방식의 solver/substitution으로 독립 검증
5. 같은 seed/template sibling을 같은 split에 배치
6. 대회 leaderboard/test와 구조 중복 감사

[DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)은 Apache-2.0 generator를 제공한다. integer subset과 challenge 분포에 가까운 family부터 사용한다.

### 6.2 STaR

[STaR](https://arxiv.org/abs/2203.14465)의 대회형 반복:

1. 현재 Qwen adapter가 문제당 여러 rationale 생성
2. deterministic parser로 integer 추출
3. gold와 일치한 후보만 1차 채택
4. 산술·대입 검증과 repetition/length filter
5. 문제별 최대 채택 수를 제한
6. 새 SFT adapter 학습
7. frozen validation으로 반복 유지 여부 결정

정답을 힌트로 주고 재생성한 rationale는 `rationalized_with_answer`로 따로 기록한다. lucky answer나 사후 합리화가 될 수 있어 clean human/self-solved trace와 섞어 평가하지 않는다.

### 6.3 DART-Math식 균형

일반 rejection sampling은 쉬운 문제에서 정답 풀이를 많이 모으고 어려운 문제에서는 거의 얻지 못한다. [DART-Math](https://arxiv.org/abs/2407.13690)는 난이도 적응형 rejection sampling으로 이를 줄인다.

대회 적용:

- 문제당 최대 accepted trace 수 고정
- medium/hard에 더 많은 generation budget
- unsolved에 무한 budget을 쓰지 않음
- 유형·template별 quota
- accepted trace 수와 원래 pass rate 모두 저장

### 6.4 반복 한계

Self-training은 2~3회까지만 우선 계획한다. 다음이면 중단한다.

- validation 상승 <0.3%p이거나 confidence interval과 겹침
- output diversity 급감
- explanation 길이·반복 증가
- fresh procedural/unseen-template 성능 하락
- 동일한 잘못된 shortcut이 증폭

## 7. Preference optimization

### 7.1 KTO

[KTO](https://arxiv.org/abs/2402.01306)는 pair 없이 candidate를 desirable/undesirable로 분류해 학습할 수 있다. exact integer로 자동 label하기 쉬워 가장 싼 preference 후보다.

비교:

- outcome-only label
- process-filtered label
- 맞는 답이지만 lucky rationale인 후보를 별도 class/제외

### 7.2 ORPO

[ORPO](https://arxiv.org/abs/2403.07691)는 SFT와 odds-ratio preference를 한 단계에 결합하고 별도 reference model을 요구하지 않는다. 메모리가 제한된 24GB 환경에서 유리할 수 있다.

### 7.3 DPO

[DPO](https://arxiv.org/abs/2305.18290)는 `(prompt, chosen, rejected)`로 policy를 직접 최적화한다.

좋은 hard negative:

- 같은 풀이에서 부호 하나만 틀림
- 산술 한 단계 오류
- 조건 하나 누락
- 맞는 final을 우연히 냈지만 과정이 모순
- 여러 숫자를 남겨 parser를 혼란시킴
- 같은 template의 숫자를 이전 문제에서 복사

무작위의 명백한 오답보다 model-own near miss가 더 유용하다.

### 7.4 선택 원칙

KTO → ORPO → DPO 순으로 소규모 비교하되, 하나만 유지한다. preference 단계가 SFT 대비:

- clean accuracy,
- unseen-template accuracy,
- pass@N,
- output diversity,
- parser success

중 최소 두 핵심 지표를 개선하지 못하면 버린다.

## 8. GRPO와 RLVR

[DeepSeekMath](https://arxiv.org/abs/2402.03300)은 value model 없이 같은 prompt의 여러 응답을 그룹 상대 보상으로 쓰는 GRPO를 제안했다. 정수 answer reward가 정확하고 싼 이 대회와 구조적으로 잘 맞는다.

### 8.1 시작 조건

- SFT checkpoint가 medium 문제에서 non-zero pass rate 보유
- group 내 전부 정답/전부 오답이 지나치게 많지 않음
- parser와 reward unit tests 완료
- reward hacking probe 통과

### 8.2 보상

```text
reward =
  1.0 * exact_integer_correct
  + 0.02 * canonical_final_marker
  + 0.03 * deterministic_process_check
  - 0.02 * invalid_or_conflicting_final
  - capped_repetition_penalty
```

계수는 예시이며 반드시 ablation한다. answer reward가 압도적이어야 한다. format reward만 받고 오답을 내는 정책이 생기지 않도록 한다.

### 8.3 모니터링

- reward/accuracy correlation
- KL, entropy
- group reward variance
- output length/repetition
- language mixing
- category별 성능
- number perturbation robustness
- fresh procedural accuracy
- pass@N와 majority@N

### 8.4 왜 조심해야 하는가

[Spurious Rewards](https://arxiv.org/abs/2506.10947)는 Qwen 계열 수학 RL에서 random·format·잘못된 reward도 benchmark 상승을 낼 수 있어, 점수 상승이 올바른 추론 학습의 증거는 아니라고 경고한다.

[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)는 큰 sampling budget에서 base가 더 넓은 solution space를 가질 수 있고 RL이 기존 행동의 확률을 재분배하는 면을 보고한다.

[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)은 GRPO의 length/clipping bias를 분석하고 Dr. GRPO 계열 수정을 제안한다.

따라서 RL 성공은 validation accuracy 하나가 아니라:

- unseen template,
- 조건·숫자 perturbation,
- 중간 계산 재검증,
- diversity,
- pass@large-N

까지 통과해야 한다.

PPO는 policy/reference/reward/value 모델과 rollout 비용이 커서 이 규모에서는 후순위다. exact reward가 있는 상황에서는 GRPO보다 먼저 쓸 이유가 약하다.

## 9. Verifier와 process supervision

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)은 여러 후보를 생성하고 verifier로 선택하는 방향을 보여 준다.

[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)과 [PRM800K](https://github.com/openai/prm800k)는 process supervision의 가능성을 보였지만, PRM800K는 MATH test 5,000개 중 4,500개를 학습에 사용한 구성이라 표준 MATH test를 깨끗한 외부 평가로 유지할 수 없다.

[Math-Shepherd](https://arxiv.org/abs/2312.08935)는 중간 단계마다 여러 continuation을 rollout해 자동 process label을 만든다. 사람 label 비용은 줄지만 생성비와 noisy label 위험이 크다.

투자 판단:

- `pass@N`이 낮다 → PRM보다 generator 개선
- `pass@N`은 높고 `majority@N`이 낮다 → verifier/selector 가치 큼
- 산술/대입으로 검증 가능하다 → learned PRM보다 deterministic check 우선
- learned verifier가 필요하다 → 같은 Qwen2.5-3B-Instruct base의 별도 adapter로 만들고 운영진 확인

## 10. Tool-integrated reasoning

[PAL](https://arxiv.org/abs/2211.10435)과 [Program of Thoughts](https://arxiv.org/abs/2211.12588)는 자연어 해석을 모델에 맡기고 계산을 Python에 위임한다. [ToRA](https://arxiv.org/abs/2309.17452)는 language reasoning과 tool interaction을 결합한다.

규칙이 허용하면 CoT와 TIR 두 경로를 비교한다.

- pure CoT
- code/Python trace
- SymPy equation solve
- CoT/TIR answer-level voting

필수 sandbox:

- network disabled
- subprocess/filesystem 금지
- AST allowlist
- `fractions`, `decimal`, 제한된 `sympy`만
- import allowlist
- CPU/wall-time/memory/stdout 제한
- deterministic seed
- 실행 코드와 결과 transcript 저장

모델이 생성한 임의 Python을 `exec`로 직접 실행하지 않는다. Tool 사용이 금지되면 동일 인터페이스를 완전히 끌 수 있게 feature flag로 만든다.

## 11. 로컬 retrieval과 dynamic few-shot

train과 leaderboard 사이에 숫자만 바뀐 템플릿이 많으므로, 인터넷 없는 로컬 retrieval은 비교적 싼 후보 전략이다. 다만 “유사 문제의 답 복사”가 아니라 “풀이 구조 제공”으로 설계한다.

### Corpus

- organizer train의 Tier A problem-only
- 검증된 concise solution이 있는 행만 few-shot 후보
- 허용된 외부 데이터는 source별 index 분리
- leaderboard/test는 index에 절대 추가하지 않음

### Retriever 우선순위

1. number-masked math-token fingerprint
2. character n-gram TF-IDF/BM25
3. topic·식 구조 filter
4. 운영진이 허용할 때만 별도 embedding model; 기본값은 사용 안 함

### Prompt 정책

- query의 exact/near duplicate는 answer copying 위험 flag
- 2~4개 example cap
- example의 숫자와 query 숫자를 명시적으로 대조
- 마지막 instruction에 “예제 답을 복사하지 말고 현재 숫자로 재계산”
- context 길이와 latency 기록

### 공정 평가

- random split이 아니라 query cluster 전체를 retrieval corpus에서 제외한 leave-cluster-out 평가
- closed-book vs retrieval
- problem-only retrieval vs problem+solution retrieval
- exact template vs semantic retrieval
- retrieved example의 답을 가린 ablation

retrieval이 public leaderboard만 올리고 unseen-template를 악화하면 사용하지 않는다. 최종 추론의 로컬 train retrieval 허용 여부와 별도 embedding 모델 사용 가능성은 운영진에게 확인한다.

## 12. Self-consistency와 test-time compute

[Self-Consistency](https://arxiv.org/abs/2203.11171)는 다양한 reasoning path의 답을 vote한다.

정수형 가중 voting:

\[
\hat a = \arg\max_a \sum_i w_i \mathbf{1}[\operatorname{extract}(y_i)=a]
\]

\(w_i\) 후보:

- 1: 단순 majority
- deterministic check 통과 여부
- same-base verifier score
- trace diversity penalty

[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)은 문제 난이도에 따라 search와 verification 예산을 다르게 배분하는 것이 균일 best-of-N보다 효율적일 수 있음을 보였다.

권장 adaptive policy:

1. greedy 1개
2. sample 4개
3. top answer share ≥0.75이고 검산 통과하면 종료
4. 불일치하면 총 8~16개
5. high difficulty 또는 CoT/TIR 충돌이면 32개
6. verifier/selector는 최종 unresolved에만

[s1](https://arxiv.org/abs/2501.19393)은 1,000개 고품질 예제와 budget forcing을 탐구한다. 단순히 “Wait”를 반복시키면 3B 범용 모델에서 장황함만 늘 수 있으므로 reasoning SFT 후 작은 ablation으로 제한한다.

## 13. Ensemble과 model merging

[Model Soups](https://arxiv.org/abs/2203.05482)는 같은 base에서 fine-tune한 모델의 weight 평균이 추가 추론비 없이 일반화를 높일 수 있음을 보였다. [SWA](https://arxiv.org/abs/1803.05407), [TIES-Merging](https://arxiv.org/abs/2306.01708), [DARE](https://arxiv.org/abs/2311.03099)도 후보지만 후순위다.

순서:

1. 서로 다른 seed/checkpoint의 prediction diversity 측정
2. answer-level ensemble
3. greedy checkpoint soup
4. LoRA delta merge/TIES는 마지막

같은 Qwen2.5-3B-Instruct에서 나온 delta만 사용한다. Qwen2.5-Math 또는 다른 base와의 merge는 금지한다. 다중 adapter 허용 여부도 운영진에게 확인한다.

## 14. 외부 데이터

### 14.1 권장 안전 순서

1. organizer train
2. 독립 검증한 procedural integer problems
3. 라이선스가 명확한 human-authored train split
4. 같은 Qwen base가 만든 verified traces
5. 운영진이 명시적으로 허용한 외부 teacher-generated data

### 14.2 후보

| 데이터 | 용도 | 라이선스·주의 |
|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 산술 word problem | MIT; train만 학습, test는 진단 |
| [MATH](https://github.com/hendrycks/math) | 고난도 competition math | 저장소 MIT; 원문 출처 권리와 contamination 별도 감사 |
| [DeepMind Mathematics](https://github.com/google-deepmind/mathematics_dataset) | procedural generation | Apache-2.0 |
| [SVAMP](https://github.com/arkilpatel/SVAMP) | perturbation robustness | MIT, 파생 provenance 확인 |
| [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) | 다국어 진단 | CC-BY-4.0 |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | 대규모 SFT 후보 | CC-BY-4.0, teacher provenance·source rights |
| [Open-R1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | reasoning data 후보 | Apache-2.0 카드, 생성모델과 원천 확인 |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | competition CoT 후보 | 카드상 Apache-2.0; AoPS·시험 PDF 원천 권리 별도 |
| [PRM800K](https://github.com/openai/prm800k) | process 연구 | MIT; MATH 평가 오염 주의 |

데이터 카드의 라이선스가 수집된 모든 포럼 글·시험지·올림피아드 문제의 권리를 자동 해결하는 것은 아니다. dataset-level license와 source-level rights를 모두 기록한다.

## 15. Contamination 방지

[A Survey on Data Contamination](https://arxiv.org/abs/2406.04244), [LLM Decontaminator](https://arxiv.org/abs/2311.04850), [ConTAM](https://arxiv.org/abs/2411.03923)은 exact n-gram만으로 paraphrase·번역·template 중복을 충분히 잡기 어렵다는 점을 보여 준다.

외부 데이터마다:

1. exact normalized hash
2. 숫자 placeholder template hash
3. token MinHash
4. LaTeX/SymPy expression canonicalization
5. embedding 후보
6. top pair 수동 검토

를 수행한다. 공개 leaderboard의 텍스트는 오직 비교 대상이며 training·generation prompt·retrieval corpus에 절대 넣지 않는다.

## 16. 채택·폐기 원칙

방법 하나는 다음을 모두 기록해야 한다.

- hypothesis
- 변경되는 한 가지 축
- compute/time cost
- rule status
- validation result와 confidence interval
- category winner/loser
- parser failure·length·latency
- 실패 시 폐기 이유

우선 채택할 최종 후보는 “가장 복잡한 시스템”이 아니라 다음 Pareto frontier다.

1. **Accuracy model:** 제한 내 최고 exact accuracy
2. **Efficient model:** 정확도 손실 ≤0.5%p에서 훨씬 빠른 모델
3. **Reproducible model:** 오프라인 organizer rerun 위험이 가장 낮은 모델

발표용 모델은 세 기준의 trade-off를 보여 주고, 최종 제출은 대회 제약에 맞는 한 점을 선택한다.
