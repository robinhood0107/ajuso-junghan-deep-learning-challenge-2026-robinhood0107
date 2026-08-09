# Deep Learning Challenge 2026 — 조사·실행 계획 및 선행 구현

기준 시각: **2026-08-10 KST**
대상 대회: Kaggle [Deep Learning Challenge 2026](https://www.kaggle.com/competitions/deep-learning-challenge-2026)
고정 베이스: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

> **현재 운영 계약:** 로컬 구현 루트는 `$PROJECT`, 데이터 루트는
> `$PROJECT/deep-learning-challenge-2026`이다. 2026-08-03 14:00 정제본인
> 831행 leaderboard와 627개 train exclusion ID를 사용한다. split v4는 재생성하지
> 않고 hard-group exclusion overlay를 적용한다. training cache-off와 eval KV-cache-on을
> 분리한 current source의 B0 preflight와 local synthetic smoke는 run tag
> `20260810T062500KST`에서 green이다. 이 tag에 bound된 production B1을 시작하기 전에는
> 직전 read-only VRAM threshold를 다시 통과한다.
> selection-eligible B1/QLoRA 점수는 아직 없다. 보강 전 source의 fold 0 base diagnostic은
> 2,942문항 중 1,210 exact match(41.1285%)를 기록했지만 parser/latency 관찰용일 뿐
> 모델 선택에 쓰지 않는다. 정확한 실행 명령은
> [Gate B CPU-ready 런북](docs/10_GATE_B_CPU_READY_RUNBOOK.md)에 있다.

## 한 줄 결론

이 대회의 가장 유망한 주 경로는 **깨끗한 그룹 분할 → 정답과 풀이가 검증된 간결한 CoT로 QLoRA/LoRA SFT → 같은 베이스 모델의 rejection self-training → 제한적인 preference/RLVR 실험 → 정수 파서·로컬 계산 검증·adaptive self-consistency**이다.

기법 수를 늘리는 것보다 다음 세 가지가 성패를 좌우한다.

1. 공개 리더보드와 구조적으로 닮은 문제가 많은 데이터에서 **템플릿 중복 누출을 막는 내부 검증**
2. 결측·오표기·불완전 문제를 걸러 내는 **데이터 품질 관리**
3. 인터넷이 완전히 차단된 상태에서도 동일 제출을 재생성하는 **end-to-end 재현성**

```mermaid
flowchart LR
    A["규칙·원본 잠금"] --> B["품질 분류·그룹 분할"]
    B --> C["Base 기준선"]
    C --> D["검증된 CoT SFT"]
    D --> E["균형 rejection self-training"]
    E --> F{"Preference/RL이 clean CV를 개선?"}
    F -- "예" --> G["승자 checkpoint"]
    F -- "아니오" --> H["pre-RL checkpoint 유지"]
    G --> I["검증·adaptive self-consistency"]
    H --> I
    I --> J["인터넷 차단 재현·제출"]
```

## 현재 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| 최신 로컬 데이터 계약 | **완료** | train 17,000행, filtered leaderboard 831행, organizer exclusion 627 ID의 hash·schema·row 검증 |
| 영상 파일 기술 감식 | 완료 | 전체 12,889프레임 디코드, 스트림·해시·슬라이드 전환 감사 |
| 영상 슬라이드 내용 분석 | 완료 | 13개 구간의 규칙·제출물·평가·일정 내용을 화면 근거로 기록 |
| 영상 음성의 완전 전사 | **미완료** | 내장/동봉 자막과 로컬 ASR 모델이 없어 음성 발화를 검증 가능한 방식으로 전사할 수 없음 |
| Kaggle authenticated API 재검증 | **완료** | 참가 중인 slug·Rules/Data/Evaluation·파일 목록·현재 제출 가능 횟수(5)를 읽기 전용 확인; 파일 목록에는 sample submission이 없음 |
| 논문·방법론 조사 | 완료 | SFT, LoRA/QLoRA, self-training, preference, GRPO/RLVR, verifier, TIR, test-time compute, 오염 감사 |
| model-free Gate A | **READY** | strict loader, filtered audit, split v4 overlay, parser, grouped evaluation, voting, uppercase-ID submission, provenance와 회귀 테스트 |
| 모델·토크나이저 preflight | **current-source B0 READY** | run tag `20260810T062500KST`에서 pinned tokenizer와 2개 full weight shard/commit, CUDA/BF16/NF4 QLoRA runtime, physical VRAM preflight, cache-on local synthetic smoke를 green으로 확인했다. |
| 실제 QLoRA 학습·모델 추론 | **B1 production 실행 대기** | old source fold 0 base diagnostic은 2,942문항·1,210 exact match(41.1285%)와 redacted parser audit을 남겼다. attention-mask/cache 보강 전 run이므로 모델 선택·QLoRA 승격에는 쓰지 않으며, current-source B0 tag와 versioned production baseline 뒤에만 QLoRA를 시작한다. |

따라서 **Gate A는 READY**이며, current-source Gate B0 GPU gate도 green이다. selection-eligible
실제 모델 점수는 아직 없다. full pinned cache와 전용 QLoRA 환경은 준비됐고, local synthetic
smoke가 최초의 실제 CUDA workload였다. current-source production GPU workload는 같은 tag의
preflight/smoke, 직전 read-only VRAM 조건, versioned artifact target을 모두 확인한 뒤에만
실행한다. 구현·테스트의 최종 상태는
[선행 구현 및 검증 상태](docs/09_IMPLEMENTATION_STATUS.md)에 기록한다.

## 가장 중요한 데이터 발견

- 현재 입력 파일은 `deep_chal_math_train.csv`, `deep_chal_math_leaderboard_filtered.csv`, `train_filtered_ids.csv`다. 과거 1,000행 leaderboard는 역사적 감사 근거일 뿐 현재 제출 ID source가 아니다.
- train 17,000행의 답은 모두 canonical signed integer지만, 문제 품질은 균일하지 않다.
- 재현 가능한 audit v3에서 train 내부 math-aware 중복은 4그룹/8행, 좁은 source-format 중복은 8그룹/16행이다. 별도의 초기 공격적 탐색에서는 13개 후보 그룹이 나왔으며, 정의가 다르므로 수치를 합치지 않는다. 최소 한 쌍은 같은 문제에 서로 다른 답이 붙은 명백한 라벨 충돌이다.
- 최신 filtered audit의 development CV는 14,736행이다. organizer 627 ID를 hard group으로 확장하면 총 629행이 제외되고 전체 eligible은 16,371행이다.
- pinned tokenizer와 같은 시스템 프롬프트 기준 CV chat 입력 최대는 1,119토큰, filtered leaderboard inference 입력 최대는 1,276토큰이다. 현재 12GB 시작 config는 sequence length 2,048로 고정한다.
- 과거 raw leaderboard의 malformed ` answer` header는 제한적으로 읽기만 한다. 현재 filtered 입력 header는 `id,question`이고, 제출은 authenticated Rules/Evaluation로 확인된 `ID,answer`를 기본 schema로 사용한다. 현재 authenticated 파일 목록에는 sample CSV가 없다.

자세한 수치와 예외 목록은 [데이터셋 포렌식](docs/02_DATASET_FORENSICS.md)에 있다.

## 권장 진행 순서

### 0. 규칙과 증거 잠금

- Kaggle Rules·Data·Evaluation·Submission을 token-authenticated API로 읽기 전용 확인했고, 응답 hash와 파일 목록을 artifact로 보존했다.
- 공개·동등 접근 가능한 외부 데이터와 상용 teacher의 **training-data 생성**은 규칙상 허용됐다. leaderboard/test 입력은 금지다. 로컬 Python/SymPy와 same-base multi-adapter/checkpoint 결합은 서면 확인 전 비활성화하며, Kaggle upload는 사용자의 명시 요청 없이는 하지 않는다.
- 원본 CSV와 영상 해시를 manifest에 고정한다.

### 1. 평가 기반부터 만든다

- 정수 answer parser와 submission validator를 모델보다 먼저 구현한다.
- train을 정규화·템플릿 cluster 단위로 나눠 반복 선택용 development 5-fold와 한 번만 여는 sealed final holdout을 만든다.
- 원본 greedy, direct answer, concise CoT, sampling N별 `greedy/pass@N/majority@N`을 측정한다.
- 리더보드는 최종 sanity check에만 사용하고 하이퍼파라미터 탐색기로 쓰지 않는다.

### 2. 주력 모델 경로

- 현재 RTX 4070 SUPER 12GB의 첫 실행은 NF4 double-quant/BF16 compute, seq 2,048, microbatch 1, gradient accumulation 16, rank 16/alpha 32, gradient checkpointing on으로 시작한다. training은 `use_cache=false`를 유지하고, `model.eval()`의 직렬 generation은 KV cache를 명시적으로 켠다.
- rank 32나 4K sequence는 실제 free-VRAM smoke와 개발 성능 근거를 통과한 뒤에만 별도 config로 비교한다. full BF16 fine-tuning은 이 12GB 장비의 경로가 아니다.
- 제공 train의 정답만 있는 예제를 그대로 장황한 풀이로 꾸미지 않는다. 같은 Qwen 베이스에서 여러 풀이를 생성하고 정답·계산을 검증해 간결한 풀이만 채택한다.
- 절차적으로 생성한 integer-only 문제와 라이선스·출처가 명확한 공개 데이터만 단계적으로 추가한다.
- SFT 다음은 STaR/rejection sampling과 난이도 균형을 우선하고, KTO/ORPO/DPO와 GRPO는 각각 고정 검증에서 실제 이득이 있을 때만 유지한다.

### 3. 추론과 제출

- greedy → 4개 후보 → 합의가 약한 문제만 8/16/32개로 늘리는 adaptive sampling을 사용한다.
- 답 수준 voting, 로컬 산술·대입 검증, 필요 시 같은 Qwen 베이스에서 학습한 selector를 결합한다.
- raw generation, 추출 답, 후보 표, seed, latency를 모두 저장한다.
- 인터넷 차단 dry-run에서 모델 로드부터 `submission.csv`까지 다시 만들어 해시와 행 수를 검증한다.

### 4. 발표 준비

- 모든 실험에 데이터 manifest, config, Git SHA, 환경 lock, 모델/adapter checksum, seed, 비용, 실패 이유를 남긴다.
- 발표는 최고 점수 하나가 아니라 **어떤 실패를 어떻게 찾아 제거했고, 어느 단계가 얼마를 올렸는지**를 ablation과 오류 사례로 증명한다.

## 문서 지도

1. [증거·대회 규칙 감사](docs/01_EVIDENCE_AND_COMPETITION_AUDIT.md)
   원본 manifest, 영상 타임라인, 현재 확인된 규칙, 출처 간 불일치와 검증 한계
2. [데이터셋 포렌식](docs/02_DATASET_FORENSICS.md)
   행 수·분포·토큰 길이·중복·유사도·이상치, 품질 tier와 안전한 분할 설계
3. [방법론·논문 지도](docs/03_METHODS_AND_LITERATURE.md)
   SFT부터 RLVR·도구 사용·test-time compute까지 근거, 기대효과, 비용, 규칙 위험
4. [실험·학습 마스터 플랜](docs/04_EXPERIMENT_AND_TRAINING_PLAN.md)
   baseline부터 최종 모델 freeze까지 단계별 실험표, 승격·중단 게이트, compute별 트랙
5. [오프라인 추론·제출·재현성](docs/05_INFERENCE_SUBMISSION_REPRODUCIBILITY.md)
   parser, voting, verifier, sandbox, submission 검증, organizer 재실행 패키지
6. [일정·리스크·발표 계획](docs/06_OPERATIONS_SCHEDULE_PRESENTATION.md)
   31일 운영 캘린더, 역할, 실험 원장, 리스크 레지스터, 발표 평가 대응
7. [운영진 규칙 확인 질문서](docs/07_RULE_CLARIFICATION.md)
   Discord/Kaggle에 그대로 보낼 수 있는 짧은 질문과 답변별 의사결정
8. [출처 목록](docs/08_SOURCES.md)
   공식 모델·라이선스·논문·공개 데이터 링크와 사용 목적
9. [선행 구현 및 검증 상태](docs/09_IMPLEMENTATION_STATUS.md)
   실제 구현 모듈, audit/split/tokenizer 산출물, 테스트·커버리지, 재현 명령과 남은 blocker
10. [Gate B CPU-ready 실행 런북](docs/10_GATE_B_CPU_READY_RUNBOOK.md)
    현재 WSL/RTX 4070 SUPER에서 CPU 검증부터 5-fold OOF freeze, one-shot holdout,
    filtered leaderboard 제출까지 그대로 실행할 명령과 중단 조건
11. [지속 실행 체크리스트와 공개 저장소 경계](docs/11_EXECUTION_CONTINUATION_PLAN.md)
    중단 후 재개 순서, GPU·holdout·submission gate, Git 공개 제외 대상을 고정

## 절대 하지 않을 것

- 금지된 Qwen2.5-Math, DeepSeek-R1, Llama 계열 가중치의 merge·ensemble·추론 사용
- leaderboard/test 문제를 외부 상용 API나 다른 모델에 넣어 답 생성
- leaderboard 질문을 합성 데이터 seed, self-training prompt 또는 training set으로 사용
- 공개 리더보드 한 번의 상승만으로 방법을 채택
- 출처·라이선스·생성 모델을 알 수 없는 대규모 CoT dump를 무검증 투입
- parser 실패를 조용히 0으로 대체
- 인터넷 연결·다운로드 cache에 의존한 상태로 최종 추론
- 영상 음성을 전사하지 못한 상태에서 “영상 전체 발언을 완전 분석했다”고 주장

## 구현 단계별 체크포인트

- **A — model-free 기반:** strict loader, manifest, quality flags, 안전한 group split, parser, grouped evaluator, voting, submission writer/validator를 구현·테스트했다. **현재 완료 상태**다.
- **B — organizer-only 단일 adapter:** base direct-answer 개발 기준선을 먼저 실행하고, 그 뒤 answer-only QLoRA와 독립 검증된 concise rationale를 분리 비교한다. pre-mask/cache-off fold 0 diagnostic은 완료되어 raw를 공개하지 않는 parser audit과 safe synthetic regression으로 전환했지만, 이 결과는 수집·검증용이다. eval KV-cache source의 새 B0와 production baseline 전에는 승격 근거가 아니다.
- **C — 확장:** 규칙상 허용된 외부 공개 데이터와 training-only teacher rationale도 provenance/오염/품질 gate 뒤에만 쓴다. Python/SymPy TIR과 same-base 다중 adapter/checkpoint 결합은 서면 확인 전 비활성화한다.

따라서 모든 질문의 답이 올 때까지 안전한 기반 구현을 멈추지는 않지만, 답이 없는 기능을 묵시적으로 허용하지도 않는다.

이 저장소의 기존 `NU_` 파일은 작업 시작 전부터 존재한 사용자 파일이므로 수정하거나 삭제하지 않았다.

## 공개 저장소와 라이선스

이 공개 저장소에는 코드·테스트·문서·재현 명령만 포함한다. Kaggle 원본 데이터, 데이터
파생 artifact, 모델/checkpoint, prediction/submission, API token과 로컬 개인 파일은
`.gitignore`로 제외한다. 대회 데이터는 Kaggle에서 직접 받아야 하며, raw data를 이
저장소에 올리지 않는다.

코드는 [GNU General Public License v3.0](LICENSE)로 배포한다.
