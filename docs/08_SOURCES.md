# 08. 출처 목록

기준일: 2026-07-31. 구현 시에는 라이브러리 문서와 데이터 카드의 revision을 다시 고정한다. 기술 주장은 가능한 한 모델 공식 페이지, 원 논문, 공식 저장소를 우선했다.

## 1. 대회·모델

| 출처 | 이 계획에서의 용도 |
|---|---|
| [Kaggle: Deep Learning Challenge 2026](https://www.kaggle.com/competitions/deep-learning-challenge-2026) | 공식 대회 locator. 현재 비로그인 본문 접근 제한으로 최신 Rules snapshot 필요 |
| [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | parameter, architecture, context, tokenizer/chat template |
| [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | pretraining·post-training 배경 |
| [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) | 비상업 연구·평가, 파생물·표시·상업 사용 조건 |
| [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122) | synthesis, rejection SFT, GRPO, CoT/TIR 방법 참고. 가중치 사용 금지 |
| [Qwen2.5-Math official repository](https://github.com/QwenLM/Qwen2.5-Math) | 평가·prompt·tool 방법 참고. 모델/RM artifact 사용 금지 |

## 2. Parameter-efficient SFT

| 논문/문서 | 핵심 연결 |
|---|---|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 같은 base에서 값싼 adapter 실험 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 4-bit NF4 base + LoRA, 24GB 주력 |
| [Hugging Face PEFT LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora) | 구현 시 target/rank/version 확인 |
| [Hugging Face PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | QLoRA preparation |
| [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) | completion/assistant-only loss와 chat template |
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | 소량 고품질 예제 우선의 참고 근거 |
| [OpenMathInstruct-2](https://arxiv.org/abs/2410.01560) | 대규모 math instruction 데이터의 품질·형식·다양성 |
| [MetaMath: Bootstrap Your Own Mathematical Questions](https://arxiv.org/abs/2309.12284) | 문제 재작성·관점 다양화 |
| [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387) | 제한된 고품질 reasoning data 연구; 재현과 일반화 범위는 별도 검증 |

## 3. Self-training·rejection·curriculum

| 논문 | 핵심 연결 |
|---|---|
| [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) | 정답 검증된 rationale로 반복 self-training |
| [DART-Math](https://arxiv.org/abs/2407.13690) | 쉬운 문제 편향을 줄이는 difficulty-aware rejection |
| [DART-Math official repository](https://github.com/hkust-nlp/dart-math) | DARS 구현 참고 |
| [Curriculum Learning](https://icml.cc/Conferences/2009/papers/119.pdf) | 데이터 순서의 고전적 근거 |
| [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) | 절차적 integer 문제와 independent solver |
| [Open-R1: Update 1](https://huggingface.co/blog/open-r1) | 공개 reasoning pipeline, data·RL 재현 경험 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 검증 가능한 reward의 대규모 reasoning RL 참고; 모델 사용 금지 |
| [rStar-Math](https://arxiv.org/abs/2501.04519) | self-evolution·search·process reward 참고; 고비용 stretch |

## 4. Preference와 RL

| 논문/문서 | 핵심 연결 |
|---|---|
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | chosen/rejected hard-negative 학습 |
| [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) | SFT+preference 통합, 메모리 절약 |
| [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | paired data 없이 correct/incorrect 후보 활용 |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | PPO 원리; 대회에서는 후순위 |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | SFT→RM→PPO 구조 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO와 mathematical reasoning |
| [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) | process-aware online RL 참고 |
| [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) | GRPO clipping/length bias와 Dr. GRPO |
| [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947) | random/format reward에서도 점수가 오르는 해석 위험 |
| [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) | RL이 solution distribution을 재배분할 가능성, pass@large-N 평가 |
| [TRL documentation](https://huggingface.co/docs/trl/en/index) | SFT/DPO/KTO/GRPO 구현. 최종 requirements에서 exact version pin |

## 5. Verifier·process supervision

| 논문/자료 | 핵심 연결 |
|---|---|
| [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) | best-of-N candidate verifier |
| [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) | process vs outcome supervision |
| [PRM800K](https://github.com/openai/prm800k) | process label; MATH test contamination 주의 |
| [Math-Shepherd](https://arxiv.org/abs/2312.08935) | rollout 기반 자동 process label |

외부 Qwen/DeepSeek PRM 가중치는 fixed-base 규칙상 사용하지 않는다. 논문에서 방법만 참고하고 필요 시 같은 Qwen2.5-3B-Instruct base로 자체 verifier adapter를 만든다.

## 6. 추론·도구·test-time compute

| 논문 | 핵심 연결 |
|---|---|
| [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) | 여러 reasoning path의 answer voting |
| [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) | 모델 해석 + Python 계산 |
| [Program of Thoughts Prompting](https://arxiv.org/abs/2211.12588) | 계산을 외부 프로그램에 위임 |
| [ToRA: A Tool-Integrated Reasoning Agent](https://arxiv.org/abs/2309.17452) | 자연어 reasoning과 tool interaction |
| [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) | 난이도별 adaptive inference budget |
| [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) | 소량 고품질 data와 budget forcing |

Tool-integrated 방법은 운영진이 로컬 Python/SymPy를 허용할 때만 사용한다.

## 7. Ensemble·weight averaging

| 논문 | 핵심 연결 |
|---|---|
| [Model Soups](https://arxiv.org/abs/2203.05482) | 같은 base의 fine-tuned weights 평균 |
| [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) | SWA |
| [TIES-Merging](https://arxiv.org/abs/2306.01708) | delta sign conflict를 고려한 merge |
| [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099) | DARE 계열 delta sparsification |

다중 checkpoint/adapter가 규칙상 허용되고 prediction diversity가 실제로 있을 때만 후순위로 시험한다.

## 8. 평가 오염·중복

| 논문 | 핵심 연결 |
|---|---|
| [A Survey on Data Contamination in Large Language Models](https://arxiv.org/abs/2406.04244) | contamination 유형과 평가 위험 |
| [LLM Decontaminator](https://arxiv.org/abs/2311.04850) | paraphrase·번역이 n-gram 검사를 우회 |
| [ConTAM: A Framework for Quantifying Data Contamination](https://arxiv.org/abs/2411.03923) | contamination 영향 정량화 참고 |

이 대회에서는 exact hash뿐 아니라 number-masked template, math-token, character n-gram, 수동 고유사도 검토가 필요하다는 근거로 사용한다.

## 9. 공개 데이터 후보

| 데이터 | 공식 locator | 계획상 상태 |
|---|---|---|
| GSM8K | [Hugging Face](https://huggingface.co/datasets/openai/gsm8k), [GitHub](https://github.com/openai/grade-school-math) | MIT, train 후보; test는 진단 |
| MATH | [GitHub](https://github.com/hendrycks/math) | train integer subset 후보; 원문 권리·오염 감사 |
| DeepMind Mathematics | [GitHub](https://github.com/google-deepmind/mathematics_dataset) | Apache-2.0 procedural 우선 |
| SVAMP | [GitHub](https://github.com/arkilpatel/SVAMP) | perturbation 진단, MIT |
| MGSM | [Google Research](https://github.com/google-research/url-nlp/tree/main/mgsm) | 다국어 진단, CC-BY-4.0 |
| OlympiadBench | [GitHub](https://github.com/OpenBMB/OlympiadBench) | 고난도 진단 우선 |
| TheoremQA | [GitHub](https://github.com/wenhuchen/TheoremQA) | theorem/tool 진단 우선 |
| OpenMathInstruct-2 | [Hugging Face](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | CC-BY-4.0 카드; teacher/source 감사 후 |
| OpenR1-Math-220k | [Hugging Face](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Apache-2.0 카드; teacher/source 감사 후 |
| NuminaMath-CoT | [Hugging Face](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Apache-2.0 카드; AoPS/PDF 원천 주의 |
| OpenMathReasoning | [NVIDIA NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) | CoT/TIR/selector 연구; teacher와 source 권리 확인 |
| PRM800K | [GitHub](https://github.com/openai/prm800k) | process 연구, MATH 평가 오염 |

## 10. 출처 사용 원칙

각 외부 source에는 다음 manifest가 필요하다.

```yaml
name:
official_url:
revision_or_commit:
downloaded_at:
artifact_sha256:
dataset_card_license:
underlying_source_rights:
generator_or_teacher:
allowed_by_competition:
allowed_evidence:
contamination_report:
transform_code_sha256:
included_rows:
excluded_rows:
```

“공개 인터넷에서 구할 수 있음”, “Hugging Face 카드에 Apache라고 적힘”, “다른 대회에서 많이 씀”은 단독 승인 근거가 아니다.
