# 02. 데이터셋 포렌식

대상:

- `deep_chal_math_train.csv`
- `deep_chal_math_leaderboard.csv`

모든 수치는 2026-07-31 KST의 로컬 원본 해시에 고정된 결과다. 이 문서는 모델 성능 측정이 왜 단순 random split으로는 신뢰할 수 없는지 설명하고, 안전한 데이터 파이프라인을 정의한다.

## 1. 구조 감사

### 1.1 Train

| 항목 | 결과 |
|---|---:|
| 논리 행 | 17,000 |
| header | `id,question,answer` |
| ID | 17,000개, 연속·유일 (개별 ID는 public repository에서 제외) |
| 3열 CSV parse | 전행 정상 |
| question 결측 | 0 |
| answer 결측 | 0 |
| answer signed integer 정규식 통과 | 17,000 |
| raw exact question duplicate | 0 |

답 문자열에는 선행/후행 whitespace, `+`, leading zero, decimal, scientific notation이 없다. 모든 현재 값은 signed 64-bit 범위에 들지만, 제출 파이프라인은 부동소수점이나 int64에 의존하지 말고 arbitrary-precision 정수 문자열을 사용한다.

### 1.2 Leaderboard

| 항목 | 결과 |
|---|---:|
| 논리 행 | 1,000 |
| ID | 1,000개, 연속·유일 (개별 ID는 public repository에서 제외) |
| raw header | `id,question, answer` |
| 실제 데이터 폭 | 모든 행 2열 |
| 로컬 sample submission | 없음 |

헤더에는 answer 앞에 공백이 있고 각 행에는 빈 세 번째 필드를 나타내는 trailing comma조차 없다. 즉 strict 3-column parser에는 전행 width mismatch다.

안전한 ingest는 다음 조건을 만족해야 한다.

1. Python CSV parser를 `newline=""`로 연다.
2. header 이름만 strip하되 question 원문은 strip/정규화하지 않는다.
3. leaderboard에서 **오직 빠진 마지막 answer 필드만** 명시적으로 null padding한다.
4. 2열이 아닌 다른 width, 중복 ID, 예상 ID 범위 이탈은 즉시 실패시킨다.
5. 제출 schema는 이 결함 있는 파일에서 복사하지 않고 공식 sample submission으로 확정한다.

## 2. 정답 분포

| 통계 | 값 |
|---|---:|
| unique answers | 2,023 |
| minimum | -5,765,435 |
| maximum | 3,431,577,212,128,939 |
| negative | 502 (2.95%) |
| zero | 222 (1.31%) |
| positive | 16,276 (95.74%) |
| \|answer\| ≥ 1,000,000 | 133 |
| \|answer\| ≥ 1,000,000,000,000 | 3 |
| median \|answer\| | 30 |
| p75 \|answer\| | 162 |
| p90 \|answer\| | 1,500 |
| p95 \|answer\| | 6,085 |
| p99 \|answer\| | 500,000 |
| 최대 자릿수 | 16 |

최빈 정답은 `2`로 646개, 3.8%다. 이 값은 모델 없는 최빈값 baseline이며 정확도가 낮더라도 end-to-end 평가 코드가 제대로 작동하는지 확인하는 smoke test로 유용하다.

위 분포가 뜻하는 실무적 위험:

- 작은 양의 정수 편향으로 “항상 2” 같은 퇴행이 validation 일부에서 생각보다 덜 나빠 보일 수 있다.
- 음수 부호, 0, 쉼표가 있는 큰 수, 16자리 정수를 parser 단위 테스트에 반드시 넣어야 한다.
- CSV를 pandas float로 한 번이라도 변환하면 자릿수·scientific notation 문제가 생길 수 있다.
- accuracy만 볼 뿐 크기 오차를 평가하지 않으므로 `42`와 `43`도 완전 오답이다.

## 3. 문제 길이·문자·토큰

### 3.1 문자와 단어

| 통계 | Train | Leaderboard |
|---|---:|---:|
| chars p50 | 203 | 206 |
| chars p95 | 492 | 489 |
| chars p99 | 791 | 753 내외 |
| chars max | 4,517 | 4,391 |
| words p50 | 36 | 36 |
| words p95 | 85 | 88 |
| words max | 770 | 729 |
| multiline | 1,983 (11.66%) | 121 (12.1%) |
| LaTeX marker | 약 8.2K (48%) | 약 508 (50.8%) |

### 3.2 실제 Qwen tokenizer

[Qwen2.5-3B-Instruct tokenizer](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)을 사용해 측정했다.

| raw question tokens | Train | Leaderboard |
|---|---:|---:|
| p50 | 57 | 58 |
| p90 | 110 | 113.1 |
| p95 | 142 | 143.05 |
| p99 | 250.01 | 287.03 |
| max | 1,699 | 1,242 |
| >1,024 | 4 | 1 |
| >2,048 | 0 | 0 |

짧은 system instruction과 Qwen chat template를 포함하면:

| chat-formatted input tokens | Train | Leaderboard |
|---|---:|---:|
| p50 | 91 | 92 |
| p90 | 144 | 147.1 |
| p95 | 176 | 177.05 |
| p99 | 284.01 | 321.03 |
| max | 1,733 | 1,276 |

이는 `Qwen/Qwen2.5-3B-Instruct` commit `aa8e72537993ba99e69dfaafa59ed015b17504d1`, Transformers 4.57.6, tokenizers 0.22.2와 다음 system prompt를 고정해 다시 측정한 값이다.

```text
Solve the math problem carefully. Return the final signed integer as `Final answer: <integer>`.
```

판정:

- 원문 입력만 보면 2,048 context로 전행을 수용할 수 있다.
- 학습에는 assistant reasoning이 추가되므로 4,096 sequence length를 기본 실험으로 둔다.
- 2K/4K의 정확도·처리량을 비교하고 4K의 이득이 없다면 2K로 줄인다.
- 긴 문제 0.1%만 별도 bucket으로 두어 truncation 여부를 명시적으로 검사한다.
- max generation length는 입력 길이와 별도로 256/512/1,024를 비교한다. 불필요하게 긴 풀이가 정확도를 보장하지 않는다.

### 3.3 언어·정규화

휴리스틱상 영어가 train 16,664/17,000, leaderboard 985/1,000으로 대부분이다. 그러나 다음 흔적이 있다.

- train CJK/일본 문자 포함 45, Cyrillic 9, non-ASCII 653
- leaderboard CJK 2, non-ASCII 28
- NFKC 적용 시 변하는 행 train 133, leaderboard 7
- NBSP, tab, control character, 번역 잔재

원본 text와 model용 normalized text를 분리한다.

| 필드 | 정책 |
|---|---|
| `question_raw` | CSV에서 읽은 exact 원문, 수정 금지 |
| `question_display` | CRLF·제어문자만 안전하게 표시 |
| `question_model` | 검증된 최소 정규화만 적용 |
| `question_canonical` | 중복 탐지 전용; 학습 입력으로 사용하지 않음 |
| `normalization_flags` | NFKC 변화, odd dollar, brace 불균형, URL 등 |

무조건 NFKC·LaTeX 제거를 하면 전각 수학 기호나 의도된 형식이 망가질 수 있다.

## 4. 문제 유형 분포

정규식 기반의 1차 primary-topic 휴리스틱이므로 정답표가 아니라 stratification과 감사용 proxy다.

| 유형 | Train | Train % | Leaderboard % |
|---|---:|---:|---:|
| 산술 word problem | 5,092 | 29.95 | 27.7 |
| 응용·단위·과학·금융 | 2,996 | 17.62 | 16.7 |
| 기하 | 2,277 | 13.39 | 14.4 |
| 정수론 | 1,890 | 11.12 | 12.1 |
| 대수 | 1,470 | 8.65 | 11.1 |
| 조합·확률 | 1,267 | 7.45 | 8.2 |
| 미분·적분 | 472 | 2.78 | 2.7 |
| 수열 | 174 | 1.02 | 소수 |
| 선형대수 | 60 | 0.35 | 소수 |
| 미분류 | 1,302 | 7.66 | 나머지 |

leaderboard는 큰 틀에서 IID로 보이지만 대수 비율이 약 +2.45 percentage points다. 유형 휴리스틱만으로 “분포가 동일하다”고 확정할 수는 없다. template/source/길이/수식/언어/결측 시각정보까지 함께 비교해야 한다.

## 5. 중복·템플릿 누출 감사

중복은 하나의 수치로 말하면 안 된다. 정규화 강도마다 의미가 다르다.

### 5.1 Train 내부

| 레이어 | 정의 | 발견 |
|---|---|---:|
| L0 | raw string exact | 0 |
| L1 | Unicode/case/space canonical | 1개 그룹, 2행 |
| L2 | LaTeX·문장부호를 공격적으로 제거한 구조 유사 | 13개 그룹, 26행 |
| L3 | 숫자를 placeholder로 바꾼 exact template | 137개 그룹, 301행, 최대 7행 |
| L4 | fuzzy semantic/template similarity | 별도 cluster 생성 필요 |

위 표의 L1–L3는 최초 exploratory 감사 정의다. 현재 영구 구현의 `data-audit-v3`은 정의와 코드를 함께 고정했고 결과는 다음과 같다.

| audit v3 신호 | 용도 | Train 결과 |
|---|---|---:|
| `math_aware` | 숫자·연산자를 보존한 보수적 exact 비교 | 4그룹, 8행, 최대 2행 |
| `source_format` | 위 비교 + 제한적 문제번호/끝 `#`/문장부호 공백 제거 | 8그룹, 16행, 최대 2행 |
| `number_masked_template` | 부호는 보존하고 숫자 크기만 가린 soft 후보 | 141그룹, 322행, 최대 10행 |

초기 공격적 정규화의 13그룹과 audit v3의 8그룹은 서로 다른 함수의 결과다. 재현 가능한 기준값은 v3이고, 과거 수치를 맞추려고 정규화를 공격적으로 만들지 않는다.

심각한 라벨 충돌:

의미상 같은 한 pair에서 서로 다른 정수 label이 발견됐다. 개별 ID, question text,
answer는 대회 데이터 파생물이므로 public repository에 넣지 않는다. 독립 계산은 한
label을 지지했지만 운영 데이터에서는 임의 보정하지 않는다.

그러나 운영 데이터에서는 임의로 한쪽을 수정하지 않는다.

1. 두 행을 모두 `label_conflict`로 quarantine한다.
2. 독립적인 symbolic/brute-force solver와 사람이 각각 검증한다.
3. 보정할 경우 원문, 기존 label, 새 label, 근거, reviewer, code hash를 기록한다.
4. 두 행과 모든 template sibling을 validation에서 제외하거나 같은 group으로 둔다.

### 5.2 Train ↔ Leaderboard

- raw/canonical exact question overlap: 0
- 매우 공격적인 formatting normalization과 별도 유사도 검색에서 사실상 같은 문제 후보가 존재
- audit v3의 sign-preserving 숫자 마스크 exact template overlap: leaderboard 18개, 1.8%; **soft 후보일 뿐 직접 중복 판정이 아님**
- fuzzy number-masked near match: leaderboard 84개, 8.4% 후보

수동·math-token 재검증으로 문제번호, 끝 표식, TeX spacing 정도만 다른 **고신뢰 직접
중복을 최소 3개** 확정했다. 개별 train/leaderboard ID와 원문은 공개하지 않으며,
private contamination audit artifact에서만 추적한다.

이 문서에는 이를 이용한 leaderboard 답을 쓰거나 제출 파일을 만들지 않는다. 감사상 결론은 공개 평가 1,000개 중 최소 0.3%가 train과 직접 중복되어 public score가 일반화 성능을 순수하게 측정하지 않는다는 것이다.

char n-gram TF-IDF 최근접 train similarity:

| threshold | Leaderboard 문제 수 |
|---|---:|
| ≥0.98 | 14 |
| ≥0.95 | 33 |
| ≥0.90 | 73 |
| ≥0.85 | 110 |
| ≥0.80 | 135 |
| ≥0.70 | 192 |
| ≥0.60 | 269 |

최근접 similarity의 중앙값은 약 0.452, p90 0.869, p95 0.933, p99 0.984, 최대 약 0.998이다.

해석:

- train 답을 그대로 복사할 수 있다는 뜻은 아니다. 숫자가 바뀐 같은 템플릿이면 계산을 다시 해야 한다.
- random row split은 같은 template family가 train/validation 양쪽에 들어가 점수를 부풀릴 가능성이 높다.
- retrieval을 쓸 경우 정확한 구조 유사 문제는 도움이 되지만, 숫자 치환을 놓치면 치명적인 answer copying 오류를 만든다.
- leaderboard 문제를 train에 넣거나 그 문제에서 합성 sibling을 만드는 것은 평가 누출이다.

## 6. 품질 결함

### 6.1 관찰된 대표 이상치

private audit은 풀이 불가능한 fragment, 실제 그림 없이 그림을 참조하는 문항, 목차/링크만
남은 문항, answer fragment·boxed number 누출, 제목만 남은 문항을 확인했다. 개별 ID와
문장 조각은 대회 데이터 파생물이므로 공개하지 않는다.

이 예시는 모델의 지능 문제가 아니라 입력 자체의 결함이다. 리더보드의 결손 문항을 수동으로 외부 검색하거나 다른 모델에 묻는 것은 규칙 위험이 있으므로 하지 않는다. parser는 이런 행을 그대로 전달하고 `quality=unanswerable`로 로그한다.

leaderboard에는 image URL 문항 6개와 별도 missing-figure 후보 4개가 있다. 모두 자동
오답으로 단정하지 않고 전용 slice에서 수동 판정한다. 개별 ID는 private audit에만 둔다.

복수 subpart 후보도 leaderboard에 13개 있다. 일부는 여러 값을 요구하지만 하나의 정수로
결합하는 규칙이 불명확하다. “마지막 숫자”를 무조건 답으로 고르는 것은 위험하므로 같은
source family의 train label convention을 규칙 안에서 분석하거나 low-confidence routing해야 한다.

### 6.2 스크래핑·표현 흔적

| 휴리스틱 | Train | Leaderboard |
|---|---:|---:|
| BBCode | 157 | 12 |
| URL | 152 | 6 |
| image/figure reference | 164 | 6 |
| 명백한 missing-visual phrase | 약 72 | 약 5 |
| Asymptote code | 64 | 4 |
| 번역 marker | 46 | 5 |
| 문제번호 prefix | 1,135 | 52 |
| odd unescaped `$` proxy | 1,080 | 54 |
| curly-brace imbalance proxy | 52 | 1 |

Asymptote가 포함된 문제는 텍스트 안에 도형을 재구성할 코드가 있으므로 단순히 missing visual로 분류하면 안 된다. 반대로 외부 image URL만 있고 필요한 조건이 이미지에만 있으면 최종 오프라인 추론에서 접근할 수 없다.

더 심각하게, train 질문 안에 정답·풀이가 섞여 있다.

- `answer` 단어 포함 301행
- answer fragment/declaration 37행
- `solution` 단어 포함 89행
- 숫자 literal `\\boxed{n}` 포함 21행
- 위 21행 중 20행은 boxed 숫자가 label과 같아 직접 shortcut이 됨

적어도 한 train 행은 장문의 worked solution 안의 boxed number와 gold label이 서로
달랐고, 독립 계산은 gold label을 지지했다. 즉 질문에 포함된 worked solution도 gold
rationale로 신뢰할 수 없다. SFT 전에 problem-only, answer-leaked, worked-solution,
contradictory-rationale를 분리해야 한다.

권장 플래그:

- `has_asymptote`
- `has_image_url`
- `requires_missing_visual`
- `answer_cue_in_prompt`
- `fragment_or_title_only`
- `latex_suspect`
- `multilingual_or_translation_artifact`

## 7. 품질 tier

| Tier | 정의 | 학습 | 내부 평가 |
|---|---|---|---|
| A | 구조적으로 완전하고 고위험 오염 flag·중복이 없음; label 검증 수준은 suffix로 구분 | 우선 사용 | 사용 |
| B | 표현 이상이 있으나 원문만으로 풀 수 있고 label 검증 | 사용 가능, flag 유지 | 별도 slice |
| C | label conflict, answer cue, lucky-answer 의심, 심각한 TeX 손상 | 기본 제외 | 진단 전용 |
| D | 문제 fragment, 필요한 그림 누락, 답을 결정할 정보 부족 | 제외 | unanswerable slice |
| E | leaderboard/test | 절대 학습 금지 | 제출 생성에만 사용 |

Tier 판정은 규칙 기반 1차 flag와 수동 이중 검토를 결합한다. 사람이 “어려워 보인다”는 이유로 D에 넣지 않는다. 조건이 실제로 누락되었는지 근거를 남긴다.

17,000개를 며칠 안에 전부 독립 수학 검증했다고 가장하지 않는다. Tier A에는 검증 수준을 별도 suffix로 둔다.

- `A-S`: schema·형식·오염 휴리스틱을 통과한 syntactic clean. label 수학 검증은 미완료
- `A-V`: deterministic solver/대입으로 label 검증
- `A-D`: solver 또는 풀이 근거와 독립 human reviewer 두 경로가 일치

첫 SFT는 A-S를 사용할 수 있지만, locked evaluation과 고난도 self-training seed는 A-V/A-D를 우선한다. 수동 검토는 모든 high-risk flag, 모든 label conflict, answer leak 전부와 각 topic·source cluster의 층화 표본을 대상으로 하고 검토율을 보고한다.

## 8. 안전한 분할 설계

### 8.1 cluster 생성 순서

1. raw hash
2. conservative Unicode/space hash
3. number-preserving canonical hash — 자동 hard union 가능
4. 좁은 source-format hash — 제거 규칙 fixture 검증 후 자동 hard union 가능
5. number-masked template hash — **후보 생성만**, 자동 union 금지
6. token 5–13gram MinHash/LSH — 후보 생성만
7. character n-gram TF-IDF 또는 embedding 후보 생성
8. 식을 parse할 수 있으면 SymPy canonical form
9. high similarity pair 수동 판정
10. 의미 동일성이 확인된 pair만 adjudicated union으로 추가

모든 확인된 rewrite와 동일 source problem은 같은 cluster에 둔다. 그러나 숫자가 바뀐 template sibling은 답과 조건이 달라질 수 있으므로 유사하다는 이유만으로 hard union하지 않는다.

Group 불변성을 최우선으로 지키면서 할당 비용함수에 row count, primary topic, quality tier, answer sign/zero, long-context를 넣어 development fold와 final holdout의 분포를 균형화한다. 큰 cluster 때문에 완전 층화가 불가능하면 group을 쪼개지 않고 각 축의 편차를 manifest에 보고한다.

현재 split v4는 math-aware/source-format exact 두 신호만 hard union해 16,992개 cluster를 만들고, stable hash 순서로 1,700행 holdout을 먼저 선택한 뒤 나머지를 whole-group 5-fold로 row-balance한다. 각 fold는 3,060행이고, holdout은 1,699그룹/1,700행이다. answer sign·절댓값 bucket·quality flag의 실제 분포는 artifact에 기록하지만 현재 할당 비용에는 넣지 않는다. 향후 stratified allocator를 채택하려면 v4와 동일한 locked protocol에서 편차 개선과 group 불변성을 먼저 검증하고 split version을 올린다.

### 8.2 추천 평가 세트

- `final_locked_holdout`: 전체 cluster의 10~15%. 생성 직후 ID·label을 sealed manifest로 잠그고, 모든 방법·하이퍼파라미터 선택이 끝난 뒤 사전 선언한 primary/fallback에 **한 번만** 사용
- `development_group_fold_0..4`: 나머지 85~90%의 GroupKFold 5개. 반복 모델 선택, threshold calibration, seed 비교에 사용
- `development_probe`: development pool 안의 고정 group-aware split. 넓은 탐색은 이 한 split·1 seed에서 successive halving하고 finalist만 5-fold로 확인
- `clean_core`: Tier A만
- `noisy_slice`: Tier B/C
- `missing_visual_slice`: D와 시각 의존
- `long_context_slice`: 상위 1% 길이
- `rare_answer_slice`: 음수·0·12자리 이상
- `unseen_template_slice`: train fold와 numeric-template가 겹치지 않는 문제
- `fresh_procedural`: 학습 seed와 완전히 분리한 생성 문제

모든 self-training generation, retrieval corpus, external-data matching과 학습은 해당 run의 **training clusters에만** 접근한다. development validation cluster와 final locked cluster에서 rationale를 생성해 다시 학습하는 것도 누출이다.

Primary와 fallback은 development GroupKFold 결과로 먼저 freeze한다. 그 후 final locked holdout을 한 번 열어 일반화 추정과 catastrophic regression 확인에만 사용하고, 결과를 본 뒤 새 hyperparameter·prompt·data를 만들지 않는다. 공개 leaderboard도 이 내부 final holdout의 대체재가 아니다.

### 8.3 모델 선택 통계

각 실험에 다음을 기록한다.

- overall exact accuracy
- category·length·quality·answer-sign별 accuracy
- greedy, pass@N, majority@N, selector@N
- extraction failure rate
- invalid/multiple-final-answer rate
- duplicate/source `cluster_id`를 통째로 재표집한 paired bootstrap 95% confidence interval; row-IID bootstrap은 사용하지 않음
- 같은 문항의 두 모델은 exact McNemar, 복수 confirmatory 비교는 Holm 보정
- seed 3회 평균과 표준편차
- baseline 대비 paired correctness change

1,000개 leaderboard에서 0.1 percentage point는 1문제다. 작은 변동을 과도하게 해석하지 않는다.

## 9. 권장 데이터 레코드

전처리 후 각 행은 최소한 다음 메타데이터를 가져야 한다.

| 필드 | 의미 |
|---|---|
| `id` | 원본 ID, 불변 |
| `question_raw` | 원본 |
| `answer_str` | canonical signed integer string |
| `source_file_sha256` | 원본 provenance |
| `row_sha256` | 행 추적 |
| `quality_tier` / `quality_flags` | 결함 |
| `topic_primary` / `topic_multi` | 유형 |
| `raw_tokens` / `chat_tokens` | 길이 |
| `canonical_hash` | 보수적 중복 |
| `template_hash` | 숫자 마스크 |
| `cluster_id` | split 단위 |
| `fold` | 고정 group fold |
| `solution` | 있는 경우 풀이 |
| `solution_provenance` | human/self-generated/procedural |
| `verifier_status` | answer/process 검증 |
| `license` / `source_url` | 외부 데이터일 때 필수 |

## 10. 정제와 증강의 중단 기준

다음 중 하나라도 발생하면 해당 데이터 배치를 학습에 넣지 않는다.

- 원천 출처 또는 라이선스를 설명할 수 없음
- 공개 전에는 organizer train+leaderboard 및 알려진 benchmark/source와 exact·template·semantic overlap 감사를 통과하지 못함
- 답을 independent solver로 검증할 수 없고 원래 label과 생성 풀이가 충돌
- teacher 모델이 규칙상 허용되는지 확인되지 않음
- solution이 답을 그대로 되풀이할 뿐 논리적 근거가 없음
- 긴 CoT가 짧은 CoT 대비 clean/unseen-template validation을 악화
- 동일 template가 전체 batch를 지배

Final test는 공개 전 접근할 수 없으므로 사전 contamination gate에 포함했다고 주장하지 않는다. 공개 후에는 test를 학습·검색 corpus에 넣지 않은 채 read-only overlap report만 만들고, 발견 결과를 이용해 모델·prompt·답을 수정하지 않는다.

## 11. 데이터 관점의 최종 진단

이 데이터셋은 분량과 정수형 보상 구조 때문에 self-training과 RLVR에 유리하지만, 그대로 random split하면 신뢰하기 어렵다.

가장 먼저 해야 할 일은 추가 모델 학습이 아니라:

1. leaderboard CSV의 width 결함을 안전하게 처리하는 loader
2. row-level 품질 flag와 quarantine
3. template/source cluster
4. immutable group split
5. contamination report

이 다섯 가지다. 이 기반 없이 얻은 1~2%p 상승은 실제 generalization인지 template 기억인지 구분할 수 없다.
