# 07. 확인된 규칙과 잔여 질문

기준일: **2026-08-10 KST**

이 문서는 추정 목록이 아니라 현재 실행 feature flag의 권위 원본이다. 사용자 제공
초기 사용자 제공 규칙 원문은 다음 local snapshot으로 고정했다. 원본 attachment 경로와
대회 데이터는 공개 Git 저장소에 넣지 않는다.

- 원문: local user-provided attachment (public repository에서 제외)
- SHA-256: `1de6e12d77334af56af9ec3851085e6a9fa1d8a6168846eadeaee9c696999aa2`
- 결정 artifact: `artifacts/analysis/kaggle-b0-20260804/rules-decision-manifest-v1.json`
- 전체 visible site 내용 구조화 snapshot:
  `artifacts/analysis/kaggle-b0-20260804/user-supplied-site-content-v1.json`
- 브라우저 근거: `artifacts/analysis/kaggle-b0-20260804/browser-access-evidence-v1.json`

2026-08-10에는 `KAGGLE_API_TOKEN`을 값 노출 없이 사용해 참가 중인
`deep-learning-challenge-2026`을 read-only authenticated API로 재확인했다. 페이지 전체
응답·Rules 응답·file list·submission-limit 응답의 SHA-256과 판단은
`artifacts/analysis/kaggle-b0-20260810/authenticated-api-snapshot-v1.json`에 고정했다.
이 artifact는 data-derived evidence이므로 공개 Git 저장소에 넣지 않는다.

## 1. 확인된 규칙

| 항목 | 현재 결정 |
|---|---|
| 베이스 모델 | `Qwen/Qwen2.5-3B-Instruct`만 허용 |
| 다른 모델 weight | merge·최종 추론 ensemble에 사용 금지 |
| 학습 방법 | Full FT, LoRA/QLoRA, SFT/RL, augmentation, quantization 허용 |
| 공개 외부 데이터 | 무료이고 모든 참가자가 동등 접근 가능한 데이터는 허용 |
| 유료·비공개 데이터 | 금지 |
| 상용 API teacher | **training data/rationale 생성 목적은 허용** |
| leaderboard/test API·검색 입력 | 금지 |
| 추론 네트워크 | 인터넷/API 없이 로컬 offline만 허용 |
| test-time compute | Majority Voting, Self-Consistency, Best-of-N 허용 |
| 평가 | 정수 exact-match accuracy |
| 제출 | `submission.csv`, 열은 정확히 `ID,answer`, 모든 expected ID 필요 |
| 재현 검증 | 학습/추론 코드, weight, data 목록, 환경, 방법론 제출 필요 |
| 팀 | 참가 가능 |

규칙 원문은 정확한 Hugging Face commit을 지정하지 않았다. 프로젝트는 재현성을 위해
`aa8e72537993ba99e69dfaafa59ed015b17504d1`을 내부 pin으로 사용하지만, 이것을
“운영진 지정 revision”이라고 표현하지 않는다.

## 2. 2026-08-03 데이터 수정으로 해결된 계약

- 현재 leaderboard는 `deep_chal_math_leaderboard_filtered.csv`, 831행,
  header `id,question`, SHA
  `032333a1361c8083093674ad19817e024c38dc7c9f4bdf05c0c9b0c71940dcf1`이다.
- train 오류 목록은 `train_filtered_ids.csv`, 627 unique ID, SHA
  `67e4674afa685b985a6dc52e9050d9fb17116a99dbd9606cba82c976c904b4f3`이다.
  이 파일에서는 `id`만 denylist로 사용한다.
- 과거 1,000행 malformed leaderboard는 역사적 감사용이며 현재 submission ID source가
  아니다.
- authenticated Rules/Evaluation로 `ID,answer`를 재확인했다. 2026-08-10 API file listing은
  4개 공식 CSV만 반환했으며 sample CSV는 없어 row order/SHA를 확보할 수 없다.
- authenticated API가 반환한 현재 제출 가능 횟수는 `numAllowedNow=5`이고 기존 제출은
  0건이다. 이 값은 daily/total 한도나 final timezone을 확정하지 않는다.

## 3. 아직 필요한 운영진 답변

다음 항목은 답변 전 feature-off다.

1. 로컬 offline Python/SymPy 계산을 모델 생성 코드에서 호출하는 TIR 허용 여부
2. 같은 고정 베이스에서 학습한 여러 adapter/checkpoint voting, weight soup, 별도
   selector/verifier adapter의 허용 범위
3. 제공 train만을 index로 쓰는 최종 offline dynamic few-shot retrieval의 허용 범위
4. sample submission 파일, 정확한 row order와 SHA-256 (현재 file listing에는 없음)
5. daily/total submission limit와 final window의 정확한 timezone (`numAllowedNow=5`만 확인)
6. 팀 최대 인원과 merge deadline
7. 상위권 checkpoint 제출 형식(LoRA only/merged)과 Qwen license 관련 추가 지침
8. 운영진이 권장하거나 지정하는 immutable Hugging Face revision

### 운영진에게 보낼 짧은 질문

```text
안녕하세요. 현재 공개 규칙에서 fixed base, 외부 공개 training data, training-only
teacher API, offline inference, ID/answer schema와 test-time voting은 확인했습니다.
재현 패키지를 정확히 고정하기 위해 아래 잔여 항목을 확인 부탁드립니다.

1) local offline Python/SymPy TIR이 허용되나요?
2) 같은 Qwen2.5-3B-Instruct에서 학습한 여러 LoRA/checkpoint의 voting, weight soup,
   selector/verifier adapter는 어디까지 허용되나요?
3) organizer train-only offline retrieval은 최종 추론에서 허용되나요?
4) sample submission 파일과 정확한 ID 순서, daily/total 제출 한도, final timezone은
   무엇인가요?
5) 팀 최대 인원/merge deadline, checkpoint 제출 형식, 권장 immutable model revision을
   알려 주실 수 있나요?

답변을 Rules/FAQ에도 고정해 주시면 감사하겠습니다.
```

## 4. 답변 전 기본 정책

| 기능 | 현재 기본값 |
|---|---|
| organizer-only direct-answer baseline | 허용, 첫 모델 실험 |
| organizer-only answer-only QLoRA | final GPU gate 뒤 허용 |
| 공개 외부 데이터 | provenance/license/contamination 검증 뒤 ablation만 |
| commercial teacher rationale | training-only, source/model/prompt 전부 기록, test/LB 금지 |
| local Python/SymPy | off |
| same-base multi-adapter/checkpoint 결합 | off |
| 다른 모델 weight/최종 inference ensemble | 금지 |
| leaderboard/test를 학습·self-training seed로 사용 | 금지 |
| leaderboard/test를 외부 API/검색에 전달 | 금지 |
| Majority/Self-Consistency/Best-of-N | 개발 근거와 runtime budget 안에서 허용 |

## 5. 답변 보존 형식

```yaml
question_id:
asked_at_kst:
channel:
thread_url:
exact_question_sha256:
official_responder:
answered_at_kst:
answer_text:
answer_screenshot_sha256:
rules_page_updated:
decision:
affected_configs:
```

답이 바뀌면 이전 기록을 덮어쓰지 않고 새 revision을 추가한다.

## 6. snapshot 상태

- [x] 사용자 제공 Rules 원문과 SHA
- [x] fixed model·외부 데이터·training teacher·offline inference·test-time methods
- [x] `submission.csv`, `ID,answer`
- [x] 2026-08-03 filtered data bundle/hash
- [x] Hugging Face 내부 pinned revision tree/model card
- [x] 비로그인 Kaggle login redirect와 browser stop 근거
- [x] authenticated Rules/Data/Evaluation/Submission contract snapshot (API hash 보존)
- [x] current file listing에서 sample submission 부재 확인
- [x] current submission allowance `numAllowedNow=5`, submissions 0
- [ ] sample row order/SHA 및 daily/total limit/final timezone
- [ ] Python/SymPy와 same-base multi-checkpoint 서면 답변
- [ ] team size/merge deadline
- [ ] license/weight-distribution 세부 지침

모든 새 근거에는 URL 또는 원문 위치, 저장 시각, SHA-256을 남긴다.
