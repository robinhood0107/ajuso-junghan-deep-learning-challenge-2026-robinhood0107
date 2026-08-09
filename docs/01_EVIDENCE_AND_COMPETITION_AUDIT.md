# 01. 증거·대회 규칙 감사

기준 시각: **2026-07-31 KST**

## 1. 감사 목적과 판정 언어

이 문서는 대회 설명, 다운로드된 파일, 안내 영상이 서로 일치하는지 확인하고, 실제 구현 전에 어떤 규칙을 반드시 다시 확인해야 하는지 고정한다.

모든 주장은 다음 증거 등급 중 하나로 취급한다.

| 등급 | 의미 | 이 문서의 예 |
|---|---|---|
| A | 현재의 공식 원문을 직접 확인 | 로그인된 Kaggle Rules 원문, 운영진의 서면 답변 |
| B | 로컬 원본을 기계적으로 검증 | CSV 행 수·해시, 영상 스트림·프레임 |
| C | 공식 제공물의 화면을 직접 판독 | 영상 슬라이드에 적힌 모델·평가·제출물 |
| D | 사용자가 제공한 텍스트 스냅샷 | 이 요청에 붙여 넣은 Overview/Dataset Description |
| E | 분석 또는 권고 | 권장 실험, 규칙 위험 추론 |

현재 Kaggle 페이지는 비로그인 HTTP 요청을 로그인 화면으로 전환했고 익명 competition API에서도 본문을 받지 못했다. 기존 브라우저 로그인 세션을 이용한 읽기 전용 재확인도 현재 WSL 작업경로와 브라우저 연결의 호환 문제로 열리지 않았다. 이는 대회가 없다는 뜻이 아니라 인증된 최신 Rules 원문을 이번 감사에서 확보하지 못했다는 뜻이다. 따라서 규칙에 관한 확정 표현은 B/C/D 근거 범위로 제한하고, 충돌하는 내용은 운영진 확인 전까지 미확정으로 둔다.

## 2. 원본 manifest

### 2.1 로컬 데이터셋

원본 위치: local Kaggle download directory (public Git repository에서 제외)

| 파일 | 바이트 | SHA-256 | 상태 |
|---|---:|---|---|
| `deep_chal_math_train.csv` | 4,335,431 | `e240dcd9752d12143162706cee4818d4025456605c991ece337df6e9abeb869a` | strict UTF-8, BOM/NUL 없음 |
| `deep_chal_math_leaderboard.csv` | 254,899 | `18cae340803dd649ce162a575dcda01bb75bbcf0759f80392d692603951ccd32` | strict UTF-8, BOM/NUL 없음 |

두 파일의 로컬 수정 시각은 2026-07-31 04:21:12 KST다. 폴더에는 이 두 파일만 있고 sample submission과 final test는 없다.

원본은 읽기 전용 근거로 취급한다. 전처리 산출물은 별도 경로에 만들고 다음 항목을 manifest에 저장해야 한다.

- 원본 경로, 크기, 해시, 수정 시각
- parser와 정규화 코드의 Git SHA
- 출력 데이터의 해시와 행 수
- 제거·보류·수정한 행 ID와 사유
- source split과 cluster split의 해시

### 2.2 안내 영상

원본 위치: local briefing-video file (public Git repository에서 제외)

| 항목 | 값 |
|---|---|
| 크기 | 30,489,956 bytes |
| SHA-256 | `05af2ef7aa57cad8878bb5a78dc8fe06bd33b4d22fb720a7b074db549b0564eb` |
| 재생 시간 | 515.58초, 약 8분 35.58초 |
| 영상 | H.264 High, 1920×1080, 25 fps, yuv420p, BT.709 |
| 음성 | AAC-LC, 48 kHz, stereo |
| 내장 자막/데이터/챕터 | 없음 |
| 전체 디코드 검사 | 12,889프레임, decode error 없음 |

같은 Desktop 범위에 `.srt` 또는 `.vtt` 자막이 없었고, 로컬 ASR 도구와 모델도 없었다. 따라서 다음 두 결과를 분리한다.

- **완료:** 영상 기술 감식, 전체 프레임 디코드 확인, 슬라이드 구간과 화면 텍스트 분석
- **미완료:** 음성 발화의 단어 단위 완전 전사와 발언 인용

음성에만 존재하는 예외 규칙이 있을 가능성을 배제할 수 없다. 완전 전사가 필요하면 공식 자막을 확보하거나, 별도의 ASR 모델 다운로드·실행을 허가한 뒤 전사 결과를 다시 청취 검증해야 한다.

## 3. 영상 슬라이드 타임라인

시간은 화면 전환 기준의 근사치다. 다음 표는 화면에서 판독 가능한 내용을 요약하며 음성의 축어록이 아니다.

| 구간 | 화면 근거 |
|---|---|
| 00:00–00:24.8 | 대회 제목, 2026-07-31~08-30, LLM 수학 챌린지 |
| 00:24.8–01:00.72 | Qwen2.5-3B-Instruct 고정, 처음 보는 정수형 수학 문제, Accuracy, 개인/Kaggle 진행 |
| 01:00.72–01:40.08 | 상금: 1위 200만원 및 아주대학교 총장상, 2~4위 각 100만원, 5~9위 각 50만원 |
| 01:40.08–02:21.64 | 고정 베이스, SFT/RL 허용, Qwen2.5-Math·DeepSeek-R1·Llama 등 다른 베이스 금지 |
| 02:21.64–02:40.64 | 제공 train과 공개 외부 데이터 허용, 상용 API를 이용한 test 답 생성 및 임의 라벨 조작 금지 |
| 02:40.64–03:29.12 | 추론 중 인터넷/API 금지, majority voting/self-consistency 허용, 제출 코드로 재현 가능해야 함; Python≥3.10, PyTorch≥2.0, CUDA≥12, requirements 권고 |
| 03:29.12–04:05.76 | 상위권은 final checkpoint, 전체 학습 파이프라인, 생성→후처리→CSV 추론 코드, 외부 데이터 출처, requirements 제출; W&B/TensorBoard log·실험 보고서·README 권고 |
| 04:05.76–05:23.36 | 실시간 leaderboard는 참고용, final test가 공식 순위; 정수 exact match; 운영진 재현 검증 실패 시 불이익 가능 |
| 05:23.36–06:21.76 | 상위 12명/팀 발표, leaderboard 50% + 모델 우수성 발표 50%, 검증 후 최종 9명/팀 수상 |
| 06:21.76–06:57.20 | 파일명 `deep_chal_math_train.csv`, `deep_chal_math_leaderboard.csv`, `deep_chal_math_test.csv` |
| 06:57.20–07:17.84 | `id`, `question`, `answer`; 모든 답은 정수 |
| 07:17.84–08:12.52 | 일정과 개인 참가 관련 안내 |
| 08:12.52–끝 | Discord 링크 `discord.gg/UnRd6f8dU` 표시 |

## 4. 현재까지 확인된 규칙

### 4.0 사용자 제공 일정 snapshot

아래는 Kaggle에서 현재 재확인한 A등급 원문이 아니라, 사용자 제공 본문의 D등급 snapshot이다.

| 일정 | 날짜 |
|---|---|
| 모집 | 2026-06-08~06-28 |
| 참여 확정 | 2026-07-20~07-27 |
| 문제 공개 | 2026-07-31 예정 |
| 이론 학습 | 2026-07-31~08-30 |
| 챌린지 | 2026-07-31~08-30 |
| 평가·검증 | 2026-08-01~09-20 예정 |
| final test 공개·제출 | dataset 설명상 2026-08-31 00:00~23:59 |
| 수상자 발표 | 2026-09-28 예정 |

평가·검증 기간이 챌린지와 겹치고 final test가 챌린지 종료 다음 날 하루만 열린다는 표현이 있으므로, KST 여부와 실제 download/upload window를 반드시 재확인한다. 사용자 제공 Data 탭 snapshot은 파일 2개, 총 약 4.59MB, CSV, Apache-2.0으로 표시되어 있으나, 최종 test와 sample submission은 현재 로컬 폴더에 없다. 데이터셋 카드 라이선스와 수집된 개별 문제의 원천 권리는 별개로 감사한다.

### 4.1 명확도가 높은 내용

다음은 사용자 제공 본문과 영상 화면이 대체로 일치한다.

- 베이스 모델은 Qwen/Qwen2.5-3B-Instruct로 고정한다.
- Qwen2.5-Math, DeepSeek-R1, Llama 등 다른 모델을 베이스로 쓰면 안 된다.
- SFT와 RL 계열 학습 기법은 허용된다.
- 제공 train을 사용하고 공개 외부 데이터도 사용할 수 있다.
- test 문제 답을 상용 API로 직접 생성하면 안 된다.
- 최종 추론 중 인터넷과 외부 API를 사용할 수 없다.
- majority voting과 self-consistency 같은 test-time 기법은 가능하다.
- 평가 답은 정수이며 exact match다.
- 공개/실시간 leaderboard는 참고용이고 final/private 평가가 공식 순위를 정한다.
- 상위 참가자는 모델뿐 아니라 학습·추론·후처리 코드와 데이터 출처를 재현 가능하게 제출해야 한다.
- 상위 12개 진출자는 모델 성능 50%와 발표 평가 50%의 검증을 거쳐 최종 9개 수상자가 된다.

### 4.2 아직 해석하면 안 되는 경계

“외부 공개 데이터 자유”만으로 다음을 자동 허용한다고 보면 안 된다.

- Qwen2.5-Math, DeepSeek, GPT 등 금지/외부 모델이 생성한 공개 CoT를 증류하는 행위
- 다른 모델의 reward model·PRM·embedding·reranker를 학습 또는 추론에 사용하는 행위
- 같은 Qwen 베이스에서 만든 여러 adapter/checkpoint를 동시에 메모리에 올리는 행위
- Python/SymPy나 모델 생성 코드를 실행하는 tool-integrated reasoning
- public leaderboard 문제를 사용한 수동 오류 분석, prompt 수정 또는 self-training
- checkpoint를 공개 저장소로 배포하는 행위

이 항목은 [운영진 질문서](07_RULE_CLARIFICATION.md)로 확인한 후 `rules_snapshot.md`에 답변 링크와 시각을 남겨야 한다.

## 5. 출처 간 불일치

| 항목 | 로컬/영상 | 사용자 제공 본문 | 조치 |
|---|---|---|---|
| train 파일명 | `deep_chal_math_train.csv` | `deep_chal_math_dataset_train.csv` | 실제 파일과 영상명을 코드 기준으로 사용하되 Kaggle Data 탭 다시 확인 |
| leaderboard 파일명 | `deep_chal_math_leaderboard.csv` | `deep_chal_math_dataset_leaderboard.csv` | 위와 동일 |
| 데이터 형식 | 실제 CSV | Overview 일부에 `test.parquet` 표현 | CSV를 현재 근거로 사용, Rules/Evaluation 최신본 확인 |
| Discord | 영상 `UnRd6f8dU` | 본문 `JhWDr73g65` | 현재 공식 공지 링크 확인, 어느 것도 권위로 가정하지 않음 |
| 참가 단위 | 영상에서 개인 참가 표현 | 본문에 팀/명 혼용 | 팀 허용 여부와 팀 병합 기한 확인 |
| final test 공개 | 영상에는 날짜 중심 | 본문은 08-31 00:00 공개, 24시간 제출 | timezone·다운로드/제출 window 서면 확인 |
| sample submission | 로컬 없음 | 예시는 `ID,answer` 대문자 ID | 실제 sample CSV로 대소문자와 열 순서 고정 |
| leaderboard CSV | 헤더 ` answer`, 모든 데이터 행 2열 | 세 필드 데이터로 설명 | 로더 보정은 가능하지만 제출 schema는 sample 기준으로 별도 생성 |

불일치는 어느 한쪽을 임의로 선택해 없애지 않는다. 각각을 `source`, `observed_at`, `text/hash`, `resolution` 필드로 기록한다.

## 6. 모델·라이선스 감사

[공식 모델 카드](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)에 따르면 Qwen2.5-3B-Instruct는 3.09B parameters, 36 layers, GQA 16 query/2 KV heads, 32,768-token context, 최대 8,192 생성 토큰을 갖는다.

중요하게도 이 3B 모델은 Apache-2.0이 아니라 [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)를 사용한다. 공식 라이선스 본문상 자동 허용 범위는 비상업적 연구·평가이고, 상업적 사용에는 별도 라이선스가 필요하다. 파생물의 배포와 표시 의무도 존재한다.

따라서 다음은 법률 자문이 아니라 실무상 확인 게이트다.

- 상금이 있는 이 대회 참가가 허용된 research/evaluation 범위인지
- 상위권 checkpoint를 운영진에게 제출하는 것이 배포에 해당하는지
- 발표·GitHub·모델 카드에 필요한 attribution 문구
- 파생 adapter만 제출할지 merged weight까지 제출할지
- 운영진이 Qwen 측과 별도 사용 허가를 확보했는지

주최 측이 모델을 지정했다는 사실만으로 참가자 개인의 라이선스 의무가 자동 소멸한다고 가정하지 않는다.

모델 snapshot manifest에는 repository locator만 쓰지 않고 base commit SHA, tokenizer/config/generation-config/chat-template 각 파일 SHA-256, LICENSE/NOTICE hash와 다운로드 시각을 기록한다. 운영진이 revision을 지정하지 않으면 첫 승인 snapshot을 pin하고 `main`을 다시 따라가지 않는다.

## 7. 재현성 요구를 구현 계약으로 바꾸기

영상 화면의 “운영진이 제출을 재생성” 요구를 다음 acceptance criteria로 바꾼다.

1. 빈 환경에서 고정된 Python/PyTorch/CUDA 버전으로 설치된다.
2. 인터넷이 끊긴 상태에서 local cache와 명시된 파일만 읽는다.
3. 원본 CSV의 해시가 다르면 즉시 중단한다.
4. 모델·tokenizer·adapter의 해시와 base revision이 manifest와 일치한다.
5. 하나의 명령으로 raw generation부터 최종 CSV까지 생성한다.
6. 모든 seed, decoding parameter, prompt version, parser version이 기록된다.
7. CSV의 ID set·순서·행 수·중복·정수 정규식·결측을 자동 검증한다.
8. 같은 하드웨어/소프트웨어 조건의 반복 실행에서 최종 CSV가 같거나, sampling 사용 시 사전에 정의된 deterministic seed 정책으로 같아진다.
9. 학습 데이터 출처, 라이선스, 변환, 제외, contamination 판정이 추적된다.
10. README의 명령만으로 운영진이 재실행할 수 있다.

## 8. 구현 전 hard gates

### Gate R0 — 규칙 snapshot

통과 조건:

- 로그인된 Kaggle Overview, Rules, Data, Evaluation, Submission, Timeline의 저장본
- sample submission 원본과 SHA-256
- 제출 횟수·파일 크기·runtime·GPU·인터넷 정책
- 팀과 외부 데이터/tool/teacher/ensemble 질문에 대한 답

미통과 시 가능한 작업:

- 로컬 train 전처리, parser 단위 테스트, 모델-free 평가 harness
- manifest·quality flags·development/final group split과 제출 validator

최신 Rules와 정확한 base/tokenizer revision만 확보되면, 모호한 기능을 모두 off한 organizer-only 단일 adapter baseline은 별도 R0-B gate로 진행할 수 있다.

미통과 시 금지할 작업:

- 모호한 외부 teacher CoT 다운로드·학습
- leaderboard를 이용한 반복 최적화
- 다중 모델/verifier/tool이 최종 설계라고 확정

### Gate R1 — 데이터 snapshot

통과 조건:

- 원본 hash manifest
- row-level quality annotation
- duplicate/template cluster
- immutable fold mapping
- contamination report

### Gate R2 — compute·offline 재현

통과 조건:

- 원격 GPU 종류·시간·저장공간·비용
- base model과 외부 데이터의 오프라인 cache 계획
- 네트워크 차단 inference smoke test

## 9. 감사 방법 manifest와 재현 한계

초기 감사는 병렬 read-only 분석으로 수행됐고 일부 임시 스크립트·contact sheet는 `/tmp`에만 존재했다. 수치의 재현 수준을 과장하지 않는다.

| 증거 | 사용 방법 | 현재 재현 수준 |
|---|---|---|
| 파일 크기·SHA-256·UTF-8 | filesystem stat, streaming SHA-256, strict UTF-8 decode | 완전 재현 가능 |
| CSV 행·ID·answer | Python RFC4180 `csv` parser, embedded newline 포함 논리행 | strict loader와 dataset SHA를 구현·테스트하고 audit v3 JSON으로 영구화 |
| quantile·topic·quality 수치 | 독립 Python 전수 집계 | quantile convention과 quality 함수는 audit v3에 고정; 초기 topic 휴리스틱은 exploratory로 유지 |
| L1/L2/template 중복 | NFKC/공백, TeX·문장부호, sign-preserving number-mask의 서로 다른 fingerprint | audit v3에 함수·version 고정; math 4그룹, source-format 8그룹, soft template 141그룹이며 초기 정의와 차이를 명시 |
| fuzzy/TF-IDF | char n-gram, 최대 feature 약 180K, top pair 수동 확인 | 최초 `ngram_range`와 전체 config가 보존되지 않아 threshold 수치는 exploratory; 재실행 전 확정 benchmark로 쓰지 않음 |
| tokenizer 길이 | Qwen2.5-3B-Instruct tokenizer + 짧은 system/chat template | commit `aa8e...04d1`, 라이브러리 버전, 파일 SHA, 정확한 prompt를 tokenizer-profile v3에 보존 |
| 영상 스트림·전체 decode | imageio-ffmpeg 0.6.0의 FFmpeg 7.0.2 static binary, video stream을 null sink까지 decode | 원본 hash와 binary 경로로 재현 가능 |
| 영상 slide timeline | 대표 프레임/contact sheet를 사람이 판독 | contact sheet는 `/tmp`였고 전환은 근사치; 음성 transcript 근거가 아님 |

사용한 FFmpeg binary snapshot:

isolated local Python environment의 `imageio_ffmpeg` bundled FFmpeg 7.0.2

재현 decode 예시:

```bash
ffmpeg-linux-x86_64-v7.0.2 -v error -i <video.mp4> -map 0:v:0 -an -f null -
```

새 구현은 audit config, source-tree hash, 정규화 함수, quantile convention을 JSON report에 넣어 이 한계를 보완했다. 최초 수치와 새 영구 구현이 다를 때 최초 값을 억지로 맞추지 않고 정의 차이와 새 결과를 함께 기록한다. 최신 구현·테스트 증거는 [09_IMPLEMENTATION_STATUS.md](09_IMPLEMENTATION_STATUS.md)에 있다.

## 10. 최종 감사 판정

현재 자료만으로 대회 방향과 구현 계획을 세우기에는 충분하다. 그러나 최신 공식 규칙을 완전 검증했다고 말할 수는 없다.

- **데이터 원본성·구조:** B등급으로 강하게 확인
- **영상 화면 규칙:** C등급으로 확인
- **사용자 제공 대회 본문:** D등급 snapshot
- **최신 Kaggle Rules 및 submission 계약:** A등급 근거 미확보
- **영상 음성 전체 발언:** 전사 근거 미확보

실행 계획은 이 한계를 전제로 보수적으로 설계되었다. 규칙 확인 결과가 바뀌면 모델 방법론 전체를 뜯어고치기보다, teacher 데이터·도구·ensemble 같은 모호한 모듈만 feature flag로 끄도록 구성해야 한다.
