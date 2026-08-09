# 11. 지속 실행 체크리스트와 공개 저장소 경계

기준 시각: **2026-08-10 KST**

이 문서는 작업을 중단했다가 다시 시작하더라도 같은 순서·동일한 안전 조건으로
이어가기 위한 실행 원장이다. 상태 값이 바뀌는 GPU 사용량이나 Kaggle 제출 한도는 이
문서의 숫자를 재사용하지 않고 해당 명령 직전 다시 관측한다.

## 1. 공개 저장소 경계

공개 GitHub 저장소에는 다음만 넣는다.

- `src/`, `tests/`, `configs/`, `docs/`, `README.md`, `pyproject.toml`, `uv.lock`
- 재현 명령, 규칙 해석, SHA-256 값, 모델·데이터를 다시 받는 방법
- GNU GPL v3.0 라이선스 전문

현재 canonical public origin은
`https://github.com/robinhood0107/ajuso-junghan-deep-learning-challenge-2026-robinhood0107`이다.
요청한 한글 대회명은 repository description에 보존하고, URL slug는 GitHub API가 허용한
ASCII 표기로 사용한다.

다음은 절대로 stage하거나 push하지 않는다.

- Kaggle 원본 CSV·ZIP와 그 복사본
- `artifacts/` 전체: development shard, split ID, raw generation, adapter manifest,
  screenshot, submission, checksum은 대회 데이터 또는 그 파생물일 수 있다.
- `data/`, `datasets/`, `downloads/`, `cache/`, generated report 및 raw
  CSV/Parquet/JSONL/weight 파일처럼 별도 전달해야 하는 데이터·model 산출물
- `env`, `.env*`, `.kaggle/`, access token, Hugging Face token 등 모든 인증 정보
- local model weight·adapter·checkpoint·prediction·submission과 experiment tracker
- 기존 사용자 파일 `NU_`, 개인 작업 프롬프트, local browser/tool cache

`.gitignore`은 위 경계를 강제한다. 공개 저장소의 문서는 데이터 내용을 포함하지 않고,
로컬 artifact는 수상 검증이 필요한 경우 별도 비공개 전달물로만 사용한다.

현재 checkout은 `.githooks/pre-commit`을 `core.hooksPath=.githooks`로 연결한다.
이 hook은 staged Git blob만 읽어 `.gitignore`를 우회하는 force-add와 recognizable token
pattern을 fail-closed로 막는다. 새 clone에서도 다음을 먼저 실행하고, 전체 tracked tree는
그 아래 명령으로 별도 점검한다.

```bash
git config core.hooksPath .githooks
PYTHONPATH=src python3 -m deep_challenge.public_repo_guard --all
```

## 2. 현재 확인된 운영 계약

- 유일한 베이스: `Qwen/Qwen2.5-3B-Instruct`, 내부 재현 pin:
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- 허용: Full FT/LoRA/QLoRA, SFT, RL, augmentation, quantization, offline
  Majority Voting/Self-Consistency/Best-of-N.
- 금지: 다른 모델 weight/merge/최종 추론 ensemble, pre-training, leaderboard/test를
  학습·self-training seed·외부 API/검색 입력으로 사용하는 행위, 추론 중 인터넷.
- 현재 Kaggle API 인증 snapshot은
  `artifacts/analysis/kaggle-b0-20260810/authenticated-api-snapshot-v1.json`에 있다.
  authenticated 파일 목록에는 sample submission이 없고, Rules/Evaluation은
  `submission.csv`의 `ID,answer`를 요구한다. 현재 API가 보인 제출 가능 횟수는 5이고
  과거 제출은 0건이다.
- local Python/SymPy TIR 및 같은 베이스의 multi-adapter/checkpoint 결합은 운영진의
  명시 답변 전 **off**다. 첫 모델 경로는 single base 또는 single adapter다.

## 3. 재개 순서와 완료 조건

| 순서 | 작업 | 시작 조건 | 완료 증거 | 중단 조건 |
|---:|---|---|---|---|
| P0 | 공개 repo 초기화 | `.gitignore` audit 완료 | GPLv3, clean staged file list, public repo | 민감 파일이 stage됨 |
| B0.1 | Kaggle rules/data snapshot | API token 존재 | page/file/limit hash, sample 부재 기록 | slug/auth 실패 |
| B0.2 | GPU preflight | model/runtime/VRAM 충족 | 새 no-overwrite preflight JSON, `training_ready=true` | GPU used >1,024MiB 또는 free <10,240MiB |
| B0.3 | final synthetic smoke | B0.2 green | `status=green`; only local `2+3` prompt | parser/load/backward/VRAM failure |
| B1.0 | fold 0 base direct-answer | B0.3 green | JSONL + manifest + raw generation/provenance | invalid parser/artifact |
| B1.1 | parser golden corpus | B1.0 real generations | added regression tests + full CPU suite | conflict is hidden or test fails |
| B2.0 | fold 0 answer-only QLoRA | same-fold base manifest | exact adapter bundle/checksum/manifest | train IDs or provenance mismatch |
| B1/B2.1 | folds 1–4 repeat | fold 0 regression green | five base + five adapter OOF runs | any fold incomplete |
| B2.2 | complete OOF comparison | all five folds | grouped paired bootstrap, exact McNemar, Holm | single fold/reused run/mixed method |
| B3 | freeze and locked holdout | B2.2 evidence, primary decided | durable claim + one receipt | no freeze, already-consumed claim |
| B4.1 | filtered leaderboard prediction | frozen policy, B3 complete | strict prediction manifest, no invalid answers | data SHA/schema/adapter mismatch |
| B4.2 | submission build/verify | B4.1 complete | writer + two independent validators + checksum | any missing/invalid ID |
| B4.3 | Kaggle upload | explicit user request only | Kaggle submission receipt | no explicit authorization |

## 4. Immediate exact commands

Set local paths without committing them:

```bash
PROJECT=/absolute/path/to/deepleaning
DATA_DIR="$PROJECT/deep-learning-challenge-2026"
GPU_ENV=/absolute/path/to/deep-challenge-gpu-venv
REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1

cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version \
  --format=csv,noheader,nounits
```

Only when the just-observed values satisfy `used <= 1024MiB` and
`free >= 10240MiB`, use a new tag and execute the B0.2/B0.3 commands in
[`10_GATE_B_CPU_READY_RUNBOOK.md`](10_GATE_B_CPU_READY_RUNBOOK.md). Never reuse
an existing preflight or smoke output path. The smoke artifact is the only
permitted first CUDA workload and contains no organizer/leaderboard/test prompt.

## 5. Per-step Git record

Commit only public code/document changes after each completed phase:

1. `chore: initialize public reproducibility repository`
2. `docs: record authenticated Kaggle contract`
3. `fix: <runtime guard or parser regression>` when a regression test accompanies it
4. `docs: record Gate B <phase> evidence` with only hashes and non-sensitive summaries

Do not commit raw artifacts merely to make a commit. Before every commit, run
`git diff --cached --check`, `git status --short`, and a staged-secret/data audit.
Push only the intended public files. A Kaggle submission is not a Git action and
is intentionally outside automatic execution.

## 6. Deferred experiments

The first pass ends after single-adapter direct-answer OOF comparison and the
one permitted holdout evaluation. Concise rationale, public external data,
self-training, preference/RL, local tool use, multi-checkpoint voting, and
full-development refit are separate versioned experiments. They may begin only
after their rule, license, contamination, compute, and development-evidence
gates are recorded; none is silently folded into the first-pass result.
