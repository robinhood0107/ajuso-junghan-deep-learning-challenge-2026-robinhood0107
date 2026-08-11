# Gate B teacher-pilot-v4 terminal failure record

기준 시각: **2026-08-11 KST**

`teacher-pilot-v4`의 유일한 bounded live evaluation은 terminal failure로 끝났다.
이 문서는 raw question, answer, rationale, ID, prompt, provider stderr, event를 포함하지 않는
공개-safe aggregate와 hash만 보존한다. v1/v2/v3/v4 ledger 모두 forensic evidence이며 재개,
repair, plan 생성에 사용하지 않는다.

## Frozen run binding

- source commit: `68ff7d4`
- run tag: `20260811T191118KST`
- teacher config semantic/file SHA-256:
  `7a7f3e117a3f454c21dd9721ae432b9da6c0813e3b806a19af7f1d9a34d8adef` /
  `063881f7d72a96e25202736a8fe729a0d64271376019b281094a5350c92a0d97`
- prompt template/policy SHA-256:
  `3029e9297bdda504e0f48e1ce4d57e363e5d3a5342edf18253b11c4f75ecd8a7` /
  `8de961862f2cabf245753ee276d4b833d8917934d4ba84fa8f9caa20a64ab924`
- frozen source-manifest file/tree SHA-256:
  `fd24fa4ee772e628e5d8f40895a3daebe2e7fc47b51c57052f3ca3d4fe847a0c` /
  `cbc0250eeccde4bd921e795093d71ac49fef373fe5ef06a737cfbb42ca06adf4`
- private pilot plan SHA-256:
  `8736b1efbeb3564ba472e1e10b716f447771c2314918811bbf9f834792bbb64c`
- immutable harness authorization payload SHA-256:
  `98fff9c8c46deebe2b36b7d90f14b6ec824a32a32969db76b02e23981548b157`

## Synthetic gate

- offline replay: qualified; report file SHA-256
  `fd667510aa16d854983aa49a119342ca8831824f2058d82f7097583d585bd9f5`
- fixed live canary: qualified; exactly two 32-row synthetic chunks, each 32/32 with
  duplicate, missing, unexpected, and order-mismatch counts all zero; report file SHA-256
  `ae0d58740494d7b016ca8cb284f28e6af4e3af7be9e297a37efab1adb4594608`

## Organizer pilot result

The deterministic fold-0 training pilot ran exactly four initial 32-row chunks with worker 1.
The raw-free final state was:

- total attempts: 4; parsed: 3; failed: 1
- accepted: **79/128**; retryable: 49; exhausted: 0; unassessed: 0
- initial threshold: 103/128; failure category:
  `initial_exact_match_below_threshold`
- terminal-marker payload/file SHA-256:
  `490e20c5bf5685eb5a975f9eeb7daf1ee01a75532450e5707109d444ae677d43` /
  `a51c11c45e4f2902ea961907af619589abac4485f8ee11ae3617dc293d0f0b28`
- raw-free status file SHA-256:
  `1f5078cc9c4c92293629d2949ff1bda759e0f0794a8cc13616c9a4da5815d4ea`

The local finalizer exited `1` as specified, created no source JSONL or source manifest,
and the marker blocks every later teacher execution for this plan. Status inspection remains
available.

## Mandatory stop

Do not run repair, logical audit, pilot receipt, full bank, corpus materialization, SFT
preflight, GPU training/generation, OOF work, holdout evaluation, leaderboard prediction, or
submission work from this candidate. Do not create or allowlist `teacher-pilot-v5` until a new
versioned synthetic live-eval harness is designed and explicitly approved. Kaggle upload still
requires a separate explicit user request.
