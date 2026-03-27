## TalkNet Refinement Decoupling

Branch:
- `feat/talknet-refinement-decouple`

Purpose:
- stop Phase 1 from blocking on expensive inline TalkNet speaker binding
- preserve higher-quality TalkNet binding as an optional second-pass refinement
- reduce GPU cost and timeout risk during `finalize_extraction`

### Problem

Before this change, Phase 1 `finalize_extraction(...)` ran:
- tracklet clustering
- TalkNet speaker binding

This made Phase 1 completion slow and expensive, even for short videos. In practice:
- clustering completed
- TalkNet dominated runtime
- `finalize_extraction` could retry across containers
- Phase 2 could not start until TalkNet finished

### New Design

Phase 1 now uses:
- heuristic speaker binding inline by default

TalkNet is still available, but moved to:
- an optional second-pass refinement call

This means:
- Phase 1 can complete faster and unblock the rest of the pipeline
- TalkNet quality can still be used when desired
- TalkNet no longer has to sit on the critical path by default

### Files Changed

- `modal_worker.py`
- `pipeline/phase_1_modal_pipeline.py`

### `modal_worker.py` Changes

#### 1. Disabled debug logging by default

Changed secret defaults:
- `CLYPT_MODEL_DEBUG: "1" -> "0"`
- `CLYPT_MODEL_DEBUG_EVERY: "10" -> "20"`

Impact:
- reduces TalkNet log spam
- lowers overhead during model execution

#### 2. Added inline TalkNet gate

New env flag:
- `CLYPT_ENABLE_TALKNET_INLINE`

Default:
- `"0"`

Behavior:
- when off, Phase 1 finalize skips TalkNet and uses heuristic speaker binding
- when on, inline finalize may still use TalkNet first and fall back to heuristic if needed

#### 3. Changed `_run_speaker_binding(...)`

Updated behavior:
- heuristic binding is now the default inline path
- TalkNet is only used inline if:
  - `force_talknet=True`, or
  - `CLYPT_ENABLE_TALKNET_INLINE=1`

Impact:
- removes TalkNet from the default critical path

#### 4. Added timing logs to Phase 1 finalize

New timing output around:
- Step 3 clustering
- Step 4 speaker binding

Impact:
- makes runtime cost visible
- helps distinguish clustering cost from speaker-binding cost

#### 5. Added `refine_speaker_bindings(...)`

New Modal method:
- `refine_speaker_bindings(video_bytes, audio_wav_bytes, words, tracks)`

Behavior:
- runs TalkNet as a dedicated second-pass refinement
- returns:
  - refined `words`
  - refined `speaker_bindings`

Impact:
- keeps TalkNet available without forcing it into Phase 1 finalize

### `pipeline/phase_1_modal_pipeline.py` Changes

#### 1. Added optional refinement hook

New helper:
- `maybe_refine_speaker_bindings(...)`

Behavior:
- by default, does nothing
- if refinement is enabled, calls the new worker method after Phase 1 returns

#### 2. Added refinement env flag

New env flag:
- `CLYPT_RUN_TALKNET_REFINEMENT`

Default:
- `"0"`

Behavior:
- `0`: heuristic Phase 1 output is used directly
- `1`: pipeline runs second-pass TalkNet refinement before writing outputs

Impact:
- makes TalkNet opt-in instead of mandatory

### Resulting Runtime Modes

#### Fast default mode

Flags:
- `CLYPT_ENABLE_TALKNET_INLINE=0`
- `CLYPT_RUN_TALKNET_REFINEMENT=0`

Behavior:
- Phase 1 uses heuristic speaker binding
- fastest and cheapest path
- Phase 2 can start sooner

Recommended for:
- normal iteration
- cost-sensitive runs
- debugging the rest of the pipeline

#### Optional quality refinement mode

Flags:
- `CLYPT_ENABLE_TALKNET_INLINE=0`
- `CLYPT_RUN_TALKNET_REFINEMENT=1`

Behavior:
- Phase 1 completes with heuristic binding
- pipeline then runs TalkNet refinement as a separate pass

Recommended for:
- selected runs where speaker quality matters
- measuring TalkNet quality without blocking every Phase 1 execution

#### Legacy inline TalkNet mode

Flags:
- `CLYPT_ENABLE_TALKNET_INLINE=1`

Behavior:
- restores inline TalkNet attempt during finalize

Recommended for:
- targeted experiments only
- not recommended as default due to cost and timeout risk

### Expected Benefits

- Phase 1 should complete more reliably
- less chance of hitting the finalize timeout
- reduced wasted GPU time on blocking TalkNet runs
- easier visibility into where runtime is spent
- TalkNet quality preserved as an optional upgrade path

### Recommended Rollout

1. Deploy this branch.
2. Run with defaults first:
   - no refinement
3. Confirm Phase 1 completes and Phase 2 starts reliably.
4. Enable `CLYPT_RUN_TALKNET_REFINEMENT=1` only when you want higher-quality speaker binding.

### Notes

- This change does not remove TalkNet.
- It changes when TalkNet runs.
- The main tradeoff is explicit:
  - default path optimizes cost and throughput
  - optional path optimizes speaker-binding quality
