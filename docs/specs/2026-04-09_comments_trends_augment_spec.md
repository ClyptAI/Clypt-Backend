# Clypt V3.1 Spec: Comment + Trend Signal Augmentation (Hard-Join, Fail-Fast)

**Status:** Active (implemented behavior reference)  
**Date:** 2026-04-09  
**Scope:** Add YouTube comments + trend signals to Phase 4 retrieval/ranking in augment mode, persist node/candidate attribution in Spanner, and use a local Phase 1 test-bank link mapping for deterministic input videos.

---

## 1. Locked Decisions

1. Signal mode: `augment` (single clip set, not separate list).
2. Join mode: `hard join` (Phase 4 must wait for signals).
3. Failure mode: `fail hard/fail fast` on signal pipeline errors.
4. Comments ordering: `relevance` only.
5. Comments dedupe: no near-duplicate dedupe across users; only obvious spam from same author.
6. Replies: fetch full replies for top-level comments.
7. Attribution scope: `edge-only inferred` (do not infer unrelated temporal neighbors).
8. Hop depth max: `2`.
9. Temporal expansion cap: `+/- 30s`.
10. Candidate attribution: weighted aggregate.
11. Cross-source agreement bonus: additive and capped.
12. Bonus/signal influence: capped to preserve base semantic quality.
13. LLM routing for signal pipeline is explicitly locked (see Section 1.1).
14. LLM failures in locked signal callpoints are terminal for the run (no silent fallback).

## 1.1 Locked LLM Call Matrix (Signal Pipeline)

| Callpoint | Purpose | Model | Thinking |
|---|---|---|---|
| `#1` | Comment/trend cluster -> clip-seeking retrieval prompt generation | Gemini 3 Flash | `low` |
| `#2` | Trend relevance adjudication against video context | Gemini 3 Flash | `minimal` |
| `#3` | Comment/reply quality classification (useful vs noise) | Gemini 3.1 Flash Lite | `low` |
| `#5` | Cluster-to-node moment span resolution | Gemini 3 Flash | `minimal` |
| `#9` | Trend query synthesis from video context | Gemini 3 Flash | `low` |
| `#10` | Top-level + full-reply thread consolidation | Gemini 3 Flash | `minimal` |
| `#11` | Candidate attribution explanation text for UI/debug | Gemini 3.1 Flash Lite | `low` |

Notes:
1. `#11` is explanatory only and does not directly alter deterministic score math.
2. For all listed callpoints, malformed/invalid responses are treated as hard failures under fail-fast policy.

---

## 2. Problem Statement

Current Phase 4 uses general semantic retrieval prompts and graph traversal only. It lacks:
- external audience signal seeding (comments),
- external zeitgeist signal seeding (trends),
- deterministic attribution of why a node/clip was externally boosted.

Also, MVP testing uses URL-input semantics with a curated link->local-video test-bank mapping for Phase 1, and this path needs deterministic validation and reproducible run behavior.

---

## 3. Non-Goals (MVP)

1. No autonomous catalog-wide trend daemon (future Phase 0/ops layer).
2. No multi-platform comment sources (YouTube only).
3. No separate "signal-only clip list" output.
4. No quota optimization beyond sensible page caps for testing.

---

## 4. Architecture Changes

## 4.1 Phase 1 Test-Bank Link Mapping

Replace URL downloading work for MVP with a deterministic local test-bank.

### Required behavior

1. Introduce a test-bank mapping from public link -> local video file path.
2. Phase 1 input resolver uses mapped local file path (no network download).
3. Comments/trends still use the provided public link for external signal APIs.
4. Support a small fixed bank (5-10 links) for reproducible testing.
5. If a link is not present in the test-bank when test-bank mode is enabled, return a clear validation error before Phase 1 starts.

### Example mapping payload

```json
{
  "https://www.youtube.com/watch?v=abc123": {
    "local_video_path": "/opt/clypt-phase1/videos/mrbeastflagrant.mp4"
  }
}
```

## 4.2 External Signals Pipeline (Parallel + Hard Join)

Start comment ingestion when comments signals are enabled and source video URL/video_id is known, in parallel with Phase 2 and Phase 3. Start trend ingestion when trend signals are enabled and after Phase 2 semantic summaries are available.

Execution flow:
1. **Phase 1 preflight (media-only):**
- if test-bank mode is enabled, resolve source URL -> local media path.
- this resolution is used only for Phase 1 media/video-audio input.
- if test-bank strict mode is enabled and link is unmapped: fail immediately before Phase 1 starts.
2. **Phase24 preflight (signal-only):**
- use the posted source URL (not local path mapping) to resolve `youtube_video_id` for comments.
- if comments signals are enabled and `youtube_video_id` cannot be resolved from URL: fail immediately before Phase 2 starts.
3. `run()` starts `comments_future` immediately after Phase24 preflight when `CLYPT_ENABLE_COMMENT_SIGNALS=1`.
4. Phase 2 runs.
5. On Phase 2 success, `trends_future` starts immediately using Phase 2 semantic summaries when `CLYPT_ENABLE_TREND_SIGNALS=1`.
6. Phase 3 runs; `trends_future` executes concurrently with Phase 3 (no wait).
7. Before Phase 4 prompt seeding: hard-join waits for all enabled signal futures:
- both futures when comments+trends are enabled,
- only `comments_future` when trends are disabled,
- only `trends_future` when comments are disabled.
8. If any enabled signal future raises or returns invalid payload: fail run immediately (terminal), do not enter Phase 4.
9. If all enabled signal futures succeed: enrich Phase 4 prompts + attribution.
10. If both `CLYPT_ENABLE_COMMENT_SIGNALS=0` and `CLYPT_ENABLE_TREND_SIGNALS=0`, skip signal futures and proceed to Phase 4 with general prompts only.

Cancellation and failure propagation:
1. If `comments_future` fails before `trends_future` starts and trends are enabled: mark run failed and skip launching `trends_future`.
2. If `comments_future` fails while `trends_future` is running: mark run failed, request cancellation for `trends_future`, and do not enter Phase 4.
3. If `trends_future` fails (when enabled): mark run failed and do not enter Phase 4.
4. If Phase 2 fails: request cancellation for `comments_future`, never start `trends_future`, terminate run.
5. If Phase 3 fails: request cancellation for any active signal future, terminate run.

Notes:
- This preserves parallelism while keeping hard correctness gate at Phase 4.
- Mapping signals to nodes still occurs after Phase 2 nodes exist.
- Test-bank mapping does not drive comments/trends API fetches; source URL does.

---

## 5. Data Model Additions (Spanner)

## 5.1 New tables

1. `external_signals`
- PK: `(run_id, signal_id)`
- `signal_type STRING(32)` (`comment_top`, `comment_reply`, `trend_topic`, `trend_query`)
- `source_platform STRING(32)` (`youtube`, `google_trends`)
- `source_id STRING(MAX)` (comment id, trend token)
- `author_id STRING(MAX)` nullable
- `text STRING(MAX)` (raw text)
- `engagement_score FLOAT64` (likes/replies/traffic-normalized)
- `published_at TIMESTAMP` nullable
- `metadata_json STRING(MAX)`

2. `external_signal_clusters`
- PK: `(run_id, cluster_id)`
- `cluster_type STRING(32)` (`comment`, `trend`)
- `summary_text STRING(MAX)`
- `member_signal_ids ARRAY<STRING(128)>`
- `cluster_weight FLOAT64`
- `embedding ARRAY<FLOAT32>`
- `metadata_json STRING(MAX)`

3. `node_signal_links`
- PK: `(run_id, node_id, cluster_id)`
- `link_type STRING(16)` (`direct`, `inferred`)
- `hop_distance INT64`
- `time_offset_ms INT64`
- `similarity FLOAT64`
- `link_score FLOAT64`
- `evidence_json STRING(MAX)`

4. `candidate_signal_links`
- PK: `(run_id, clip_id, cluster_id)`
- `cluster_type STRING(32)`
- `aggregated_link_score FLOAT64`
- `coverage_ms INT64`
- `direct_node_count INT64`
- `inferred_node_count INT64`
- `agreement_flags ARRAY<STRING(32)>` (`general`, `comment`, `trend`)
- `bonus_applied FLOAT64`
- `evidence_json STRING(MAX)`

5. `prompt_source_links`
- PK: `(run_id, prompt_id)`
- `prompt_source_type STRING(16)` (`general`, `comment`, `trend`)
- `source_cluster_id STRING(128)` nullable (set for `comment`/`trend`, null for `general`)
- `source_cluster_type STRING(32)` nullable (`comment`, `trend`)
- `metadata_json STRING(MAX)`

This table is the durable source of truth for prompt source attribution and agreement scoring inputs.

6. `subgraph_provenance`
- PK: `(run_id, subgraph_id)`
- `seed_source_set ARRAY<STRING(16)>` (values from `general|comment|trend`)
- `seed_prompt_ids ARRAY<STRING(128)>`
- `source_cluster_ids ARRAY<STRING(128)>`
- `support_summary_json STRING(MAX)` (counts/weights by source type)
- `canonical_selected BOOL`
- `dedupe_overlap_ratio FLOAT64` nullable
- `selection_reason STRING(128)` nullable
- `metadata_json STRING(MAX)`

This table is the durable source of truth for subgraph-level provenance used by subgraph review context and downstream debugging.

## 5.2 Referential integrity constraints

Add explicit FK constraints in Spanner:

1. `node_signal_links (run_id, node_id)` -> `semantic_nodes (run_id, node_id)`
2. `node_signal_links (run_id, cluster_id)` -> `external_signal_clusters (run_id, cluster_id)`
3. `candidate_signal_links (run_id, clip_id)` -> `clip_candidates (run_id, clip_id)`
4. `candidate_signal_links (run_id, cluster_id)` -> `external_signal_clusters (run_id, cluster_id)`
5. `prompt_source_links (run_id, source_cluster_id)` -> `external_signal_clusters (run_id, cluster_id)` when `source_cluster_id IS NOT NULL`

Additional integrity rules:
1. `candidate_signal_links.cluster_type` must match the linked cluster type.
2. `prompt_source_links.prompt_source_type` and `source_cluster_id/source_cluster_type` must be consistent:
- `general` -> both cluster fields null
- `comment|trend` -> both cluster fields non-null and type-aligned
3. `subgraph_provenance.seed_source_set` must be non-empty.
4. Run-level delete/cleanup must delete child link rows before parent rows if cascade is not configured.

## 5.3 Existing table extension

`clip_candidates`:
- add `external_signal_score FLOAT64` nullable
- add `agreement_bonus FLOAT64` nullable
- add `external_attribution_json STRING(MAX)` nullable

---

## 6. Signal Ingestion Design

## 6.1 Comments (YouTube Data API)

### Fetch plan
1. Resolve `youtube_video_id` from posted source URL.
2. Call `commentThreads.list` with:
- `part=snippet,replies`
- `videoId=<id>`
- `order=relevance`
- `textFormat=plainText`
- paginated up to configured cap.
3. Determine scalable top-thread count from total available threads:
- `total_threads = pageInfo.totalResults`
- `target_top_threads = clamp(ceil(sqrt(total_threads)), CLYPT_COMMENT_TOP_THREADS_MIN, CLYPT_COMMENT_TOP_THREADS_MAX)`
- default ceiling for MVP: `CLYPT_COMMENT_TOP_THREADS_MAX=40`
4. Dynamic formula is the only selection rule for top-thread count in this spec (no separate fixed max-thread override).
5. Select the first `target_top_threads` in YouTube relevance order (after same-author spam collapse only).
6. For selected threads, fetch full replies using `comments.list(parentId=...)` (paginated).
7. Keep top-level + replies as independent signals.
8. Run `#10` thread consolidation call (Gemini 3 Flash, `minimal`) to synthesize thread intent + moment hints.
9. Run `#3` quality classification call (Gemini 3.1 Flash Lite, `low`) to filter obvious low-signal comments/replies before clustering.

### Spam rule
Only collapse duplicates when same `author_id` repeats effectively identical text in spam-like pattern.

### Frequency semantics
Do not cross-user dedupe semantically similar comments; frequency is intentionally retained as signal strength.

## 6.2 Trends (Python path)

Use Python-native trend client (`trendspyg`) for MVP integration simplicity in current worker stack.

### Retrieval logic (video-guided)
1. Build trend queries using `#9` synthesis call (Gemini 3 Flash, `low`) from Phase 2 semantic summaries + video metadata.
2. Query related trends/topics/queries for those seeds.
3. Retain trend items using `#2` relevance adjudication call (Gemini 3 Flash, `minimal`) against full video context.
4. Convert retained trend items into trend clusters.

This is not autonomous catalog polling; it is run-scoped trend lookup for the target video.

---

## 7. Cluster + Mapping Logic

## 7.1 Clustering

1. Embed all comment signals.
2. Build comment clusters with deterministic agglomerative thresholding (cosine).
3. Convert each cluster into retrieval prompts with `#1` call (Gemini 3 Flash, `low`).
4. Embed trend clusters similarly.

## 7.2 Node linking

For each cluster:
1. `direct` links: retrieve top seed nodes by embedding similarity.
2. Resolve cluster moment span with `#5` call (Gemini 3 Flash, `minimal`) over local candidate node neighborhoods.
3. `inferred` links: expand via graph edges only, constrained by:
- max hop depth `2`
- max temporal window `+/- 30s`
- edge-existence required (`edge-only inferred`).
4. Inferred expansion uses the exact same edge weighting and traversal behavior as the existing general Phase 4 pipeline (no augmented-specific edge allow-list or extra penalties).
5. Write `node_signal_links`.

No free temporal-only expansion without edge support.

## 7.3 Subgraph-level dedupe (pre-Gemini)

Before subgraph review calls, perform cross-source subgraph dedupe using node-overlap Jaccard:
- if overlap ratio > `CLYPT_PHASE4_SUBGRAPH_OVERLAP_DEDUPE_THRESHOLD`, keep one canonical subgraph (highest deterministic expansion score).
- preserve provenance metadata so canonical subgraph still records contributing seed sources (`general/comment/trend`).

This is intentional and not redundant:
1. pre-Gemini dedupe reduces duplicate LLM review calls/cost,
2. post-review candidate dedupe still runs because distinct subgraphs can yield overlapping clip windows.
3. merged subgraph provenance is passed into subgraph-review prompt context.

Subgraph provenance payload (minimum):
- `seed_source_set` (array of `general|comment|trend`)
- `seed_prompt_ids` (union across merged seeds)
- `source_cluster_ids` (comment/trend cluster ids that contributed)
- `support_summary` (counts by source type)

---

## 8. Phase 4 Integration (Augment Mode)

## 8.1 Prompt pool

Final prompt pool = general meta prompts + comment cluster prompts + trend cluster prompts.

During subgraph review prompting, include subgraph provenance context so Gemini can weigh confidence with awareness of independent source convergence.

Prompt metadata contract:
- each prompt carries `prompt_source_type = general | comment | trend`
- each prompt is durably persisted in `prompt_source_links`
- this field is propagated into seeds/subgraphs/candidates for deterministic source attribution and agreement scoring

## 8.2 Seed retrieval

Use existing retrieval path across semantic + multimodal embeddings. No separate candidate generator.

## 8.3 Candidate scoring additions

Base candidate ranking remains existing semantic pipeline output (`base_score`). Add deterministic external components using the formulas below.

Symbol-to-env mapping used in formulas:
- `W_TOP_LIKE = CLYPT_SIGNAL_ENGAGEMENT_TOP_LIKE_WEIGHT`
- `W_TOP_REPLY = CLYPT_SIGNAL_ENGAGEMENT_TOP_REPLY_WEIGHT`
- `W_REPLY_LIKE = CLYPT_SIGNAL_ENGAGEMENT_REPLY_LIKE_WEIGHT`
- `W_REPLY_PARENT = CLYPT_SIGNAL_ENGAGEMENT_REPLY_PARENT_WEIGHT`
- `W_CLUSTER_MEAN = CLYPT_SIGNAL_CLUSTER_MEAN_WEIGHT`
- `W_CLUSTER_MAX = CLYPT_SIGNAL_CLUSTER_MAX_WEIGHT`
- `W_CLUSTER_FREQ = CLYPT_SIGNAL_CLUSTER_FREQ_WEIGHT`
- `FREQ_REF = CLYPT_SIGNAL_CLUSTER_FREQ_REF`
- `HOP_DECAY_1 = CLYPT_SIGNAL_HOP_DECAY_1`
- `HOP_DECAY_2 = CLYPT_SIGNAL_HOP_DECAY_2`
- `W_COVERAGE = CLYPT_SIGNAL_COVERAGE_WEIGHT`
- `W_DIRECT_RATIO = CLYPT_SIGNAL_DIRECT_RATIO_WEIGHT`
- `EPSILON = CLYPT_SIGNAL_EPSILON`

### 8.3.1 Per-signal engagement score

For each external signal `s`:

1. `likes_term = log1p(like_count_s)`
2. `reply_term = log1p(reply_count_s)` for top-level comments, else `0` for replies
3. `quality_mult` from callpoint `#3`:
- `high_signal=1.00`
- `contextual=0.75`
- `low_signal=0.30`
- `spam=0.00`

Raw score:
- top-level comment:  
  `raw_s = quality_mult * (W_TOP_LIKE * likes_term + W_TOP_REPLY * reply_term)`
- reply comment:  
  `raw_s = quality_mult * (W_REPLY_LIKE * likes_term + W_REPLY_PARENT * log1p(parent_reply_count))`

Run-normalized score:
- `denom = max(p95(raw_nonspam_scores), EPSILON)`
- if `raw_nonspam_scores` is empty: `signal_score_s = 0` for all signals
- otherwise: `signal_score_s = clip(raw_s / denom, 0, 1)`

### 8.3.2 Cluster weight (frequency preserved)

For cluster `k`:
- `mean_eng_k = mean(signal_score_s in k)`
- `max_eng_k = max(signal_score_s in k)`
- `freq_term_k = clip(log1p(num_signals_k) / log1p(FREQ_REF), 0, 1)`

`cluster_weight_k = clip(W_CLUSTER_MEAN * mean_eng_k + W_CLUSTER_MAX * max_eng_k + W_CLUSTER_FREQ * freq_term_k, 0, 1)`

### 8.3.3 Node link score

For node `i` linked to cluster `k`:
- `direct_score_ik = cluster_weight_k * similarity_ik`

Decay terms:
- `hop_decay = 1.00` (direct), `HOP_DECAY_1` (1-hop inferred), `HOP_DECAY_2` (2-hop inferred)
- `time_decay = max(0, 1 - abs(time_offset_ms) / CLYPT_SIGNAL_TIME_WINDOW_MS)`

Final:
- `link_score_ik = direct_score_ik * hop_decay * time_decay`

Persist in `node_signal_links.link_score`.

### 8.3.4 Candidate cluster aggregation (weighted aggregate)

For candidate `c` and cluster `k`:
- Let each included linked node have overlap duration weight `w_i` in candidate span.
- `agg_den_ck = sum(w_i)`
- if `agg_den_ck <= EPSILON`, set `agg_ck = 0`
- else `agg_ck = sum(w_i * link_score_ik) / agg_den_ck`
- if `candidate_duration_ms <= EPSILON`, set `coverage_ck = 0`
- else `coverage_ck = clip(total_overlap_ms / candidate_duration_ms, 0, 1)`
- `direct_ratio_ck = direct_overlap_ms / total_overlap_ms` (0 when denominator is 0)

`cluster_contrib_ck = agg_ck * ((1 - W_COVERAGE) + W_COVERAGE * coverage_ck) * ((1 - W_DIRECT_RATIO) + W_DIRECT_RATIO * direct_ratio_ck)`

Per-cluster cap:
- `cluster_contrib_ck = min(cluster_contrib_ck, CLYPT_SIGNAL_CLUSTER_CAP)` (default `0.12`)

### 8.3.5 External score and agreement bonus

External signal score:
- `external_signal_score_c = min(sum(cluster_contrib_ck over k), CLYPT_SIGNAL_TOTAL_CAP)` (default `0.20`)

Source coverage definition (used by meaningful-support checks):
- for source `S` in candidate `c`, compute union-overlap duration across all linked nodes from that source:
  - `source_union_overlap_ms_c,S = union_overlap_ms(linked_nodes_for_source_S within candidate_window_c)`
- if `candidate_duration_ms <= EPSILON`, set `source_coverage_c,S = 0`
- else `source_coverage_c,S = clip(source_union_overlap_ms_c,S / candidate_duration_ms, 0, 1)`

Cross-source agreement bonus (additive, capped):
- Meaningful support thresholds:
  - for comment/trend: max `cluster_contrib_ck` for that source >= `CLYPT_SIGNAL_MEANINGFUL_MIN_CLUSTER_CONTRIB`
  - and `source_coverage_c,S >= CLYPT_SIGNAL_MEANINGFUL_MIN_SOURCE_COVERAGE`
  - for general: candidate has at least one supporting prompt with `prompt_source_type=general` from `prompt_source_links`
- Tier 1: has meaningful support from `general + comment` -> `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER1`
- Tier 2: has meaningful support from `general + comment + trend` -> `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER2`
- `agreement_bonus_c = min(tier_bonus, CLYPT_SIGNAL_AGREEMENT_CAP)` (default `0.10`)

Final candidate score:
- `final_score_c = base_score_c + external_signal_score_c + agreement_bonus_c`

`external_signal_score + agreement_bonus` is capped to prevent external signals from overpowering intrinsic semantic quality.

Constants above are sourced from env configuration (Section 11). Numeric literals in examples are defaults only.

## 8.4 Attribution persistence

1. Write `candidate_signal_links` for every kept candidate and supporting cluster.
2. Persist compact explanation in `clip_candidates.external_attribution_json`.
3. Generate explanation text via `#11` call (Gemini 3.1 Flash Lite, `low`) using deterministic evidence payloads.
4. Persist subgraph provenance debug artifact/row so subgraph-level and candidate-level provenance remain auditable end-to-end.
5. Persist subgraph-level provenance row in `subgraph_provenance` for every deduped/kept subgraph.

---

## 9. Clarification: Why not rely only on existing subgraph expansion?

Existing expansion finds connected moments but does not label causal provenance of external signals. The new attribution layer is needed for:
1. explainability in Cortex graph,
2. deterministic weighted aggregate scoring,
3. reliable debugging ("what external source moved this rank").

We do not add a second heavy traversal system; we add deterministic attribution over the existing traversal output.

---

## 10. Failure Policy + Logging

## 10.1 Hard-fail conditions

Any of the following fails the run immediately:
1. comments fetch/parse failure (when `CLYPT_ENABLE_COMMENT_SIGNALS=1`),
2. replies fetch failure (when `CLYPT_ENABLE_COMMENT_SIGNALS=1`),
3. trends fetch/parse failure (when `CLYPT_ENABLE_TREND_SIGNALS=1`),
4. cluster build failure,
5. future failure in enabled signal futures,
6. missing test-bank mapping in test-bank mode (pre-Phase1 gate).
7. unresolved `youtube_video_id` from source URL when comments signals are enabled (pre-Phase2 gate).

## 10.2 Structured logging

Add log events:
1. `signals_fetch_start`, `signals_fetch_done`
2. `comments_threads_count`, `comments_replies_count`
3. `trend_fetch_start_after_phase2`, `trend_items_count`, `trend_retained_count`
4. `signal_clusters_built` with counts/sizes
5. `signals_hard_join_wait_start`, `signals_hard_join_wait_done`
6. `signals_node_linking_done`
7. `signals_candidate_attribution_done`
8. `signals_llm_call_start` / `signals_llm_call_done` with `callpoint_id`, `model`, `latency_ms`
9. terminal `signals_failure` with typed error code and `failed_callpoint_id` where applicable

---

## 11. Configuration Surface

Add env vars:
- `CLYPT_ENABLE_COMMENT_SIGNALS=1|0`
- `CLYPT_ENABLE_TREND_SIGNALS=1|0`
- `CLYPT_SIGNAL_MODE=augment`
- `CLYPT_SIGNAL_FAIL_FAST=1`
- `CLYPT_SIGNAL_MAX_HOPS=2`
- `CLYPT_SIGNAL_TIME_WINDOW_MS=30000`
- `CLYPT_COMMENT_ORDER=relevance`
- `CLYPT_COMMENT_MAX_REPLIES_PER_THREAD`
- `CLYPT_COMMENT_TOP_THREADS_MIN=15`
- `CLYPT_COMMENT_TOP_THREADS_MAX=40`
- `CLYPT_COMMENT_CLUSTER_SIM_THRESHOLD`
- `CLYPT_TREND_MAX_ITEMS`
- `CLYPT_TREND_RELEVANCE_THRESHOLD`
- `CLYPT_SIGNAL_EPSILON=1e-6`
- `CLYPT_SIGNAL_CLUSTER_CAP`
- `CLYPT_SIGNAL_AGREEMENT_CAP`
- `CLYPT_SIGNAL_TOTAL_CAP`
- `CLYPT_SIGNAL_ENGAGEMENT_TOP_LIKE_WEIGHT=0.65`
- `CLYPT_SIGNAL_ENGAGEMENT_TOP_REPLY_WEIGHT=0.35`
- `CLYPT_SIGNAL_ENGAGEMENT_REPLY_LIKE_WEIGHT=0.85`
- `CLYPT_SIGNAL_ENGAGEMENT_REPLY_PARENT_WEIGHT=0.15`
- `CLYPT_SIGNAL_CLUSTER_MEAN_WEIGHT=0.45`
- `CLYPT_SIGNAL_CLUSTER_MAX_WEIGHT=0.25`
- `CLYPT_SIGNAL_CLUSTER_FREQ_WEIGHT=0.30`
- `CLYPT_SIGNAL_CLUSTER_FREQ_REF=30`
- `CLYPT_SIGNAL_HOP_DECAY_1=0.75`
- `CLYPT_SIGNAL_HOP_DECAY_2=0.55`
- `CLYPT_SIGNAL_COVERAGE_WEIGHT=0.30`
- `CLYPT_SIGNAL_DIRECT_RATIO_WEIGHT=0.15`
- `CLYPT_SIGNAL_MEANINGFUL_MIN_CLUSTER_CONTRIB`
- `CLYPT_SIGNAL_MEANINGFUL_MIN_SOURCE_COVERAGE=0.15`
- `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER1=0.04`
- `CLYPT_SIGNAL_AGREEMENT_BONUS_TIER2=0.07`
- `CLYPT_PHASE4_SUBGRAPH_OVERLAP_DEDUPE_THRESHOLD=0.70`
- `CLYPT_SIGNAL_LLM_FAIL_FAST=1`
- `CLYPT_SIGNAL_LLM_MODEL_1=gemini-3-flash`
- `CLYPT_SIGNAL_LLM_MODEL_2=gemini-3-flash`
- `CLYPT_SIGNAL_LLM_MODEL_3=gemini-3.1-flash-lite`
- `CLYPT_SIGNAL_LLM_MODEL_5=gemini-3-flash`
- `CLYPT_SIGNAL_LLM_MODEL_9=gemini-3-flash`
- `CLYPT_SIGNAL_LLM_MODEL_10=gemini-3-flash`
- `CLYPT_SIGNAL_LLM_MODEL_11=gemini-3.1-flash-lite`
- `CLYPT_PHASE1_INPUT_MODE=direct|test_bank` (default `test_bank` for this MVP path)
- `CLYPT_PHASE1_TEST_BANK_PATH` (JSON/YAML mapping file)
- `CLYPT_PHASE1_TEST_BANK_STRICT=1|0` (if `1`, unmapped links are rejected)

---

## 12. File-Level Implementation Plan (Spec-Level)

1. `backend/phase1_runtime/input_resolver.py` (new)
- resolve source link -> local Phase 1 video path via test-bank mapping.
- this module is media-only and does not own comments/trends API identity resolution.

2. `backend/providers/config.py`
- add signal + test-bank env settings dataclasses.

3. `backend/runtime/phase14_live.py`
- start `comments_future` right after Phase24 preflight.
- start `trends_future` immediately after Phase 2 completes (while Phase 3 runs).
- hard-join before Phase 4.
- enforce cancellation/failure propagation rules from Section 4.2.
- wire enriched prompts and attribution outputs.

4. New package `backend/pipeline/signals/`
- `comments_client.py`
- `trends_client.py`
- `cluster.py`
- `linking.py`
- `scoring.py`
- `llm_runtime.py`
- `contracts.py`

5. `backend/repository/models.py`
- add record models for new tables/fields.

6. `backend/repository/spanner_phase14_repository.py`
- DDL for new tables (`external_*`, `node_signal_links`, `candidate_signal_links`, `prompt_source_links`, `subgraph_provenance`) + write/read methods.

7. `backend/runtime/phase24_worker_app.py`
- include signal settings in worker initialization and payload handling.

8. Tests:
- `tests/backend/pipeline/signals/*`
- `tests/backend/repository/test_spanner_phase14_repository.py`
- `tests/backend/runtime/test_phase24_worker_app.py`
- `tests/backend/runtime/test_phase14_live.py`
- `tests/backend/phase1_runtime/test_input_resolver.py`
- `tests/backend/repository/test_prompt_source_links.py`
- `tests/backend/repository/test_subgraph_provenance.py`
- `tests/backend/runtime/test_phase24_signal_futures.py`

---

## 13. Acceptance Criteria

1. End-to-end URL-input run (MVP-mocked via test-bank link mapping for Phase 1 media) succeeds with comments+trends enabled and writes:
- `external_signals`,
- `external_signal_clusters`,
- `node_signal_links`,
- `candidate_signal_links`,
- `prompt_source_links`,
- `subgraph_provenance`.
2. Final `clip_candidates` contain external attribution fields.
3. Signal pipeline starts in parallel with Phase 2/3 and hard-joins before Phase 4.
4. Any signal subsystem failure produces immediate terminal run failure with clear typed logs.
5. Test-bank link resolution consistently routes Phase 1 input to mapped local videos for reproducible runs.

---

## 14. External References

1. YouTube `commentThreads.list`: https://developers.google.com/youtube/v3/docs/commentThreads/list  
2. YouTube `comments.list`: https://developers.google.com/youtube/v3/docs/comments/list  
3. YouTube quota guide: https://developers.google.com/youtube/v3/determine_quota_cost  
4. `trendspyg` repo: https://github.com/flack0x/trendspyg  
5. `trends-js` repo: https://github.com/Shaivpidadi/trends-js
