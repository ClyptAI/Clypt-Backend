# ReID-First Global Person Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace signature-based non-face global-person attachment with actual person ReID embeddings, while preserving co-visibility repair as the safety net against bad merges.

**Architecture:** Keep the current face-first clustering strategy for anchored identities, but remove geometry/signature as the primary non-face identity mechanism. Non-face tracklets must attach to global identities through learned person ReID embeddings plus explicit confidence thresholds, with co-visibility repair remaining enabled to split any remaining bad merges. Box geometry may still be logged for diagnostics, but it must not decide identity assignment.

**Tech Stack:** Existing Phase 1 worker pipeline, InsightFace face anchors, BPBreID-class or `osnet_ain_x1_0` ReID embeddings, FAISS or direct cosine similarity for per-job matching, pytest.

---

## File / Module Plan

### Existing files to modify
- Modify: `backend/do_phase1_worker.py`
  - Remove signature-only non-face attachment from Step 3 clustering.
  - Replace it with ReID-backed attachment and group assignment.
  - Keep co-visibility repair enabled and measurable.
- Modify: `backend/speaker_binding/visual_features.py`
  - Expose reusable person ReID feature extraction APIs for clustering.
- Modify: `backend/speaker_binding/visual_identity.py`
  - Add helpers for identity centroids / per-tracklet ReID evidence if the code belongs there rather than the worker.
- Modify: `backend/pipeline/phase1_contract.py`
  - Add optional diagnostics for ReID-driven global clustering evidence if needed.
- Modify: `tests/backend/do_phase1_service/test_extract.py`
  - Add end-to-end clustering regression coverage for the new attachment path.

### New files to create
- Create: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`
  - Focused tests for non-face attachment, ambiguity rejection, and co-visibility repair behavior.

### Existing helpers to retire or demote
- Modify or remove use of:
  - `_choose_signature_attachment_label(...)`
  - `_choose_signature_attachment_label_for_group(...)`
  - `_cluster_signature_only_tracklets(...)`
  - `_clusters_have_compatible_seat_signature(...)`
  - `_tracklet_signature(...)`
  - `_tracklet_signature_distance(...)`
- These may remain temporarily for diagnostics/backward-compat comparison, but they must no longer decide non-face global identity assignment.

---

## Target Behavior

### What must remain true
- Face-anchored identity formation remains the highest-precision path.
- Co-visibility repair remains active as a split pass after attachment.
- Same-frame collisions for a single global identity should fall dramatically on couch/panel scenes.

### What must change
- Non-face tracklets must no longer attach to globals because their median box geometry looks similar.
- “Same seat, similar box size” must not be enough to merge two identities.
- Signature/geometry can be used only for:
  - debugging
  - instrumentation
  - optional post-hoc analysis
- Signature/geometry must not be used as a primary matcher or fallback matcher for identity assignment.

### Acceptance criteria
- In a seated multi-person scene, distinct simultaneously visible people must not collapse into one global identity due only to spatial compatibility.
- The renderer should no longer show several different simultaneous people with the same global label in cases like the screenshot you shared.
- Regression metrics should show a drop in:
  - `same_identity_frame_collision_pairs_*`
  - `same_identity_frame_collision_frames_*`

---

## ReID-First Clustering Design

### Stage 1: Keep face-first seed clusters
- Build face-track clusters exactly as today.
- Use face embeddings and face-track associations to seed reliable global identities.
- Each face-seeded global identity should maintain:
  - member local track IDs
  - one or more ReID embedding centroids
  - evidence counts / embedding quality stats

### Stage 2: Build per-tracklet ReID evidence
- For every non-face or weak-face tracklet, compute a bounded set of person ReID embeddings.
- Reuse the branch’s real visual feature extraction path rather than inventing a parallel model loader.
- Each tracklet should yield:
  - sampled embedding vectors
  - a centroid or robust aggregate embedding
  - quality metadata (sample count, confidence, coverage)

### Stage 3: Attach to existing face-seeded globals by embedding similarity
- Compare each non-face tracklet against face-seeded global identity centroids.
- Use cosine similarity (or equivalent normalized distance) as the primary decision signal.
- Require:
  - sufficient embedding quality
  - similarity above threshold
  - clear margin over the second-best candidate
- If the tracklet is ambiguous, do not attach it yet.

### Stage 4: Cluster unattached non-face tracklets with ReID only
- Group remaining tracklets by ReID similarity alone.
- Allow them to form new globals only when:
  - they have enough embedding support
  - they are mutually similar by ReID
  - the cluster is not immediately contradicted by co-visibility conflict
- No geometry/signature fallback here either.

### Stage 5: Run co-visibility repair
- Keep `_repair_covisible_cluster_merges(...)` in the path.
- If a merged global contains tracklets that are clearly distinct people in overlapping frames, split it.
- Repair skip logic must be tightened so severe collision metrics can block the skip.

---

## Threshold / Decision Policy

### Attachment thresholds
- Introduce explicit env-tunable thresholds for:
  - minimum ReID sample count per tracklet
  - minimum centroid quality
  - minimum similarity to attach
  - minimum best-vs-second-best margin

### Ambiguity handling
- If a tracklet has no confident ReID match:
  - leave it unattached for the next stage
- If it still has no confident ReID cluster:
  - mint a new singleton global rather than forcing a bad merge

### No signature fallback
- If ReID evidence is weak or ambiguous, the system must prefer:
  - unresolved / singleton identity
over:
  - signature-based merge

This is the central product choice in this plan.

---

## Repair-Skip Tightening

### Current problem
- Repair can be skipped because coarse counts look healthy even when collision metrics are terrible.

### Required change
- `_should_skip_cluster_repair(...)` must take collision severity into account.
- A cluster state with severe same-frame collisions must not skip repair just because:
  - `face_cluster_count`
  - `clusters_before_repair`
  - `visible_people_est`
happen to line up.

### Minimum rule
- Do not skip repair if any of these exceed threshold:
  - `same_identity_frame_collision_pairs_before_repair`
  - `same_identity_frame_collision_frames_before_repair`
  - `same_identity_labels_with_collisions_before_repair`

---

## Instrumentation / Debug Outputs

### Add clustering diagnostics
- For each non-face attachment attempt, log:
  - chosen global candidate
  - best similarity
  - second-best similarity
  - attach / reject reason
- For each newly formed ReID-only cluster, log:
  - member count
  - centroid quality
  - ambiguity / rejection reason if not formed

### Add manifest-safe debug summaries
- Add optional summaries like:
  - `reid_attach_attempts`
  - `reid_attach_successes`
  - `reid_attach_ambiguous_rejections`
  - `reid_singleton_fallbacks`
  - `repair_skip_blocked_by_collisions`

---

## Task Plan

### Task 1: Lock down current bad behavior with tests

**Files:**
- Create: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`
- Modify: `tests/backend/do_phase1_service/test_extract.py`

- [ ] **Step 1: Write a failing regression test for simultaneous distinct seated people**

```python
def test_non_face_attachment_does_not_merge_covisible_distinct_people():
    ...
    assert len({label_for_tid["42"], label_for_tid["43"], label_for_tid["44"]}) == 3
```

- [ ] **Step 2: Run the focused test and verify it fails on current signature-based logic**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k covisible`
Expected: FAIL with merged identities.

- [ ] **Step 3: Add a failing test for repair-skip gating**

```python
def test_repair_skip_is_blocked_when_collision_metrics_are_severe():
    assert should_skip is False
```

- [ ] **Step 4: Run the second focused test and verify it fails**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k repair_skip`
Expected: FAIL because current gate ignores collisions.

- [ ] **Step 5: Commit**

```bash
git add tests/backend/speaker_binding/test_global_person_reid_clustering.py tests/backend/do_phase1_service/test_extract.py
git commit -m "test: lock down global clustering over-merge regressions"
```

### Task 2: Build reusable ReID tracklet evidence

**Files:**
- Modify: `backend/speaker_binding/visual_features.py`
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`

- [ ] **Step 1: Write the failing test for per-tracklet ReID evidence extraction**

```python
def test_tracklet_reid_evidence_returns_centroid_and_quality():
    evidence = build_tracklet_reid_evidence(...)
    assert evidence.centroid is not None
    assert evidence.sample_count >= 1
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k tracklet_reid`
Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement minimal ReID evidence builder**

```python
@dataclass
class TrackletReIDEvidence:
    centroid: np.ndarray | None
    sample_count: int
    quality: float
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k tracklet_reid`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/speaker_binding/visual_features.py backend/do_phase1_worker.py tests/backend/speaker_binding/test_global_person_reid_clustering.py
git commit -m "feat: add reid evidence extraction for global clustering"
```

### Task 3: Replace signature-based attachment to face-seeded globals

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`

- [ ] **Step 1: Write a failing test for ReID-first attachment**

```python
def test_non_face_tracklet_attaches_by_reid_similarity_not_geometry():
    ...
    assert assigned_label == face_seeded_label
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k reid_similarity`
Expected: FAIL.

- [ ] **Step 3: Implement ReID-first attachment with explicit ambiguity rejection**

```python
if best_similarity < attach_threshold:
    return None
if (best_similarity - second_best_similarity) < margin_threshold:
    return None
return best_label
```

- [ ] **Step 4: Remove signature helper calls from this attachment path**

Expected code effect:
- `_choose_signature_attachment_label(...)` no longer decides label assignment here.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k "reid_similarity or covisible"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/speaker_binding/test_global_person_reid_clustering.py
git commit -m "feat: replace signature attachment with reid matching"
```

### Task 4: Cluster unattached non-face tracklets by ReID only

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`

- [ ] **Step 1: Write a failing test for ReID-only grouping of unattached non-face tracklets**

```python
def test_unattached_non_face_tracklets_cluster_by_reid_without_geometry():
    ...
    assert formed_cluster_count == 2
```

- [ ] **Step 2: Run the test to verify failure**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k unattached_non_face`
Expected: FAIL.

- [ ] **Step 3: Implement ReID-only grouping with singleton fallback**

```python
if cluster_confident:
    assign_group_label(...)
else:
    assign_singleton_label(...)
```

- [ ] **Step 4: Run the focused test to verify pass**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k unattached_non_face`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/speaker_binding/test_global_person_reid_clustering.py
git commit -m "feat: cluster unattached body tracklets with reid only"
```

### Task 5: Tighten repair-skip gating with collision metrics

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Test: `tests/backend/speaker_binding/test_global_person_reid_clustering.py`

- [ ] **Step 1: Extend the failing repair-skip test with concrete collision thresholds**

```python
def test_repair_skip_false_when_collision_pairs_exceed_threshold():
    assert _should_skip_cluster_repair(...) is False
```

- [ ] **Step 2: Run the test and verify failure**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k repair_skip`
Expected: FAIL.

- [ ] **Step 3: Update `_should_skip_cluster_repair(...)` to consider collision metrics**

```python
if collision_pairs > 0 or collision_labels > 0:
    return False
```

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py -k "repair_skip or covisible"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py tests/backend/speaker_binding/test_global_person_reid_clustering.py
git commit -m "fix: force cluster repair when collision metrics are severe"
```

### Task 6: Add diagnostics and end-to-end regression coverage

**Files:**
- Modify: `backend/do_phase1_worker.py`
- Modify: `backend/do_phase1_service/test_extract.py`
- Modify: `backend/pipeline/phase1_contract.py`

- [ ] **Step 1: Write failing tests for ReID clustering summaries**

```python
def test_tracking_metrics_include_reid_clustering_debug_fields():
    assert metrics["reid_attach_attempts"] >= 0
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `PYTHONPATH=. pytest -q tests/backend/do_phase1_service/test_extract.py -k reid_clustering`
Expected: FAIL.

- [ ] **Step 3: Add manifest-safe diagnostics and worker logs**

Expected logs:
- `ReID attach: tid=... best=... second=... decision=...`
- `ReID body-only clusters formed: ...`
- `Repair skip blocked by collisions: ...`

- [ ] **Step 4: Run the end-to-end and focused test suites**

Run: `PYTHONPATH=. pytest -q tests/backend/speaker_binding/test_global_person_reid_clustering.py tests/backend/do_phase1_service/test_extract.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/do_phase1_worker.py backend/do_phase1_service/test_extract.py backend/pipeline/phase1_contract.py tests/backend/speaker_binding/test_global_person_reid_clustering.py
git commit -m "chore: add reid clustering diagnostics"
```

### Task 7: Validate on the couch-panel regression video

**Files:**
- Modify: none required unless regressions appear
- Use existing debug outputs and metrics

- [ ] **Step 1: Run the full Phase 1 pipeline on the known failing video**

Run:
```bash
cd /Users/rithvik/CascadeProjects/Clypt-V2/.worktrees/pyannote-visual-max-accuracy && ./scripts/do_phase1/run_remote_job.sh
```

- [ ] **Step 2: Inspect clustering metrics**

Expected improvements:
- lower `same_identity_frame_collision_pairs_before_repair`
- much lower `same_identity_frame_collision_pairs_after_repair`
- no obvious multi-person same-global collapse in the couch scene

- [ ] **Step 3: Inspect raw `phase_1_visual.json` for simultaneous same-global conflicts**

```bash
ssh -i ~/.ssh/clypt_do_ed25519 root@162.243.100.226 "python3 - <<'PY'\n...\nPY"
```

- [ ] **Step 4: Commit any final threshold tuning only if metrics justify it**

```bash
git add ...
git commit -m "tune: stabilize reid-first global clustering thresholds"
```

---

## Testing Strategy

### Unit tests
- ReID evidence extraction
- confident attach vs ambiguous reject
- singleton fallback for weak evidence
- repair-skip gating with collision metrics

### Integration tests
- face-seeded cluster + non-face attachment path
- ReID-only clustering for body-only tracklets
- manifest-safe debug output

### Manual validation
- run the known couch/panel video
- verify that simultaneous distinct seated people no longer share the same global label
- inspect collision metrics before/after repair

---

## Rollout Notes

- Keep the old signature helpers in the tree only until the ReID-first path is stable.
- Do not silently fall back to signature attachment if the ReID path is weak.
- If the ReID model is unavailable, fail loudly behind the experiment flag rather than shipping geometry-based merges under the same branch behavior.

