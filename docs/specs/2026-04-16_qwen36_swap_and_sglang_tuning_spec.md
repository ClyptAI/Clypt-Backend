# Clypt V3.1 Spec: Qwen3.6-35B-A3B Swap + SGLang Tuning

**Status:** Active (supersedes `2026-04-15_qwen_sglang_full_cutover_spec.md` on Qwen serving)
**Date:** 2026-04-16
**Owner:** Backend runtime / inference
**Scope:** Replace the Phase 2-4 generation model from `Qwen/Qwen3.5-27B` to `Qwen/Qwen3.6-35B-A3B` (MoE, ~3 B active / 35.95 B total) on the SGLang serving stack, and capture the FP8-KV / MTP / memory-fraction wins the current launch line leaves on the table. Scope covers SGLang launch flags, sampler defaults, concurrency caps, env/docs drift control, and the full-transition doctrine (no Qwen3.5-27B fallback retained anywhere).

---

## 0. Full-Transition Doctrine (No Fallback Retained)

This is an **atomic cutover**. After this spec lands:

- `Qwen/Qwen3.6-35B-A3B` is the only model id referenced by active code, config defaults, env files, systemd units, deploy scripts, and runbook docs.
- No `Qwen/Qwen3.5-27B` literal survives outside of (a) this spec's "why/history" paragraphs, (b) historical specs explicitly tagged Inactive/Superseded, and (c) one-time benchmark snapshots under `tmp/qwen36_swap/baseline/` used as evidence for the acceptance gate.
- No conditional `if model == "Qwen/Qwen3.5-27B"` shims, compatibility fallbacks, or per-model code paths are introduced. The swap is a literal-for-literal replacement plus a tightened sampler, nothing more.
- The legacy `scripts/do_phase1/systemd/clypt-vllm-qwen.service` unit (a 27B-vLLM-Docker fallback path from the pre-SGLang era) is removed from the repo. Its operator-facing "stop-and-disable" cleanup in the deploy script is retained defensively so hosts that still have the unit installed on disk get it disabled on next deploy.
- Any missing env (`GENAI_GENERATION_MODEL`, `GENAI_FLASH_MODEL`, `SG_MODEL`, `CLYPT_LOCAL_LLM_MODEL`, etc.) falls back to `Qwen/Qwen3.6-35B-A3B`. There is no split-brain state where code boots on a missing env and quietly serves 27B.

The `rg` gate in §8 enforces this doctrine.

---

## 1. Locked Decisions

1. **Model id:** `Qwen/Qwen3.6-35B-A3B` (exact HF repo id, case-sensitive).
2. **SGLang version:** pinned `>= 0.5.10` via `SG_PACKAGE_SPEC` in the deploy script.
3. **Launch flags (mandatory):**
   - `--kv-cache-dtype fp8_e4m3` (halves KV memory on H200; matches pre-SGLang vLLM unit).
   - Radix cache stays enabled (SGLang default in 0.5.10; only `--disable-radix-cache` exists, so we emit no flag unless the operator sets `SG_ENABLE_RADIX_CACHE=0`).
   - `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (NextN MTP; Qwen3.6 ships draft heads).
   - `--mamba-scheduler-strategy extra_buffer` plus `SGLANG_ENABLE_SPEC_V2=1` as an environment variable. Qwen3.6 is a hybrid Mamba/Attention MoE and SGLang 0.5.10 refuses to start with speculative decoding + radix cache unless both are set.
   - `--mem-fraction-static 0.78` (VibeVoice no longer co-resident after `2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md`; H200 has headroom).
   - `--context-length 131072`, `--chunked-prefill-size 8192`, `--schedule-policy lpm` (unchanged).
   - `--reasoning-parser qwen3` (required for Qwen3.6 response format).
   - `--grammar-backend xgrammar` (kept for step 1 to isolate the model swap from a grammar-backend swap; §6 R1 describes the `llguidance` follow-up).
4. **Thinking mode:** disabled at every call site via `chat_template_kwargs.enable_thinking=False`. Qwen3.6 **does not** support the `/think` / `/nothink` soft switch; the chat-template kwarg is the only correct toggle. `LocalOpenAIQwenClient.generate_json` sends it as a top-level field on the chat-completion payload (alongside `top_k` and `min_p`) because the client uses stdlib `urllib.request`, not the OpenAI Python SDK, so SDK-side `extra_body` unwrapping does not apply.
5. **Production sampler defaults** (strict-schema JSON path):
   - `temperature=0.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
   - These replace the Qwen3.5-era generic defaults (`temp=0.7, top_p=0.8, presence=1.5`) which were actively harmful for strict JSON (presence penalty pushes the sampler away from repeated schema keys).
6. **Bench sampler == prod sampler.** `scripts/bench_phase24_llm_concurrency.py` constructs its client with `LocalGenerationSettings` values identical to §1.5 so concurrency curves are faithful to production load.
7. **Concurrency caps** stay at the spec-2026-04-16 L4-offload floor unless rebench data in §7 unambiguously permits raising them: Phase 2 merge=8, boundary=10, Phase 3 local=8, long-range=8, Phase 4 subgraph=10.
8. **No Qwen3.5-27B fallback anywhere.** Enforced by the §0 doctrine and the §8 `rg` gate.

---

## 2. Why This Change

1. **Capability:** Qwen3.6-35B-A3B is a strictly better instruction-follower on structured-output benchmarks with roughly the same active parameter footprint as 27B dense. MoE with 256 routed / 8 active experts gives ~3 B active compute per token (much cheaper decode) at the cost of 72 GB of weights vs. the 27B dense footprint of ~54 GB.
2. **Serving efficiency left on the table (fixed here):**
   - No speculative decoding. Qwen3.6 ships NextN draft heads; SGLang MTP gives 1.5-2× decode throughput on our 1.5-2 KB structured outputs.
   - No FP8 KV cache on the current SGLang unit. Re-enabling halves KV memory → ~2× concurrency at the same context length.
   - `mem-fraction-static=0.55` is a holdover from VibeVoice co-residency; ~30 GB of H200 HBM was unused.
   - Sampler used `presence_penalty=1.5` (inherited from Qwen general recommendation), which is actively wrong for strict-schema JSON because schemas require repeated keys.
   - Bench script used a different sampler than production, so concurrency curves were not comparable.
   - `xgrammar` schema-compile was the cause of the 2026-04-15 Phase 4 crash documented in `docs/ERROR_LOG.md`; `llguidance` is a tested SGLang alternative worth A/B-ing (§6 R1, §7 follow-up).
3. **H200 headroom is now available.** After `2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md` moved VibeVoice ASR + node-media prep to Cloud Run L4, the H200 hosts only RF-DETR + SGLang Qwen + Phase 2-4 worker. 141 GB VRAM − ~72 GB weights − ~6 GB RF-DETR activations leaves ample room for KV and working set at higher static memory fractions.

---

## 3. Model + Hardware Facts

| Item | Value | Source |
|---|---|---|
| Model id | `Qwen/Qwen3.6-35B-A3B` | HF model card |
| Total / active params | 35.95 B BF16 / ~3 B | model card |
| Weight footprint | ~72 GB BF16 on disk and in VRAM | `safetensors.totalFileSize` |
| Architecture tag | `qwen3_5_moe` | model card `tags` |
| Experts | 256 routed (8 active) + 1 shared, expert intermediate dim 512 | model card |
| Attention | Gated DeltaNet (linear) + Gated Attention with GQA (16 Q heads, 2 KV heads, head dim 256) | model card |
| Native context | 262,144 tokens (extensible to 1.01 M via YaRN) | model card |
| Required SGLang | `>= 0.5.10` | model card |
| Reasoning parser | `--reasoning-parser qwen3` | model card |
| MTP draft method | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | model card |
| Soft `/think /nothink` switch | **Not supported** — must use `chat_template_kwargs.enable_thinking=False` | model card |
| Default thinking | **On** — must explicitly disable at every call site | model card |
| H200 VRAM | 141 GB HBM3e | hardware spec |
| Target `mem-fraction-static` | `0.78` canary (model card says 0.8; L4-offload spec ladder: 0.72 → 0.78 → higher) | `2026-04-16_phase14_l4_offload_concurrency_revamp_spec.md:374-378` |

Per-sequence KV cost estimate at 131K context: GQA-2 × head-dim 256 × two layers of attention × FP8 ≈ ~5 GB worst-case, well within the §1.7 concurrency caps at mem-fraction 0.78.

---

## 4. Code Changes

### 4.1 Source code (defaults flipped, no per-model conditionals)

- `backend/providers/config.py`
  - `VertexSettings.generation_model` default → `"Qwen/Qwen3.6-35B-A3B"`.
  - `VertexSettings.flash_model` default → `"Qwen/Qwen3.6-35B-A3B"`.
  - Env fallback `_read_env("GENAI_GENERATION_MODEL") or "Qwen/Qwen3.6-35B-A3B"`.
  - Env fallback `_read_env("GENAI_FLASH_MODEL") or "Qwen/Qwen3.6-35B-A3B"`.
  - `LocalGenerationSettings` dataclass defaults: `temperature=0.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
  - Matching env-fallback literals for `CLYPT_LOCAL_LLM_TEMPERATURE`, `CLYPT_LOCAL_LLM_TOP_P`, `CLYPT_LOCAL_LLM_TOP_K`, `CLYPT_LOCAL_LLM_MIN_P`, `CLYPT_LOCAL_LLM_PRESENCE_PENALTY`, `CLYPT_LOCAL_LLM_REPETITION_PENALTY` updated to the same defaults.

- `backend/runtime/phase14_live.py`
  - Both `flash_model` defaults in the dataclass and the constructor → `"Qwen/Qwen3.6-35B-A3B"`.

- `backend/pipeline/config.py`
  - `SignalLLMCallConfig.model_1`, `model_2`, `model_3`, `model_5`, `model_9`, `model_10`, `model_11` factory defaults all → `"Qwen/Qwen3.6-35B-A3B"`.

- `scripts/bench_phase24_llm_concurrency.py`
  - `--model` arg default → `"Qwen/Qwen3.6-35B-A3B"`.
  - Client construction replaced with a `LocalGenerationSettings` mirroring the §1.5 sampler so the bench is a faithful proxy for production load.

- `tests/backend/runtime/test_phase24_worker_app.py`
  - Update the two assertions that pinned `"Qwen/Qwen3.5-27B"` to pin `"Qwen/Qwen3.6-35B-A3B"`.

No client-side logic changes. `LocalOpenAIQwenClient.generate_json` already sets `chat_template_kwargs.enable_thinking=False`, emits strict `response_format` with `additionalProperties=false`, and posts to `/v1/chat/completions` on `SG_BASE_URL`.

### 4.2 Legacy unit removed

- `scripts/do_phase1/systemd/clypt-vllm-qwen.service` is **deleted**. It was a Qwen3.5-27B vLLM-on-Docker fallback from the pre-SGLang era; retaining it violates §0. The deploy script's defensive `systemctl stop/disable clypt-vllm-qwen.service 2>/dev/null || true` lines (lines 118-120) stay, so hosts with the unit still installed on disk get it quiesced on next deploy.

### 4.3 Serving (SGLang / systemd)

- `scripts/do_phase1/systemd/clypt-sglang-qwen.service` — the literal `ExecStart` line is rewritten to:
  ```
  ExecStart=/opt/clypt-phase1/venvs/sglang/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port 8001 \
    --reasoning-parser qwen3 --grammar-backend xgrammar \
    --schedule-policy lpm --chunked-prefill-size 8192 \
    --mem-fraction-static 0.78 --context-length 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-scheduler-strategy extra_buffer \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  # plus Environment=SGLANG_ENABLE_SPEC_V2=1
  ```
  (The deploy script regex-rewrites this line at install time, but the committed unit must match so a direct-boot without the deploy script still serves the right model.)

- `scripts/do_phase1/deploy_sglang_qwen_service.sh` — top-of-file default block bumped and extended:
  - `SG_PACKAGE_SPEC` default pinned to `sglang[all]==0.5.10` (or the current stable `>= 0.5.10`).
  - `SG_MODEL` default → `Qwen/Qwen3.6-35B-A3B`.
  - `SG_MEM_FRACTION_STATIC` default → `0.78`.
  - New env knobs with defaults: `SG_KV_CACHE_DTYPE=fp8_e4m3`, `SG_ENABLE_RADIX_CACHE=1`, `SG_SPECULATIVE_MODE=nextn`, `SG_SPECULATIVE_NUM_STEPS=3`, `SG_SPECULATIVE_TOPK=1`, `SG_SPECULATIVE_DRAFT_TOKENS=4`.
  - Same defaults replicated in the post-`ENV_FILE` source block.
  - Inline Python templating block extended to emit `--kv-cache-dtype` and the four `--speculative-*` flags into `optional_flags` when their env-controlled inputs are non-empty. Radix cache is the SGLang default and only needs a flag to *disable* it, so `SG_ENABLE_RADIX_CACHE=0` emits `--disable-radix-cache` and any truthy value emits nothing.
  - Trailing `echo` summary extended to print each new env so operators see the effective values.

### 4.4 Env and runbook doc sweep

All `Qwen/Qwen3.5-27B` literals replaced with `Qwen/Qwen3.6-35B-A3B` in:

- `.env.example`
- `docs/runtime/known-good.env` (including the commented reference `sglang.launch_server` command — updated to the §4.3 line)
- `docs/runtime/ENV_REFERENCE.md`
- `docs/deployment/P1_DEPLOY.md`
- `docs/runtime/RUNTIME_GUIDE.md` (if any references exist)
- `README.md` (if any references exist)

`docs/ERROR_LOG.md` is appended **only if** something regresses during canary per the §8 acceptance gate.

---

## 5. Rollout (Ordered Steps)

Each step is independently runnable; commits happen per step so rollbacks are surgical. All shell commands assume the H200 droplet with the repo at `/opt/clypt-phase1/repo` unless marked "(local)".

### Step 1 — Pre-flight (host): verify capacity and capture 27B baseline benchmarks

```bash
df -h /opt/clypt-phase1/hf-cache
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
# disk ≥ 90 GB free, VRAM 141 GB total, ≥ 100 GB free at capture time

cd /opt/clypt-phase1/repo && . .venv/bin/activate
mkdir -p tmp/qwen36_swap/baseline
for s in phase2_merge phase3_local phase3_long_range phase4_meta phase4_subgraph phase4_pool; do
  python scripts/bench_phase24_llm_concurrency.py \
    --scenario "$s" --concurrency-values "1,2,4,6,8,10" --rounds 2 \
    --base-url http://127.0.0.1:8001/v1 \
    --model "Qwen/Qwen3.5-27B" \
    --output-json "tmp/qwen36_swap/baseline/${s}.json"
done

HF_HOME=/opt/clypt-phase1/hf-cache hf download Qwen/Qwen3.6-35B-A3B
du -sh /opt/clypt-phase1/hf-cache/hub/models--Qwen--Qwen3.6-35B-A3B
```

The baseline JSONs under `tmp/qwen36_swap/baseline/` are the sole tolerated `Qwen/Qwen3.5-27B` residue and are kept in-repo by temporarily whitelisting `tmp/qwen36_swap/` in `.gitignore` (or carried as PR-attached artifacts — pick one, document in PR body).

### Step 2 — Host: bump SGLang venv to the Qwen3.6-capable version

```bash
sudo SG_MODEL="Qwen/Qwen3.6-35B-A3B" \
     SG_PACKAGE_SPEC="sglang[all]==0.5.10" \
     bash scripts/do_phase1/deploy_sglang_qwen_service.sh

/opt/clypt-phase1/venvs/sglang/bin/python -c "import sglang, sys; print(sglang.__version__)"
curl -fsS http://127.0.0.1:8001/health
```

This run still serves the old model (the systemd ExecStart literal is swapped in Step 4); we are only validating that the new SGLang installs cleanly in the dedicated venv.

### Step 3 — Repo (local): atomic code + test literal swap + sampler tightening

Edit the files listed in §4.1. Then:

```bash
python -m pytest tests/backend/providers -q
python -m pytest tests/backend/runtime -q
python -m pytest tests/backend/pipeline -q
```

All green. If a test still asserts `Qwen/Qwen3.5-27B`, fix the assertion — never weaken a production default to match a stale test.

### Step 4 — Repo (local) + host: SGLang launch line

Rewrite `scripts/do_phase1/systemd/clypt-sglang-qwen.service` per §4.3, delete `scripts/do_phase1/systemd/clypt-vllm-qwen.service`, and extend `scripts/do_phase1/deploy_sglang_qwen_service.sh` per §4.3. Then on the host:

```bash
sudo SG_MODEL="Qwen/Qwen3.6-35B-A3B" \
     SG_MEM_FRACTION_STATIC="0.78" \
     SG_KV_CACHE_DTYPE="fp8_e4m3" \
     SG_SPECULATIVE_MODE="nextn" \
     bash scripts/do_phase1/deploy_sglang_qwen_service.sh

curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool
nvidia-smi
```

Expected: `/v1/models` reports `Qwen/Qwen3.6-35B-A3B`; `nvidia-smi` shows ~72 GB weights + ~30-40 GB working set.

### Step 5 — Repo (local): env + doc sweep

Edit the files listed in §4.4, then run the §8 `rg` gate. Commit on green.

### Step 6 — Host: rebench on the new serving stack

```bash
cd /opt/clypt-phase1/repo && . .venv/bin/activate
mkdir -p tmp/qwen36_swap/qwen36
for s in phase2_merge phase3_local phase3_long_range phase4_meta phase4_subgraph phase4_pool; do
  python scripts/bench_phase24_llm_concurrency.py \
    --scenario "$s" --concurrency-values "1,2,4,6,8,10" --rounds 2 \
    --base-url http://127.0.0.1:8001/v1 \
    --model "Qwen/Qwen3.6-35B-A3B" \
    --output-json "tmp/qwen36_swap/qwen36/${s}.json"
done

python3 - <<'PY'
import json, pathlib
base = pathlib.Path("tmp/qwen36_swap/baseline")
new  = pathlib.Path("tmp/qwen36_swap/qwen36")
for path in sorted(base.glob("*.json")):
    b = json.loads(path.read_text()); n = json.loads((new / path.name).read_text())
    bb, nb = b.get("best") or {}, n.get("best") or {}
    print(f"{path.stem:24s}  rps  27B={bb.get('requests_per_s', 0):.2f}  3.6={nb.get('requests_per_s', 0):.2f}  "
          f"p95 27B={bb.get('latency_ms',{}).get('p95',0):.0f}ms  3.6={nb.get('latency_ms',{}).get('p95',0):.0f}ms  "
          f"err 27B={bb.get('error_count',-1)}  3.6={nb.get('error_count',-1)}")
PY
```

Target: ≥ 1.3× rps at zero-error concurrency on at least one of `phase4_subgraph`, `phase3_long_range`, `phase2_merge`. `phase4_pool` may show < 1.1× due to short-output path; that is acceptable.

Raise Phase 2-4 concurrency caps only where the data is unambiguous; update `docs/runtime/known-good.env` and `docs/deployment/P1_DEPLOY.md` accordingly.

### Step 7 — Host: end-to-end pipeline eval (the ±2% gate)

```bash
python -m pytest tests/backend/pipeline -q
python -m pytest tests/backend/pipeline/test_subgraph_review_schema_compat.py -q

python -m backend.runtime.run_phase1 \
  --job-id "qwen36_eval_$(date +%Y%m%d_%H%M%S)" \
  --source-path /opt/clypt-phase1/videos/<eval-video>.mp4 \
  --run-phase14
```

Compare Phase 4 candidate set against the 27B baseline per `docs/EVALS.md`. Accuracy must not drop > 2%. If it drops more, §6 R5 applies (revert Step 3's sampler change in isolation and re-run).

Log regressions in `docs/ERROR_LOG.md` per `AGENTS.md` template. Do not log a green run.

---

## 6. Risks and Mitigations

### R1. `xgrammar` schema-compile crash on the new model
**Signal:** `test_subgraph_review_schema_compat.py` fails, or `clypt-sglang-qwen.service` logs `xgrammar` / `compile_json_schema` / `EngineCore` errors.
**Mitigation:** Retry the deploy first (the 2026-04-15 `ERROR_LOG` entry shows `xgrammar` is sensitive to schema shape, not model identity). If it reproduces, re-deploy with `SG_GRAMMAR_BACKEND=llguidance` and rerun §5 Step 6. Do not mix the grammar-backend swap with the initial model swap — run as a separate follow-up to preserve attribution.

### R2. NextN MTP draft heads cause instability or first-token slowdown
**Signal:** p50 @ concurrency=1 worse than 27B baseline; MTP-related log lines.
**Mitigation:** Redeploy with `SG_SPECULATIVE_MODE=off`; the launch line will omit the four `--speculative-*` flags. Isolates MTP from other variables.

### R3. FP8 KV cache numerical instability on H200
**Signal:** garbage tokens, NaN loss, schema validation failures unrelated to grammar backend.
**Mitigation:** Redeploy with `SG_KV_CACHE_DTYPE=auto` (KV reverts to BF16). Step `SG_MEM_FRACTION_STATIC` down to 0.65 to absorb the doubled KV cost.

### R4. `mem-fraction-static=0.78` OOMs at peak concurrency
**Signal:** SGLang logs `CUDA out of memory` in steady state, not at boot.
**Mitigation:** Step down to 0.72 then 0.65, redeploying each time. Trust measurement over the model card's 0.8 recommendation.

### R5. Sampler tightening causes Phase 4 quality regression
**Signal:** §5 Step 7 shows > 2% accuracy drop.
**Mitigation:** Revert Step 3's sampler change only (keep model + serving). Re-run Step 7 to isolate. Most likely cause is `temperature=0.0` collapsing diversity on Phase 4 meta-prompt generation — an intermediate `temperature=0.3` for that scenario (per-call override in the Phase 4 signal config) is the next try.

### R6. Documentation drift
**Signal:** A future agent finds `Qwen/Qwen3.5-27B` in an active file.
**Mitigation:** The §8 `rg` gate. Any match outside the historical exclusions fails review.

### R7. Operator boots a host directly from the committed systemd unit without running the deploy script
**Signal:** SGLang starts with the old ExecStart.
**Mitigation:** The §4.3 rewrite ensures the committed unit is correct. The `clypt-vllm-qwen.service` deletion in §4.2 prevents accidental start of the legacy 27B vLLM path. Per §0, there is no configuration under which an un-deployed host serves 27B.

---

## 7. Optional follow-up: `llguidance` A/B

Isolated from the initial swap to preserve attribution. Only run after §5 Step 7 passes.

```bash
sudo SG_GRAMMAR_BACKEND=llguidance bash scripts/do_phase1/deploy_sglang_qwen_service.sh
# rerun §5 Step 6 with output dir tmp/qwen36_swap/qwen36_llguidance/
```

Switch the deploy-script default from `xgrammar` to `llguidance` only if it shows ≥ 10% rps gain at the same error rate **and** zero schema-compile crashes across all six scenarios.

---

## 8. Acceptance Criteria

This swap is complete when **all** of the following hold:

1. `Qwen/Qwen3.6-35B-A3B` is the only model id referenced by active code, env, systemd, deploy scripts, and runbook docs.
2. `scripts/do_phase1/systemd/clypt-vllm-qwen.service` no longer exists in the repo.
3. SGLang `>= 0.5.10` is pinned in `deploy_sglang_qwen_service.sh`.
4. The systemd ExecStart on H200 includes `--kv-cache-dtype fp8_e4m3`, `--mamba-scheduler-strategy extra_buffer`, and the four `--speculative-*` flags for NextN MTP. The unit exports `Environment=SGLANG_ENABLE_SPEC_V2=1`. Radix cache is inherited from SGLang's default-on behavior; absence of `--disable-radix-cache` is the correct acceptance signal.
5. `SG_MEM_FRACTION_STATIC` is at least `0.72` and validated under load.
6. Production `LocalGenerationSettings` defaults are `temperature=0.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
7. `tmp/qwen36_swap/qwen36/` shows non-regression on all six scenarios and ≥ 1.3× rps on at least one of `phase4_subgraph`, `phase3_long_range`, `phase2_merge`.
8. `pytest tests/backend/{pipeline,runtime,providers}` is green.
9. End-to-end Phase 1-4 accuracy on the eval video bank is within 2% of the 27B baseline (per `docs/EVALS.md`).
10. This spec is registered in `docs/specs/SPEC_INDEX.md` under Active, and `2026-04-15_qwen_sglang_full_cutover_spec.md` is moved to Superseded.
11. The drift gate passes with zero matches:

    ```bash
    rg 'Qwen/Qwen3\.5-27B' \
      --glob '!docs/specs/2026-04-1[45]_*.md' \
      --glob '!docs/specs/2026-04-16_qwen36_swap_and_sglang_tuning_spec.md' \
      --glob '!tmp/qwen36_swap/baseline/**' \
      --glob '!.gitnexus/**'
    ```

    (This spec itself is excluded because §0 and §2 narrate the old model id intentionally. `.gitnexus/` is excluded because it holds the local knowledge-graph binary index which is gitignored and regenerated by `npx gitnexus analyze`.)

---

## 9. Out of Scope

- VibeVoice ASR (remains on the Cloud Run L4 combined service or local vLLM per `CLYPT_PHASE1_ASR_BACKEND`).
- Vertex embeddings, RF-DETR visual, Phase 1 ASR chain.
- Phase 4 candidate-scoring semantics / prompt rewrites.
- Multi-GPU tensor parallelism (single H200 stays single-GPU).
- Agentic multi-turn `preserve_thinking` use (reserved for a future spec; our flows are single-turn JSON extraction).
- Migration of `2026-04-10_phase5_6_spec.md` to the new model id (that spec is planned, not implemented; it will naturally pick up §4.1's defaults).
