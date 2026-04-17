# ERROR LOG

Persistent record of major runtime/deployment/pipeline errors and their recoveries.

> **2026-04-17 note:** Entries below that reference GCE L4 / Cloud Run L4 combined
> service, the VibeVoice `bfloat16` Dockerfile patch, and Cloud Tasks dispatch
> describe infrastructure that has since been torn down. The code paths,
> deploy scripts, and Docker images referenced by those incidents no longer
> exist in this repository; the entries are retained for historical context
> only. Current Phase 1 ASR and node-media prep are both single-host.

## Entry Template

- **Date/Time (UTC):**
- **Subsystem:**
- **Environment:**
- **Symptom / Error signature:**
- **Root cause:**
- **Fix applied:**
- **Verification evidence:**
- **Follow-up guardrails:**

---

## 2026-04-16 - Cloud Run L4 VibeVoice OOM during vLLM profile_run; cut over to GCE L4 with bf16 patch

- **Date/Time (UTC):** 2026-04-16 (diagnosis and pivot across the day)
- **Subsystem:** Phase 1 ASR / Phase 2 node-media prep combined L4 service
- **Environment:** Cloud Run L4 (`clypt-phase1-l4-combined`) → GCE `g2-standard-8` L4 VM (`clypt-phase1-l4-gce`, `us-central1-a`, external IP `34.59.190.134`)
- **Symptom / Error signature:** Container started vLLM, passed warmup, then crashed mid-`profile_run` with CUDA OOM. Cold-start also took 15-20 min because the ~12 GB model was re-downloaded on every revision.
- **Root cause:**
  1. VibeVoice's `vllm_plugin/model.py` defaults `_audio_encoder_dtype = torch.float32` "for numerical precision" even though the HF checkpoint ships every sub-module in `bfloat16`. The fp32 upcast inflated the model from ~10 GB to ~18 GB on a 24 GB L4, starving vLLM's KV-cache sizing during `profile_run`.
  2. Cloud Run ephemeral filesystem forced a full HF model re-download on every cold start; no place to persist `/root/.cache/huggingface`.
  3. Cloud Run L4 operationally hides GPU state (no `nvidia-smi` access, no persistent FS) making iterative debugging of (1) impractical.
- **Fix applied:**
  1. `docker/phase24-media-prep/Dockerfile` now (a) sets `HF_HOME=/root/.cache/huggingface` + `HF_HUB_ENABLE_HF_TRANSFER=1`, (b) installs `hf_transfer`, (c) bakes `microsoft/VibeVoice-ASR` into the image via `snapshot_download`, and (d) sed-patches `vllm_plugin/model.py` to force `_audio_encoder_dtype = torch.bfloat16` (grep-guarded so the build fails if upstream reshapes that line). The sed step is placed late so the 12 GB model-bake layer stays cached across rebuilds.
  2. `backend/runtime/l4_combined_bootstrap.py` retuned defaults for L4: `max_num_seqs=4`, `max_model_len=16384`, `gpu_memory_utilization=0.90`, `VIBEVOICE_FFMPEG_MAX_CONCURRENCY=16`, startup health-wait `1500 s`, `--skip-deps`.
  3. Cut over the deploy target from Cloud Run L4 to a GCE `g2-standard-8` L4 VM via new `scripts/deploy_l4_gce.sh` (firewall-gated to droplet egress IP, `AUTH_MODE=none`, persistent host cache at `/var/clypt/hf-cache`, multi-zone probing to survive transient L4 stockouts). Requested and obtained `GPUS_ALL_REGIONS=1` via `gcloud alpha quotas preferences create`.
  4. Left Cloud Run deploy script in the repo but marked it deprecated in docs.
- **Verification evidence:** Build succeeded with bf16-patched layer (`gce-bf16-20260416-181315`). GCE VM came up after multi-zone retry; droplet-scoped firewall rule `clypt-l4-combined-ingress` active on tcp:8080. Container launched with tuned CLI args visible in `docker inspect`; the baked image shipped `_audio_encoder_dtype = torch.bfloat16` in `/app/vllm_plugin/model.py`. Cold-start on fresh VM re-downloaded once (bind mount overlays baked layer) in ~3 min with `hf_transfer`, then cached to host disk for subsequent restarts.
- **Follow-up guardrails:**
  - Keep the sed grep-guard in the Dockerfile; the build must fail if the upstream model.py line changes shape.
  - Keep `scripts/deploy_l4_gce.sh` as the canonical L4 deploy path. Do not resurrect the Cloud Run path unless the container can be proven to fit under 24 GB with the bf16 patch.
  - Treat `GPUS_ALL_REGIONS=0` as a hard prerequisite to fix before first deploy; document it in P1_DEPLOY.md §3.4.1.
  - The bind mount overlays the baked-in HF cache layer, so first boot on a fresh VM always re-downloads; treat this as expected and rely on the `/var/clypt/hf-cache` host volume for persistence across restarts.

## 2026-04-17 - L4 Dockerfile bf16 sed guard undercounted bfloat16 assignments, failing the build

- **Date/Time (UTC):** 2026-04-17 ~03:20
- **Subsystem:** L4 combined service Docker build (`docker/phase24-media-prep/Dockerfile`, bf16 patch step)
- **Environment:** Cloud Build targeting `us-east4-docker.pkg.dev/clypt-v3/cloud-run-source-deploy/clypt-phase1-l4-combined` — builds `gce-bf16-20260416-194909` and `gce-bf16-20260416-201857`.
- **Symptom / Error signature:** Both builds failed at the bf16 patch step with `returned a non-zero code: 1`. The failing chain was the final `[ $(grep -c 'self._audio_encoder_dtype = torch.bfloat16' ...) -ge 3 ]` count check inside the `RUN grep ... && sed ... && ! grep ... && [ ... -ge 3 ]` guard.
- **Root cause:** Upstream `vllm_plugin/model.py` resolves `self._audio_encoder_dtype` via a **three-way** branch:
  ```python
  root_torch_dtype = get_cfg(config, "torch_dtype", None)
  if root_torch_dtype is not None:
      if isinstance(root_torch_dtype, str):
          self._audio_encoder_dtype = getattr(torch, root_torch_dtype)   # (A)
      else:
          self._audio_encoder_dtype = root_torch_dtype                   # (B)
  else:
      self._audio_encoder_dtype = torch.float32                          # (C)
  ```
  The previous guard only `sed`-rewrote (A) and (B) and commented that (C) was already `torch.bfloat16`. But upstream's (C) is `torch.float32`, so post-patch the file contained only **two** `torch.bfloat16` assignments while the guard asserted `-ge 3`. Separately, leaving (C) as `torch.float32` was also a latent correctness bug: any future HF checkpoint that drops `torch_dtype` from its config would fall through the `else:` branch and re-trigger the original fp32 OOM.
- **Fix applied:**
  - Extended the `sed` step in `docker/phase24-media-prep/Dockerfile` with a third `-e 's|...= torch.float32|...= torch.bfloat16|'` expression so all three assignments are pinned to bfloat16.
  - Added a pre-`sed` `grep -q 'self\._audio_encoder_dtype = torch\.float32'` guard so the build fails loudly if upstream ever drops branch (C).
  - Added a post-`sed` `! grep -q` anti-assertion for branch (C) and tightened the final count check from `-ge 3` to `-eq 3` so over-substitution is caught too.
  - Validated the rewritten guard locally against `vllm_plugin/model.py@main` before rebuilding (all three pre-grep checks pass, all three anti-greps pass, final count is exactly 3).
- **Verification evidence:** Local dry-run of the new `sed` chain against the upstream file produces the expected block shape (three `self._audio_encoder_dtype = torch.bfloat16  # Clypt: forced bf16 ...` lines in the if/else ladder) and the final `-eq 3` check succeeds.
- **Follow-up guardrails:**
  - Count check is now `-eq 3`, not `-ge 3` — if upstream adds a fourth assignment site the build fails, forcing us to re-read the dtype resolver before shipping.
  - Keep all three pre-grep and all three anti-grep guards; they collectively pin the shape of the upstream branch.

## 2026-04-17 - VibeVoice embed_multimodal returned [] during vLLM v1 profile_run; crash-loop on sanity_check_mm_encoder_outputs

- **Date/Time (UTC):** 2026-04-17 03:10
- **Subsystem:** Phase 1 ASR / Phase 2 node-media prep combined L4 service (VibeVoice vLLM plugin)
- **Environment:** GCE `clypt-phase1-l4-gce` (`us-central1-a`, external IP `34.59.75.53`), image `clypt-phase1-l4-combined:gce-bf16-20260416-185650` (bf16 patch landed correctly, `/app/vllm_plugin/model.py` confirmed `_audio_encoder_dtype = torch.bfloat16`).
- **Symptom / Error signature:** With OOM fixed by the bf16 patch, the container progressed past model load and `determine_available_memory` but crashed inside `profile_run`:
  ```
  File ".../vllm/v1/worker/utils.py", line 192, in sanity_check_mm_encoder_outputs
    assert len(mm_embeddings) == expected_num_items, (
  AssertionError: Expected number of multimodal embeddings to match number of input
  items: 1, but got len(mm_embeddings)=0 instead. This is most likely due to
  incorrect implementation of the model's `embed_multimodal` method.
  RuntimeError: Engine core initialization failed.
  ```
  Docker restart-policy `always` drove a crash loop (`restartCount=1`, uptime ~3 min/cycle). GPU held 0 MiB used on subsequent cycles because the crash aborts boot before weights reload.
- **Root cause:** VibeVoice `vllm_plugin/model.py:988` `embed_multimodal` short-circuits with `return []` when `raw_audio is None` or empty, with an inline comment "this happens during memory profiling". vLLM v1's profile_run (present in 0.14.1+, the officially recommended version per `docs/vibevoice-vllm-asr.md`) builds **one** synthetic multimodal input and then asserts that `embed_multimodal` returns exactly one embedding per input item. Upstream's plugin was written against an older vLLM that tolerated zero-length returns during profile. No upstream fix exists: VibeVoice repo `main` at `4a78d3e` still has the `return []` branch; the only adjacent PR (#291) addresses vLLM 0.16+ processor-API compat, not this assertion, and is still open/unmerged.
- **Fix applied:** Added `docker/phase24-media-prep/patches/vibevoice_profile_run.py`, a Python patcher invoked from the Dockerfile after the existing bf16 `sed` step. It rewrites the two `return []` branches to synthesize one second of silence (`torch.zeros(sample_rate, dtype=self._audio_encoder_dtype, device=encoder_device)`) and fall through to the normal encoder path, so the encoder emits one correctly-shaped embedding and vLLM also gets an honest profile-run memory read. The patcher is idempotent (sentinel-guarded) and fails the Docker build with exit 2 if upstream reshapes the target block. The Dockerfile step additionally greps the sentinel post-patch and runs `py_compile` on the rewritten `model.py`.
- **Verification evidence:** Patcher dry-run against the live container's `model.py` produced clean output, `py_compile` passed, idempotent re-run printed "already applied", drift simulation (stripped sentinel) failed loudly as designed. Pending: rebuilt image tag `gce-bf16-20260417-03*` deployed to `clypt-phase1-l4-gce`, container reaches `/health` 200, and ASR round-trip from droplet succeeds.
- **Follow-up guardrails:**
  - Dockerfile must keep both the bf16 `sed` patch and the `vibevoice_profile_run.py` patch. Neither alone is sufficient for a functional 24 GB L4 deploy.
  - When upstream ships a real fix (track PRs #291, #223 and any successors that touch `embed_multimodal` profile-run semantics), retire the Python patcher but keep the sentinel to detect rebase conflicts.
  - Any future plugin-source patch MUST go through a sentinel-guarded, idempotent Python patcher (not sed) if it spans multiple lines of logic — blind sed on this surface already cost us one silent-no-op cycle (see previous entry).

## 2026-04-17 - VibeVoice bf16 Dockerfile patch was a no-op; model.py sets audio_encoder_dtype from config.torch_dtype

- **Date/Time (UTC):** 2026-04-17 02:46
- **Subsystem:** Phase 1 ASR / Phase 2 node-media prep combined L4 service (VibeVoice vLLM plugin)
- **Environment:** GCE `clypt-phase1-l4-gce` (`us-central1-a`, external IP `34.59.75.53`), image `clypt-phase1-l4-combined:gce-bf16-20260416-185650`.
- **Symptom / Error signature:** Container warmed fine through model load (~18.22 GiB reported by vLLM), then died during `profile_run`:
  ```
  [VibeVoice] Converted acoustic_tokenizer to torch.float32 (was torch.bfloat16)
  [VibeVoice] Converted semantic_tokenizer to torch.float32 (was torch.bfloat16)
  [VibeVoice] Converted acoustic_connector to torch.float32 (was torch.bfloat16)
  [VibeVoice] Converted semantic_connector to torch.float32 (was torch.bfloat16)
  ...
  [VibeVoice] Error encoding audio 0: CUDA out of memory. Tried to allocate 704.00 MiB.
  GPU 0 has a total capacity of 22.03 GiB of which 555.12 MiB is free.
  ...
  AssertionError: Expected number of multimodal embeddings to match number of input
  items: 1, but got len(mm_embeddings)=0
  RuntimeError: Engine core initialization failed.
  ```
  Docker-restart loop, port 8080 never listened.
- **Root cause:** The bf16 Dockerfile patch shipped in the previous ERROR_LOG entry (`sed 's|self._audio_encoder_dtype = torch.float32|... = torch.bfloat16|'`) did not match anything in upstream `vllm_plugin/model.py`. The real code reads the dtype from the HF checkpoint's `config.torch_dtype` (= `float32` for VibeVoice-ASR):
  ```python
  root_torch_dtype = get_cfg(config, "torch_dtype", None)
  if root_torch_dtype is not None:
      if isinstance(root_torch_dtype, str):
          self._audio_encoder_dtype = getattr(torch, root_torch_dtype)
      else:
          self._audio_encoder_dtype = root_torch_dtype
  else:
      self._audio_encoder_dtype = torch.bfloat16
  ```
  The `grep -q 'self._audio_encoder_dtype = torch.bfloat16'` guard passed against the `else:` branch, so the build reported success even though the `if` branches kept producing `torch.float32`. At runtime the audio encoder and the four submodules that follow it (`acoustic_tokenizer`, `semantic_tokenizer`, `acoustic_connector`, `semantic_connector`) all upcast to fp32, bloating the model to 18.22 GiB and leaving only 555 MiB free. `profile_run` then OOM'd on the 704 MiB audio-encoder forward pass.
- **Fix applied:** Rewrote the Dockerfile patch in `docker/phase24-media-prep/Dockerfile` to:
  1. `grep -q` the two config-driven assignments (`self._audio_encoder_dtype = getattr(torch, root_torch_dtype)` and `self._audio_encoder_dtype = root_torch_dtype`) to fail the build if upstream reshapes them.
  2. `sed` both lines to `self._audio_encoder_dtype = torch.bfloat16  # Clypt: forced bf16 to fit on 24 GB L4`.
  3. Negative-grep to prove the original strings no longer exist.
  4. Count that `self._audio_encoder_dtype = torch.bfloat16` appears at least 3 times post-patch (the two rewritten branches + the pre-existing `else:`).
  The `_ensure_audio_encoder_dtype` helper then converts tokenizers/connectors bf16→bf16 (a no-op), so all five "Converted ... to torch.float32" log lines should disappear.
- **Verification evidence:** Pending: new image tag `gce-bf16-20260417-0*` (writing to `/tmp/new_image_tag.txt` in the dev shell). Will re-pull on `clypt-phase1-l4-gce`, restart container, and confirm `Model loading took <12 GiB memory` + port 8080 green.
- **Follow-up guardrails:**
  - Any sed-based patch in this Dockerfile must include a negative-grep that proves the original string is gone, not just that the target string is present.
  - When adding future guards, assume upstream will split a hot path across multiple conditional branches. Match against every branch, not the default/fallback.

## 2026-04-17 - GCE startup-script failed on nvidia-container-toolkit version skew

- **Date/Time (UTC):** 2026-04-17 02:28
- **Subsystem:** GCE VM provisioning (`scripts/deploy_l4_gce.sh` startup-script)
- **Environment:** GCE `clypt-phase1-l4-gce` (`us-central1-a`, external IP `34.59.75.53`), Deep Learning VM image family `common-cu129-ubuntu-2204-nvidia-580`.
- **Symptom / Error signature:** Startup script failed under `set -e` at `apt-get install -y nvidia-container-toolkit` with:
  ```
  The following packages have unmet dependencies:
   nvidia-container-toolkit : Depends: nvidia-container-toolkit-base (= 1.19.0-1) but 1.17.8-1 is to be installed
                              Depends: libnvidia-container-tools (= 1.19.0-1) but 1.17.8-1 is to be installed
  E: Unable to correct problems, you have held broken packages.
  ```
  `docker run --gpus all` then failed because `nvidia-ctk runtime configure` was never executed and `docker info` showed no `nvidia` runtime.
- **Root cause:** The `common-cu129-ubuntu-2204-nvidia-580` DLVM image ships `nvidia-container-toolkit`, `-base`, `libnvidia-container-tools`, and `libnvidia-container1` preinstalled and held at `1.17.8-1`. The startup script unconditionally added the upstream `https://nvidia.github.io/libnvidia-container/stable/deb` repo on top, which offered `nvidia-container-toolkit=1.19.0-1` as the top-level but kept the `-base`/`-tools` held at `1.17.8-1`, producing an unmet-dependency hard-fail. Net effect: the DLVM already had a working toolkit and only needed `nvidia-ctk runtime configure --runtime=docker` + `systemctl restart docker`, but the script forced an apt install anyway.
- **Fix applied:**
  1. Manually SSH'd in, removed `/etc/apt/sources.list.d/nvidia-container-toolkit.list`, ran `nvidia-ctk runtime configure --runtime=docker` against the preinstalled 1.17.8 toolkit, restarted docker, then continued with `gcloud auth configure-docker` / `docker pull` / `docker run`. Container came up cleanly with the NVIDIA runtime wired.
  2. Long-term fix in `scripts/deploy_l4_gce.sh`: the startup script now checks `command -v nvidia-ctk` first. If the toolkit is already present (which it is on every `common-cu*` DLVM image we use), the script skips the repo-add + `apt-get install` entirely and only runs `nvidia-ctk runtime configure` + `systemctl restart docker`. The upstream apt-repo path is retained as a fallback for bare images.
- **Verification evidence:** After the manual fix `docker info` reported the `nvidia` runtime; `docker pull` succeeded against the Artifact Registry image; `docker ps` showed `clypt-l4-combined` with `--gpus all`; vLLM began its in-container dep install (`librosa`, `scipy`, `vibevoice` editable build) and is currently warming.
- **Follow-up guardrails:**
  - Never unconditionally add upstream NVIDIA apt sources on DLVM images. Always gate on `command -v nvidia-ctk`.
  - Any future version bump of the NVIDIA toolkit on DLVM must go through an explicit `apt-get install -y nvidia-container-toolkit=<VER> nvidia-container-toolkit-base=<VER> libnvidia-container-tools=<VER> libnvidia-container1=<VER>` with a matched quadruplet, not a plain top-level install.

## 2026-04-16 - GCE startup-script failed on NVIDIA toolkit install (gpg TTY)

- **Date/Time (UTC):** 2026-04-16
- **Subsystem:** GCE VM provisioning (`scripts/deploy_l4_gce.sh` startup-script)
- **Environment:** GCE `clypt-phase1-l4-gce` first-boot startup script
- **Symptom / Error signature:** `gpg: cannot open '/dev/tty': No such device or address` and `curl: (23) Failed writing body`; startup script exited before installing `nvidia-container-toolkit`, so `docker run --gpus all` failed.
- **Root cause:** `gpg --dearmor` expects a controlling TTY by default, which GCE startup scripts don't have. The keyring fetch pipeline failed silently and the script exited on `set -e`.
- **Fix applied:** Manually SSH'd to the VM and re-ran the NVIDIA keyring import with `gpg --batch --yes --dearmor`, then completed Docker authentication, image pull, and `docker run` for the combined service. Long-term fix is to propagate the `--batch --yes` flags into the embedded startup script in `scripts/deploy_l4_gce.sh`.
- **Verification evidence:** After the manual fix, `sudo docker ps` showed `clypt-l4-combined` running with `--gpus all`; subsequent `nvidia-smi` inside the container reported the L4 GPU.
- **Follow-up guardrails:** All future `gpg --dearmor` invocations in non-interactive startup scripts must include `--batch --yes`. The `scripts/deploy_l4_gce.sh` embedded startup script now enforces that.

## 2026-04-15 - TensorRT detector load failed on binding shape type mismatch

- **Date/Time (UTC):** 2026-04-15 19:07-19:23
- **Subsystem:** Phase 1 visual extraction (`backend/phase1_runtime/tensorrt_detector.py`)
- **Environment:** DO H200 droplet with `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- **Symptom / Error signature:** Phase 1 visual extraction failed during detector startup with `TypeError: empty() received an invalid combination of arguments - got (tensorrt_bindings.tensorrt.Dims, device=str, dtype=torch.dtype)`.
- **Root cause:** TensorRT binding shapes were passed through as `tensorrt.Dims` objects and then handed directly to `torch.empty(...)`, which expects a Python tuple of ints. The engine loaded successfully, but buffer allocation crashed before any frame inference started.
- **Fix applied:** Normalized TensorRT binding shapes to tuples of Python ints before buffer allocation, refreshed the detector regression tests, and verified the focused TensorRT regression now passes.
- **Verification evidence:** New focused regression test for `_allocate_buffers()` passes locally; subsequent rerun on the droplet advanced past TensorRT engine discovery and into audio-side completion/Phase 2 queue handoff instead of dying during detector load.
- **Follow-up guardrails:** Keep a focused regression around TensorRT binding-shape normalization so engine/runtime library changes do not reintroduce `Dims`-vs-tuple buffer bugs.

## 2026-04-15 - TensorRT visual path was bottlenecked by host-side resize and preprocessing

- **Date/Time (UTC):** 2026-04-15 21:11-21:14
- **Subsystem:** Phase 1 visual extraction (`backend/phase1_runtime/frame_decode.py`, `backend/phase1_runtime/tensorrt_detector.py`, `backend/phase1_runtime/visual.py`)
- **Environment:** DO H200 droplet with `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- **Symptom / Error signature:** The visual-only RF-DETR pass ran cleanly but plateaued around `51.5 fps` even after the TensorRT correctness fixes, indicating the detector was no longer the primary bottleneck.
- **Root cause:** The runtime was still decoding full-resolution `1920x1080` RGB frames to host memory, resizing them on CPU with OpenCV, and only then moving normalized tensors onto CUDA. That left substantial host memory bandwidth, CPU resize, and host-to-device transfer overhead in the hot path.
- **Fix applied:** Moved the fast path to GPU-first preprocessing: `decode_video_frames()` now optionally applies `scale_cuda` directly to detector resolution before `hwdownload`, `TensorRTDetector._preprocess_batch()` now converts, resizes, and normalizes on CUDA with `torch`, and the visual pipeline now preserves original source dimensions separately so postprocess box rescaling remains correct after decode-time downscaling.
- **Verification evidence:** Targeted droplet regressions passed for resized GPU decode, explicit original-size rescaling, and CUDA preprocess behavior; a fresh visual-only droplet replay improved from about `51.5 fps` to `240.1 fps` on the same Billy Carson reference video (`35705` frames, `148678.5 ms` pipeline elapsed, `568.0 ms` warmup).
- **Follow-up guardrails:** If TensorRT visual throughput regresses back toward `~50 fps` on the H200 reference workload, verify the synced runtime still uses `scale_cuda` during decode, CUDA-side preprocess in `TensorRTDetector`, and preserved source dimensions for postprocess scaling.

## 2026-04-15 - H200 canonical-audio URL signing failed with user ADC credentials

- **Date/Time (UTC):** 2026-04-15 18:58-19:05
- **Subsystem:** Phase 1 ASR canonical audio URL resolution
- **Environment:** DO H200 droplet, local VibeVoice vLLM service
- **Symptom / Error signature:** Phase 1 failed with `[vibevoice-vllm] failed to sign canonical audio_gcs_uri ... Cannot get legacy ACL for an object when uniform bucket-level access is enabled`.
- **Root cause:** `/opt/clypt-phase1/sa-key.json` contained `authorized_user` ADC credentials, which cannot generate signed URLs. The fallback `blob.make_public()` path also failed because the bucket uses uniform bucket-level access.
- **Fix applied:** Replaced the droplet credential with a real service account key for `clypt-phase1-worker@clypt-v3.iam.gserviceaccount.com`; verified the runtime now resolves `google.oauth2.service_account.Credentials`.
- **Verification evidence:** Host credential probe switched from `google.oauth2.credentials.Credentials` to `google.oauth2.service_account.Credentials`; rerun successfully generated a signed URL for `gs://clypt-storage-v3/test-bank/canonical/audio/billycarsonflagrant.wav`.
- **Follow-up guardrails:** Keep `/opt/clypt-phase1/sa-key.json` as a signing-capable service account key, not a copied user ADC file.

## 2026-04-15 - H200 local node-media clip encoding failed because the GPU has no NVENC

- **Date/Time (UTC):** 2026-04-15 19:13-19:16
- **Subsystem:** Phase 2 node media clip preparation
- **Environment:** DO H200 droplet, local Phase 2-4 worker
- **Symptom / Error signature:** `ffmpeg ... -c:v h264_nvenc ...` failed with `OpenEncodeSessionEx failed: unsupported device (2)`.
- **Root cause:** NVIDIA H200 exposes NVDEC but no NVENC hardware encoder. The worker attempted local GPU video encode on a compute-only GPU class.
- **Fix applied:** Added optional direct-HTTP offload to a dedicated Cloud Run L4 media-prep service (`cloud_run_l4` backend) so H200-hosted Phase 2-4 runs can prepare/upload node clips on an NVENC-capable GPU.
- **Verification evidence:** Direct host reproduction of the failing ffmpeg command produced `unsupported device (2)`; current NVIDIA support matrix confirms H200 has 0 NVENC while L4 has NVENC support; repo now includes client, service, Dockerfile, deploy script, and config for Cloud Run L4 media prep.
- **Follow-up guardrails:** Do not target `h264_nvenc` locally on H200/H100-class hosts; use `CLYPT_PHASE24_MEDIA_PREP_BACKEND=cloud_run_l4` with an NVENC-capable L4 service.

## 2026-04-15 - Shared inference venv drift between Phase 1 and SGLang

- **Date/Time (UTC):** 2026-04-15 06:20-06:45
- **Subsystem:** Host runtime packaging / service isolation
- **Environment:** DO H200 droplet, `deploy_vllm_service.sh` + `deploy_sglang_qwen_service.sh`
- **Symptom / Error signature:** After SGLang install, the shared repo `.venv` had different `torch`, `torchvision`, `torchaudio`, and `transformers` packages than the previously working Phase 1 runtime.
- **Root cause:** Both Phase 1 services and the SGLang Qwen service were installing into the same Python environment, so serving-side package changes mutated the runtime used by Phase 1 and the Phase 2-4 local worker.
- **Fix applied:** Split host deployment into dedicated envs: `/opt/clypt-phase1/venvs/phase1` for Phase 1 + local worker and `/opt/clypt-phase1/venvs/sglang` for `clypt-sglang-qwen.service`; updated systemd units, deploy scripts, and runbooks accordingly.
- **Verification evidence:** Post-fix systemd units reference distinct interpreters; `deploy_vllm_service.sh` and `deploy_sglang_qwen_service.sh` now create separate envs by default; host verification confirms separate Python paths for Phase 1 and SGLang services.
- **Follow-up guardrails:** Do not run SGLang installs inside the Phase 1 env again; keep service units pinned to the dedicated env paths.

## 2026-04-15 - SGLang startup failed on fresh host because `ninja` was missing

- **Date/Time (UTC):** 2026-04-15 06:29
- **Subsystem:** Qwen serving bootstrap (`clypt-sglang-qwen.service`)
- **Environment:** Fresh DO H200 droplet
- **Symptom / Error signature:** SGLang scheduler crashed during JIT kernel compilation with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.
- **Root cause:** Fresh host bootstrap and SGLang deploy automation did not install `ninja-build`, which SGLang requires to compile kernels at startup.
- **Fix applied:** Added `ninja-build` to droplet bootstrap and to `deploy_sglang_qwen_service.sh` host prerequisites.
- **Verification evidence:** After installing `ninja-build`, `clypt-sglang-qwen.service` reached healthy `/health` and `/v1/models` responses on the droplet.
- **Follow-up guardrails:** Treat `ninja-build` as a mandatory SGLang host prerequisite in bootstrap and deploy automation.

## 2026-04-15 - TensorRT runtime missing on intended TensorRT Phase 1 host

- **Date/Time (UTC):** 2026-04-15 06:35-06:42
- **Subsystem:** Phase 1 visual extraction / TensorRT bring-up
- **Environment:** DO H200 droplet with `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`
- **Symptom / Error signature:** TensorRT path could not initialize because the Python `tensorrt` module and host `trtexec` binary were absent.
- **Root cause:** The env was switched to the TensorRT backend, but deploy automation did not install the host/runtime dependencies required by `backend/phase1_runtime/tensorrt_detector.py`.
- **Fix applied:** Updated `deploy_vllm_service.sh` to install `libnvinfer-bin` and `tensorrt-cu13` automatically whenever `CLYPT_PHASE1_VISUAL_BACKEND` selects a TensorRT backend; updated env/docs to pin the known-good host to `tensorrt_fp16`.
- **Verification evidence:** Host verification confirmed `trtexec` is present and the Phase 1 env can import `tensorrt`; the working env baseline now records the TensorRT backend and engine dir explicitly.
- **Follow-up guardrails:** Keep TensorRT dependency installation tied to the deploy script instead of relying on manual host fixes.

## 2026-04-08 - vLLM model ID mismatch

- **Subsystem:** Phase 1 ASR
- **Environment:** Droplet + vLLM service
- **Symptom / Error signature:** HTTP 404 when ASR requests used model ID `microsoft/VibeVoice-ASR`.
- **Root cause:** vLLM served model name is `vibevoice` (`--served-model-name`), not HF repo ID.
- **Fix applied:** Standardized env/docs on `VIBEVOICE_VLLM_MODEL=vibevoice`.
- **Verification evidence:** `/v1/models` includes `id: vibevoice`; ASR smoke tests complete.
- **Follow-up guardrails:** Keep explicit model-id checks in startup/runbook.

## 2026-04-08 - ASR/audio-chain callback delayed behind visual completion

- **Subsystem:** Phase 1 orchestration
- **Environment:** `backend/phase1_runtime/extract.py`
- **Symptom / Error signature:** Audio sidecars started late, only after RF-DETR completion.
- **Root cause:** Callback scheduling depended on an `as_completed` pattern that effectively waited on both futures.
- **Fix applied:** Start audio chain immediately after `asr_future.result()`.
- **Verification evidence:** Audio artifacts become available significantly before visual completion.
- **Follow-up guardrails:** Preserve this concurrency invariant in runtime docs/tests.

## 2026-04-08 - GPU ffmpeg unavailable on queue worker

- **Subsystem:** Phase 2 node media extraction
- **Environment:** Cloud Run Phase 2-4 worker
- **Symptom / Error signature:** `GPU ffmpeg unavailable ... falling back to CPU encoder`.
- **Root cause:** Worker runtime lacked usable GPU ffmpeg path.
- **Fix applied:** Provision tuned worker profile on `us-east4` L4 with GPU ffmpeg configuration.
- **Verification evidence:** Tuned replay reduced Phase 2-4 duration from ~13m41s to ~2m24s on reference clip.
- **Follow-up guardrails:** Keep ffmpeg device/runtime checks in worker startup and runbook.

## 2026-04-09 - Phase 3 long-range strict validation failure

- **Subsystem:** Phase 3 graph long-range edges
- **Environment:** Queue worker runs
- **Symptom / Error signature:** `Qwen returned an edge for a non-shortlisted candidate long-range pair`.
- **Root cause:** Strict validation rejected model output outside shortlisted pair set.
- **Fix applied:** Treat as hard failure with retry/rerun strategy; no silent accept.
- **Verification evidence:** Replays succeed after transient retries/reruns.
- **Follow-up guardrails:** Keep shortlist/output validation strict and observable.

## 2026-04-11 - Spanner schema drift for signal provenance tables

- **Subsystem:** Phase 4 persistence / Spanner repository
- **Environment:** Cloud Run Phase 2-4 worker + `clypt-spanner-v3/clypt-graph-db-v3`
- **Symptom / Error signature:** `404 Table not found: subgraph_provenance` during Phase 4 writes.
- **Root cause:** Worker code expected comments/trends provenance schema, but live Spanner DB was missing newly introduced tables/columns.
- **Fix applied:** Added and executed `scripts/spanner/ensure_phase24_signal_schema.py` (idempotent schema sync), and documented it in runtime/deploy checklists.
- **Verification evidence:** Post-migration runs reached `PHASE24_DONE` with Phase 4 success and provenance/candidate writes.
- **Follow-up guardrails:** Run schema sync after backend changes that touch Phase 4 signal/provenance persistence.

## 2026-04-11 - Cloud Tasks/Cloud Run dispatch instability

- **Subsystem:** Queue handoff (Cloud Tasks -> Cloud Run GPU worker)
- **Environment:** `clypt-phase24` queue + `clypt-phase24-worker` (`us-east4` L4)
- **Symptom / Error signature:** repeated `POST 429 ... no available instance`, intermittent dispatch delays, and confusing cross-region service behavior.
- **Root cause:** Combined factors: stale duplicate Cloud Run service in `us-central1`, missing Cloud Tasks service identity role bindings, and tight L4 quota/capacity pressure in `us-east4`.
- **Fix applied:** removed stale `us-central1` service, restored Cloud Tasks service identity + IAM (`roles/cloudtasks.serviceAgent` project binding and token creator on worker SA), enforced serial queue dispatch (`maxConcurrentDispatches=1`), and redeployed clean worker in `us-east4`.
- **Verification evidence:** post-redeploy runs (`...postredeploy`, `...next`) completed with `PHASE24_DONE`.
- **Follow-up guardrails:** keep only one active worker region, keep queue serial for single-GPU profile, and monitor L4 quota/capacity before burst replays.

## 2026-04-15 - VibeVoice ASR JSON truncation on long canonical clip

- **Date/Time (UTC):** 2026-04-15 01:27-01:34
- **Subsystem:** Phase 1 ASR (`backend/providers/vibevoice_vllm.py`)
- **Environment:** DO H200 droplet, local vLLM VibeVoice service (`clypt-vllm-vibevoice`)
- **Symptom / Error signature:** Phase 1 failed with `RuntimeError: [vibevoice-vllm] content is not parseable as turns: Unterminated string...`; failed runs `job_50210e03f0e942afa727bf9e9bdaa99c` and `job_3970f92c7fa147bcbdff47e0dd2f28fb`.
- **Root cause:** For canonical `joeroganflagrant.wav` (~788.7s), model output degenerated into repetition and hit generation cap (`finish_reason=length`), producing incomplete JSON.
- **Fix applied:** Increased repetition control from `VIBEVOICE_REPETITION_PENALTY=0.97` to `1.0` in `/etc/clypt-phase1/v3_1_phase1.env`, restarted `clypt-v31-phase1-worker.service`, and verified runtime env load.
- **Verification evidence:** Direct diagnostic call showed `finish_reason=length`, invalid JSON, and repeated phrase count (`they hide from this`) ~5440 times before fix; post-fix rerun `job_90ae1546b9b8410db9825bbcb63a65e8` logged `[vibevoice-vllm] done in 43.3 s — 203 turns`, confirming ASR JSON path no longer failed.
- **Follow-up guardrails:** Add ASR chunk-stream diagnostics (`finish_reason`, output length, repetition signals) and guardrails for `finish_reason=length` before parse/commit.

## 2026-04-15 - Phase 1 visual backend drifted to TensorRT

- **Date/Time (UTC):** 2026-04-15 01:51
- **Subsystem:** Phase 1 visual extraction (RF-DETR backend selection)
- **Environment:** DO H200 droplet phase1 worker
- **Symptom / Error signature:** Rerun failed with `RuntimeError: tensorrt Python package is required for TensorRT inference`.
- **Root cause:** `CLYPT_PHASE1_VISUAL_BACKEND` was not pinned to `cuda_fp16` for this worker process, causing a TensorRT code path without TensorRT runtime installed.
- **Fix applied:** Set `CLYPT_PHASE1_VISUAL_BACKEND=cuda_fp16` in `/etc/clypt-phase1/v3_1_phase1.env`, restarted `clypt-v31-phase1-worker.service`, verified env in `/proc/<pid>/environ`.
- **Verification evidence:** Subsequent run `job_d253a02d99b64be78b456a0ec3fe2a83` loaded RF-DETR with `backend=cuda_fp16` and progressed through visual inference.
- **Follow-up guardrails:** Keep visual backend pinned in env templates/runbook for non-TensorRT hosts and assert backend selection at worker startup.

## 2026-04-15 - Phase 2 merge contiguity validation failure (diagnosed)

- **Date/Time (UTC):** 2026-04-15 01:54
- **Subsystem:** Phase 2 merge/classify (`backend/pipeline/semantics/merge_and_classify.py`)
- **Environment:** Phase24 local worker (`clypt-v31-phase24-local-worker`) on DO H200
- **Symptom / Error signature:** `ValueError: merged node source_turn_ids must form a contiguous target partition` for run `job_d253a02d99b64be78b456a0ec3fe2a83`.
- **Root cause:** LLM merge output in neighborhood `nb_0007` returned interleaved/non-contiguous `source_turn_ids` (for example, `t_000154,t_000156,t_000158,...`) that violate partition contiguity contract.
- **Fix applied:** No production fix applied yet; issue reproduced offline against run artifacts and offending nodes isolated for deterministic debugging.
- **Verification evidence:** Reproduction script over `metadata/phase24_handoff.json` reproduced exact contract violation and printed failing node payloads/positions; phase24 worker logs show phase2 `phase_error` and terminal failure at 01:54:36 UTC.
- **Follow-up guardrails:** Add merge-output repair/normalization or deterministic retry path for non-contiguous partitions before hard-fail; persist failing merge debug payloads even when phase aborts.

## 2026-04-15 - Qwen structured-output serving crash during schema compile

- **Date/Time (UTC):** 2026-04-15 06:08
- **Subsystem:** Phase 2-4 local generation service (Qwen vLLM path)
- **Environment:** DO GPU host, local OpenAI-compatible Qwen endpoint, phase24 local worker
- **Symptom / Error signature:** Engine crash with `compile_json_schema` / `xgrammar` errors including `minItems is greater than the number of prefixItems...`; downstream `500` then `Connection refused`.
- **Root cause:** Structured-output grammar compilation failure in serving backend before model generation step.
- **Fix applied:** Began migration path to SGLang Qwen serving, removed dynamic `oneOf` usage from high-risk Phase 4 schema, and added fail-fast crash classification/queue behavior.
- **Verification evidence:** Local tests added for schema compatibility and fail-fast error policy; runtime now classifies `connection refused` / `xgrammar` signatures as fail-fast.
- **Follow-up guardrails:** Keep schema constructs in portable subset for server-side decoding and keep deterministic Python-side relational validation.

## 2026-04-15 - GPU host image incompatibility for vLLM container

- **Date/Time (UTC):** 2026-04-15
- **Subsystem:** Infrastructure provisioning / GPU runtime
- **Environment:** New DigitalOcean GPU droplet on generic Ubuntu image
- **Symptom / Error signature:** vLLM service start failure with `nvidia-container-cli ... libnvidia-ml.so.1: cannot open shared object file`.
- **Root cause:** Host image lacked expected NVIDIA userland libraries for Docker GPU runtime.
- **Fix applied:** Updated deployment guidance to require GPU-ready base image for new droplet provisioning.
- **Verification evidence:** Failure reproduced in service logs; deployment docs now encode GPU-base-image requirement explicitly.
- **Follow-up guardrails:** Treat non-GPU-ready base images as invalid for runtime rollout.
