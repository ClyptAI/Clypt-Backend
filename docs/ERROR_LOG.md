# ERROR LOG

Persistent record of major runtime/deployment/pipeline errors and their recoveries.

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
