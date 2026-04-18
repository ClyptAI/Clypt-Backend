# ERROR LOG

Persistent record of major runtime/deployment/pipeline errors and their recoveries.

> Historical entries below may mention legacy file paths from the retired
> H200/RTX split. Some of those files have since been deleted as part of the
> `phase1` + `phase26` + Modal cleanup; treat those paths as historical context,
> not current operator entrypoints.

## 2026-04-18 - Phase1 vLLM sidecar booted the wrong model because systemd swallowed `MODEL_PATH`

- **Date/Time (UTC):** 2026-04-18 00:40 UTC
- **Subsystem:** Phase1 VibeVoice vLLM sidecar (`scripts/do_phase1/systemd/clypt-phase1-vllm-vibevoice.service`, `scripts/do_phase1/deploy_phase1_services.sh`)
- **Environment:** Fresh Rithvik-team H200 droplet (`clypt-phase1-h200-nyc2`) during first bring-up of the new Phase1 host topology
- **Symptom / Error signature:** All top-level Phase1 services were `active`, and `GET /health` on the local VibeVoice wrapper returned `{"status":"ok","vibevoice_asr_ready":true}`. But `systemctl status clypt-phase1-vllm-vibevoice` showed `MODEL_PATH=` in the final docker command, `generate_tokenizer_files.py: error: argument --output/-o: expected one argument`, and the underlying vLLM server actually booted `Qwen/Qwen3-0.6B` instead of `microsoft/VibeVoice-ASR`.
- **Root cause:** The systemd unit embedded a long `/bin/bash -lc` docker one-liner that relied on nested `$$` escaping for the `snapshot_download(...)` command substitution. On the fresh droplet, systemd consumed the escape sequence in a way that left bash with `MODEL_PATH=` and an empty positional value for `vllm serve`, so vLLM silently fell back to its default demo model while the outer wrapper still passed health.
- **Fix applied:** Replaced the fragile inline docker command with a dedicated helper script, [`scripts/do_phase1/run_vllm_vibevoice_container.sh`](../scripts/do_phase1/run_vllm_vibevoice_container.sh), and changed the systemd unit to launch that script directly. The helper resolves the VibeVoice snapshot path in plain bash, hard-fails if it is empty, then starts vLLM with the explicit snapshot path. Also updated [`scripts/do_phase1/deploy_phase1_services.sh`](../scripts/do_phase1/deploy_phase1_services.sh) to prewarm the `microsoft/VibeVoice-ASR` cache into `/opt/clypt-phase1/hf-cache` before the unit starts, reducing fresh-host startup variance.
- **Verification evidence:** On the live H200, the pre-fix unit consistently showed `MODEL_PATH=` and container logs proved vLLM loaded `Qwen/Qwen3-0.6B`. After installing the helper-script launch path and restarting the sidecar, `GET http://127.0.0.1:8000/v1/models` returned the VibeVoice-served model instead of the Qwen stub, and the local wrapper health remained green.
- **Follow-up guardrails:** Do not embed multi-layered shell interpolation directly inside systemd `ExecStart` when model selection is load-bearing. Keep the helper script as the single authoritative launch surface, and always verify the inner sidecar with `/v1/models` on fresh hosts instead of trusting only the wrapper `/health`.

## 2026-04-18 - Two-H200 fresh-host run failed because copied ADC creds could not sign and Phase1 deploy no longer installed TensorRT

- **Date/Time (UTC):** 2026-04-18 00:49 UTC
- **Subsystem:** Phase1/Phase26 host credential setup and Phase1 visual deploy (`scripts/do_phase1/deploy_phase1_services.sh`, `scripts/do_phase26/deploy_phase26_services.sh`)
- **Environment:** First full Phase 1-4 test attempt on the new Rithvik-team Phase1 H200 plus Ming-team Phase26 H200
- **Symptom / Error signature:** The Phase1 worker resolved the test-bank mapping and started sidecars, then failed with two parallel errors: `failed to sign canonical audio_gcs_uri=... you need a private key to sign credentials ... <class 'google.oauth2.credentials.Credentials'> just contains a token` from `/tasks/vibevoice-asr`, and `visual extraction failed: tensorrt Python package is required for TensorRT inference. Install the NVIDIA TensorRT runtime.` from `/tasks/visual-extract`.
- **Root cause:** Two fresh-host regressions. First, both H200s were provisioned with copied `application_default_credentials.json` files (`type=authorized_user`) instead of signing-capable service-account keys, so any code path that needs V4 signed URLs failed mid-run. Second, the new `phase1` deploy script had lost the older TensorRT dependency installation that the visual path requires when `CLYPT_PHASE1_VISUAL_BACKEND=tensorrt_fp16`.
- **Fix applied:** Restored fail-fast credential validation in the shared deploy preamble: both host deploy scripts now reject `GOOGLE_APPLICATION_CREDENTIALS` files unless they are real `service_account` JSON keys with a private key. Restored Phase1 TensorRT provisioning by teaching [`scripts/do_phase1/deploy_phase1_services.sh`](../scripts/do_phase1/deploy_phase1_services.sh) to install `libnvinfer-bin` plus `tensorrt-cu13` and verify both `trtexec` and the Python `tensorrt` module whenever the visual backend is TensorRT. Updated Phase1/Phase26/Modal deploy docs to explicitly ban copying `authorized_user` ADC onto hosts.
- **Verification evidence:** The failing run logs on `clypt-phase1-h200-nyc2` showed both errors deterministically. The host credential files were confirmed to be `authorized_user` JSONs containing only a refresh token, and `pip show tensorrt*` plus `command -v trtexec` confirmed the TensorRT runtime was absent on the fresh Phase1 H200 before the fix.
- **Follow-up guardrails:** Host deploys must use service-account keys, not local workstation ADC files. Keep TensorRT dependency installation tied to the Phase1 deploy script so the default `tensorrt_fp16` path remains self-contained on fresh H200s.

## 2026-04-18 - First live Phase1 run still tried to cold-download NFA on the H200

- **Date/Time (UTC):** 2026-04-18 00:58 UTC
- **Subsystem:** Phase1 audio-post chain warmup (`scripts/do_phase1/deploy_phase1_services.sh`, `backend/providers/forced_aligner.py`, `backend/providers/emotion2vec.py`, `backend/providers/yamnet.py`)
- **Environment:** Fresh Rithvik-team Phase1 H200 during the first full run after the credential/TensorRT fixes
- **Symptom / Error signature:** The rerun got past ASR and visual startup, then the worker logged `loading NFA model 'stt_en_fastconformer_hybrid_large_pc' on cuda (attempt 1/2) ...` followed by `ValueError: Not able to download url right now, please try again.` while Phase1 was already live. The job log showed forced-alignment stage failure instead of using a preloaded model cache.
- **Root cause:** The old H200 deploy flow used to prewarm NFA + emotion2vec+ (and effectively YAMNet via first-load) before any services restarted. That warmup step was lost in the `phase1` topology cleanup, so the first live job became responsible for downloading the NeMo aligner model and other audio-post assets on demand.
- **Fix applied:** Restored explicit audio-post prewarm to [`scripts/do_phase1/deploy_phase1_services.sh`](../scripts/do_phase1/deploy_phase1_services.sh). The deploy now instantiates `ForcedAlignmentProvider`, `Emotion2VecPlusProvider`, and `YAMNetProvider` after dependency install, forcing their model/runtime initialization before services are restarted.
- **Verification evidence:** The failing job log on `job_e4fc042b193c4e0ca5dd7353bc12a940` showed the forced aligner trying to download from NGC during the live run. Manual reachability check to the NeMo model URL succeeded (`HTTP 200`), confirming the issue was missing prewarm rather than a permanently inaccessible model source.
- **Follow-up guardrails:** Keep audio-post prewarm in the Phase1 deploy path. The first customer run should never be the thing that warms NFA/emotion2vec+/YAMNet on a fresh host.

## 2026-04-17 - Phase26 fresh-host startup failed before Qwen was cached locally

- **Date/Time (UTC):** 2026-04-17 23:45 UTC
- **Subsystem:** Phase26 H200 SGLang startup (`scripts/do_phase26/deploy_phase26_services.sh`, `scripts/do_phase26/systemd/clypt-phase26-sglang-qwen.service`)
- **Environment:** Fresh Ming-team H200 droplet (`clypt-phase26-h200-nyc2`) during first downstream-host bring-up for the two-H200 + Modal topology
- **Symptom / Error signature:** `clypt-phase26-sglang-qwen.service` entered a restart loop. The first failure was `sglang serve: error: unrecognized arguments:` with nothing after the colon. After fixing that, the service still failed with `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.` while `HF_HUB_OFFLINE=1` was active.
- **Root cause:** Two separate fresh-host gaps compounded. First, the unit passed `${SG_EXTRA_ARGS}` directly in `ExecStart`; when the env var was blank, systemd still handed SGLang an empty argument, which argparse rejected. Second, the service intentionally enforced offline Hugging Face mode, but `deploy_phase26_services.sh` never prewarmed the `Qwen/Qwen3.6-35B-A3B` cache on a new droplet before starting SGLang.
- **Fix applied:** Updated [`scripts/do_phase26/systemd/clypt-phase26-sglang-qwen.service`](../scripts/do_phase26/systemd/clypt-phase26-sglang-qwen.service) to launch through `/bin/bash -lc` and append `SG_EXTRA_ARGS` only when present. Updated [`scripts/do_phase26/deploy_phase26_services.sh`](../scripts/do_phase26/deploy_phase26_services.sh) to prewarm the Qwen snapshot into `/opt/clypt-phase26/hf-cache` with `snapshot_download(...)` before installing/restarting the SGLang unit, preserving `HF_HUB_OFFLINE=1` as the steady-state runtime behavior.
- **Verification evidence:** On `clypt-phase26-h200-nyc2`, the blank-argument failure disappeared immediately after installing the patched unit. The follow-on failure reproduced deterministically until the Qwen snapshot prewarm was started against the fresh host cache, confirming the missing-cache root cause rather than a launch-flag regression.
- **Follow-up guardrails:** Keep the SGLang unit authoritative for the launch flags, but never pass optional trailing args directly unless the shell conditionally omits them. Any host that keeps `HF_HUB_OFFLINE=1` must prewarm the exact model revision before service start; otherwise the first boot will fail by design.

## 2026-04-17 - Modal node-media-prep app deployed without the local `backend` package

- **Date/Time (UTC):** 2026-04-17 23:38 UTC
- **Subsystem:** Modal node-media-prep deployment (`scripts/modal/node_media_prep_app.py`)
- **Environment:** First deployment of the new L4-backed Modal `node-media-prep` app for the two-H200 + Modal topology
- **Symptom / Error signature:** Modal app deploy succeeded, but every request to `/health` hung and app logs showed `ModuleNotFoundError: No module named 'backend'` from `/root/node_media_prep_app.py`.
- **Root cause:** The Modal app file imports `backend.runtime.node_media_prep`, `backend.providers.storage`, and `backend.providers.config`, but Modal 1.x does not implicitly ship unrelated local packages into the container. The image installed Python dependencies but never mounted the repo-local `backend` package.
- **Fix applied:** Updated [`scripts/modal/node_media_prep_app.py`](../scripts/modal/node_media_prep_app.py) to include `.add_local_python_source("backend")` in the Modal image definition, after the image build steps. This makes the local `backend` package importable inside the container without changing the external API contract.
- **Verification evidence:** After redeploying the patched app, `curl -i https://rithviks84--clypt-node-media-prep-node-media-prep.modal.run/health` returned `HTTP/2 200` with `{"status":"ok"}`. The focused regression suite for the Modal app also passed locally: `tests/scripts/test_modal_node_media_prep_app.py`, `tests/backend/runtime/test_node_media_prep.py`, and `tests/backend/providers/test_remote_node_media_prep_client.py`.
- **Follow-up guardrails:** Any Modal app that imports repo-local code outside its own module path must explicitly include that package with `add_local_python_source(...)` or `add_local_dir(...)`. Do not assume deploy success implies the container has all local imports available.

## 2026-04-18 - Fresh droplet deploy inherited repo-root `.env.local` from copied workspace

- **Date/Time (UTC):** 2026-04-18 00:15 UTC
- **Subsystem:** Host deploy workflow (`scripts/do_phase1/deploy_phase1_services.sh`, `scripts/do_phase26/deploy_phase26_services.sh`, Python dotenv loading in `backend/providers/config.py`)
- **Environment:** Ming-team Phase26 H200 brought up by copying the local workstation tree directly into `/opt/clypt-phase26/repo`
- **Symptom / Error signature:** Phase26 services loaded `VIBEVOICE_BACKEND=native` even though `/etc/clypt-phase26/phase26.env` did not set it. Dispatch then failed fast with `Unsupported VIBEVOICE_BACKEND='native'; only 'vllm' is supported on main.`
- **Root cause:** The copied repo included the workstation's repo-root `.env.local`. `backend/providers/config.py` intentionally loads `.env` / `.env.local` from the current repo root via `dotenv` when present, so the droplet inherited local-development overrides that should never have reached host runtime.
- **Fix applied:** Added a shared deploy guard in [`scripts/lib/preamble.sh`](../scripts/lib/preamble.sh) and wired it into both host deploy scripts. [`deploy_phase1_services.sh`](../scripts/do_phase1/deploy_phase1_services.sh) and [`deploy_phase26_services.sh`](../scripts/do_phase26/deploy_phase26_services.sh) now fail immediately if the copied repo contains `.env` or `.env.local`, forcing operators to rely only on `/etc/clypt-phase1/phase1.env` or `/etc/clypt-phase26/phase26.env` plus the committed `/etc/clypt/*` runtime env templates. Updated the host deploy runbooks to explicitly require excluding `.env` / `.env.local` during copy/sync.
- **Verification evidence:** After renaming the copied `/opt/clypt-phase26/repo/.env.local` out of the way on the Ming H200, dispatch and worker booted cleanly and no longer saw `VIBEVOICE_BACKEND=native`. The new guard now prevents the same bad state from recurring on fresh hosts.
- **Follow-up guardrails:** Keep repo-root `.env` / `.env.local` strictly local-development artifacts. Any droplet sync command should exclude them explicitly, and any future host deploy entrypoint should call the same preamble guard before sourcing host env files.

## 2026-04-17 - Phase 1 / downstream host coupling blocked the two-H200 + Modal split

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 1 orchestration, downstream enqueue boundary, deployment/runtime documentation
- **Environment:** repository migration from the old mixed host naming into the new `phase1` + `phase26` + Modal topology
- **Symptom / Error signature:** The repo still assumed the Phase 1 host directly owned the downstream SQLite queue and still documented the old H200/RTX split in canonical runtime/deploy surfaces. That made the intended topology ambiguous: Phase 1 could not cleanly dispatch to a second H200, and operators had no canonical host-scoped env/runbook set for the new split.
- **Root cause:** Host-level responsibilities had evolved faster than the naming and service boundaries. The codebase still mixed Phase 1 orchestration, local queue ownership, and older deployment vocabulary, so the new architecture existed only as intent rather than as committed runtime surfaces.
- **Fix applied:** Added persistent local Phase 1 service boundaries for VibeVoice and visual extraction, added a remote Phase26 dispatch client/service, moved host-level entrypoints and deploy assets to explicit `phase1` / `phase26` names, added new env baselines (`known-good-phase1-h200.env`, `known-good-phase1-h100-backup.env`, `known-good-phase26-h200.env`), added new deployment docs (`PHASE1_HOST_DEPLOY.md`, `PHASE26_HOST_DEPLOY.md`, `MODAL_NODE_MEDIA_PREP_DEPLOY.md`), and rewrote the canonical README/runtime/architecture docs around the two-H200 + Modal split.
- **Verification evidence:** Focused regression suites passed after the migration: `tests/backend/runtime/phase1_visual_service/test_phase1_visual_service_app.py`, `tests/backend/runtime/phase26_dispatch_service/test_phase26_dispatch_service_app.py`, `tests/backend/providers/test_phase1_visual_and_phase26_clients.py`, `tests/backend/phase1_runtime/test_factory.py`, `tests/backend/phase1_runtime/test_runner.py`, `tests/backend/runtime/test_phase24_worker_app.py`, and the relevant config tests in `tests/backend/providers/test_provider_config_and_clients.py`.
- **Follow-up guardrails:** Keep host-level naming semantic: use `phase1` for Phase 1 host surfaces, `phase26` for downstream host surfaces, and reserve `phase24` for the current business-logic modules only. Any future Phase 6 render/export rollout should follow the same pattern: Phase26 owns orchestration, Modal owns the stateless remote worker surface.

## 2026-04-17 - Fresh H200 visual deploy failed because `youtokentome` needed explicit `--no-build-isolation`

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** H200 Phase 1 deploy automation (`scripts/do_phase1_visual/deploy_visual_service.sh`)
- **Environment:** Fresh DigitalOcean NVIDIA AI/ML Ready image in `atl1` on `gpu-h200x1-141gb`, Phase 1 venv bootstrap from `requirements-do-phase1-visual.txt`
- **Symptom / Error signature:** `deploy_visual_service.sh` failed during the `requirements-do-phase1-visual.txt` install when `nemo-toolkit[asr]` pulled `youtokentome>=1.0.5`; pip aborted with `ModuleNotFoundError: No module named 'Cython'` while getting build requirements for the `youtokentome` wheel.
- **Root cause:** `youtokentome`'s build path imported `Cython` without declaring it in isolated build requirements, so a fresh venv on the H200 could not satisfy the NeMo ASR dependency tree. An initial script patch that exported `PIP_NO_BUILD_ISOLATION=1` was not sufficient on the host's pip 26.0.1 path; the build only became deterministic once pip was invoked with the explicit `--no-build-isolation` flag.
- **Fix applied:** Updated [`scripts/do_phase1_visual/deploy_visual_service.sh`](../scripts/do_phase1_visual/deploy_visual_service.sh) to preinstall `Cython<4` and then install `youtokentome>=1.0.5` with `python -m pip install --no-build-isolation ...` before the main `requirements-do-phase1-visual.txt` solve. This forces the build to reuse the current venv where `Cython` is already present.
- **Verification evidence:** On the fresh Atlanta H200, `source /opt/clypt-phase1/venvs/phase1/bin/activate && python -m pip install --no-build-isolation youtokentome==1.0.6` completed successfully and built the wheel in-venv. The failing `ModuleNotFoundError: No module named 'Cython'` only reproduced when relying on the earlier env-var-based workaround.
- **Follow-up guardrails:** Treat NeMo's ASR extras as a fragile transitive graph on fresh GPU hosts. When adding or updating audio-post dependencies, validate the full H200 deploy script on a clean droplet rather than relying on an already-warmed venv.

## 2026-04-17 - Fresh H200 visual deploy started Phase 2-4 worker before SGLang existed

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** H200 Phase 1 deploy automation (`scripts/do_phase1_visual/deploy_visual_service.sh`)
- **Environment:** Fresh DigitalOcean NVIDIA AI/ML Ready image in `atl1` on `gpu-h200x1-141gb`, first-pass H200 bring-up before `deploy_sglang_qwen_service.sh`
- **Symptom / Error signature:** `deploy_visual_service.sh` completed the Phase 1 venv install, TensorRT install, and NFA/emotion2vec+ prewarm, then failed at `systemctl restart clypt-v31-phase24-local-worker.service` with `Unit clypt-sglang-qwen.service not found.`
- **Root cause:** `clypt-v31-phase24-local-worker.service` has `Requires=clypt-sglang-qwen.service`, but `deploy_visual_service.sh` restarted the Phase 2-4 worker before the documented next step (`deploy_sglang_qwen_service.sh`) had installed the SGLang unit. The script order contradicted the deploy order described in its own final message.
- **Fix applied:** Updated [`scripts/do_phase1_visual/deploy_visual_service.sh`](../scripts/do_phase1_visual/deploy_visual_service.sh) to restart only the Phase 1 API and Phase 1 worker during the visual deploy. The script now starts `clypt-v31-phase24-local-worker.service` only when `/etc/systemd/system/clypt-sglang-qwen.service` already exists; otherwise it logs a skip message and leaves Phase 2-4 startup to `deploy_sglang_qwen_service.sh`.
- **Verification evidence:** On the Atlanta H200, the failure reproduced exactly once on a fresh host before the patch (`Unit clypt-sglang-qwen.service not found`). After the sequencing fix, the visual deploy can complete without attempting to start Phase 2-4 early, and `deploy_sglang_qwen_service.sh` remains the step that restarts `clypt-v31-phase24-local-worker.service` after SGLang becomes healthy.
- **Follow-up guardrails:** Keep the H200 bring-up order strict: `bootstrap_h200.sh` → `deploy_visual_service.sh` → `deploy_sglang_qwen_service.sh`. Do not reintroduce any unconditional Phase 2-4 worker restart in the visual deploy unless the SGLang unit is guaranteed to be installed first.

## 2026-04-17 - Fresh H200 Phase 1 worker crashed on Python 3.10 because `TypeVar(default=...)` is too new

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 1 worker / Spanner repository import path (`backend/runtime/run_phase1_worker.py` → `backend/repository/spanner_phase14_repository.py`)
- **Environment:** Fresh DigitalOcean NVIDIA AI/ML Ready image in `atl1` on `gpu-h200x1-141gb`, Ubuntu 22.04 / Python 3.10.12
- **Symptom / Error signature:** `clypt-v31-phase1-worker.service` entered a restart loop immediately after the visual deploy. `journalctl -u clypt-v31-phase1-worker.service` showed `TypeError: TypeVar.__init__() got an unexpected keyword argument 'default'` at `backend/repository/spanner_phase14_repository.py:48`.
- **Root cause:** `backend/repository/spanner_phase14_repository.py` used `TypeVar("T", default=Any)`, which is not supported by the H200 host's Python 3.10 interpreter. The import failed before the worker could initialize `build_default_phase1_job_runner()`.
- **Fix applied:** Updated [`backend/repository/spanner_phase14_repository.py`](../backend/repository/spanner_phase14_repository.py) to use `T = TypeVar("T")` instead. The `default=Any` clause was only type-hint sugar for the overloaded helpers and had no runtime value, so removing it preserves behavior while restoring Python 3.10 compatibility.
- **Verification evidence:** The worker crash signature reproduced consistently in the H200 systemd journal before the patch. After syncing the compatibility change and restarting the service, the Phase 1 worker can import the repository module on Python 3.10 instead of failing during module import.
- **Follow-up guardrails:** Avoid Python-3.12+/3.13-only typing syntax in runtime codepaths unless the deploy images are upgraded first. If newer typing features are desired for static checking, prefer `typing_extensions` shims or keep them out of modules imported by production workers on Python 3.10.

## 2026-04-17 - Rsynced `.env.local` poisoned the H200 worker with stale local VibeVoice settings

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** H200 Phase 1 worker config loading (`backend/providers/config.py` + `scripts/do_phase1_visual/deploy_visual_service.sh`)
- **Environment:** Fresh DigitalOcean NVIDIA AI/ML Ready image in `atl1` on `gpu-h200x1-141gb`, repo rsynced from a local workstation that contained `.env.local`
- **Symptom / Error signature:** `clypt-v31-phase1-worker.service` kept crashing after the Python 3.10 `TypeVar(default=...)` fix with `ValueError: Unsupported VIBEVOICE_BACKEND='native'; only 'vllm' is supported on main.` even though `/etc/clypt-phase1/v3_1_phase1.env` contained only the remote-ASR `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_*` settings and no `VIBEVOICE_*` keys.
- **Root cause:** `backend/providers/config.py` auto-loads repo-root `.env` and `.env.local` from `Path.cwd()`. The rsynced droplet repo still had a workstation-local `.env.local` with `VIBEVOICE_BACKEND=native`, so the worker imported stale local-dev settings on top of the correct systemd env file.
- **Fix applied:** Updated [`scripts/do_phase1_visual/deploy_visual_service.sh`](../scripts/do_phase1_visual/deploy_visual_service.sh) to delete repo-local `.env` and `.env.local` files on the droplet before any services start. Live H200 recovery also removes the stray `.env.local` from `/opt/clypt-phase1/repo` before restarting the worker.
- **Verification evidence:** On the Atlanta H200, `grep -R '^VIBEVOICE_BACKEND=' /opt/clypt-phase1/repo` identified `/opt/clypt-phase1/repo/.env.local:VIBEVOICE_BACKEND=native` as the only live source of the bad setting. After removing the stray dotenv file, the worker no longer inherits the unsupported local VibeVoice backend from the repo checkout.
- **Follow-up guardrails:** Do not rely on rsync alone to keep deploy repos clean; local dotenv files are operationally hazardous on the H200 because provider config auto-loads them. Keep `deploy_visual_service.sh` as the guardrail that strips `.env` / `.env.local` on the droplet before Phase 1 or Phase 2-4 services are started.

## 2026-04-17 - Fresh H200 bootstrap failed because base-image pip lacked `--break-system-packages`

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** H200 bootstrap automation (`scripts/do_phase1_visual/bootstrap_h200.sh`)
- **Environment:** Fresh DigitalOcean NVIDIA AI/ML Ready image in `atl1` on `gpu-h200x1-141gb`
- **Symptom / Error signature:** `bootstrap_h200.sh` progressed through apt provisioning and then failed during Qwen cache prewarm with `pip3 install --break-system-packages --quiet "huggingface_hub>=0.28"` → `no such option: --break-system-packages`.
- **Root cause:** The script assumed a newer pip CLI than the Ubuntu 22.04 NVIDIA AI/ML base image currently provides. On this image, `pip` predates the `--break-system-packages` flag, so the one-time `huggingface_hub` install aborts before `snapshot_download(...)` can run.
- **Fix applied:** Updated [`scripts/do_phase1_visual/bootstrap_h200.sh`](../scripts/do_phase1_visual/bootstrap_h200.sh) to probe `python3 -m pip install --help` for `--break-system-packages` and only pass the flag when supported; otherwise it falls back to a normal `python3 -m pip install --quiet "huggingface_hub>=0.28"`.
- **Verification evidence:** Re-running bootstrap on the fresh Atlanta H200 no longer aborts at the `huggingface_hub` install step and proceeds into the Qwen pre-cache flow.
- **Follow-up guardrails:** Treat the DO GPU base image as Ubuntu 22.04 unless proven otherwise, and feature-detect pip flags in bootstrap scripts instead of assuming parity with newer Debian/Ubuntu packaging behavior.

> **2026-04-17 note:** Entries below that reference GCE L4 / Cloud Run L4 combined
> service, the VibeVoice `bfloat16` Dockerfile patch, and Cloud Tasks dispatch
> describe infrastructure that has since been torn down. The code paths,
> deploy scripts, and Docker images referenced by those incidents no longer
> exist in this repository; the entries are retained for historical context
> only. Phase 1 ASR and node-media prep now live on a dedicated RTX 6000
> Ada audio host that the H200 calls over HTTP — see the 2026-04-17 entry
> on the H200/RTX split for the final resolution of the NVENC issue.

> **2026-04-17 close-out note:** The two 2026-04-17 entries below — "RTX 6000
> Ada vLLM VibeVoice starved NeMo Forced Aligner into per-turn fallback" and
> "H200 NVENC gap finally resolved via dedicated RTX 6000 Ada audio host" —
> captured the architecture where NFA + emotion2vec+ + YAMNet were co-tenant
> with VibeVoice vLLM on the 48 GiB RTX 6000 Ada. See the newer 2026-04-17
> entry ("Reverted NFA/emotion2vec+/YAMNet to H200; narrowed RTX to VibeVoice
> ASR + node-media-prep sole tenant") for the follow-up decision to move
> NFA/emotion/YAMNet back to the H200 and run the RTX as a VibeVoice sole
> tenant. The flag band and `PYTORCH_CUDA_ALLOC_CONF` hint called out in the
> co-tenancy entry are no longer in effect.

## 2026-04-17 - F2.2 typed-response fan-out landed test fixtures and a phase14 live dict-vs-Pydantic mismatch

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** `backend/pipeline/{candidates,graph,signals}/` response typing (F2.2 fan-out); `backend/runtime/phase14_live.py` candidate attribution enrichment.
- **Environment:** Offline regression suite (`tests/backend/pipeline/ tests/backend/runtime/`) at HEAD of `DO-speedup-and-OSS-swap` immediately after Wave 3 parallel dispatch (PR-Q candidates, PR-R graph, PR-S signals).
- **Symptom / Error signature:** Five post-merge regressions: (1) `pydantic_core._pydantic_core.ValidationError: 4 validation errors for CandidatesPooledCandidateReviewResponse … score_breakdown.virality / coherence / engagement Field required … score_breakdown.overall_clip_quality Extra inputs are not permitted` in `tests/backend/pipeline/test_candidate_review_phase4.py` (2 tests) and `tests/backend/pipeline/test_orchestrator_phase14.py`; (2) `TypeError: 'SignalsThreadConsolidationResponse' object is not subscriptable` in `tests/backend/pipeline/signals/test_llm_runtime.py`; (3) `AttributeError: 'dict' object has no attribute 'quality'` at `backend/pipeline/signals/runtime.py:264` in `tests/backend/pipeline/signals/test_runtime.py`.
- **Root cause:** Two concurrent issues. First, **test-fixture drift**: `score_breakdown={"overall_clip_quality": X}` predated the current `candidates/prompts.py` JSON schema (which requires `virality`/`coherence`/`engagement`), and Q's `CandidatesRankedCandidateScoreBreakdownResponse` (`StrictModel extra="forbid"`) correctly surfaced the drift. Second, **scope-creep dict-vs-Pydantic mismatch**: PR-S migrated the signals `llm_runtime` wrappers to return typed `SignalsXxxResponse` models; existing tests and `backend/runtime/phase14_live.py:1534` (`enriched["explanation"] = explanation`) still assumed dicts/strings. In production, the `enriched` dict later gets JSON-serialized, and `json.dumps(Pydantic_model)` raises — so S's change introduced a silent downstream hazard. A separate procedural contributor: PR-Q and PR-S both committed while their local test runs were blocked by a transient cross-agent `ModuleNotFoundError` (PR-R had a mid-edit state referencing `LongRangeEdgeAdjudicationResponse` before finalization in `graph/responses.py`); they took the import error as "unrelated" and shipped, missing their own stale-fixture and downstream regressions.
- **Fix applied:** One fixup commit (`f4bead0`) covering (a) `tests/backend/pipeline/test_candidate_review_phase4.py` and `tests/backend/pipeline/test_orchestrator_phase14.py` — replaced `{"overall_clip_quality": X}` with `{"virality": X, "coherence": X, "engagement": X}` at all three fixture sites; (b) `tests/backend/pipeline/signals/test_llm_runtime.py` — converted 4 subscript accesses to attribute access (`.thread_summary`, `.quality`, `.prompt`, `.node_ids`); (c) `tests/backend/pipeline/signals/test_runtime.py` — rewrote 3 monkeypatches to return matching `SignalsThreadConsolidationResponse` / `SignalsCommentClassificationResponse` / `SignalsClusterPromptResponse` instances instead of dicts/strings; (d) `tests/backend/runtime/test_phase14_live.py` — rewrote the `explain_candidate_attribution_with_llm` monkeypatch to return `SignalsCandidateAttributionResponse(explanation="boosted by comments")`; (e) **production fix** in `backend/runtime/phase14_live.py:1534` — `enriched["explanation"] = explanation.explanation` so the dict stays JSON-safe when the real LLM returns a Pydantic model.
- **Verification evidence:** `python -m pytest tests/backend/pipeline tests/backend/phase1_runtime tests/backend/runtime tests/backend/repository -q --ignore=tests/backend/phase1_runtime/test_rfdetr_visual_modules.py --ignore=tests/backend/phase1_runtime/test_jobs_and_service.py` → `166 passed in 4.49s` at HEAD `f4bead0`. Full regression re-run after F3.1 (`aced7a4`) and F3.2 (`e7cd041`) landed on top: same `166 passed`. Pushed to `origin/DO-speedup-and-OSS-swap` as part of the Wave 3 + F2.2 fan-out batch.
- **Follow-up guardrails:** (1) When dispatching parallel `F2.2`-style Protocol/response-type agents, include `backend/runtime/` call sites in the "MAY touch" list so downstream dict-assuming consumers (`enriched["..."] = result`) can be migrated in the same commit. (2) If a dispatched subagent's local tests are blocked by a cross-agent import error, the agent must **not** commit; it must wait or re-sync. Reiterate this rule in future parallel-dispatch prompts. (3) `StrictModel extra="forbid"` catches fixture drift loudly at the response boundary — keep that default for every F2.2-style typed response; do not loosen to `extra="allow"` to make tests pass. (4) Treat any Pydantic-return migration as potentially introducing `json.dumps`-unsafe values downstream; grep the function's call sites for dict-assignment patterns (`enriched[...] = result`, `metadata["..."] = result`, `row["..."] = result`) before landing.

## 2026-04-16 - GCSStorageClient.get_https_url silently made objects public on signing failure

- **Date/Time (UTC):** 2026-04-16
- **Subsystem:** `backend/providers/storage.GCSStorageClient.get_https_url`
- **Environment:** All Phase 1 / Phase 2-4 runtimes that resolve GCS URIs to HTTPS (H200 audio chain, VibeVoice vLLM `build_gcs_uri_url_resolver`, node-media prep, phase24 worker).
- **Symptom / Error signature:** Silent public-URL fallback on V4 signed-URL generation failure. `get_https_url` caught any `Exception` from `blob.generate_signed_url`, called `blob.make_public()`, and returned `blob.public_url` — so a rotated service-account key, a quota hit, a uniform-bucket-level-access mismatch, or any transient auth error would flip customer audio/video objects to public ACL with no alarm and no audit trail. Discovered during a code-quality pass as finding `R1` in `tmp/code-quality-reports/06-try-except.md`.
- **Root cause:** Exception-swallowing `try/except` around signed URL generation, paired with a `blob.make_public()` side effect that mutates bucket state. This violates the fail-fast doctrine and confuses a *signing* failure (a credentials/config problem the operator must fix) with a *graceful degradation* path (there is no such thing for object ACLs on a multi-tenant media bucket). Prior incident `2026-04-15 - H200 canonical-audio URL signing failed with user ADC credentials` already documented the real failure mode — ADC creds can't sign — and the fallback was independently broken under uniform bucket-level access, which means in practice the fallback never helped; it only increased blast radius when signing worked against a bucket that *didn't* have uniform access enabled.
- **Fix applied:** Removed the silent fallback. Introduced domain exception `GcsSigningError(RuntimeError)` in `backend/providers/storage.py` carrying `gcs_uri` and chained via `raise ... from signing_err`. `get_https_url` now (1) generates the V4 signed URL, (2) on any exception raises `GcsSigningError`, and (3) never calls `blob.make_public()` or returns `blob.public_url`. `__all__` updated to export the new exception. No changes to callers were needed — every production caller (`vibevoice_vllm.build_gcs_uri_url_resolver`, phase1/phase24 runtime factories) only consumes the returned URL and benefits from surfacing a hard failure rather than silently leaking an object.
- **Verification evidence:** `python -m pytest tests/backend/providers/ -q` → `70 passed, 1 skipped`. Targeted `-k storage` → `4 passed`. Import smoke `python -c "from backend.providers.storage import GCSStorageClient, GcsSigningError; print('ok')"` → `ok`. Caller grep (`get_https_url|GCSStorageClient` across `backend/` and `tests/`) confirmed no caller relied on the public-URL fallback; no test asserted `make_public` was invoked. Only remaining `make_public` mention in the tree is the historical 2026-04-15 entry in this file.
- **Follow-up guardrails:** Do not reintroduce `blob.make_public()` in any storage client path. If a caller ever legitimately needs a public URL (e.g. an explicitly public CDN-backed asset), introduce a separate method (`get_public_url`) with the intent spelled out in its name, and wire ACL provisioning through the bucket provisioner, not at request time. The exception-swallow pattern around external-service calls is a repeating anti-pattern in this codebase (tracked under `tmp/code-quality-reports/06-try-except.md`); future audits should continue to treat "catch Exception → silent compensating side effect" as a fail-fast violation.

## 2026-04-17 - Phase 2 merge/boundary token cap too high (32768) wasted MTP budget

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 2 merge + boundary local LLM calls (`backend/pipeline/phase2/`)
- **Environment:** DO H200, `clypt-v31-phase24-local-worker`, SGLang Qwen3.6-35B-A3B with NextN MTP + FP8 KV.
- **Symptom / Error signature:** Per-call wall-clock for Phase 2 merge/boundary was higher than predicted from the SGLang bench curves. Investigation showed SGLang was decoding against a 32768-token output budget even when actual responses came back in the 1.5–5 kB range (~1500 tokens). MTP speculative decoding efficiency drops off when the server is pre-allocating KV slots for a max output that is ~4x the realized length: concurrent slots are tied up unnecessarily and the scheduler falls back to single-sequence draft acceptance more often.
- **Root cause:** `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS` and `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS` were both set to `32768` — the historical Qwen3.5-27B-era default that predates MTP + FP8 KV and predates the Phase 2 prompt trimming from 2026-04-16. At those prompt sizes, 32768 is approximately 4x the realistic upper bound on Phase 2 merge/boundary responses.
- **Fix applied:** Lowered both caps to `8192`: `CLYPT_PHASE2_MERGE_MAX_OUTPUT_TOKENS=8192` and `CLYPT_PHASE2_BOUNDARY_MAX_OUTPUT_TOKENS=8192`. Updated `.env.example`, `docs/runtime/known-good.env`, and `docs/runtime/ENV_REFERENCE.md`. Code defaults in `backend/providers/config.py` and the Phase 2 pipeline stage config also updated so the values apply uniformly whether or not the operator has overridden them.
- **Verification evidence:** Re-run on `openclawyc.mp4` showed Phase 2 merge/boundary wall-clock dropped into the expected band without any increase in `finish_reason=length` warnings (all responses continued to finish well below 8192 tokens).
- **Follow-up guardrails:** Keep Phase 2 output caps around 4–6x the p95 response length observed in prod. Do not raise them back toward 32768 without evidence that Phase 2 is actually producing longer structured outputs. If Phase 2 prompts are rewritten to request substantially larger outputs in the future, re-profile first and adjust as a deliberate change (not as a copy-paste from the Phase 1 Qwen3.5 baseline).

## 2026-04-17 - SGLang Qwen3.6 serving flags consolidated (context, MTP, scheduler, grammar, offline mode)

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** SGLang Qwen3.6-35B-A3B serving (`clypt-sglang-qwen.service`, H200); launch flag contract in `scripts/do_phase1_visual/deploy_sglang_qwen_service.sh`.
- **Environment:** DO H200 droplet; SGLang 0.5.10; model cached locally at `/opt/clypt-phase1/hf-cache`; serving on `127.0.0.1:8001`.
- **Symptom / Error signature:** Several serving defects accumulated on the pre-2026-04-17 flag set: (a) the Qwen3.6 hybrid Mamba/Attention path refused to run MTP + radix cache together until `SGLANG_ENABLE_SPEC_V2=1` was exported, (b) the configured `--context-length 131072` left too little KV headroom for concurrent Phase 2-4 calls under FP8 KV + MTP, (c) `--speculative-eagle-topk` was omitted so the NextN MTP draft heads fell back to a non-optimal top-1 behavior without being explicitly pinned, (d) `--mamba-scheduler-strategy extra_buffer`, `--grammar-backend xgrammar`, and `--reasoning-parser qwen3` were not all pinned in the committed unit, leaving direct-boot hosts at risk of drifting to SGLang defaults that do not match the Phase 4 schema-compile contract.
- **Root cause:** The launch flag set had been accumulating across the 2026-04-16 Qwen3.5→3.6 cutover and the 2026-04-17 MTP + FP8-KV enablement, and the committed systemd unit had drifted behind the effective deploy-script-rewritten command. A direct-boot without the deploy script was producing a launch line that differed from the canary-validated line.
- **Fix applied:**
  - **Launch flags consolidated into the committed unit.** `scripts/do_phase1_visual/systemd/clypt-sglang-qwen.service` `ExecStart` is now the authoritative line: `--context-length 65536 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.78 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --mamba-scheduler-strategy extra_buffer --schedule-policy lpm --chunked-prefill-size 8192 --grammar-backend xgrammar --reasoning-parser qwen3`, with `Environment=HF_HUB_OFFLINE=1` and `Environment=SGLANG_ENABLE_SPEC_V2=1`. Direct-boot hosts now serve the same contract as deploy-script-run hosts.
  - **`SG_CONTEXT_LENGTH` default lowered to `65536`** in both `deploy_sglang_qwen_service.sh` and `docs/runtime/known-good.env`. The historical `131072` default is explicitly stale.
  - **Docs pass:** `docs/ARCHITECTURE.md`, `docs/runtime/RUNTIME_GUIDE.md`, `docs/runtime/ENV_REFERENCE.md`, `docs/runtime/RUN_REFERENCE.md`, `docs/deployment/P1_DEPLOY.md`, `AGENTS.md`, and the active spec `docs/specs/2026-04-16_qwen36_swap_and_sglang_tuning_spec.md` were all updated to list the full flag set and the two env vars.
- **Verification evidence:** `clypt-sglang-qwen.service` starts cleanly with the consolidated unit; `/health` returns 200; `/v1/models` lists the Qwen3.6 model; bench of `phase4_subgraph` / `phase3_long_range` / `phase2_merge` shows the expected MTP speedups without any Mamba-scheduler boot failures.
- **Follow-up guardrails:** Any change to the SGLang flag set must update the committed systemd unit, `deploy_sglang_qwen_service.sh`, `docs/runtime/known-good.env`, `docs/runtime/ENV_REFERENCE.md`, `docs/ARCHITECTURE.md`, and `docs/runtime/RUNTIME_GUIDE.md` in the same commit. Direct-boot drift is the specific risk the consolidated unit is guarding against; do not "simplify" the unit by removing flags that match SGLang defaults, because those defaults drift between SGLang releases.

## 2026-04-17 - SGLang HF_HUB_OFFLINE missing caused DNS failures on startup

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** SGLang Qwen3.6-35B-A3B startup (`clypt-sglang-qwen.service`, H200)
- **Environment:** DO H200 droplet; SGLang service at `127.0.0.1:8001`; model loaded from local HF cache.
- **Symptom / Error signature:** SGLang scheduler crashed during startup with DNS resolution errors when the H200 droplet had limited or no outbound internet (private VPC, no default route, or transient DNS failure): errors of the form `socket.gaierror: [Errno -3] Temporary failure in name resolution` while SGLang attempted to contact `huggingface.co` to check for model updates.
- **Root cause:** SGLang (and the underlying `transformers`/`huggingface_hub` libraries) will attempt to reach HuggingFace Hub on startup to resolve the latest model revision unless explicitly told not to. On a production H200 droplet where the model is already fully cached locally and outbound DNS/HTTP is constrained, this lookup stalls or errors, causing the service to crash before reaching healthy state.
- **Fix applied:** Added `HF_HUB_OFFLINE=1` to the `clypt-sglang-qwen.service` systemd unit `Environment=` block. This instructs `huggingface_hub` to skip all network calls and use only the local cache. Verified model is fully resident in the HF cache on the H200 before relying on this flag.
- **Verification evidence:** After adding `HF_HUB_OFFLINE=1`, `clypt-sglang-qwen.service` starts cleanly without DNS calls; `curl -fsS http://127.0.0.1:8001/health` returns 200 and `/v1/models` lists the Qwen model.
- **Follow-up guardrails:** Keep `HF_HUB_OFFLINE=1` in the SGLang systemd unit for all production droplet deployments. If the model needs to be updated, temporarily remove the flag, pull the new revision to cache, then re-add the flag before restarting the service. Add this env to `docs/runtime/ENV_REFERENCE.md` SGLang flag documentation.

## 2026-04-17 - Phase 4 pool and meta token cap was truncating LLM responses

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 4 pool subgraph generation (`backend/pipeline/phase4/`)
- **Environment:** DO H200, `clypt-v31-phase24-local-worker`, SGLang Qwen3.6-35B-A3B.
- **Symptom / Error signature:** Phase 4 pool LLM calls returned `finish_reason=length` with truncated JSON responses. The worker logged warnings such as `[local-openai] finish_reason=length — structured JSON reply may be token-capped` and downstream JSON parse occasionally failed on the truncated output.
- **Root cause:** `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS` and `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS` were both set to `2048` (the historical default). Phase 4 pool and meta prompts for longer videos (22+ minutes) routinely generate responses in the 2500–3800 token range, so the 2048 cap was consistently triggering `finish_reason=length` and silently truncating structured output.
- **Fix applied:** Raised both caps to `4096`: `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS=4096` and `CLYPT_PHASE4_META_MAX_OUTPUT_TOKENS=4096`. Updated `docs/runtime/ENV_REFERENCE.md` to reflect new defaults.
- **Verification evidence:** Re-run on `openclawyc.mp4` (22m 36s) completed Phase 4 pool without any `finish_reason=length` warnings; all structured JSON responses parsed cleanly.
- **Follow-up guardrails:** Monitor `finish_reason=length` log warnings as a sentinel for token cap underprovisioning. If longer source videos are added to the test bank, re-profile Phase 4 pool output lengths and adjust `CLYPT_PHASE4_POOL_MAX_OUTPUT_TOKENS` accordingly.

## 2026-04-17 - NVENC silent CPU fallback in node-media prep: full fix chain

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 2 node-media prep ffmpeg clip extraction (RTX 6000 Ada host, `POST /tasks/node-media-prep`)
- **Environment:** DO RTX 6000 Ada droplet; ffmpeg with NVDEC/NVENC; VibeVoice vLLM at `--gpu-memory-utilization 0.77`.
- **Symptom / Error signature:** Clip extraction was silently falling back to CPU decode/encode despite NVDEC and NVENC being available on the card. ffmpeg ran without error but CPU utilization was unexpectedly high and throughput was lower than expected. Investigating `ffmpeg` process arguments and strace confirmed no NVDEC or NVENC codecs were actually in use.
- **Root cause (three-stage fix chain):**
  1. **Original bug:** ffmpeg command used `-hwaccel cuda -hwaccel_output_format cuda` with a seek (`-ss`) flag. The CUDA hwaccel filter graph fails to reinitialize after seek when `hwaccel_output_format cuda` is active, producing a filter reinit error that ffmpeg silently recovers from by falling back to CPU decode. The output was valid but fully software-decoded.
  2. **First fix attempt:** Removed `-hwaccel_output_format cuda`, kept `-hwaccel cuda`. This avoided the filter reinit error but did **not** actually engage NVDEC. On ffmpeg 4.4, `-hwaccel cuda` alone does not force hardware decode codec selection — ffmpeg still selects a software decoder unless an explicit hardware decoder is named.
  3. **Final fix:** Replaced the hwaccel flags with the explicit codec selection `-c:v h264_cuvid`. This directly names the NVDEC decoder and forces hardware decoding for H.264 input. Combined with `-c:v h264_nvenc` on the output side, both NVDEC and NVENC are fully engaged. Verified with `ffmpeg -report` that the selected decoder is `h264_cuvid` and encoder is `h264_nvenc`.
- **Fix applied:** Updated node-media prep ffmpeg invocation to use `-c:v h264_cuvid` for decode. Max concurrency kept at 8 (`CLYPT_PHASE24_NODE_MEDIA_PREP_MAX_CONCURRENCY=8`); testing at concurrency 16 OOM'd the NVENC input buffers with vLLM seated at 0.77 GPU utilization (~8 GiB free). `-ss` seek remains output-side (after `-i`) — input-side seek was evaluated but rejected due to imprecise keyframe alignment on the test assets.
- **Verification evidence:** `ffmpeg -report` confirmed `h264_cuvid` decoder and `h264_nvenc` encoder active on the RTX host. `nvidia-smi` showed NVDEC engine utilization during concurrent clip extraction. End-to-end `openclawyc.mp4` run completed node-media prep at ~90–124s wall-clock (GCS upload is the new bottleneck, not decode/encode).
- **Follow-up guardrails:** Always use `-c:v h264_cuvid` (not just `-hwaccel cuda`) to force NVDEC. Do not raise node-media prep concurrency above 8 without re-profiling VRAM with the current vLLM footprint. GCS upload latency (~90–124s for 1080p clips) is the dominant bottleneck — the Phase 3 long-range prefetch window was added specifically to hide this latency.

## 2026-04-17 - Reverted NFA/emotion2vec+/YAMNet to H200; narrowed RTX to VibeVoice ASR + node-media-prep sole tenant

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 1 audio chain topology (RTX 6000 Ada ↔ H200 split)
- **Environment:** DO H200 (Phase 1 orchestrator + RF-DETR + worker) and DO RTX 6000 Ada (VibeVoice vLLM + FastAPI audio host + ffmpeg NVENC). Job `run_20260417_001240_openclawyc` plus repeated restart attempts under the co-tenancy flag band `[0.60, 0.62]` for `--gpu-memory-utilization`.
- **Symptom / Error signature:** Even after the co-tenancy tuning captured in the two prior 2026-04-17 entries (`--max-num-seqs 1 --max-model-len 32768 --gpu-memory-utilization 0.60 --enforce-eager` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), NeMo Forced Aligner still intermittently OOM'd on global alignment (`OutOfMemoryError: CUDA out of memory. Tried to allocate N MiB. GPU 0 has a total capacity of 47.37 GiB ...`). The 48 GiB RTX 6000 Ada simply does not have enough residual headroom for NFA's global-alignment tensor shapes once VibeVoice vLLM's fixed overhead (~26 GiB) and the 32768-token KV cache are seated. Further tightening vLLM would have dropped `max_model_len` below the user-approved floor.
- **Root cause:** Co-tenancy. NFA global alignment and VibeVoice vLLM have overlapping peak-memory profiles that do not both fit into 48 GiB with any safe tuning. Emotion2Vec+ and YAMNet do not individually need a lot of VRAM, but they pile onto the same residual window and compound fragmentation. Meanwhile the H200 has 141 GiB and RF-DETR + ByteTrack alone uses a small fraction of it, so moving NFA + emotion2vec+ + YAMNet back to the H200 is strictly better on capacity.
- **Fix applied:**
  - **RTX narrowed to VibeVoice ASR + node-media-prep (sole tenant).** `/tasks/phase1-audio` renamed to `/tasks/vibevoice-asr`; response drops `diarization_payload`, `emotion2vec_payload`, `yamnet_payload`. `/tasks/node-media-prep` unchanged. NFA / Emotion2Vec+ / YAMNet providers, their deps (`nemo-toolkit`, `funasr`, `tensorflow-cpu`, `tensorflow-hub`, `librosa`, `resampy`, `setuptools==69.5.1`, `datasets`, `protobuf`), and their env vars (`CLYPT_PHASE1_YAMNET_DEVICE`, `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT`, `FUNASR_MODEL_SOURCE`) are removed from `requirements-do-phase1-audio.txt`, `scripts/do_phase1_audio/*`, and the `clypt-audio-host.service` unit. vLLM flags reverted to sole-tenant defaults — no more `--max-num-seqs 1`, `--max-model-len 32768`, `--gpu-memory-utilization 0.60`, `--enforce-eager`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` dropped from `clypt-audio-host.service`. The `start_server.py` sed-patch that injected `--enforce-eager` into the upstream wrapper argv is removed; `ExecStart` now calls `vllm serve` directly.
  - **H200 restored to full visual + audio post-processing chain.** NFA → emotion2vec+ → YAMNet run in-process in a `ThreadPoolExecutor` worker thread on the H200, concurrent with RF-DETR + ByteTrack. `backend/phase1_runtime/extract.py` re-introduces `_run_audio_chain`; `run_phase1_sidecars` now takes `vibevoice_asr_client`, `forced_aligner`, `emotion_provider`, `yamnet_provider`. `Phase1JobRunner` and `build_default_phase1_job_runner` instantiate the local providers again. `requirements-do-phase1-visual.txt` regains all Python audio deps. `scripts/do_phase1_visual/*` re-adds NFA + emotion2vec+ prewarm (with the API/worker temporarily stopped during prewarm to avoid cache contention).
  - **Env var rename with one-release deprecation aliases.** `CLYPT_PHASE1_AUDIO_HOST_URL` → `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL`, `CLYPT_PHASE1_AUDIO_HOST_TOKEN` → `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_AUTH_TOKEN`. `load_provider_settings` prefers new names, falls back to legacy with a `DeprecationWarning`. Python class renames: `RemoteAudioChainClient` → `RemoteVibeVoiceAsrClient`, `RemoteAudioChainError` → `RemoteVibeVoiceAsrError`, `PhaseOneAudioResponse` → `VibeVoiceAsrResponse`, `AudioHostSettings` → `VibeVoiceAsrServiceSettings`. Old names re-exported as deprecated aliases for one release.
  - **Docs rewrite.** `docs/ARCHITECTURE.md`, `docs/deployment/P1_DEPLOY.md`, `docs/deployment/P1_AUDIO_HOST_DEPLOY.md`, `docs/runtime/RUNTIME_GUIDE.md`, `docs/runtime/RUN_REFERENCE.md`, `docs/runtime/ENV_REFERENCE.md`, `docs/runtime/known-good.env`, `docs/runtime/known-good-audio-host.env`, `README.md`, `AGENTS.md`, `CLAUDE.md`, and `.env.example` all updated to describe the RTX as a sole-tenant VibeVoice + ffmpeg NVENC box and the H200 as the owner of the full audio post chain.
  - **Tests updated.** `tests/backend/runtime/phase1_audio_service/test_audio_chain.py` deleted (module gone); `test_app.py` narrowed to `/tasks/vibevoice-asr`; `tests/backend/providers/test_remote_audio_host_client.py` rewritten against `RemoteVibeVoiceAsrClient`; `tests/backend/providers/test_storage_and_phase1_runtime.py` now fakes VibeVoice ASR + the three local providers; `tests/backend/phase1_runtime/test_runner.py` updated to the new `Phase1JobRunner` signature. `tests/backend/conftest.py` populates the new env names by default.
- **Verification evidence:** `python -m pytest tests/backend/pipeline tests/backend/providers tests/backend/phase1_runtime tests/backend/runtime -q` passes (the only failure is an unrelated pre-existing `torch` import in `tests/backend/phase1_runtime/test_rfdetr_visual_modules.py::test_require_cuda_fails_when_unavailable` that has no bearing on this refactor). End-to-end Phase 1–4 rerun on the revised topology is pending the rsync/redeploy to both droplets.
- **Follow-up guardrails:**
  - Do not move NFA / emotion2vec+ / YAMNet back onto the RTX 6000 Ada. VibeVoice vLLM is intentionally the only GPU tenant on that card.
  - Do not set `CLYPT_PHASE1_YAMNET_DEVICE`, `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT`, or `FUNASR_MODEL_SOURCE` on the RTX host — the FastAPI process will hard-fail at startup if they are set.
  - Do not set `VIBEVOICE_*` envs on the H200 — VibeVoice vLLM never runs there. The H200 only needs `CLYPT_PHASE1_VIBEVOICE_ASR_SERVICE_URL` + `_AUTH_TOKEN` to reach the RTX.
  - `CLYPT_PHASE1_AUDIO_HOST_URL` / `_TOKEN` are accepted for one release as compat aliases; delete them from operational envs when convenient and remove the aliases in the next release window.
  - If NVENC ever lands on H200, the node-media-prep service can collapse back onto the H200 — but VibeVoice vLLM should stay on the RTX regardless because it is the one workload that benefits from having the card to itself.

## 2026-04-17 - RTX 6000 Ada vLLM VibeVoice starved NeMo Forced Aligner into per-turn fallback

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** RTX 6000 Ada audio host (`clypt-vllm-vibevoice.service` + `clypt-audio-host.service`)
- **Environment:** `POST /tasks/phase1-audio` during end-to-end Phase 1–4 test for job `run_20260417_001240_openclawyc`.
- **Symptom / Error signature:**
  - Under `clypt-audio-host.service`: `WARNING backend.providers.forced_aligner [forced_aligner] global alignment failed (OutOfMemoryError: CUDA out of memory. Tried to allocate 664.00 MiB. GPU 0 has a total capacity of 47.37 GiB of which 156.38 MiB is free. Process 14433 has 42.00 GiB memory in use. ...); falling back to per-turn alignment.`
  - `nvidia-smi` showed vLLM holding 43010 MiB / 49140 MiB with the audio-host python having only ~5 GiB headroom.
- **Root cause:** Initial cold-start of vLLM with the upstream defaults (`--gpu-memory-utilization 0.45 --max-model-len 65536 --max-num-seqs 2`) hit `Available KV cache memory: -4.29 GiB` because the VibeVoice-ASR activations at 65k × 2-seqs + CUDA graph capture consumed ~25.9 GiB before any KV cache was reserved. The quick fix of bumping `--gpu-memory-utilization 0.85` (≈41 GiB) let vLLM start, but that left NeMo Forced Aligner only ~5 GiB on the shared card — enough for per-turn alignment but not enough for the single global-alignment pass NFA prefers. Global alignment silently degraded to per-turn, which is acceptable but not the design target.
- **Fix applied:**
  - `scripts/do_phase1_audio/systemd/clypt-vllm-vibevoice.service`: switched flags to `--max-num-seqs 1 --max-model-len 32768 --gpu-memory-utilization 0.60`, plus a deploy-time sed patch of `start_server.py` that injects `--enforce-eager` into the hardcoded vllm-serve argv (the upstream wrapper's argparse doesn't expose that flag). Total vLLM footprint is ~29.5 GiB. An empirical pass at util=0.55 proved the "fixed overhead" (weights + engine + profile_run activations) is ~26 GiB on VibeVoice-ASR regardless of `max-num-seqs` / `max-model-len` / `enforce-eager` — so util=0.60 is the minimum that can satisfy the 32768-token KV-cache requirement, and `max-model-len` must stay at 32768 (the user-approved floor, half of the upstream 65536 default).
  - `scripts/do_phase1_audio/systemd/clypt-audio-host.service`: added `Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` per the PyTorch allocator hint in the OOM traceback — so NFA's PyTorch-allocated tensors don't strand GiB via fragmentation.
  - Updated `scripts/do_phase1_audio/deploy_vllm_service.sh` header + `docs/deployment/P1_AUDIO_HOST_DEPLOY.md` to document the new budget and explicitly warn against raising `--gpu-memory-utilization` above 0.60.
- **Verification evidence:** (pending) vLLM restart with new flags shows positive KV cache allocation at startup; end-to-end Phase 1–4 rerun completes audio chain without the global-alignment fallback warning; `nvidia-smi` reports vLLM ≤ 28 GiB and NFA able to allocate ≥ 10 GiB during alignment.
- **Follow-up guardrails:** Keep `--gpu-memory-utilization` in the narrow band `[0.60, 0.62]` on the RTX audio host. Below 0.60 vLLM can't reserve enough KV cache for `max_model_len=32768` and refuses to start (`estimated maximum model length is 8432`). Above ~0.62 NFA's global-alignment pass starts losing headroom and falls back to per-turn. Do not drop `max_model_len` below 32768 without user sign-off (that is the explicit user-approved floor). The deploy script's sed-patch that injects `--enforce-eager` into the upstream wrapper's argv is load-bearing: removing it adds ~2 GiB of CUDA-graph memory and will re-trigger the KV-cache shortfall.

## 2026-04-17 - H200 NVENC gap finally resolved via dedicated RTX 6000 Ada audio host

- **Date/Time (UTC):** 2026-04-17
- **Subsystem:** Phase 1 audio chain + Phase 2 node-media prep topology
- **Environment:** DO H200 droplet (Phase 1 orchestrator + visual + worker) calling a new DO RTX 6000 Ada droplet (Phase 1 audio + NVENC node-media prep) over HTTP
- **Symptom / Error signature:** Historical `h264_nvenc: unsupported device (2)` on H200 (see 2026-04-15 entry) and the failed Cloud Run / GCE L4 follow-ups.
- **Root cause:** H200 has no NVENC encoder block, so any co-located ffmpeg clip-extraction path fails. Earlier workarounds (Cloud Run L4, GCE L4) were operationally fragile.
- **Fix applied:** Split Phase 1 across two droplets. All VibeVoice vLLM + NFA + emotion2vec+ + YAMNet work and all ffmpeg NVENC clip extraction now live on an RTX 6000 Ada droplet serving `POST /tasks/phase1-audio` and `POST /tasks/node-media-prep`. The H200 always calls the RTX host through `RemoteAudioChainClient` / `RemoteNodeMediaPrepClient`; `backend/providers/config.py` fails fast at load if the URL or bearer token is missing, i.e. there is no local fallback. All prior L4/Cloud-Run-related envs, scripts, and Docker images are deleted.
- **Verification evidence:** New unit tests (`tests/backend/providers/test_remote_audio_host_client.py`, `test_remote_node_media_prep_client.py`, `tests/backend/runtime/phase1_audio_service/test_app.py`, `test_node_media_prep.py`, `test_audio_chain.py`) and updated runner/worker tests pass. Deploy scripts under `scripts/do_phase1_audio/` provision an NVENC smoke test before starting services.
- **Follow-up guardrails:** Do not reintroduce an in-process audio chain or ffmpeg path on the H200; config load must stay fail-fast on missing remote endpoints. Do not set `VIBEVOICE_*` envs on the H200.

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

## 2026-04-16 - Phase 1 factory silently disabled Spanner Phase14 repository on init failure (F0.3)

- **Date/Time (UTC):** 2026-04-16
- **Subsystem:** `backend/phase1_runtime/factory._build_phase14_repository`
- **Environment:** Phase 1 local runtime (`run_phase14` / `Phase1JobRunner`), all deploy targets.
- **Symptom / Error signature:** Silent `None` return from `_build_phase14_repository` on any Spanner init failure. A rotated credential, missing IAM, or DDL regression in `SpannerPhase14Repository.from_settings(...)` / `bootstrap_schema()` would be swallowed by a bare `except Exception`, logged at WARNING, and the Phase 1 runner would proceed with `phase14_repository=None`. Downstream `upsert_run` / `upsert_phase24_job` / `write_phase_metric` calls all no-op when the repository is `None`, so durable persistence vanished while Phase 1 kept burning GPU (NeMo NFA, emotion2vec+, YAMNet, remote VibeVoice ASR), producing nothing durable. Indistinguishable from the intentional opt-out path where Spanner is simply unconfigured.
- **Root cause:** The `try/except Exception: return None` wrapper conflated two very different cases — "operator deliberately did not configure Spanner" and "Spanner is configured but init failed" — and silently degraded to in-memory-only operation in both. This violated fail-fast: a rotated service-account key or a schema regression should crash at startup so the operator notices immediately, not fifteen minutes into a GPU run.
- **Fix applied:**
  - Added `SpannerSettings.is_configured` property in `backend/providers/config.py` (True iff `project`, `instance`, and `database` are all non-empty). This is the single sanctioned "opt-out" predicate.
  - Rewrote `_build_phase14_repository` in `backend/phase1_runtime/factory.py` to: (a) early-return `None` when `settings.spanner.is_configured` is False (the intentional opt-out), and (b) construct `SpannerPhase14Repository.from_settings(...)` + `bootstrap_schema()` with **no** `try/except`, so init failures propagate with the original traceback.
  - Updated `tests/backend/phase1_runtime/test_factory.py`: existing bootstrap test now passes a configured spanner stub; added coverage for the unconfigured-opt-out path and for the "init raises → exception propagates" path (the latter is what regressed into silent `None` under the old code).
- **Verification evidence:** `python -m pytest tests/backend/phase1_runtime -k "factory or spanner or repository"` → 3 passed (factory) plus `tests/backend/repository/test_spanner_phase14_repository.py` → 15 passed. `python -c "from backend.phase1_runtime.factory import _build_phase14_repository; print('ok')"` → ok. `SpannerSettings()` returns `is_configured=False` (default, empty project); `SpannerSettings(project="p")` returns `is_configured=True`; partially-empty configs return False.
- **Follow-up guardrails:** `_build_phase14_repository` must never grow a broad `except Exception` again; any refactor that needs to catch a specific, well-understood exception (e.g. a transient-deadline-on-bootstrap case we later decide to retry) must catch only that exception class and must not return `None`. The Phase 1 runner's `phase14_repository is None` branches in `backend/phase1_runtime/runner.py` remain only for the genuine unconfigured-Spanner case; do not add silent-degradation paths there. Discovered in code quality audit (`tmp/code-quality-reports/06-try-except.md` finding R4); shipped as PR-C of the F0 fail-fast sweep.

---

## 2026-04-16 - Phase 1 audio fail-fast cleanup: silent fallbacks removed across forced-aligner, VibeVoice probe, and audio_gcs_uri shim (F0.4 + F0.5 + F0.6 + StageEvent C2)

- **Date/Time (UTC):** 2026-04-16
- **Subsystem:** Phase 1 audio chain — `backend/providers/forced_aligner.py`, `backend/providers/vibevoice_vllm.py`, `backend/runtime/phase1_audio_service/app.py`
- **Environment:** DO H200 (forced aligner, extract orchestration) + DO RTX 6000 Ada (VibeVoice ASR service). Discovered during the code-quality audit that produced `tmp/code-quality-reports/06-try-except.md` and `tmp/code-quality-reports/03-unused-code.md`.
- **Symptom / Error signature:** Phase 1 runs could "succeed" while silently emitting degraded telemetry and missing data. Specifically: (a) a forced-aligner model-load failure returned `[]` with a `warning`, causing the upstream `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1` check to detect zero words and fail with an opaque "no alignments" error instead of the actual torch/NeMo load exception; (b) a per-turn forced-aligner fallback exception returned `all_words = []` and kept going, hiding real misalignment; (c) VibeVoice's `_probe_duration` swallowed every `ffprobe` exception and returned `0.0`, which then poisoned RTF/duration telemetry downstream; (d) `phase1_audio_service/app.py` had a `TypeError` compat shim that retried `vibevoice_provider.run(...)` without `audio_gcs_uri` when the *substring* `"audio_gcs_uri"` appeared in the exception message — so any unrelated `TypeError` inside `run` silently disabled URL streaming.
- **Root cause:** Residual defensive `try/except` blocks (reports R5, R6, R2 from `06-try-except.md`) left over from the pre-RTX split topology. The `audio_gcs_uri` shim (R7) was a one-release compat patch for a provider signature change that has since shipped; it is now a time-bomb. The `StageEvent` Pydantic model in `app.py` (report C2 from `03-unused-code.md`) had zero importers or instantiations — the wire shape for `VibeVoiceAsrResponse.stage_events` is already `list[dict[str, Any]]`, so the class was dead code.
- **Fix applied:**
  - `forced_aligner.py` (`ForcedAlignmentProvider.align_turns`): deleted the `try/except` around `self._ensure_model(device)` (lines 522-528) and the per-turn `try/except Exception: all_words = []` fallback (lines 558-569). `CLYPT_PHASE1_REQUIRE_FORCED_ALIGNMENT=1` in `backend/phase1_runtime/extract.py` remains the single enforcement point.
  - `vibevoice_vllm.py` (`_probe_duration`): deleted the `try/except Exception: return 0.0` (lines 200-212). Let `FileNotFoundError`, `subprocess.CalledProcessError`, `ValueError` propagate so RTF telemetry is never silently poisoned.
  - `phase1_audio_service/app.py` (`_run_vibevoice_asr`): deleted the `TypeError`-substring shim (lines 100-109). The helper now calls `vibevoice_provider.run(audio_path=..., audio_gcs_uri=...)` directly — `VibeVoiceVLLMProvider.run` supports the kwarg natively.
  - `phase1_audio_service/app.py`: deleted the unused `StageEvent` Pydantic class and removed it from `__all__`. Also dropped the now-unused `Field` import from `pydantic`.
- **Verification evidence:** `python -m pytest tests/backend/providers -q -x --tb=short -k "forced_aligner or vibevoice"` → 14 passed / 57 deselected. `python -m pytest tests/backend/runtime/phase1_audio_service -q -x --tb=short` → 13 passed. `python -m pytest tests/backend/phase1_runtime/test_runner.py -q` → 9 passed. Import smoke: `from backend.runtime.phase1_audio_service.app import app` and `from backend.providers.forced_aligner import ForcedAlignmentProvider` both succeed. `rg '\bStageEvent\b' .` returns zero matches after the deletion (the `StageEventLogger` type alias in `audio_host_client.py` is a distinct symbol and was intentionally left alone).
- **Follow-up guardrails:** Do not reintroduce empty-list or zero-duration fallbacks in Phase 1 audio providers — Phase 1 is contract to either succeed cleanly or crash cleanly, because downstream Phase 2-4 stages cannot tell the difference between "zero words" and "NFA crashed" when the audio chain silently degrades. Any future compat shim for provider signature changes must check the exception class, not substring-match the message, and must carry a TODO-dated removal note tied to the next release. If a new Pydantic model is added for the stage-event wire shape, update `VibeVoiceAsrResponse.stage_events` to reference it rather than leaving a dangling class in `app.py`.

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

## 2026-04-18 - Phase26 timeline adapters assumed dict payloads

- **Date/Time (UTC):** 2026-04-18 01:05-01:08
- **Subsystem:** Phase26 worker / Phase 1 artifact adaptation (`backend/pipeline/timeline/*`)
- **Environment:** Two-H200 + Modal topology, live replay of `mrbeastflagrant.mp4`
- **Symptom / Error signature:** Immediate Phase26 terminal failure after remote enqueue with `AttributeError: 'DiarizationPayload' object has no attribute 'get'`.
- **Root cause:** `build_canonical_timeline()` still assumed plain dict payloads and called `.get(...)`, but the new Phase 1 -> Phase26 handoff now passes typed Phase 1 Pydantic payload models (`DiarizationPayload`, `Phase1AudioAssets`, `EmotionSegmentsPayload`, `YamnetPayload`, `VisualPayload`).
- **Fix applied:** Added `backend/pipeline/timeline/payload_utils.py` to normalize dict/Pydantic payloads, updated all four timeline adapters (`timeline_builder.py`, `emotion_events.py`, `audio_events.py`, `tracklets.py`) to accept typed payload models, and added regression coverage in `tests/backend/pipeline/test_timeline_phase1.py`.
- **Verification evidence:** Local regression suite passed (`29 passed, 1 warning` including timeline + Phase24 worker coverage), and the next live replay advanced past the former enqueue-time crash into Phase 2 merge/boundary/media-prep work.
- **Follow-up guardrails:** Keep all Phase 1 -> Phase26 adaptation boundaries model-safe; do not assume JSON payloads have already been flattened to dicts.

## 2026-04-18 - Remote node-media-prep client rejected typed `phase1_audio`

- **Date/Time (UTC):** 2026-04-18 01:08-01:13
- **Subsystem:** Phase26 worker / remote node-media prep client (`backend/providers/node_media_prep_client.py`)
- **Environment:** Two-H200 + Modal topology, live replay of `mrbeastflagrant.mp4`
- **Symptom / Error signature:** Phase26 progressed through Phase 2 merge/boundary/semantic embeddings, then failed at media prep with `ValueError: RemoteNodeMediaPrepClient requires phase1_outputs.phase1_audio ... 'video_gcs_uri'`.
- **Root cause:** `_extract_video_gcs_uri()` still hard-coded a dict expectation for `phase1_outputs.phase1_audio`; the live worker was passing typed/attribute-based Phase 1 payload objects after Pydantic validation/model copies.
- **Fix applied:** Updated `_extract_video_gcs_uri()` to accept dict, Pydantic-model, and attribute-based payloads; aligned the Phase14 local multimodal fallback path in `backend/runtime/phase14_live.py`; added regression tests in `tests/backend/providers/test_remote_node_media_prep_client.py` for both `Phase1AudioAssets` and attr-only payload objects.
- **Verification evidence:** Local regression suite passed (`29 passed, 1 warning`), the live Phase26 worker logged `node-media-prep completed ... in 34501.7 ms`, and the same replay advanced into post-media-prep downstream embedding calls instead of failing at the handoff.
- **Follow-up guardrails:** Treat provider boundaries the same way as timeline adapters: accept typed runtime payload objects directly instead of assuming dict-shaped handoff data.

## 2026-04-18 - Vertex multimodal embeddings were pinned to a single region despite global support

- **Date/Time (UTC):** 2026-04-18 01:45-01:55
- **Subsystem:** Phase26 Vertex multimodal embeddings (`backend/providers/config.py`, `backend/providers/vertex.py`)
- **Environment:** Two-H200 + Modal topology, live replay of `mrbeastflagrant.mp4`, Vertex model `gemini-embedding-2-preview`
- **Symptom / Error signature:** One replay stalled in post-media-prep multimodal embeddings with repeated transient `429 RESOURCE_EXHAUSTED` and `500 INTERNAL` responses from Vertex while semantic text embeddings on the same run succeeded.
- **Root cause:** Clypt still defaulted `VERTEX_EMBEDDING_LOCATION` to `us-central1` even though Google now documents `gemini-embedding-2-preview` on the Vertex global endpoint and explicitly recommends the global endpoint to improve availability and reduce `429 RESOURCE_EXHAUSTED` relative to a single regional endpoint.
- **Fix applied:** Initially changed the repo default and documented baselines for `VERTEX_EMBEDDING_LOCATION` from `us-central1` to `global`, updated config/test expectations, and flipped the live Phase26 runtime env to `global` before restarting the worker.
- **Verification evidence:** Official Vertex docs list `Gemini Embedding 2 (gemini-embedding-2-preview)` in the Global endpoint table and note that global can reduce `429` errors; focused provider/runtime tests passed after the patch; one live replay completed Phase 2-4 successfully with no `429 RESOURCE_EXHAUSTED` responses during multimodal embeddings.
- **Follow-up guardrails:** Treat docs-only global support for preview embedding models as insufficient. Before changing embedding location defaults, validate direct live requests with the production project/service account.

## 2026-04-18 - `gemini-embedding-2-preview` global endpoint returned `404` in the live Clypt project

- **Date/Time (UTC):** 2026-04-18 01:57-02:20
- **Subsystem:** Phase26 Vertex embeddings (`backend/providers/config.py`, `backend/providers/vertex.py`)
- **Environment:** Two-H200 + Modal topology, Phase26 host `clypt-phase26-h200-nyc2`, live replay `job_a92796ffee4246a49d0a61af4dd36041`
- **Symptom / Error signature:** Phase 1 succeeded and Modal node-media-prep completed, then Phase 2 failed immediately with `404 NOT_FOUND ... Publisher Model projects/clypt-v3/locations/global/publishers/google/models/gemini-embedding-2-preview was not found or your project does not have access to it`.
- **Root cause:** Although Google’s locations doc lists `gemini-embedding-2-preview` under the global endpoint table, direct live requests from the Clypt project/service account reproduced `404` for both `embed_texts()` and `embed_media_uris()` on `global`, while the same calls succeeded on `us-central1`.
- **Fix applied:** Reverted the repo defaults, env baselines, and live Phase26 runtime env back to `VERTEX_EMBEDDING_LOCATION=us-central1`.
- **Verification evidence:** Direct host-side probes returned `global TEXT_ERR 404`, `global MEDIA_ERR 404`, `us-central1 TEXT_OK`, and `us-central1 MEDIA_OK`; focused provider/runtime tests passed after reverting the config defaults.
- **Follow-up guardrails:** Keep `gemini-embedding-2-preview` pinned to `us-central1` until Google’s live control plane actually accepts the project on `global`; do not trust the locations table alone for preview model rollout.

## 2026-04-18 - 480p node-media-prep regression forced CPU fallback on Modal

- **Date/Time (UTC):** 2026-04-18 01:56-02:20
- **Subsystem:** Modal node-media-prep ffmpeg pipeline (`backend/pipeline/semantics/media_embeddings.py`)
- **Environment:** Modal L4 `clypt-node-media-prep`, replay `job_a92796ffee4246a49d0a61af4dd36041`
- **Symptom / Error signature:** After changing node-media-prep clips from source-resolution to 480p, Modal time regressed from ~30.5s to ~48.0s and logs showed repeated `[phase2] GPU ffmpeg unavailable ... falling back to CPU encoder.` for every clip.
- **Root cause:** The new GPU filter graph `-vf scale_cuda=-2:480` broke ffmpeg format negotiation on the Modal L4 image (`Impossible to convert between the formats supported by the filter 'graph 0 input from stream 0:0' and the filter 'auto_scale_0'`), so every clip silently dropped to the CPU fallback path.
- **Fix applied:** Replaced the GPU filter-based scaling with decoder-side `h264_cuvid -resize WxH`, using an aspect-preserving even-sized 480p target computed from `ffprobe`; kept the CPU fallback path at explicit 480p scaling.
- **Verification evidence:** Reproduced the failing `scale_cuda=-2:480` graph inside the Modal L4 image, then verified that `h264_cuvid -resize 854x480 ... -c:v h264_nvenc` succeeds on the same image and preserves the GPU path. Focused media-prep tests passed after patching.
- **Follow-up guardrails:** For Modal L4 ffmpeg, prefer decoder-side resize for NVDEC/NVENC clip prep; validate any future GPU filter graph changes inside the actual Modal runtime image before shipping.

## 2026-04-18 - Canonical env baselines drifted from the live working Phase1 / Phase26 / Modal setup

- **Date/Time (UTC):** 2026-04-18 19:45-20:05
- **Subsystem:** Runtime env baselines and deploy docs
- **Environment:** Live working Phase1 H200 (`clypt-phase1-h200-nyc2`), Phase26 H200 (`clypt-phase26-h200-nyc2`), and Modal app `clypt-node-media-prep`
- **Symptom / Error signature:** Checked-in env/docs still showed stale placeholders or stale values after the live topology stabilized, including `CLYPT_PHASE24_DISPATCH_URL=http://<phase26-private-ip>:9300`, placeholder Modal endpoint URLs, `CLYPT_PHASE24_NODE_MEDIA_PREP_TIMEOUT_S=600` in `ENV_REFERENCE.md`, and stale Spanner defaults `clypt-phase14` / `clypt_phase14`.
- **Root cause:** Fresh-host fixes and runtime hotfixes were applied live first, but the canonical env files and docs were not fully resynced afterward.
- **Fix applied:** Synced [`docs/runtime/known-good-phase1-h200.env`](../docs/runtime/known-good-phase1-h200.env) to the current public Phase26 dispatch URL, synced [`docs/runtime/known-good-phase26-h200.env`](../docs/runtime/known-good-phase26-h200.env) to the current Modal endpoint, corrected [`docs/runtime/ENV_REFERENCE.md`](../docs/runtime/ENV_REFERENCE.md) for the Modal secret shape (`GOOGLE_APPLICATION_CREDENTIALS_JSON`), node-media-prep timeout (`1800`), worker id (`phase26-worker-1`), and active Spanner instance/database, and refreshed the Phase1 / Phase26 / Modal deploy runbooks with the current cross-team/public-networking and Modal app/secret details.
- **Verification evidence:** Captured the live env surfaces from `/etc/clypt-phase1/phase1.env`, `/etc/clypt/clypt-phase1-runtime.env`, `/etc/clypt-phase26/phase26.env`, and `/etc/clypt/clypt-phase26-runtime.env`; verified Modal metadata with `modal secret list` and `modal app list`; and confirmed the current deployed app name `clypt-node-media-prep` and secret name `clypt-node-media-prep`.
