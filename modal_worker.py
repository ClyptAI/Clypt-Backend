"""
Clypt Phase 1 — Modal GPU Worker
=================================
Serverless GPU microservice that performs deterministic multimodal extraction:

  1. NVIDIA Parakeet-TDT → word-level ASR with timestamps + punctuation
  2. YOLOv26 + BoT-SORT  → dense person tracking with persistent IDs
  3. TalkNet ASD         → active speaker binding (audio-visual sync)

Media is downloaded locally by the calling pipeline and sent to this worker
as raw bytes. This avoids YouTube bot detection on datacenter IPs.

Deployed via: modal deploy modal_worker.py
Test via:     modal serve modal_worker.py  (hot-reload dev mode)
"""

import modal

# ──────────────────────────────────────────────
# App + Image
# ──────────────────────────────────────────────
app = modal.App("clypt-sota-worker")
TRACKING_VOLUME = modal.Volume.from_name("clypt-phase1-tracking", create_if_missing=True)
MODEL_DEBUG_SECRET = modal.Secret.from_dict(
    {
        "CLYPT_MODEL_DEBUG": "1",
        "CLYPT_MODEL_DEBUG_EVERY": "10",
    }
)

ASR_MODEL_NAME = "nvidia/parakeet-tdt-1.1b"
TALKNET_MODEL_PATH = "/root/.cache/clypt/pretrain_TalkSet.model"
TALKNET_REPO_ROOT = "/root/talknet_asd"
YOLO_WEIGHTS_PATH = "yolo26s.pt"
YOLO_ONNX_PATH = "/root/.cache/clypt/yolo26s.onnx"
YOLO_ENGINE_PATH = "/root/.cache/clypt/yolo26s.engine"
YOLO_OPENVINO_DIR = "/root/.cache/clypt/yolo26s_openvino_model"
PHASE1_SCHEMA_VERSION = "2.0.0"
PHASE1_TASK_TYPE = "person_tracking"
PHASE1_COORDINATE_SPACE = "absolute_original_frame_xyxy"
PHASE1_GEOMETRY_TYPE = "aabb"
PHASE1_CLASS_TAXONOMY = {"0": "person"}


def download_asr_model():
    """Download Parakeet weights at image build time so they're cached."""
    import nemo.collections.asr as nemo_asr

    print(f"Downloading {ASR_MODEL_NAME} weights into container cache...")
    nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)


def download_yolo_model():
    """Download YOLO26 weights at image build time so they're cached."""
    from ultralytics import YOLO

    print("Downloading YOLO26s weights into container cache...")
    YOLO(YOLO_WEIGHTS_PATH)


def prepare_yolo_onnx_tensorrt():
    """Best-effort export to ONNX + TensorRT engine at build time.

    If TensorRT tooling is unavailable during image build, we keep the worker
    functional by falling back to PyTorch weights at runtime.
    """
    import os
    import subprocess
    from ultralytics import YOLO

    os.makedirs("/root/.cache/clypt", exist_ok=True)
    model = YOLO(YOLO_WEIGHTS_PATH)

    if not os.path.exists(YOLO_ONNX_PATH):
        try:
            print("Exporting YOLO26s to ONNX...")
            onnx_out = model.export(format="onnx", dynamic=True, simplify=True, opset=17)
            if isinstance(onnx_out, str) and os.path.exists(onnx_out) and onnx_out != YOLO_ONNX_PATH:
                os.replace(onnx_out, YOLO_ONNX_PATH)
        except Exception as e:
            print(f"Warning: ONNX export failed ({type(e).__name__}: {e})")

    if not os.path.exists(YOLO_ENGINE_PATH) and os.path.exists(YOLO_ONNX_PATH):
        try:
            print("Compiling YOLO26s ONNX -> TensorRT engine...")
            subprocess.run(
                [
                    "trtexec",
                    f"--onnx={YOLO_ONNX_PATH}",
                    f"--saveEngine={YOLO_ENGINE_PATH}",
                    "--fp16",
                    "--workspace=4096",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(
                "Warning: TensorRT engine compile unavailable; "
                f"using PyTorch/ONNX fallback ({type(e).__name__}: {e})"
            )

    if not os.path.exists(YOLO_OPENVINO_DIR):
        try:
            print("Exporting YOLO26s to OpenVINO...")
            model.export(format="openvino")
        except Exception as e:
            print(f"Warning: OpenVINO export failed ({type(e).__name__}: {e})")


def download_talknet_model():
    """Cache TalkNet checkpoint + architecture files at image build time."""
    import os
    import subprocess
    import urllib.request

    os.makedirs(os.path.dirname(TALKNET_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.join(TALKNET_REPO_ROOT, "model"), exist_ok=True)

    if not os.path.exists(TALKNET_MODEL_PATH):
        print("Downloading TalkNet weights...")
        file_id = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        subprocess.run(
            ["gdown", "--id", file_id, "-O", TALKNET_MODEL_PATH],
            check=True,
        )
    else:
        print("TalkNet checkpoint already cached.")

    files = {
        os.path.join(TALKNET_REPO_ROOT, "model", "talkNetModel.py"):
        "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/model/talkNetModel.py",
        os.path.join(TALKNET_REPO_ROOT, "model", "audioEncoder.py"):
        "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/model/audioEncoder.py",
        os.path.join(TALKNET_REPO_ROOT, "model", "visualEncoder.py"):
        "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/model/visualEncoder.py",
        os.path.join(TALKNET_REPO_ROOT, "model", "attentionLayer.py"):
        "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/model/attentionLayer.py",
        os.path.join(TALKNET_REPO_ROOT, "loss.py"):
        "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/loss.py",
    }
    for out_path, url in files.items():
        if not os.path.exists(out_path):
            print(f"Downloading TalkNet source file: {os.path.basename(out_path)}")
            urllib.request.urlretrieve(url, out_path)

    init_file = os.path.join(TALKNET_REPO_ROOT, "model", "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, "w", encoding="utf-8").close()


def download_insightface_model():
    """Cache InsightFace buffalo_l model files at image build time."""
    try:
        from insightface.app import FaceAnalysis

        print("Downloading InsightFace buffalo_l model pack into container cache...")
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:
        print(f"Warning: could not pre-cache InsightFace model pack ({type(e).__name__}: {e})")


clypt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "cmake",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
        "libsndfile1",
    )
    # Step 1: Core ML deps (cached once torch is resolved)
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "fastapi[standard]",
        "pydantic",
        "ultralytics",
        "lap>=0.5.12",
        "scikit-learn",
        "opencv-python-headless",
        "decord",
        "insightface",
        "onnxruntime-gpu",
        "onnx",
        "python_speech_features",
        "pandas",
        "scipy",
        "tqdm",
        "matplotlib",
        "imageio",
        "Pillow",
        "resampy",
        "soundfile",
        "gdown",
    )
    # Step 2: NeMo ASR
    .pip_install("nemo_toolkit[asr]")
    # Step 3: Cache model weights at build time
    .run_function(download_asr_model)
    .run_function(download_yolo_model)
    .run_function(prepare_yolo_onnx_tensorrt)
    .run_function(download_talknet_model)
    .run_function(download_insightface_model)
)


# ──────────────────────────────────────────────
# GPU Worker (class-based for VRAM persistence)
# ──────────────────────────────────────────────
@app.cls(
    image=clypt_image,
    gpu="H100",
    timeout=3600,
    max_containers=8,
    min_containers=0,
    scaledown_window=900,
    enable_memory_snapshot=False,
    secrets=[MODEL_DEBUG_SECRET],
    volumes={"/vol/clypt-chunks": TRACKING_VOLUME},
)
class ClyptWorker:

    @staticmethod
    def _load_talknet_checkpoint(model, loss_av, ckpt_path: str):
        """Load TalkNet checkpoint into model + AV classifier head."""
        import torch

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            for k in ("state_dict", "model_state_dict"):
                if k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(state)}")

        normalized_state = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module."):]
            normalized_state[key] = value

        model_state = {}
        for key, value in normalized_state.items():
            if key.startswith("model."):
                model_state[key[len("model."):]] = value
            elif key.startswith("module."):
                model_state[key[len("module."):]] = value
            elif key in model.state_dict():
                model_state[key] = value

        loss_state = {}
        for key, value in normalized_state.items():
            if key.startswith("lossAV."):
                loss_state[key[len("lossAV."):]] = value
            elif key in loss_av.state_dict():
                loss_state[key] = value

        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "TalkNet checkpoint mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        if loss_state:
            loss_av.load_state_dict(loss_state, strict=False)

    @staticmethod
    def _build_track_indexes(
        tracks: list[dict],
    ) -> tuple[dict[int, list[dict]], dict[str, list[dict]]]:
        """Build per-frame and per-track indexes (sorted by frame)."""
        from collections import defaultdict

        frame_to_dets: dict[int, list[dict]] = defaultdict(list)
        track_to_dets: dict[str, list[dict]] = defaultdict(list)
        for d in tracks:
            fi = int(d.get("frame_idx", -1))
            tid = str(d.get("track_id", ""))
            if fi < 0 or not tid:
                continue
            frame_to_dets[fi].append(d)
            track_to_dets[tid].append(d)

        for tid in list(track_to_dets.keys()):
            track_to_dets[tid].sort(key=lambda x: int(x["frame_idx"]))
        return frame_to_dets, track_to_dets

    @staticmethod
    def _validate_tracking_contract(tracks: list[dict]):
        """Contract-driven schema validation for phase handoff."""
        required = (
            "frame_idx",
            "track_id",
            "class_id",
            "label",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "geometry_type",
        )
        for i, d in enumerate(tracks):
            for k in required:
                if k not in d:
                    raise RuntimeError(f"Track contract violation at index {i}: missing '{k}'")
            if float(d["x2"]) <= float(d["x1"]) or float(d["y2"]) <= float(d["y1"]):
                raise RuntimeError(f"Track contract violation at index {i}: invalid xyxy box")

    @staticmethod
    def _tracking_contract_pass_rate(tracks: list[dict]) -> float:
        """Compute explicit schema pass rate for rollout gating metrics."""
        required = (
            "frame_idx",
            "track_id",
            "class_id",
            "label",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "geometry_type",
        )
        if not tracks:
            return 1.0
        passed = 0
        for d in tracks:
            ok = True
            for k in required:
                if k not in d:
                    ok = False
                    break
            if ok:
                if float(d["x2"]) <= float(d["x1"]) or float(d["y2"]) <= float(d["y1"]):
                    ok = False
            if ok:
                passed += 1
        return float(passed / max(1, len(tracks)))

    def _detect_face_in_person_det(self, frame_rgb, det: dict):
        """Detect a face inside a person bbox and return (112x112 crop, relative anchor)."""
        import cv2
        import numpy as np

        if self.face_analyzer is None or frame_rgb is None:
            return None, None

        fh, fw = frame_rgb.shape[:2]
        cx = float(det.get("x_center", 0.0))
        cy = float(det.get("y_center", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))
        if w <= 1e-6 or h <= 1e-6:
            return None, None

        # Broad ROI around upper body/head for robust profile-face recall.
        x1 = max(0, int(cx - 0.62 * w))
        x2 = min(fw, int(cx + 0.62 * w))
        y1 = max(0, int(cy - 0.82 * h))
        y2 = min(fh, int(cy + 0.30 * h))
        if x2 <= x1 or y2 <= y1:
            return None, None

        roi_rgb = frame_rgb[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            return None, None

        try:
            faces = self.face_analyzer.get(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            return None, None
        if not faces:
            return None, None

        best_face = max(
            faces,
            key=lambda f: float(
                max(0.0, f.bbox[2] - f.bbox[0])
                * max(0.0, f.bbox[3] - f.bbox[1])
                * getattr(f, "det_score", 1.0)
            ),
        )
        fb = np.asarray(best_face.bbox, dtype=np.float32)
        fx1, fy1, fx2, fy2 = fb.tolist()
        fw_face = max(2.0, fx2 - fx1)
        fh_face = max(2.0, fy2 - fy1)
        fx1 = max(0, int(np.floor(fx1 - 0.10 * fw_face)))
        fy1 = max(0, int(np.floor(fy1 - 0.10 * fh_face)))
        fx2 = min(roi_rgb.shape[1], int(np.ceil(fx2 + 0.10 * fw_face)))
        fy2 = min(roi_rgb.shape[0], int(np.ceil(fy2 + 0.10 * fh_face)))
        if fx2 <= fx1 or fy2 <= fy1:
            return None, None

        gx1 = x1 + fx1
        gy1 = y1 + fy1
        gx2 = x1 + fx2
        gy2 = y1 + fy2
        if gx2 <= gx1 or gy2 <= gy1:
            return None, None

        face_rgb = frame_rgb[gy1:gy2, gx1:gx2]
        if face_rgb.size == 0:
            return None, None
        face_rgb = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        anchor = {
            "x_offset": (float(gx1) - cx) / w,
            "y_offset": (float(gy1) - cy) / h,
            "w_ratio": (float(gx2 - gx1)) / w,
            "h_ratio": (float(gy2 - gy1)) / h,
        }
        return face_rgb, anchor

    @staticmethod
    def _interpolate_track_detections(dets: list[dict], max_gap: int = 5) -> dict[int, dict]:
        """Fill short detection gaps via linear interpolation."""
        best_by_frame: dict[int, dict] = {}
        for d in dets:
            fi = int(d.get("frame_idx", -1))
            if fi < 0:
                continue
            old = best_by_frame.get(fi)
            if old is None or float(d.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
                best_by_frame[fi] = d

        if not best_by_frame:
            return {}

        out = dict(best_by_frame)
        frames = sorted(best_by_frame.keys())
        for left, right in zip(frames, frames[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            dl = best_by_frame[left]
            dr = best_by_frame[right]
            for fi in range(left + 1, right):
                alpha = (fi - left) / float(right - left)
                out[fi] = {
                    "frame_idx": fi,
                    "track_id": str(dl.get("track_id", "")),
                    "x_center": (1.0 - alpha) * float(dl.get("x_center", 0.0))
                    + alpha * float(dr.get("x_center", 0.0)),
                    "y_center": (1.0 - alpha) * float(dl.get("y_center", 0.0))
                    + alpha * float(dr.get("y_center", 0.0)),
                    "width": max(
                        1.0,
                        (1.0 - alpha) * float(dl.get("width", 1.0))
                        + alpha * float(dr.get("width", 1.0)),
                    ),
                    "height": max(
                        1.0,
                        (1.0 - alpha) * float(dl.get("height", 1.0))
                        + alpha * float(dr.get("height", 1.0)),
                    ),
                    "confidence": 0.5
                    * min(float(dl.get("confidence", 0.0)), float(dr.get("confidence", 0.0))),
                    "interpolated": True,
                }
        return out

    @staticmethod
    def _tensor_debug_stats(name, tensor):
        """Compact tensor diagnostics for model-debug logging."""
        import torch

        t = tensor.detach().float()
        if t.numel() == 0:
            return f"{name}: empty"
        finite_mask = torch.isfinite(t)
        finite_count = int(finite_mask.sum().item())
        total = int(t.numel())
        if finite_count == 0:
            return f"{name}: shape={tuple(t.shape)} finite=0/{total}"
        tf = t[finite_mask]
        return (
            f"{name}: shape={tuple(t.shape)} "
            f"finite={finite_count}/{total} "
            f"min={float(tf.min()):.5f} max={float(tf.max()):.5f} "
            f"mean={float(tf.mean()):.5f} std={float(tf.std(unbiased=False)):.5f}"
        )

    def _talknet_forward_scores(
        self,
        audio_t,
        visual_t,
    ):
        """Batched TalkNet forward.

        Returns per-frame speaking probabilities in [0, 1].
        """
        import torch

        b, t = visual_t.shape[:2]
        self._talknet_debug_calls = getattr(self, "_talknet_debug_calls", 0) + 1
        debug_now = (
            getattr(self, "model_debug", False)
            and (
                self._talknet_debug_calls
                % max(1, getattr(self, "model_debug_every", 20))
                == 1
            )
        )
        if debug_now:
            print("  [TALKNET DEBUG] Input tensors:")
            print("   " + self._tensor_debug_stats("audio_t", audio_t))
            print("   " + self._tensor_debug_stats("visual_t", visual_t))

        audio_embed = self.talknet_model.forward_audio_frontend(audio_t)
        visual_embed = self.talknet_model.forward_visual_frontend(visual_t)
        if debug_now:
            print("   " + self._tensor_debug_stats("audio_embed_pre_xattn", audio_embed))
            print("   " + self._tensor_debug_stats("visual_embed_pre_xattn", visual_embed))

        audio_embed, visual_embed = self.talknet_model.forward_cross_attention(
            audio_embed, visual_embed
        )
        if debug_now:
            print("   " + self._tensor_debug_stats("audio_embed_post_xattn", audio_embed))
            print("   " + self._tensor_debug_stats("visual_embed_post_xattn", visual_embed))

        outs_av = self.talknet_model.forward_audio_visual_backend(audio_embed, visual_embed)
        if outs_av.shape[0] != b * t:
            raise RuntimeError(
                f"TalkNet output shape mismatch: outs_av={tuple(outs_av.shape)}, expected first dim {b*t}"
            )
        if debug_now:
            print("   " + self._tensor_debug_stats("outs_av", outs_av))

        av_logits = self.talknet_loss_av.FC(outs_av)
        av_prob = torch.softmax(av_logits, dim=-1)[:, 1].reshape(b, t)
        if debug_now:
            print("   " + self._tensor_debug_stats("av_logits", av_logits))
            print("   " + self._tensor_debug_stats("av_prob", av_prob))
            print(f"  [TALKNET DEBUG] forward_call={self._talknet_debug_calls} b={b} t={t}")
        return av_prob

    @modal.enter()
    def load_model(self):
        """Load Parakeet, YOLO, and TalkNet into GPU VRAM."""
        import os
        import sys
        import nemo.collections.asr as nemo_asr
        import torch
        from insightface.app import FaceAnalysis
        from ultralytics import YOLO
        from omegaconf import open_dict

        self.model_debug = os.getenv("CLYPT_MODEL_DEBUG", "0") == "1"
        self.model_debug_every = int(os.getenv("CLYPT_MODEL_DEBUG_EVERY", "20"))
        self._talknet_debug_calls = 0
        if self.model_debug:
            print(
                "Model debug logging enabled: "
                f"CLYPT_MODEL_DEBUG=1, every={self.model_debug_every} TalkNet forwards"
            )

        # --- Load Parakeet ---
        print(f"Loading {ASR_MODEL_NAME} into GPU VRAM...")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=ASR_MODEL_NAME
        )
        self.asr_model.eval()

        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.preserve_alignments = True
            decoding_cfg.compute_timestamps = True
            decoding_cfg.word_seperator = " "
        self.asr_model.change_decoding_strategy(decoding_cfg)

        self.time_stride = 8 * self.asr_model.cfg.preprocessor.window_stride

        # --- Load YOLOv26 ---
        print("Loading YOLO26s into GPU VRAM...")
        self.yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        try:
            # Avoid runtime TensorRT export on cold start; only load an existing
            # prebuilt engine if it is already present.
            if os.path.exists(YOLO_ENGINE_PATH):
                print("Loading YOLO26s TensorRT engine...")
                self.yolo_model = YOLO(YOLO_ENGINE_PATH)
            else:
                print("Using PyTorch YOLO26s weights (no TensorRT engine found).")
        except Exception as e:
            print(
                "Warning: TensorRT engine load failed; using PyTorch YOLO model "
                f"({type(e).__name__}: {e})"
            )

        self.gpu_device = torch.device("cuda")
        self.face_analyzer = None
        try:
            print("Loading InsightFace (buffalo_l) models...")
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # Lower threshold to retain extreme profile faces in podcast footage.
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.15)
            print("InsightFace ready.")
        except Exception as e:
            print(
                "Warning: InsightFace initialization failed; clustering will use histogram fallback "
                f"({type(e).__name__}: {e})"
            )

        # --- Load TalkNet ---
        self.talknet_model = None
        self.talknet_loss_av = None
        try:
            print("Loading TalkNet model into GPU VRAM...")
            if TALKNET_REPO_ROOT not in sys.path:
                sys.path.append(TALKNET_REPO_ROOT)
            from model.talkNetModel import talkNetModel
            from loss import lossAV

            self.talknet_model = talkNetModel()
            self.talknet_loss_av = lossAV()
            self._load_talknet_checkpoint(
                self.talknet_model,
                self.talknet_loss_av,
                TALKNET_MODEL_PATH,
            )
            self.talknet_model = self.talknet_model.to(self.gpu_device)
            self.talknet_model.eval()
            self.talknet_loss_av = self.talknet_loss_av.to(self.gpu_device)
            self.talknet_loss_av.eval()
            print("TalkNet ready.")
        except Exception as e:
            # Keep worker alive; binding step falls back if this fails at runtime.
            self.talknet_model = None
            self.talknet_loss_av = None
            print(
                "Warning: failed to load TalkNet checkpoint "
                f"({type(e).__name__}: {e})"
            )

        print("Models ready in VRAM.")

    # ──────────────────────────────────────────
    # ASR (NVIDIA Parakeet-TDT-1.1B)
    # ──────────────────────────────────────────
    def _run_asr(self, audio_wav_path: str) -> list[dict]:
        """Run Parakeet ASR on a 16kHz mono WAV and return word-level timestamps."""
        print("Running Parakeet ASR inference...")
        hypotheses = self.asr_model.transcribe(
            [audio_wav_path],
            return_hypotheses=True,
        )

        # RNNT/TDT models return a tuple (best_hypotheses, all_hypotheses)
        if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        hypothesis = hypotheses[0]
        words = []

        # NeMo uses 'timestep' in older versions, 'timestamp' in newer
        ts_dict = None
        for attr in ("timestep", "timestamp"):
            val = getattr(hypothesis, attr, None)
            if isinstance(val, dict) and "word" in val:
                ts_dict = val
                break

        if ts_dict:
            for stamp in ts_dict["word"]:
                start_s = stamp["start_offset"] * self.time_stride
                end_s = stamp["end_offset"] * self.time_stride
                word = stamp.get("char") or stamp.get("word", "")
                words.append({
                    "word": word,
                    "start_time_ms": int(start_s * 1000),
                    "end_time_ms": int(end_s * 1000),
                    "speaker_track_id": None,  # populated by TalkNet later
                })
        else:
            print("Warning: no timestamp dict with 'word' key found")

        print(f"ASR complete: {len(words)} words transcribed")
        return words

    # ──────────────────────────────────────────
    # Visual tracking (YOLOv11 + BoT-SORT)
    # ──────────────────────────────────────────
    def _ensure_h264(self, video_path: str) -> str:
        """Re-encode to H.264 if the video uses AV1 or another codec OpenCV can't decode."""
        import subprocess, os
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        codec = result.stdout.strip()
        print(f"  Video codec: {codec}")
        if codec in ("av1", "vp9", "hevc", "h265"):
            h264_path = video_path.replace(".mp4", "_h264.mp4")
            print(f"  Re-encoding {codec} → H.264 (NVENC preferred)...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p4",
                        "-cq",
                        "23",
                        "-an",
                        h264_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                # Fallback in case nvenc isn't available in the runtime FFmpeg build.
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "23",
                        "-an",
                        h264_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print(f"  Re-encoded: {os.path.getsize(h264_path) / 1e6:.1f} MB")
            return h264_path
        return video_path

    @staticmethod
    def _probe_video_meta(video_path: str) -> dict:
        """Return fps/size/frame_count metadata for a local video."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "fps": 25.0,
                "width": 0,
                "height": 0,
                "total_frames": 0,
                "duration_s": 0.0,
            }
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_s": float(total_frames / max(1e-6, fps)),
        }

    @staticmethod
    def _build_chunk_plan(total_frames: int, fps: float) -> list[dict]:
        """Build overlapping chunk windows for parallel tracking."""
        chunk_seconds = 60.0
        overlap_seconds = 2.0
        chunk_frames = max(1, int(round(chunk_seconds * fps)))
        overlap_frames = max(1, int(round(overlap_seconds * fps)))
        stride = max(1, chunk_frames - overlap_frames)

        plan = []
        start = 0
        idx = 0
        while start < total_frames:
            end = min(total_frames, start + chunk_frames)
            plan.append(
                {
                    "chunk_idx": idx,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "overlap_frames": int(overlap_frames),
                }
            )
            if end >= total_frames:
                break
            idx += 1
            start += stride
        return plan

    @staticmethod
    def _ensure_botsort_reid_yaml() -> str:
        """Write a strict BoT-SORT config with ReID + GMC enabled."""
        import os

        out = "/tmp/clypt/botsort_reid.yaml"
        os.makedirs("/tmp/clypt", exist_ok=True)
        if not os.path.exists(out):
            with open(out, "w", encoding="utf-8") as f:
                f.write(
                    "tracker_type: botsort\n"
                    "track_high_thresh: 0.35\n"
                    "track_low_thresh: 0.1\n"
                    "new_track_thresh: 0.6\n"
                    "track_buffer: 45\n"
                    "match_thresh: 0.78\n"
                    "fuse_score: True\n"
                    "gmc_method: sparseOptFlow\n"
                    "proximity_thresh: 0.5\n"
                    "appearance_thresh: 0.25\n"
                    "with_reid: True\n"
                    "model: auto\n"
                )
        return out

    @staticmethod
    def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        w = max(1.0, float(x2) - float(x1))
        h = max(1.0, float(y2) - float(y1))
        cx = float(x1) + 0.5 * w
        cy = float(y1) + 0.5 * h
        return cx, cy, w, h

    @staticmethod
    def _xyxy_abs_to_xywhn(
        x1: float, y1: float, x2: float, y2: float, width: int, height: int
    ) -> tuple[float, float, float, float]:
        """Deterministic absolute-xyxy -> normalized-xywh conversion."""
        w = max(1.0, float(width))
        h = max(1.0, float(height))
        cx, cy, bw, bh = ClyptWorker._xyxy_to_xywh(x1, y1, x2, y2)
        return cx / w, cy / h, bw / w, bh / h

    @staticmethod
    def _xywhn_to_xyxy_abs(
        xcn: float, ycn: float, wn: float, hn: float, width: int, height: int
    ) -> tuple[float, float, float, float]:
        """Deterministic normalized-xywh -> absolute-xyxy conversion."""
        w = max(1.0, float(width))
        h = max(1.0, float(height))
        cx = float(xcn) * w
        cy = float(ycn) * h
        bw = max(1.0, float(wn) * w)
        bh = max(1.0, float(hn) * h)
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        return x1, y1, x2, y2

    @staticmethod
    def _compute_letterbox_meta(
        orig_w: int, orig_h: int, input_w: int, input_h: int
    ) -> dict:
        """Cache letterbox scale/padding metadata for explicit inverse-affine writes."""
        ow = max(1, int(orig_w))
        oh = max(1, int(orig_h))
        iw = max(1, int(input_w))
        ih = max(1, int(input_h))
        scale = min(iw / float(ow), ih / float(oh))
        new_w = int(round(ow * scale))
        new_h = int(round(oh * scale))
        pad_x = 0.5 * (iw - new_w)
        pad_y = 0.5 * (ih - new_h)
        return {
            "orig_w": ow,
            "orig_h": oh,
            "input_w": iw,
            "input_h": ih,
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }

    @staticmethod
    def _forward_letterbox_xyxy(
        x1: float, y1: float, x2: float, y2: float, lb: dict
    ) -> tuple[float, float, float, float]:
        """Map absolute box coords into letterboxed tensor space."""
        s = float(lb["scale"])
        px = float(lb["pad_x"])
        py = float(lb["pad_y"])
        return (x1 * s + px, y1 * s + py, x2 * s + px, y2 * s + py)

    @staticmethod
    def _inverse_letterbox_xyxy(
        lx1: float, ly1: float, lx2: float, ly2: float, lb: dict
    ) -> tuple[float, float, float, float]:
        """Map letterboxed tensor coords back to absolute original-frame coords."""
        s = max(1e-9, float(lb["scale"]))
        px = float(lb["pad_x"])
        py = float(lb["pad_y"])
        x1 = (lx1 - px) / s
        y1 = (ly1 - py) / s
        x2 = (lx2 - px) / s
        y2 = (ly2 - py) / s
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        ub = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return float(inter / max(1e-6, ua + ub - inter))

    @staticmethod
    def _cosine_dist(a, b) -> float:
        import numpy as np

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den <= 1e-9:
            return 1.0
        return 1.0 - float(np.dot(a, b) / den)

    @staticmethod
    def _propagate_gaps_in_tracklets(tracks: list[dict], max_gap: int = 2) -> list[dict]:
        """Lightweight propagation over short frame gaps (confidence-guided frame skipping fill)."""
        from collections import defaultdict

        by_tid = defaultdict(list)
        for d in tracks:
            by_tid[str(d.get("track_id", ""))].append(d)

        out = list(tracks)
        for tid, dets in by_tid.items():
            dets = sorted(dets, key=lambda x: int(x["frame_idx"]))
            for left, right in zip(dets, dets[1:]):
                lf = int(left["frame_idx"])
                rf = int(right["frame_idx"])
                gap = rf - lf - 1
                if gap <= 0 or gap > max_gap:
                    continue
                lc = float(left.get("confidence", 0.0))
                rc = float(right.get("confidence", 0.0))
                if min(lc, rc) < 0.15:
                    continue
                for fi in range(lf + 1, rf):
                    alpha = (fi - lf) / float(rf - lf)
                    l_local = int(left.get("local_frame_idx", lf))
                    r_local = int(right.get("local_frame_idx", rf))
                    local_fi = int(round((1.0 - alpha) * l_local + alpha * r_local))
                    x1 = (1 - alpha) * float(left["x1"]) + alpha * float(right["x1"])
                    y1 = (1 - alpha) * float(left["y1"]) + alpha * float(right["y1"])
                    x2 = (1 - alpha) * float(left["x2"]) + alpha * float(right["x2"])
                    y2 = (1 - alpha) * float(left["y2"]) + alpha * float(right["y2"])
                    cx, cy, w, h = ClyptWorker._xyxy_to_xywh(x1, y1, x2, y2)
                    out.append(
                        {
                            **left,
                            "frame_idx": int(fi),
                            "local_frame_idx": int(local_fi),
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "x_center": float(cx),
                            "y_center": float(cy),
                            "width": float(w),
                            "height": float(h),
                            "confidence": float(min(lc, rc) * 0.85),
                            "source": "propagated",
                        }
                    )
        return out

    def _compute_track_embeddings_for_chunk(
        self,
        chunk_video_path: str,
        chunk_tracks: list[dict],
    ) -> dict[str, list[float]]:
        """Compute per-track ArcFace embeddings on sampled person ROIs."""
        import cv2
        import numpy as np
        from collections import defaultdict

        if self.face_analyzer is None or not chunk_tracks:
            return {}

        by_tid = defaultdict(list)
        for d in chunk_tracks:
            by_tid[str(d["track_id"])].append(d)
        samples_by_tid = {}
        for tid, dets in by_tid.items():
            ranked = sorted(
                dets,
                key=lambda d: float(d.get("confidence", 0.0))
                * max(1.0, float(d.get("width", 1.0)) * float(d.get("height", 1.0))),
                reverse=True,
            )
            seen = set()
            sampled = []
            for d in ranked:
                fi = int(d["local_frame_idx"])
                if fi in seen:
                    continue
                seen.add(fi)
                sampled.append(d)
                if len(sampled) >= 4:
                    break
            samples_by_tid[tid] = sampled

        cap = cv2.VideoCapture(chunk_video_path)
        if not cap.isOpened():
            return {}

        out = {}
        for tid, sampled in samples_by_tid.items():
            vecs = []
            for d in sampled:
                fi = int(d["local_frame_idx"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                x1 = max(0, min(w - 1, int(round(float(d["x1"])))))
                y1 = max(0, min(h - 1, int(round(float(d["y1"])))))
                x2 = max(0, min(w, int(round(float(d["x2"])))))
                y2 = max(0, min(h, int(round(float(d["y2"])))))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                try:
                    faces = self.face_analyzer.get(crop)
                except Exception:
                    continue
                if not faces:
                    continue
                best_face = max(
                    faces,
                    key=lambda f: float(
                        max(0.0, f.bbox[2] - f.bbox[0])
                        * max(0.0, f.bbox[3] - f.bbox[1])
                        * getattr(f, "det_score", 1.0)
                    ),
                )
                emb = np.asarray(getattr(best_face, "normed_embedding", None), dtype=np.float32)
                if emb.size > 0:
                    vecs.append(emb)
            if vecs:
                out[tid] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32).tolist()
        cap.release()
        return out

    def _roi_refine_interpolated_tracks(
        self,
        chunk_video_path: str,
        tracks: list[dict],
        model,
        width: int,
        height: int,
        infer_imgsz: int,
    ) -> tuple[list[dict], int]:
        """Refine propagated detections using detector-on-ROI rather than full-frame inference."""
        import cv2
        import numpy as np
        from collections import defaultdict

        if not tracks:
            return tracks, 0

        # Group propagated detections by local frame for single frame decode.
        by_local_fi: dict[int, list[int]] = defaultdict(list)
        for i, d in enumerate(tracks):
            if str(d.get("source", "")) != "propagated":
                continue
            fi = int(d.get("local_frame_idx", -1))
            if fi < 0:
                continue
            by_local_fi[fi].append(i)

        if not by_local_fi:
            return tracks, 0

        cap = cv2.VideoCapture(chunk_video_path)
        if not cap.isOpened():
            return tracks, 0

        refined = 0
        for local_fi in sorted(by_local_fi.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            fh, fw = frame.shape[:2]
            for idx in by_local_fi[local_fi]:
                d = tracks[idx]
                x1 = float(d.get("x1", 0.0))
                y1 = float(d.get("y1", 0.0))
                x2 = float(d.get("x2", x1 + 1.0))
                y2 = float(d.get("y2", y1 + 1.0))
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)

                # ROI predicted around propagated box (expanded context).
                rx1 = max(0, int(np.floor(x1 - 0.35 * bw)))
                ry1 = max(0, int(np.floor(y1 - 0.35 * bh)))
                rx2 = min(fw, int(np.ceil(x2 + 0.35 * bw)))
                ry2 = min(fh, int(np.ceil(y2 + 0.35 * bh)))
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue

                try:
                    pred = model.predict(
                        source=roi,
                        classes=[0],
                        conf=0.15,
                        verbose=False,
                        imgsz=infer_imgsz,
                    )
                except Exception:
                    continue
                if not pred:
                    continue
                r = pred[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                roi_boxes = r.boxes.xyxy.cpu().numpy()
                roi_confs = r.boxes.conf.cpu().numpy()
                best = None
                for bxyxy, conf in zip(roi_boxes, roi_confs):
                    bx1, by1, bx2, by2 = [float(v) for v in bxyxy]
                    gx1 = min(max(0.0, bx1 + rx1), float(max(0, width - 1)))
                    gy1 = min(max(0.0, by1 + ry1), float(max(0, height - 1)))
                    gx2 = min(max(gx1 + 1.0, bx2 + rx1), float(max(1, width)))
                    gy2 = min(max(gy1 + 1.0, by2 + ry1), float(max(1, height)))
                    iou = self._bbox_iou_xyxy((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                    score = (0.7 * iou) + (0.3 * float(conf))
                    if best is None or score > best[0]:
                        best = (score, gx1, gy1, gx2, gy2, float(conf), iou)

                if best is None:
                    continue
                _, gx1, gy1, gx2, gy2, conf_best, iou_best = best
                if iou_best < 0.2 and conf_best < 0.45:
                    continue

                cx, cy, nw, nh = self._xyxy_to_xywh(gx1, gy1, gx2, gy2)
                d["x1"] = float(gx1)
                d["y1"] = float(gy1)
                d["x2"] = float(gx2)
                d["y2"] = float(gy2)
                d["x_center"] = float(cx)
                d["y_center"] = float(cy)
                d["width"] = float(nw)
                d["height"] = float(nh)
                d["confidence"] = float(max(float(d.get("confidence", 0.0)), conf_best))
                d["source"] = "roi_refine"
                refined += 1

        cap.release()
        return tracks, refined

    def _track_single_chunk(
        self,
        video_path: str,
        meta: dict,
        chunk: dict,
        tracker_cfg: str,
        chunk_dir: str,
    ) -> dict:
        """Track one chunk independently and emit per-chunk NDJSON."""
        import json
        import os
        import subprocess
        import time
        from ultralytics import YOLO

        fps = float(meta["fps"])
        width = int(meta["width"])
        height = int(meta["height"])
        infer_imgsz = max(320, int(os.getenv("CLYPT_YOLO_IMGSZ", "640")))
        start_f = int(chunk["start_frame"])
        end_f = int(chunk["end_frame"])
        start_s = start_f / max(1e-6, fps)
        dur_s = max(1.0 / max(1e-6, fps), (end_f - start_f) / max(1e-6, fps))
        chunk_idx = int(chunk["chunk_idx"])

        chunk_video_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.mp4")
        ffmpeg_result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start_s:.3f}",
                "-i", video_path,
                "-t", f"{dur_s:.3f}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-an",
                chunk_video_path,
            ],
            capture_output=True,
        )
        if ffmpeg_result.returncode != 0:
            stderr_msg = ffmpeg_result.stderr.decode(errors='replace')
            raise RuntimeError(f"ffmpeg chunk failed (exit {ffmpeg_result.returncode}): {stderr_msg}")

        model_path = YOLO_ENGINE_PATH if os.path.exists(YOLO_ENGINE_PATH) else YOLO_WEIGHTS_PATH
        model = YOLO(model_path)
        stride = 2 if dur_s >= 50.0 else 1
        started = time.time()
        processed_frames = 0
        lb_meta = self._compute_letterbox_meta(width, height, infer_imgsz, infer_imgsz)

        results = model.track(
            source=chunk_video_path,
            tracker=tracker_cfg,
            persist=True,
            classes=[0],
            stream=True,
            verbose=False,
            vid_stride=stride,
            imgsz=infer_imgsz,
        )

        tracks = []
        for local_fi, r in enumerate(results):
            processed_frames += 1
            global_fi = start_f + local_fi * stride
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            obb_polys = None
            if getattr(r, "obb", None) is not None and getattr(r.obb, "xyxyxyxy", None) is not None:
                try:
                    obb_polys = r.obb.xyxyxyxy.cpu().numpy()
                except Exception:
                    obb_polys = None

            for i, (xyxy, tid_raw, conf) in enumerate(zip(boxes_xyxy, ids, confs)):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                # Deterministic conversion path:
                # 1) absolute xyxy -> normalized xywh
                # 2) normalized xywh -> absolute xyxy
                # 3) explicit letterbox forward+inverse affine
                xcn, ycn, wn, hn = self._xyxy_abs_to_xywhn(x1, y1, x2, y2, width, height)
                x1, y1, x2, y2 = self._xywhn_to_xyxy_abs(xcn, ycn, wn, hn, width, height)
                lx1, ly1, lx2, ly2 = self._forward_letterbox_xyxy(x1, y1, x2, y2, lb_meta)
                x1, y1, x2, y2 = self._inverse_letterbox_xyxy(lx1, ly1, lx2, ly2, lb_meta)
                x1 = min(max(0.0, x1), float(max(0, width - 1)))
                y1 = min(max(0.0, y1), float(max(0, height - 1)))
                x2 = min(max(x1 + 1.0, x2), float(max(1, width)))
                y2 = min(max(y1 + 1.0, y2), float(max(1, height)))
                cx, cy, bw, bh = self._xyxy_to_xywh(x1, y1, x2, y2)
                out = {
                    "frame_idx": int(global_fi),
                    "local_frame_idx": int(local_fi),
                    "chunk_idx": int(chunk_idx),
                    "track_id": f"chunk_{chunk_idx}_track_{int(tid_raw)}",
                    "local_track_id": int(tid_raw),
                    "class_id": 0,
                    "label": "person",
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "x_center": float(cx),
                    "y_center": float(cy),
                    "width": float(bw),
                    "height": float(bh),
                    "source": "detector",
                    "geometry_type": "aabb",
                    "bbox_norm_xywh": {
                        "x_center": float(xcn),
                        "y_center": float(ycn),
                        "width": float(wn),
                        "height": float(hn),
                    },
                }
                if obb_polys is not None and i < len(obb_polys):
                    pts = obb_polys[i].reshape(-1, 2).tolist()
                    out["geometry_type"] = "obb"
                    out["polygon"] = [[float(px), float(py)] for px, py in pts]
                tracks.append(out)

        if stride > 1:
            # Confidence-guided fallback: if sparse pass has many weak detections,
            # rerun dense tracking to re-anchor.
            weak_ratio = 0.0
            if tracks:
                weak_ratio = sum(1 for d in tracks if float(d.get("confidence", 0.0)) < 0.3) / len(tracks)
            if weak_ratio > 0.25:
                dense_results = model.track(
                    source=chunk_video_path,
                    tracker=tracker_cfg,
                    persist=True,
                    classes=[0],
                    stream=True,
                    verbose=False,
                    vid_stride=1,
                    imgsz=infer_imgsz,
                )
                tracks = []
                for local_fi, r in enumerate(dense_results):
                    processed_frames += 1
                    global_fi = start_f + local_fi
                    if r.boxes is None or r.boxes.id is None:
                        continue
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    for xyxy, tid_raw, conf in zip(boxes_xyxy, ids, confs):
                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        xcn, ycn, wn, hn = self._xyxy_abs_to_xywhn(x1, y1, x2, y2, width, height)
                        x1, y1, x2, y2 = self._xywhn_to_xyxy_abs(xcn, ycn, wn, hn, width, height)
                        lx1, ly1, lx2, ly2 = self._forward_letterbox_xyxy(x1, y1, x2, y2, lb_meta)
                        x1, y1, x2, y2 = self._inverse_letterbox_xyxy(lx1, ly1, lx2, ly2, lb_meta)
                        x1 = min(max(0.0, x1), float(max(0, width - 1)))
                        y1 = min(max(0.0, y1), float(max(0, height - 1)))
                        x2 = min(max(x1 + 1.0, x2), float(max(1, width)))
                        y2 = min(max(y1 + 1.0, y2), float(max(1, height)))
                        cx, cy, bw, bh = self._xyxy_to_xywh(x1, y1, x2, y2)
                        tracks.append(
                            {
                                "frame_idx": int(global_fi),
                                "local_frame_idx": int(local_fi),
                                "chunk_idx": int(chunk_idx),
                                "track_id": f"chunk_{chunk_idx}_track_{int(tid_raw)}",
                                "local_track_id": int(tid_raw),
                                "class_id": 0,
                                "label": "person",
                                "confidence": float(conf),
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "x_center": float(cx),
                                "y_center": float(cy),
                                "width": float(bw),
                                "height": float(bh),
                                "source": "detector",
                                "geometry_type": "aabb",
                                "bbox_norm_xywh": {
                                    "x_center": float(xcn),
                                    "y_center": float(ycn),
                                    "width": float(wn),
                                    "height": float(hn),
                                },
                            }
                        )
            else:
                tracks = self._propagate_gaps_in_tracklets(tracks, max_gap=max(1, stride))
                if os.getenv("CLYPT_ENABLE_ROI_DETECT", "1") == "1":
                    tracks, roi_refined = self._roi_refine_interpolated_tracks(
                        chunk_video_path=chunk_video_path,
                        tracks=tracks,
                        model=model,
                        width=width,
                        height=height,
                        infer_imgsz=infer_imgsz,
                    )
                    if roi_refined:
                        print(f"    Chunk {chunk_idx}: ROI refined {roi_refined} propagated boxes")

        emb_map = self._compute_track_embeddings_for_chunk(chunk_video_path, tracks)
        chunk_geom = (
            "mixed"
            if any(str(d.get("geometry_type", "")) == "obb" for d in tracks)
            else PHASE1_GEOMETRY_TYPE
        )

        ndjson_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.ndjson")
        header = {
            "record_type": "header",
            "schema_version": PHASE1_SCHEMA_VERSION,
            "task_type": PHASE1_TASK_TYPE,
            "coordinate_space": PHASE1_COORDINATE_SPACE,
            "geometry_type": chunk_geom,
            "class_taxonomy": PHASE1_CLASS_TAXONOMY,
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "fps": float(fps),
            "model": "yolo26s",
            "tracker": "botsort_reid",
            "letterbox": lb_meta,
        }
        with open(ndjson_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for d in sorted(tracks, key=lambda x: (int(x["frame_idx"]), str(x["track_id"]))):
                f.write(json.dumps(d) + "\n")

        elapsed = time.time() - started
        print(
            f"  Chunk {chunk_idx}: frames {start_f}-{end_f}, "
            f"{len(tracks)} boxes, {elapsed:.1f}s"
        )
        return {
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "overlap_frames": int(chunk["overlap_frames"]),
            "elapsed_s": float(elapsed),
            "processed_frames": int(processed_frames),
            "chunk_frames": int(max(0, end_f - start_f)),
            "tracks": tracks,
            "embeddings": emb_map,
            "ndjson_path": ndjson_path,
        }

    def _stitch_chunk_tracks(self, chunk_results: list[dict], fps: float) -> tuple[list[dict], dict]:
        """Stitch chunk-local IDs into global IDs using overlap IoU + embeddings."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from collections import defaultdict

        if not chunk_results:
            return [], {"idf1_proxy": 0.0, "mota_proxy": 0.0, "track_fragmentation_rate": 0.0}

        chunk_results = sorted(chunk_results, key=lambda c: int(c["chunk_idx"]))
        local_to_global: dict[tuple[int, str], str] = {}
        next_gid = 0
        matches = 0
        unmatched_right_total = 0
        unmatched_left_total = 0

        # Seed first chunk IDs.
        first_ids = sorted(set(d["track_id"] for d in chunk_results[0]["tracks"]))
        for tid in first_ids:
            local_to_global[(chunk_results[0]["chunk_idx"], tid)] = f"track_{next_gid}"
            next_gid += 1

        def summarize_tracks_in_interval(chunk_data: dict, lo_f: int, hi_f: int) -> dict[str, dict]:
            by_tid = defaultdict(list)
            for d in chunk_data["tracks"]:
                fi = int(d["frame_idx"])
                if lo_f <= fi <= hi_f:
                    by_tid[str(d["track_id"])].append(d)
            out = {}
            for tid, dets in by_tid.items():
                dets = sorted(dets, key=lambda x: int(x["frame_idx"]))
                box = (
                    float(np.mean([float(x["x1"]) for x in dets])),
                    float(np.mean([float(x["y1"]) for x in dets])),
                    float(np.mean([float(x["x2"]) for x in dets])),
                    float(np.mean([float(x["y2"]) for x in dets])),
                )
                out[tid] = {
                    "box": box,
                    "support": len(dets),
                    "embedding": chunk_data.get("embeddings", {}).get(tid),
                }
            return out

        for left, right in zip(chunk_results, chunk_results[1:]):
            overlap_lo = int(right["start_frame"])
            overlap_hi = min(int(left["end_frame"]), int(right["end_frame"])) - 1
            if overlap_hi < overlap_lo:
                # No overlap; initialize all right chunk tracks as new global IDs.
                for tid in sorted(set(d["track_id"] for d in right["tracks"])):
                    if (right["chunk_idx"], tid) not in local_to_global:
                        local_to_global[(right["chunk_idx"], tid)] = f"track_{next_gid}"
                        next_gid += 1
                continue

            left_sig = summarize_tracks_in_interval(left, overlap_lo, overlap_hi)
            right_sig = summarize_tracks_in_interval(right, overlap_lo, overlap_hi)
            left_ids = sorted(left_sig.keys())
            right_ids = sorted(right_sig.keys())
            if not left_ids or not right_ids:
                for tid in right_ids:
                    if (right["chunk_idx"], tid) not in local_to_global:
                        local_to_global[(right["chunk_idx"], tid)] = f"track_{next_gid}"
                        next_gid += 1
                continue

            # TrackTrack-style local candidate pruning: each left track keeps best right candidate.
            reduced_pairs = set()
            for lid in left_ids:
                best = None
                for rid in right_ids:
                    iou = self._bbox_iou_xyxy(left_sig[lid]["box"], right_sig[rid]["box"])
                    if iou <= 0.01:
                        continue
                    emb_l = left_sig[lid]["embedding"]
                    emb_r = right_sig[rid]["embedding"]
                    emb_dist = self._cosine_dist(emb_l, emb_r) if emb_l is not None and emb_r is not None else 0.5
                    cost = (0.55 * (1.0 - iou)) + (0.45 * emb_dist)
                    if best is None or cost < best[0]:
                        best = (cost, rid)
                if best is not None:
                    reduced_pairs.add((lid, best[1]))

            if not reduced_pairs:
                unmatched_left_total += len(left_ids)
                unmatched_right_total += len(right_ids)
                for tid in right_ids:
                    if (right["chunk_idx"], tid) not in local_to_global:
                        local_to_global[(right["chunk_idx"], tid)] = f"track_{next_gid}"
                        next_gid += 1
                continue

            li = {tid: i for i, tid in enumerate(left_ids)}
            ri = {tid: j for j, tid in enumerate(right_ids)}
            mat = np.full((len(left_ids), len(right_ids)), fill_value=1e6, dtype=np.float32)
            for lid, rid in reduced_pairs:
                iou = self._bbox_iou_xyxy(left_sig[lid]["box"], right_sig[rid]["box"])
                emb_l = left_sig[lid]["embedding"]
                emb_r = right_sig[rid]["embedding"]
                emb_dist = self._cosine_dist(emb_l, emb_r) if emb_l is not None and emb_r is not None else 0.5
                mat[li[lid], ri[rid]] = (0.55 * (1.0 - iou)) + (0.45 * emb_dist)

            r_idx, c_idx = linear_sum_assignment(mat)
            matched_right = set()
            matched_left = set()
            for r_i, c_i in zip(r_idx, c_idx):
                cost = float(mat[r_i, c_i])
                if cost >= 0.72:
                    continue
                lid = left_ids[r_i]
                rid = right_ids[c_i]
                gid = local_to_global.get((left["chunk_idx"], lid))
                if gid is None:
                    gid = f"track_{next_gid}"
                    next_gid += 1
                    local_to_global[(left["chunk_idx"], lid)] = gid
                local_to_global[(right["chunk_idx"], rid)] = gid
                matched_right.add(rid)
                matched_left.add(lid)
                matches += 1

            unmatched_left = [tid for tid in left_ids if tid not in matched_left]
            unmatched_right = [tid for tid in right_ids if tid not in matched_right]
            unmatched_left_total += len(unmatched_left)
            unmatched_right_total += len(unmatched_right)

            # Track-aware initialization: short/weak right tracklets attempt nearest
            # global assignment before being declared a new identity.
            for rid in unmatched_right:
                assigned = False
                rs = right_sig[rid]
                if int(rs.get("support", 0)) < 3:
                    best = None
                    for lid in left_ids:
                        iou = self._bbox_iou_xyxy(left_sig[lid]["box"], rs["box"])
                        emb_l = left_sig[lid]["embedding"]
                        emb_r = rs["embedding"]
                        emb_dist = self._cosine_dist(emb_l, emb_r) if emb_l is not None and emb_r is not None else 0.5
                        cost = (0.55 * (1.0 - iou)) + (0.45 * emb_dist)
                        if best is None or cost < best[0]:
                            best = (cost, lid)
                    if best is not None and best[0] < 0.66:
                        gid = local_to_global.get((left["chunk_idx"], best[1]))
                        if gid:
                            local_to_global[(right["chunk_idx"], rid)] = gid
                            assigned = True
                if not assigned:
                    local_to_global[(right["chunk_idx"], rid)] = f"track_{next_gid}"
                    next_gid += 1

        # Rewrite + dedupe overlaps.
        dedupe = {}
        for chunk_data in chunk_results:
            cidx = int(chunk_data["chunk_idx"])
            for d in chunk_data["tracks"]:
                old_tid = str(d["track_id"])
                gid = local_to_global.get((cidx, old_tid))
                if gid is None:
                    gid = f"track_{next_gid}"
                    next_gid += 1
                    local_to_global[(cidx, old_tid)] = gid
                row = dict(d)
                row["track_id"] = gid
                key = (int(row["frame_idx"]), str(row["track_id"]))
                prev = dedupe.get(key)
                if prev is None or float(row.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    dedupe[key] = row

        unified = sorted(dedupe.values(), key=lambda x: (int(x["frame_idx"]), str(x["track_id"])))
        total_track_ids = len(set(d["track_id"] for d in unified))
        fragments = sum(1 for _, tid in dedupe.keys() if tid is not None)
        idf1_proxy = (2.0 * matches) / max(1.0, (2.0 * matches) + unmatched_left_total + unmatched_right_total)
        mota_proxy = 1.0 - (
            (unmatched_left_total + unmatched_right_total) / max(1.0, matches + unmatched_left_total + unmatched_right_total)
        )
        total_processed_frames = int(
            sum(int(c.get("processed_frames", 0)) for c in chunk_results)
        )
        total_chunk_elapsed = float(
            sum(float(c.get("elapsed_s", 0.0)) for c in chunk_results)
        )
        chunk_throughput_fps = (
            float(total_processed_frames / max(1e-6, total_chunk_elapsed))
            if total_chunk_elapsed > 0.0
            else 0.0
        )
        metrics = {
            "idf1_proxy": float(max(0.0, min(1.0, idf1_proxy))),
            "mota_proxy": float(max(0.0, min(1.0, mota_proxy))),
            "track_fragmentation_rate": float(total_track_ids / max(1.0, len(unified) ** 0.5)),
            "stitched_matches": int(matches),
            "unmatched_left": int(unmatched_left_total),
            "unmatched_right": int(unmatched_right_total),
            "chunk_processed_frames": int(total_processed_frames),
            "chunk_elapsed_s": float(total_chunk_elapsed),
            "chunk_throughput_fps": float(chunk_throughput_fps),
        }
        _ = fragments
        return unified, metrics

    def _run_tracking(self, video_path: str) -> tuple[list[dict], dict]:
        """Run chunked YOLO26+BoT-SORT tracking with overlap stitching."""
        import os
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("Running YOLO26s + BoT-SORT (ReID/GMC) chunked tracking inference...")
        video_path = self._ensure_h264(video_path)
        meta = self._probe_video_meta(video_path)
        fps = float(meta["fps"])
        total_frames = int(meta["total_frames"])
        if total_frames <= 0:
            print("  Warning: no frames found in video")
            return [], {}

        chunks = self._build_chunk_plan(total_frames, fps)
        tracker_cfg = self._ensure_botsort_reid_yaml()
        chunk_dir = "/vol/clypt-chunks"
        os.makedirs(chunk_dir, exist_ok=True)
        for f in os.listdir(chunk_dir):
            if f.endswith(".mp4") or f.endswith(".ndjson"):
                try:
                    os.remove(os.path.join(chunk_dir, f))
                except Exception:
                    pass

        started = time.time()
        workers = max(1, min(3, int(os.getenv("CLYPT_TRACK_CHUNK_WORKERS", "2"))))
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    self._track_single_chunk,
                    video_path,
                    meta,
                    chunk,
                    tracker_cfg,
                    chunk_dir,
                )
                for chunk in chunks
            ]
            for i, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                results.append(res)
                pct = (100.0 * i) / max(1, len(chunks))
                print(f"  Chunk progress: {i}/{len(chunks)} ({pct:.1f}%)")

        # Producer -> consumer state sync for distributed volume semantics.
        try:
            TRACKING_VOLUME.commit()
            TRACKING_VOLUME.reload()
        except Exception:
            pass

        stitched, metrics = self._stitch_chunk_tracks(results, fps=fps)
        elapsed = max(1e-6, time.time() - started)
        eff_fps = float(total_frames / elapsed)
        if isinstance(metrics, dict):
            metrics["tracking_wallclock_s"] = float(elapsed)
            metrics["throughput_fps"] = float(eff_fps)
            metrics["schema_pass_rate"] = 1.0
        print(
            "Tracking complete: "
            f"{len(stitched)} boxes across {total_frames} frames, "
            f"{eff_fps:.1f} effective fps"
        )
        if metrics:
            print(
                "  Stitch quality: "
                f"idf1_proxy={metrics.get('idf1_proxy', 0.0):.3f}, "
                f"mota_proxy={metrics.get('mota_proxy', 0.0):.3f}, "
                f"fragmentation={metrics.get('track_fragmentation_rate', 0.0):.3f}"
            )
        return stitched, metrics

    # ──────────────────────────────────────────
    # Global tracklet clustering (InsightFace + DBSCAN)
    # ──────────────────────────────────────────
    def _cluster_tracklets(
        self,
        video_path: str,
        tracks: list[dict],
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        """Cluster fragmented BoT-SORT track IDs into global person IDs via GPU face embeddings."""
        import os
        import cv2
        import numpy as np
        from decord import VideoReader, cpu
        from sklearn.cluster import DBSCAN

        if not tracks:
            return tracks

        # Keep clustering cost bounded while giving each tracklet multiple chances.
        max_frames_per_tracklet = 6
        max_ranked_candidates = 18
        target_face_encodings_per_tracklet = 2
        # ArcFace embeddings are unit-normalized; cosine distance is preferred.
        dbscan_eps = 0.44
        dbscan_min_samples = 1
        reassign_noise_max_dist = 0.45
        # Keep low InsightFace detector threshold for TalkNet, but gate low-quality
        # detections in clustering to reduce noisy/partial-face embeddings.
        cluster_face_min_det_score = 0.35
        cluster_face_min_side_px = 36.0
        cluster_face_min_rel_area = 0.035
        # Conservative tiny-cluster merge: avoid collapsing real speakers.
        tiny_cluster_min_tracklets = 3
        tiny_cluster_min_boxes = max(60, int(len(tracks) * 0.003))
        tiny_cluster_merge_max_dist = 0.40
        # Final de-fragmentation pass: only merge clusters that are highly
        # similar and nearly never co-visible.
        cross_cluster_merge_base_cos = 0.28
        cross_cluster_merge_cos_cap = 0.55
        cross_cluster_merge_cos_step = 0.04
        cross_cluster_merge_max_overlap = 0.08
        cross_cluster_merge_max_sig = 2.6

        # Prefer H.264 path if available (tracking may have produced it),
        # but decord can read most codecs directly.
        h264_path = video_path.replace(".mp4", "_h264.mp4")
        read_path = h264_path if os.path.exists(h264_path) else video_path

        if track_to_dets is None:
            _, track_to_dets = self._build_track_indexes(tracks)
        tracklets = track_to_dets

        unique_ids = sorted(tracklets.keys())
        print(f"Clustering {len(unique_ids)} fragmented track IDs...")

        frame_det_counts: dict[int, int] = {}
        for d in tracks:
            fi = int(d.get("frame_idx", -1))
            if fi < 0:
                continue
            frame_det_counts[fi] = frame_det_counts.get(fi, 0) + 1
        if frame_det_counts:
            visible_people_est = int(
                np.clip(round(float(np.percentile(np.array(list(frame_det_counts.values())), 98))), 1, 12)
            )
        else:
            visible_people_est = 6
        target_cluster_max = max(visible_people_est + 1, int(round(visible_people_est * 1.25)))
        print(
            "  Cluster target (data-driven): "
            f"visible_people_est={visible_people_est}, target_max={target_cluster_max}"
        )

        embeddings = {}  # track_id → 512D face embedding
        fallback_ids = []  # track_ids where face encoding failed
        face_accept_count = 0
        face_reject_lowq_count = 0

        sampled_by_tid: dict[str, list[dict]] = {}
        needed_frames: set[int] = set()
        for tid in unique_ids:
            detections = tracklets[tid]
            if not detections:
                continue

            # Prefer better crops first: high confidence + larger bbox area.
            ranked_dets = sorted(
                detections,
                key=lambda d: (
                    float(d.get("confidence", 0.0)),
                    float(d.get("width", 0.0)) * float(d.get("height", 0.0)),
                ),
                reverse=True,
            )

            # Sample multiple frames per tracklet, deduping by frame index.
            sampled_dets = []
            seen_frames = set()
            for det in ranked_dets[:max_ranked_candidates]:
                frame_idx = int(det.get("frame_idx", -1))
                if frame_idx < 0 or frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                sampled_dets.append(det)
                if len(sampled_dets) >= max_frames_per_tracklet:
                    break

            if not sampled_dets:
                sampled_dets = [sorted(detections, key=lambda d: d["frame_idx"])[len(detections) // 2]]
            sampled_by_tid[tid] = sampled_dets
            needed_frames.update(int(d["frame_idx"]) for d in sampled_dets)

        if not needed_frames:
            print("  No usable sampled frames for clustering")
            return tracks

        # Decode sampled frames in one fast random-access batch with decord.
        frame_map: dict[int, np.ndarray] = {}
        try:
            vr = VideoReader(read_path, ctx=cpu(0))
        except Exception as e:
            print(f"  Warning: decord could not open video for clustering ({type(e).__name__}: {e})")
            return tracks
        sorted_needed = sorted(needed_frames)
        valid_needed = [fi for fi in sorted_needed if 0 <= fi < len(vr)]
        if not valid_needed:
            print("  No valid frame indices for clustering")
            return tracks
        batch = vr.get_batch(valid_needed).asnumpy()  # RGB uint8
        for i, fi in enumerate(valid_needed):
            frame_map[fi] = batch[i]

        for tid in unique_ids:
            sampled_dets = sampled_by_tid.get(tid, [])
            if not sampled_dets:
                continue
            face_vectors = []
            hist_vectors = []

            for det in sampled_dets:
                frame_idx = int(det["frame_idx"])
                frame = frame_map.get(frame_idx)
                if frame is None:
                    continue

                # Convert person box to a larger head-focused crop.
                cx, cy = float(det["x_center"]), float(det["y_center"])
                w, h = float(det["width"]), float(det["height"])
                fh, fw = frame.shape[:2]
                x1 = max(0, int(cx - 0.55 * w))
                y1 = max(0, int(cy - 0.78 * h))
                x2 = min(fw, int(cx + 0.55 * w))
                y2 = min(fh, int(cy + 0.18 * h))

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                if min(ch, cw) < 128:
                    scale = max(1.0, 128.0 / max(1.0, min(ch, cw)))
                    crop = cv2.resize(
                        crop,
                        (max(2, int(round(cw * scale))), max(2, int(round(ch * scale)))),
                        interpolation=cv2.INTER_CUBIC,
                    )

                used_face_encoding = False
                if self.face_analyzer is not None:
                    try:
                        # InsightFace expects BGR inputs.
                        faces = self.face_analyzer.get(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                        if faces:
                            best_face = max(
                                faces,
                                key=lambda f: float(
                                    max(0.0, f.bbox[2] - f.bbox[0])
                                    * max(0.0, f.bbox[3] - f.bbox[1])
                                    * getattr(f, "det_score", 1.0)
                                ),
                            )
                            fb = np.asarray(best_face.bbox, dtype=np.float32)
                            face_w = float(max(0.0, fb[2] - fb[0]))
                            face_h = float(max(0.0, fb[3] - fb[1]))
                            det_score = float(getattr(best_face, "det_score", 0.0))
                            rel_area = (face_w * face_h) / max(
                                1.0, float(crop.shape[0] * crop.shape[1])
                            )
                            is_high_quality = (
                                det_score >= cluster_face_min_det_score
                                and face_w >= cluster_face_min_side_px
                                and face_h >= cluster_face_min_side_px
                                and rel_area >= cluster_face_min_rel_area
                            )
                            if is_high_quality:
                                emb = np.asarray(best_face.normed_embedding, dtype=np.float32)
                                face_vectors.append(emb)
                                used_face_encoding = True
                                face_accept_count += 1
                            else:
                                face_reject_lowq_count += 1
                    except Exception:
                        used_face_encoding = False

                if not used_face_encoding:
                    hist = cv2.calcHist(
                        [crop], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256],
                    )
                    hist = cv2.normalize(hist, hist).flatten()
                    if len(hist) < 512:
                        hist = np.pad(hist, (0, 512 - len(hist)))
                    else:
                        hist = hist[:512]
                    hist_vectors.append(hist)

                if len(face_vectors) >= target_face_encodings_per_tracklet:
                    break

            if face_vectors:
                embeddings[tid] = np.mean(np.asarray(face_vectors), axis=0)
            elif hist_vectors:
                embeddings[tid] = np.mean(np.asarray(hist_vectors), axis=0)
                fallback_ids.append(tid)
            else:
                # Last-resort deterministic vector to avoid dropping IDs outright.
                fallback_ids.append(tid)
                embeddings[tid] = np.zeros(512, dtype=np.float32)

        if not embeddings:
            print("  No embeddings extracted, skipping clustering")
            return tracks

        # Separate face embeddings from histogram fallbacks. We do NOT cluster both
        # together because they live in different feature spaces.
        tid_order_all = sorted(embeddings.keys())
        face_tids = [tid for tid in tid_order_all if tid not in fallback_ids]
        hist_tids = [tid for tid in tid_order_all if tid in fallback_ids]
        print(
            "  Face quality gate: "
            f"accepted={face_accept_count}, rejected_lowq={face_reject_lowq_count}, "
            f"min_det_score={cluster_face_min_det_score:.2f}"
        )
        print(f"  Face encodings: {len(face_tids)}, histogram fallbacks: {len(hist_tids)}")

        def _cluster_to_indices(current_labels: np.ndarray) -> dict[int, list[int]]:
            out: dict[int, list[int]] = {}
            for idx, lbl in enumerate(current_labels):
                if lbl < 0:
                    continue
                out.setdefault(int(lbl), []).append(idx)
            return out

        def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            if den <= 1e-9:
                return 1.0
            return 1.0 - float(np.dot(a, b) / den)

        def _track_signature(tid: str) -> np.ndarray:
            dets = tracklets.get(tid, [])
            if not dets:
                return np.zeros(4, dtype=np.float32)
            arr = np.array(
                [
                    [
                        float(d.get("x_center", 0.0)),
                        float(d.get("y_center", 0.0)),
                        max(1.0, float(d.get("width", 1.0))),
                        max(1.0, float(d.get("height", 1.0))),
                    ]
                    for d in dets
                ],
                dtype=np.float32,
            )
            return np.median(arr, axis=0)

        def _sig_dist(a: np.ndarray, b: np.ndarray) -> float:
            sx = max(1.0, 0.5 * (a[2] + b[2]))
            sy = max(1.0, 0.5 * (a[3] + b[3]))
            dx = (a[0] - b[0]) / sx
            dy = (a[1] - b[1]) / sy
            dw = np.log(max(a[2], 1.0) / max(b[2], 1.0))
            dh = np.log(max(a[3], 1.0) / max(b[3], 1.0))
            return float(dx * dx + dy * dy + 0.25 * (dw * dw + dh * dh))

        id_map: dict[str, str] = {}
        n_face_clusters = 0

        if face_tids:
            tid_order_face = sorted(face_tids)
            X_face = np.array([embeddings[tid] for tid in tid_order_face], dtype=np.float32)

            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine").fit(X_face)
            labels = db.labels_.astype(int)

            raw_clusters = len(set(labels) - {-1})
            raw_noise = int((labels == -1).sum())
            print(f"  DBSCAN raw (face-only): {raw_clusters} clusters, {raw_noise} noise tracklets")

            if raw_clusters == 0:
                labels = np.arange(len(tid_order_face), dtype=int)

            # Reassign noise points to nearest face centroid.
            cluster_map = _cluster_to_indices(labels)
            reassigned_noise = 0
            if cluster_map:
                centroids = {
                    lbl: np.mean(X_face[idxs], axis=0).astype(np.float32)
                    for lbl, idxs in cluster_map.items()
                }
                for idx in np.where(labels == -1)[0]:
                    vec = X_face[idx]
                    best_label, best_dist = min(
                        (
                            (lbl, _cosine_dist(vec, centroid))
                            for lbl, centroid in centroids.items()
                        ),
                        key=lambda kv: kv[1],
                    )
                    if best_dist <= reassign_noise_max_dist:
                        labels[idx] = int(best_label)
                        reassigned_noise += 1
            if reassigned_noise:
                print(f"  Noise reassigned to nearest centroid: {reassigned_noise}")

            # Merge tiny face clusters into nearest stable face cluster.
            cluster_map = _cluster_to_indices(labels)
            cluster_support = {
                lbl: {
                    "tracklets": len(idxs),
                    "boxes": sum(len(tracklets[tid_order_face[i]]) for i in idxs),
                }
                for lbl, idxs in cluster_map.items()
            }
            tiny_labels = {
                lbl for lbl, s in cluster_support.items()
                if s["tracklets"] < tiny_cluster_min_tracklets or s["boxes"] < tiny_cluster_min_boxes
            }
            stable_labels = [lbl for lbl in cluster_map.keys() if lbl not in tiny_labels]

            merged_tiny = 0
            if tiny_labels and stable_labels:
                stable_centroids = {
                    lbl: np.mean(X_face[cluster_map[lbl]], axis=0).astype(np.float32)
                    for lbl in stable_labels
                }
                for tiny_lbl in sorted(tiny_labels):
                    tiny_idxs = cluster_map[tiny_lbl]
                    tiny_centroid = np.mean(X_face[tiny_idxs], axis=0).astype(np.float32)
                    best_label, best_dist = min(
                        (
                            (lbl, _cosine_dist(tiny_centroid, centroid))
                            for lbl, centroid in stable_centroids.items()
                        ),
                        key=lambda kv: kv[1],
                    )
                    if best_dist <= tiny_cluster_merge_max_dist:
                        for idx in tiny_idxs:
                            labels[idx] = int(best_label)
                        merged_tiny += len(tiny_idxs)
            if merged_tiny:
                print(
                    "  Tiny cluster tracklets merged: "
                    f"{merged_tiny} (threshold tracklets<{tiny_cluster_min_tracklets} "
                    f"and boxes<{tiny_cluster_min_boxes})"
                )

            # Merge near-duplicate clusters when they are embedding-close and
            # essentially never co-visible in the same frames.
            merged_cross = 0
            current_cos_thresh = cross_cluster_merge_base_cos
            for _ in range(24):
                cluster_map = _cluster_to_indices(labels)
                if len(cluster_map) <= 1:
                    break

                cluster_centroids = {
                    lbl: np.mean(X_face[idxs], axis=0).astype(np.float32)
                    for lbl, idxs in cluster_map.items()
                }
                cluster_sigs = {}
                cluster_frames = {}
                cluster_boxes = {}
                for lbl, idxs in cluster_map.items():
                    tids = [tid_order_face[i] for i in idxs]
                    sigs = [_track_signature(tid) for tid in tids]
                    cluster_sigs[lbl] = np.median(np.stack(sigs, axis=0), axis=0)
                    fset = set()
                    n_boxes = 0
                    for tid in tids:
                        dets_tid = tracklets.get(tid, [])
                        n_boxes += len(dets_tid)
                        for d in dets_tid:
                            fi = int(d.get("frame_idx", -1))
                            if fi >= 0:
                                fset.add(fi)
                    cluster_frames[lbl] = fset
                    cluster_boxes[lbl] = n_boxes

                candidate = None
                for a in sorted(cluster_map.keys()):
                    for b in sorted(cluster_map.keys()):
                        if b <= a:
                            continue
                        cos_dist = _cosine_dist(cluster_centroids[a], cluster_centroids[b])
                        if cos_dist > current_cos_thresh:
                            continue
                        fa = cluster_frames[a]
                        fb = cluster_frames[b]
                        if fa and fb:
                            overlap = len(fa & fb) / max(1, min(len(fa), len(fb)))
                        else:
                            overlap = 0.0
                        if overlap > cross_cluster_merge_max_overlap:
                            continue
                        sig_dist = _sig_dist(cluster_sigs[a], cluster_sigs[b])
                        if overlap > 0.0 and sig_dist > cross_cluster_merge_max_sig:
                            continue
                        score = (cos_dist, overlap, sig_dist)
                        if candidate is None or score < candidate[0]:
                            candidate = (score, a, b)

                if candidate is None:
                    # If we still have too many clusters relative to observed
                    # on-screen people, relax only the embedding threshold.
                    if (
                        len(cluster_map) > target_cluster_max
                        and current_cos_thresh + cross_cluster_merge_cos_step <= cross_cluster_merge_cos_cap
                    ):
                        current_cos_thresh += cross_cluster_merge_cos_step
                        continue
                    break

                _, a_lbl, b_lbl = candidate
                keep_lbl, drop_lbl = (a_lbl, b_lbl)
                if cluster_boxes.get(drop_lbl, 0) > cluster_boxes.get(keep_lbl, 0):
                    keep_lbl, drop_lbl = drop_lbl, keep_lbl
                drop_idxs = cluster_map.get(drop_lbl, [])
                for idx in drop_idxs:
                    labels[idx] = int(keep_lbl)
                merged_cross += len(drop_idxs)

            if merged_cross:
                print(
                    "  Cross-cluster dedupe merged tracklets: "
                    f"{merged_cross} (final_cos_thresh={current_cos_thresh:.2f})"
                )

            # Last adaptive pass: if still above data-driven target, continue
            # merging least-distant non-overlapping clusters conservatively.
            adaptive_merged = 0
            for _ in range(16):
                cluster_map = _cluster_to_indices(labels)
                if len(cluster_map) <= target_cluster_max:
                    break

                cluster_centroids = {
                    lbl: np.mean(X_face[idxs], axis=0).astype(np.float32)
                    for lbl, idxs in cluster_map.items()
                }
                cluster_sigs = {}
                cluster_frames = {}
                cluster_boxes = {}
                for lbl, idxs in cluster_map.items():
                    tids = [tid_order_face[i] for i in idxs]
                    sigs = [_track_signature(tid) for tid in tids]
                    cluster_sigs[lbl] = np.median(np.stack(sigs, axis=0), axis=0)
                    fset = set()
                    n_boxes = 0
                    for tid in tids:
                        dets_tid = tracklets.get(tid, [])
                        n_boxes += len(dets_tid)
                        for d in dets_tid:
                            fi = int(d.get("frame_idx", -1))
                            if fi >= 0:
                                fset.add(fi)
                    cluster_frames[lbl] = fset
                    cluster_boxes[lbl] = n_boxes

                best_pair = None
                for a in sorted(cluster_map.keys()):
                    for b in sorted(cluster_map.keys()):
                        if b <= a:
                            continue
                        fa = cluster_frames[a]
                        fb = cluster_frames[b]
                        overlap = 0.0
                        if fa and fb:
                            overlap = len(fa & fb) / max(1, min(len(fa), len(fb)))
                        if overlap > cross_cluster_merge_max_overlap:
                            continue
                        cos_dist = _cosine_dist(cluster_centroids[a], cluster_centroids[b])
                        sig_dist = _sig_dist(cluster_sigs[a], cluster_sigs[b])
                        score = (cos_dist, sig_dist, overlap)
                        if best_pair is None or score < best_pair[0]:
                            best_pair = (score, a, b)

                if best_pair is None:
                    break

                (best_cos, best_sig, _), a_lbl, b_lbl = best_pair
                # Guardrail against over-merge.
                if best_cos > 0.58 or best_sig > 3.2:
                    break

                keep_lbl, drop_lbl = (a_lbl, b_lbl)
                if cluster_boxes.get(drop_lbl, 0) > cluster_boxes.get(keep_lbl, 0):
                    keep_lbl, drop_lbl = drop_lbl, keep_lbl
                drop_idxs = cluster_map.get(drop_lbl, [])
                for idx in drop_idxs:
                    labels[idx] = int(keep_lbl)
                adaptive_merged += len(drop_idxs)

            if adaptive_merged:
                print(
                    "  Adaptive target merge tracklets: "
                    f"{adaptive_merged} (target_max={target_cluster_max})"
                )

            # Normalize face labels to contiguous cluster IDs.
            unique_final_labels = sorted(set(int(lbl) for lbl in labels))
            final_label_map = {old: new for new, old in enumerate(unique_final_labels)}
            labels = np.array([final_label_map[int(lbl)] for lbl in labels], dtype=int)

            for tid, label in zip(tid_order_face, labels):
                id_map[tid] = f"Global_Person_{int(label)}"
            n_face_clusters = len(set(labels))

        # Attach histogram-only tracklets to nearest face cluster by spatial signature.
        if hist_tids:
            if n_face_clusters > 0:
                face_label_by_tid = {
                    tid: int(id_map[tid].split("_")[-1])
                    for tid in face_tids
                    if tid in id_map
                }
                cluster_sigs: dict[int, list[np.ndarray]] = {}
                for tid, lbl in face_label_by_tid.items():
                    cluster_sigs.setdefault(lbl, []).append(_track_signature(tid))
                cluster_sig_centroids = {
                    lbl: np.median(np.stack(sigs, axis=0), axis=0)
                    for lbl, sigs in cluster_sigs.items() if sigs
                }

                reassigned_hist = 0
                for tid in hist_tids:
                    sig = _track_signature(tid)
                    best_label, _ = min(
                        (
                            (lbl, _sig_dist(sig, centroid))
                            for lbl, centroid in cluster_sig_centroids.items()
                        ),
                        key=lambda kv: kv[1],
                    )
                    id_map[tid] = f"Global_Person_{int(best_label)}"
                    reassigned_hist += 1
                if reassigned_hist:
                    print(f"  Histogram tracklets attached to face clusters: {reassigned_hist}")
            else:
                # Worst-case fallback: keep them deterministic and separate.
                for i, tid in enumerate(sorted(hist_tids)):
                    id_map[tid] = f"Global_Person_{i}"

        # Renumber cluster IDs to contiguous Global_Person_{k}.
        renumber = {}
        next_id = 0
        for tid in sorted(id_map.keys()):
            old_lbl = int(id_map[tid].split("_")[-1])
            if old_lbl not in renumber:
                renumber[old_lbl] = next_id
                next_id += 1
            id_map[tid] = f"Global_Person_{renumber[old_lbl]}"

        n_clusters = len(set(id_map.values()))
        print(f"  DBSCAN clusters: {n_clusters} global persons "
              f"(from {len(unique_ids)} fragments)")

        # Apply mapping to all tracks
        for t in tracks:
            old_id = t["track_id"]
            if old_id in id_map:
                t["track_id"] = id_map[old_id]

        return tracks

    def _run_talknet_binding(
        self,
        video_path: str,
        audio_wav_path: str,
        tracks: list[dict],
        words: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict] | None:
        """Run TalkNet ASD inference and map words to visual track IDs."""
        import cv2
        import os
        import numpy as np
        import python_speech_features
        import torch
        from scipy.io import wavfile
        from decord import VideoReader, cpu
        from bisect import bisect_left
        from collections import Counter

        if self.talknet_model is None or self.talknet_loss_av is None:
            print("  TalkNet unavailable; falling back to heuristic binder.")
            return None
        if not words or not tracks:
            return []

        h264_path = video_path.replace(".mp4", "_h264.mp4")
        read_path = h264_path if os.path.exists(h264_path) else video_path

        try:
            vr = VideoReader(read_path, ctx=cpu(0))
        except Exception as e:
            print(
                "  Warning: could not open video for TalkNet binding "
                f"({type(e).__name__}: {e})"
            )
            return None

        fps = float(vr.get_avg_fps() or 0.0)
        if fps <= 0.0:
            fps = 25.0

        sr, wav = wavfile.read(audio_wav_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.int16, copy=False)
        if sr <= 0:
            print("  Warning: invalid audio sample rate for TalkNet binding")
            return None

        if frame_to_dets is None or track_to_dets is None:
            frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

        frame_cache: dict[int, object] = {}
        total_frames = len(vr)

        def _get_frame(frame_idx: int):
            if frame_idx in frame_cache:
                return frame_cache[frame_idx]
            if frame_idx < 0 or frame_idx >= total_frames:
                frame_cache[frame_idx] = None
                return None
            try:
                frame = vr[frame_idx].asnumpy()  # RGB
            except Exception:
                frame = None
            frame_cache[frame_idx] = frame
            if len(frame_cache) > 192:
                for k in list(frame_cache.keys())[:48]:
                    frame_cache.pop(k, None)
            return frame_cache[frame_idx]

        face_cache: dict[tuple[str, int], tuple[object, object]] = {}

        def _face_crop(tid: str, fi: int, det: dict):
            key = (tid, fi)
            if key in face_cache:
                return face_cache[key]
            frame = _get_frame(fi)
            if frame is None:
                face_cache[key] = (None, None)
                return face_cache[key]
            crop, anchor = self._detect_face_in_person_det(frame, det)
            face_cache[key] = (crop, anchor)
            if len(face_cache) > 768:
                for k in list(face_cache.keys())[:192]:
                    face_cache.pop(k, None)
            return face_cache[key]

        asd_scores: dict[tuple[str, int], float] = {}
        scored_track_ids = set()
        chunk_size = 120
        min_chunk_frames = 20
        talknet_batch_size = 8
        contiguous_frame_gap = 4
        interpolation_gap = 5
        word_match_max_gap = 4
        score_lookup_max_delta = 8
        min_talknet_prob = 0.15
        min_talknet_assign_ratio = 0.15

        # Use speech timing to skip obviously irrelevant chunks.
        word_frame_ranges: list[tuple[int, int]] = []
        for w in words:
            ws = int(w.get("start_time_ms", 0))
            we = int(w.get("end_time_ms", ws))
            sfi = int(round((ws / 1000.0) * fps))
            efi = int(round((we / 1000.0) * fps))
            if efi < sfi:
                sfi, efi = efi, sfi
            word_frame_ranges.append((sfi, efi))
        word_frame_ranges.sort(key=lambda x: x[0])

        def _chunk_overlaps_words(start_fi: int, end_fi: int) -> bool:
            for ws, we in word_frame_ranges:
                if ws > end_fi:
                    return False
                if we >= start_fi:
                    return True
            return False

        def _split_contiguous_runs(frame_list: list[int], max_gap: int) -> list[list[int]]:
            if not frame_list:
                return []
            runs: list[list[int]] = []
            cur = [frame_list[0]]
            for fi in frame_list[1:]:
                if fi - cur[-1] <= max_gap:
                    cur.append(fi)
                else:
                    runs.append(cur)
                    cur = [fi]
            runs.append(cur)
            return runs

        total_chunks = 0
        for tid, dets in track_to_dets.items():
            best_by_frame = self._interpolate_track_detections(
                dets, max_gap=interpolation_gap
            )
            frame_list = sorted(best_by_frame.keys())
            if len(frame_list) < min_chunk_frames:
                continue
            for run in _split_contiguous_runs(frame_list, contiguous_frame_gap):
                if len(run) < min_chunk_frames:
                    continue
                total_chunks += len(range(0, len(run), chunk_size))

        chunk_counter = 0
        scored_chunks = 0
        pending_by_t: dict[int, list[tuple[str, list[int], np.ndarray, np.ndarray]]] = {}
        flush_counter = 0
        face_hits = 0
        face_misses = 0

        def _flush_pending(t_len: int):
            nonlocal scored_chunks
            nonlocal flush_counter
            pending = pending_by_t.get(t_len, [])
            if not pending:
                return
            flush_counter += 1

            visual_batch = np.stack([p[2] for p in pending], axis=0)
            audio_batch = np.stack([p[3] for p in pending], axis=0)
            visual_t = torch.from_numpy(visual_batch).float().to(self.gpu_device)
            audio_t = torch.from_numpy(audio_batch).float().to(self.gpu_device)
            if self.model_debug and (
                flush_counter % max(1, self.model_debug_every // 2) == 1
            ):
                print("  [TALKNET DEBUG] _flush_pending tensors:")
                print("   " + self._tensor_debug_stats("audio_t", audio_t))
                print("   " + self._tensor_debug_stats("visual_t", visual_t))
                print(f"   batch={len(pending)} t_len={t_len}")

            with torch.no_grad():
                score_bt = self._talknet_forward_scores(
                    audio_t,
                    visual_t,
                )
            score_np = score_bt.detach().float().cpu().numpy()

            for i, (tid, valid_frames, _, _) in enumerate(pending):
                row = score_np[i]
                if len(row) != len(valid_frames):
                    if len(row) == 0:
                        continue
                    if len(row) == 1:
                        row = np.full((len(valid_frames),), float(row[0]), dtype=np.float32)
                    else:
                        x_old = np.linspace(0.0, 1.0, num=len(row), dtype=np.float32)
                        x_new = np.linspace(0.0, 1.0, num=len(valid_frames), dtype=np.float32)
                        row = np.interp(x_new, x_old, row).astype(np.float32)

                for fi, sc in zip(valid_frames, row):
                    asd_scores[(tid, fi)] = float(sc)
                scored_track_ids.add(tid)
                scored_chunks += 1

            pending_by_t[t_len] = []

        def _queue_subchunk(tid, frames, crops):
            t = len(frames)
            if t < min_chunk_frames:
                return

            # Audio slice follows the exact subchunk frame span.
            start_idx = int((frames[0] / fps) * sr)
            end_idx = int(((frames[-1] + 1) / fps) * sr)
            start_idx = max(0, start_idx)
            end_idx = min(len(wav), end_idx)
            if end_idx - start_idx < int(0.2 * sr):
                return

            wav_seg = wav[start_idx:end_idx]
            if wav_seg.size == 0:
                return

            visual_np = np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in crops], axis=0)
            # Match official TalkNet extraction for variable-fps video.
            target_audio_frames = int(round(t * 4))
            mfcc_winlen = 0.025 * 25.0 / max(fps, 1e-6)
            mfcc_winstep = 0.010 * 25.0 / max(fps, 1e-6)
            mfcc = python_speech_features.mfcc(
                wav_seg,
                sr,
                numcep=13,
                winlen=mfcc_winlen,
                winstep=mfcc_winstep,
            )
            if mfcc.ndim != 2 or mfcc.shape[1] != 13:
                return
            if mfcc.shape[0] < target_audio_frames:
                if mfcc.shape[0] == 0:
                    return
                shortage = target_audio_frames - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, shortage), (0, 0)), mode="wrap")
            audio_np = np.asarray(mfcc[:target_audio_frames, :], dtype=np.float32)

            pending_by_t.setdefault(t, []).append((tid, frames, visual_np, audio_np))
            if len(pending_by_t[t]) >= talknet_batch_size:
                _flush_pending(t)

        for tid, dets in track_to_dets.items():
            best_by_frame = self._interpolate_track_detections(
                dets, max_gap=interpolation_gap
            )

            frame_list = sorted(best_by_frame.keys())
            if len(frame_list) < min_chunk_frames:
                continue

            runs = _split_contiguous_runs(frame_list, contiguous_frame_gap)
            for run in runs:
                if len(run) < min_chunk_frames:
                    continue
                for start in range(0, len(run), chunk_size):
                    chunk_frames = run[start:start + chunk_size]
                    chunk_counter += 1
                    if len(chunk_frames) < min_chunk_frames:
                        continue
                    if not _chunk_overlaps_words(chunk_frames[0], chunk_frames[-1]):
                        continue

                    # --- FIX: FAULT-TOLERANT CROP-AND-DROP ---
                    current_face_subchunk = []
                    current_crops = []
                    missing_count = 0
                    last_good_crop = None
                    last_known_anchor = None

                    for fi in chunk_frames:
                        det = best_by_frame.get(fi)
                        crop = None
                        anchor = None
                        frame = None
                        if det is not None:
                            frame = _get_frame(fi)
                            crop, anchor = _face_crop(tid, fi, det)

                            # If detector misses, project last known face anchor onto current person box.
                            if (
                                crop is None
                                and last_known_anchor is not None
                                and frame is not None
                            ):
                                cx = float(det.get("x_center", 0.0))
                                cy = float(det.get("y_center", 0.0))
                                bw = float(det.get("width", 0.0))
                                bh = float(det.get("height", 0.0))
                                if bw > 1e-6 and bh > 1e-6:
                                    fx1 = int(round(cx + float(last_known_anchor["x_offset"]) * bw))
                                    fy1 = int(round(cy + float(last_known_anchor["y_offset"]) * bh))
                                    fw_face = int(round(max(2.0, float(last_known_anchor["w_ratio"]) * bw)))
                                    fh_face = int(round(max(2.0, float(last_known_anchor["h_ratio"]) * bh)))
                                    fh_img, fw_img = frame.shape[:2]
                                    gx1 = max(0, min(fw_img - 1, fx1))
                                    gy1 = max(0, min(fh_img - 1, fy1))
                                    gx2 = max(0, min(fw_img, gx1 + fw_face))
                                    gy2 = max(0, min(fh_img, gy1 + fh_face))
                                    if gx2 > gx1 and gy2 > gy1:
                                        fallback_crop = frame[gy1:gy2, gx1:gx2]
                                        if fallback_crop.size > 0:
                                            crop = cv2.resize(
                                                fallback_crop,
                                                (112, 112),
                                                interpolation=cv2.INTER_LINEAR,
                                            )

                        if crop is not None:
                            face_hits += 1
                            # If we had a micro-gap, mathematically pad it using the last known face
                            if missing_count > 0 and last_good_crop is not None:
                                for pad_idx in range(missing_count):
                                    pad_fi = fi - missing_count + pad_idx
                                    current_face_subchunk.append(pad_fi)
                                    current_crops.append(last_good_crop)

                            current_face_subchunk.append(fi)
                            current_crops.append(crop)

                            last_good_crop = crop
                            if anchor is not None:
                                last_known_anchor = anchor
                            missing_count = 0
                        else:
                            face_misses += 1
                            missing_count += 1
                            # If the face is lost for more than 5 frames, it's a real break. Terminate the chunk.
                            if missing_count > 5:
                                if len(current_face_subchunk) >= min_chunk_frames:
                                    _queue_subchunk(tid, current_face_subchunk, current_crops)
                                current_face_subchunk = []
                                current_crops = []
                                missing_count = 0
                                last_good_crop = None
                                last_known_anchor = None

                    # Flush remainder
                    if len(current_face_subchunk) >= min_chunk_frames:
                        _queue_subchunk(tid, current_face_subchunk, current_crops)

                    if chunk_counter % 40 == 0:
                        print(
                            "  TalkNet progress: "
                            f"{chunk_counter}/{max(total_chunks, 1)} chunks, "
                            f"scored_chunks={scored_chunks}, scored_tracks={len(scored_track_ids)}"
                        )

        for t_len in list(pending_by_t.keys()):
            _flush_pending(t_len)

        if not asd_scores:
            print("  TalkNet produced no valid frame scores")
            return None
        if self.model_debug:
            total_face_events = face_hits + face_misses
            hit_rate = (face_hits / total_face_events) if total_face_events > 0 else 0.0
            print(
                "  [TALKNET DEBUG] face crop stats: "
                f"hits={face_hits}, misses={face_misses}, hit_rate={hit_rate:.1%}, "
                f"flushes={flush_counter}, scored_chunks={scored_chunks}"
            )

        score_vals = np.asarray(list(asd_scores.values()), dtype=np.float32)
        print(
            "  TalkNet score stats: "
            f"min={float(np.min(score_vals)):.3f}, "
            f"p10={float(np.percentile(score_vals, 10)):.3f}, "
            f"p50={float(np.percentile(score_vals, 50)):.3f}, "
            f"p90={float(np.percentile(score_vals, 90)):.3f}, "
            f"max={float(np.max(score_vals)):.3f}"
        )
        score_spread = float(np.percentile(score_vals, 90) - np.percentile(score_vals, 50))
        min_assignment_margin = max(0.004, 0.18 * score_spread)
        print(
            "  TalkNet assignment gates: "
            f"min_prob={min_talknet_prob:.3f}, margin={min_assignment_margin:.4f}"
        )

        frame_keys = sorted(frame_to_dets.keys())

        def _nearest_frame_idx(target: int, max_gap: int = word_match_max_gap):
            if not frame_keys:
                return None
            pos = bisect_left(frame_keys, target)
            cands = []
            if pos < len(frame_keys):
                cands.append(frame_keys[pos])
            if pos > 0:
                cands.append(frame_keys[pos - 1])
            if not cands:
                return None
            best = min(cands, key=lambda x: abs(x - target))
            if abs(best - target) > max_gap:
                return None
            return best

        def _score_near(tid: str, fi: int) -> float | None:
            deltas = [0]
            for d in range(1, score_lookup_max_delta + 1):
                deltas.extend((-d, d))
            for delta in deltas:
                key = (tid, fi + delta)
                if key in asd_scores:
                    return asd_scores[key]
            return None

        assigned = 0
        words_with_frame = 0
        words_with_dets = 0
        words_with_scored_candidate = 0
        for w in words:
            mid_ms = (int(w["start_time_ms"]) + int(w["end_time_ms"])) // 2
            target_fi = int(round((mid_ms / 1000.0) * fps))
            fi = _nearest_frame_idx(target_fi, max_gap=word_match_max_gap)
            if fi is None:
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"
                continue
            words_with_frame += 1

            dets = frame_to_dets.get(fi, [])
            if not dets:
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"
                continue
            words_with_dets += 1

            best_by_track: dict[str, dict] = {}
            for d in dets:
                tid = str(d["track_id"])
                cur = best_by_track.get(tid)
                if cur is None or float(d.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                    best_by_track[tid] = d

            scored_candidates: list[tuple[float, float, str]] = []
            has_scored_candidate = False
            for tid, d in best_by_track.items():
                s = _score_near(tid, fi)
                if s is None:
                    continue
                has_scored_candidate = True
                # Keep LASER probability dominant; priors only break ties.
                conf = float(d.get("confidence", 0.0))
                total = (0.995 * float(s)) + (0.005 * conf)
                scored_candidates.append((float(total), float(s), tid))
            if has_scored_candidate:
                words_with_scored_candidate += 1

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            if scored_candidates:
                best_total, best_prob, best_tid = scored_candidates[0]
                second_total = scored_candidates[1][0] if len(scored_candidates) > 1 else -1e9
                confident_pick = (
                    best_prob >= min_talknet_prob
                    and (
                        len(scored_candidates) == 1
                        or (best_total - second_total) >= min_assignment_margin
                    )
                )
            else:
                best_tid = None
                confident_pick = False

            if best_tid is not None and confident_pick:
                w["speaker_track_id"] = best_tid
                w["speaker_tag"] = best_tid
                assigned += 1
            else:
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"

        # Smooth local flicker.
        seq = [w.get("speaker_track_id") for w in words]
        smoothed = seq[:]
        win = 2
        for i in range(len(seq)):
            lo = max(0, i - win)
            hi = min(len(seq), i + win + 1)
            neigh = [t for t in seq[lo:hi] if t]
            if not neigh:
                continue
            major, cnt = Counter(neigh).most_common(1)[0]
            if cnt >= 2:
                smoothed[i] = major
        for w, tid in zip(words, smoothed):
            w["speaker_track_id"] = tid
            w["speaker_tag"] = tid or "unknown"

        bindings: list[dict] = []
        cur = None
        for w in words:
            tid = w.get("speaker_track_id")
            ws = int(w["start_time_ms"])
            we = int(w["end_time_ms"])
            if not tid:
                continue
            if cur and cur["track_id"] == tid and ws <= cur["end_time_ms"] + 600:
                cur["end_time_ms"] = we
                cur["word_count"] += 1
            else:
                if cur:
                    bindings.append(cur)
                cur = {
                    "track_id": tid,
                    "start_time_ms": ws,
                    "end_time_ms": we,
                    "word_count": 1,
                }
        if cur:
            bindings.append(cur)

        print(
            "  TalkNet word matching: "
            f"with_frame={words_with_frame}/{len(words)}, "
            f"with_dets={words_with_dets}/{len(words)}, "
            f"with_scored_candidate={words_with_scored_candidate}/{len(words)}"
        )
        assigned_ratio = assigned / max(1, len(words))
        print(f"  TalkNet assignment ratio: {assigned}/{len(words)}={assigned_ratio:.1%}")
        if assigned_ratio < min_talknet_assign_ratio or len(scored_track_ids) < 2:
            print(
                "  TalkNet confidence too low for final binding "
                f"(assigned_ratio={assigned_ratio:.1%}, scored_tracks={len(scored_track_ids)}); "
                "falling back to heuristic binder."
            )
            return None

        print(
            "TalkNet binding complete: "
            f"{assigned}/{len(words)} words assigned, "
            f"{len(scored_track_ids)} scored tracks, {len(bindings)} segments"
        )
        return bindings

    # ──────────────────────────────────────────
    # Active speaker binding fallback (heuristic AV synchrony)
    # ──────────────────────────────────────────
    def _run_speaker_binding_heuristic(
        self,
        video_path: str,
        tracks: list[dict],
        words: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        """Bind each word to a visual track ID using AV synchrony and lip activity cues.

        This is a lightweight in-worker implementation that follows the same
        high-level objective as TalkNet: align speech timing with the most
        likely active on-screen speaker track.
        """
        import cv2
        import math
        from decord import VideoReader, cpu
        from bisect import bisect_left
        from collections import Counter

        if not words or not tracks:
            return []

        face_recognition = None
        try:
            import face_recognition as _face_recognition
            face_recognition = _face_recognition
        except BaseException:
            # Lip-landmark refinement is optional.
            face_recognition = None

        # Ensure we can read the video (use H.264 version if it exists).
        import os
        h264_path = video_path.replace(".mp4", "_h264.mp4")
        read_path = h264_path if os.path.exists(h264_path) else video_path

        try:
            vr = VideoReader(read_path, ctx=cpu(0))
        except Exception as e:
            print(
                "  Warning: could not open video for speaker binding "
                f"({type(e).__name__}: {e})"
            )
            return []

        fps = float(vr.get_avg_fps() or 0.0)
        if fps <= 0.0 or math.isnan(fps):
            fps = 30.0

        # Build or reuse per-track/per-frame indexes.
        if frame_to_dets is None or track_to_dets is None:
            frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

        # Precompute short-term motion per (track, frame) as a speech proxy.
        # Larger local movement around the face/body center tends to correlate
        # with speaking activity in conversational clips.
        motion_score: dict[tuple[str, int], float] = {}
        for tid, dets in track_to_dets.items():
            n = len(dets)
            for i, cur in enumerate(dets):
                prev = dets[i - 1] if i > 0 else cur
                nxt = dets[i + 1] if i + 1 < n else cur
                dx = abs(float(nxt["x_center"]) - float(prev["x_center"]))
                dy = abs(float(nxt["y_center"]) - float(prev["y_center"]))
                dh = abs(float(nxt["height"]) - float(prev["height"]))
                h = max(float(cur.get("height", 1.0)), 1.0)
                # Normalize by bbox height to make scores comparable across scales.
                motion = (0.45 * dx + 0.8 * dy + 1.1 * dh) / h
                motion_score[(tid, int(cur["frame_idx"]))] = float(motion)

        # Frame retrieval cache to avoid repeated random access reads.
        frame_cache: dict[int, object] = {}
        total_frames = len(vr)

        def _get_frame(frame_idx: int):
            if frame_idx in frame_cache:
                return frame_cache[frame_idx]
            if frame_idx < 0 or frame_idx >= total_frames:
                frame_cache[frame_idx] = None
            else:
                try:
                    frame_cache[frame_idx] = vr[frame_idx].asnumpy()  # RGB
                except Exception:
                    frame_cache[frame_idx] = None
            # Keep cache bounded.
            if len(frame_cache) > 96:
                for k in list(frame_cache.keys())[:24]:
                    frame_cache.pop(k, None)
            return frame_cache[frame_idx]

        lip_open_cache: dict[tuple[str, int], float] = {}

        def _lip_open_score(det: dict) -> float:
            """Optional lip landmark openness score for tie-breaking."""
            if face_recognition is None:
                return 0.0

            tid = str(det["track_id"])
            fi = int(det["frame_idx"])
            key = (tid, fi)
            if key in lip_open_cache:
                return lip_open_cache[key]

            frame = _get_frame(fi)
            if frame is None:
                lip_open_cache[key] = 0.0
                return 0.0

            cx, cy = float(det["x_center"]), float(det["y_center"])
            w, h = float(det["width"]), float(det["height"])
            fh, fw = frame.shape[:2]

            # Slightly expand crop to include lower-face region.
            x1 = max(0, int(cx - 0.6 * w))
            y1 = max(0, int(cy - 0.65 * h))
            x2 = min(fw, int(cx + 0.6 * w))
            y2 = min(fh, int(cy + 0.65 * h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                lip_open_cache[key] = 0.0
                return 0.0

            landmarks_list = face_recognition.face_landmarks(crop)
            if not landmarks_list:
                lip_open_cache[key] = 0.0
                return 0.0

            lm = landmarks_list[0]
            top = lm.get("top_lip", [])
            bottom = lm.get("bottom_lip", [])
            if not top or not bottom:
                lip_open_cache[key] = 0.0
                return 0.0

            top_y = sum(p[1] for p in top) / len(top)
            bot_y = sum(p[1] for p in bottom) / len(bottom)
            mouth_open = max(0.0, bot_y - top_y) / max(float(crop.shape[0]), 1.0)
            lip_open_cache[key] = float(mouth_open)
            return lip_open_cache[key]

        frame_keys = sorted(frame_to_dets.keys())

        def _nearest_frame_idx(target: int, max_gap: int = 2):
            if not frame_keys:
                return None
            pos = bisect_left(frame_keys, target)
            cands = []
            if pos < len(frame_keys):
                cands.append(frame_keys[pos])
            if pos > 0:
                cands.append(frame_keys[pos - 1])
            if not cands:
                return None
            best = min(cands, key=lambda x: abs(x - target))
            if abs(best - target) > max_gap:
                return None
            return best

        def _assign_word(word: dict) -> str | None:
            mid_ms = (int(word["start_time_ms"]) + int(word["end_time_ms"])) // 2
            target_fi = int(round((mid_ms / 1000.0) * fps))
            fi = _nearest_frame_idx(target_fi, max_gap=2)
            if fi is None:
                return None

            dets = frame_to_dets.get(fi, [])
            if not dets:
                return None

            # Choose strongest detection per track for this frame.
            best_by_track: dict[str, dict] = {}
            for d in dets:
                tid = str(d["track_id"])
                cur_score = float(d.get("confidence", 0.0))
                old = best_by_track.get(tid)
                if old is None or cur_score > float(old.get("confidence", 0.0)):
                    best_by_track[tid] = d
            candidates = list(best_by_track.values())

            if not candidates:
                return None

            # Primary synchrony score: motion + confidence + area.
            scored = []
            for d in candidates:
                tid = str(d["track_id"])
                conf = float(d.get("confidence", 0.0))
                area = max(1.0, float(d.get("width", 1.0)) * float(d.get("height", 1.0))
                           )
                area_n = min(1.0, area / 50000.0)
                motion = float(motion_score.get((tid, int(d["frame_idx"])), 0.0))
                base = 0.60 * motion + 0.25 * conf + 0.15 * area_n
                scored.append((base, d))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Tie-break with lip-landmark openness if candidates are close.
            if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < 0.08:
                rescored = []
                for base, d in scored[:3]:
                    lip = _lip_open_score(d)
                    rescored.append((base + 0.20 * lip, d))
                rescored.extend(scored[3:])
                rescored.sort(key=lambda x: x[0], reverse=True)
                scored = rescored

            best = scored[0][1]
            return str(best["track_id"])

        assigned = 0
        for w in words:
            tid = _assign_word(w)
            if tid is not None:
                w["speaker_track_id"] = tid
                # Compatibility for downstream phases that still read speaker_tag.
                w["speaker_tag"] = tid
                assigned += 1
            else:
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"

        # Smooth flicker: majority vote in a small local word window.
        track_seq = [w.get("speaker_track_id") for w in words]
        smoothed = track_seq[:]
        win = 2
        for i in range(len(track_seq)):
            lo = max(0, i - win)
            hi = min(len(track_seq), i + win + 1)
            neigh = [t for t in track_seq[lo:hi] if t]
            if not neigh:
                continue
            major, cnt = Counter(neigh).most_common(1)[0]
            if cnt >= 2:
                smoothed[i] = major

        for w, tid in zip(words, smoothed):
            w["speaker_track_id"] = tid
            w["speaker_tag"] = tid or "unknown"

        # Build compact speaker bindings timeline.
        bindings: list[dict] = []
        cur = None
        for w in words:
            tid = w.get("speaker_track_id")
            ws = int(w["start_time_ms"])
            we = int(w["end_time_ms"])
            if not tid:
                continue
            if cur and cur["track_id"] == tid and ws <= cur["end_time_ms"] + 600:
                cur["end_time_ms"] = we
                cur["word_count"] += 1
            else:
                if cur:
                    bindings.append(cur)
                cur = {
                    "track_id": tid,
                    "start_time_ms": ws,
                    "end_time_ms": we,
                    "word_count": 1,
                }
        if cur:
            bindings.append(cur)

        print(
            "Heuristic speaker binding complete: "
            f"{assigned}/{len(words)} words assigned, {len(bindings)} segments"
        )
        return bindings

    # ──────────────────────────────────────────
    # Active speaker binding entrypoint
    # ──────────────────────────────────────────
    def _run_speaker_binding(
        self,
        video_path: str,
        audio_wav_path: str,
        tracks: list[dict],
        words: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        """Bind words to track IDs using TalkNet, with heuristic fallback."""
        bindings = self._run_talknet_binding(
            video_path=video_path,
            audio_wav_path=audio_wav_path,
            tracks=tracks,
            words=words,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
        )
        if bindings is not None:
            return bindings

        print("Running fallback speaker binding heuristic...")
        return self._run_speaker_binding_heuristic(
            video_path,
            tracks,
            words,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
        )

    def _enforce_rollout_gates(self, tracking_metrics: dict | None):
        """Apply rollout gates for continuity quality before downstream steps."""
        import os

        min_idf1 = float(os.getenv("CLYPT_GATE_MIN_IDF1_PROXY", "0.45"))
        min_mota = float(os.getenv("CLYPT_GATE_MIN_MOTA_PROXY", "0.35"))
        max_frag = float(os.getenv("CLYPT_GATE_MAX_FRAGMENTATION", "6.0"))
        min_thr_fps = float(os.getenv("CLYPT_GATE_MIN_THROUGHPUT_FPS", "0.0"))
        max_wallclock_s = float(os.getenv("CLYPT_GATE_MAX_WALLCLOCK_S", "0.0"))
        min_schema_pass = float(os.getenv("CLYPT_GATE_MIN_SCHEMA_PASS_RATE", "1.0"))
        gate_enforce = os.getenv("CLYPT_ENFORCE_ROLLOUT_GATES", "0") == "1"
        metrics = tracking_metrics if isinstance(tracking_metrics, dict) else {}
        idf1_v = float(metrics.get("idf1_proxy", 0.0))
        mota_v = float(metrics.get("mota_proxy", 0.0))
        frag_v = float(metrics.get("track_fragmentation_rate", 999.0))
        thr_v = float(metrics.get("throughput_fps", 0.0))
        wc_v = float(metrics.get("tracking_wallclock_s", 0.0))
        schema_v = float(metrics.get("schema_pass_rate", 1.0))
        gate_ok = idf1_v >= min_idf1 and mota_v >= min_mota and frag_v <= max_frag
        if min_thr_fps > 0.0:
            gate_ok = gate_ok and (thr_v >= min_thr_fps)
        if max_wallclock_s > 0.0:
            gate_ok = gate_ok and (wc_v <= max_wallclock_s)
        gate_ok = gate_ok and (schema_v >= min_schema_pass)
        print(
            "  Rollout gates: "
            f"idf1_proxy={idf1_v:.3f}>={min_idf1:.3f}, "
            f"mota_proxy={mota_v:.3f}>={min_mota:.3f}, "
            f"frag={frag_v:.3f}<={max_frag:.3f}, "
            f"throughput={thr_v:.2f}>={min_thr_fps:.2f}, "
            f"wallclock={wc_v:.1f}<={max_wallclock_s:.1f}, "
            f"schema_pass={schema_v:.3f}>={min_schema_pass:.3f} -> "
            f"{'PASS' if gate_ok else 'FAIL'}"
        )
        if gate_enforce and not gate_ok:
            raise RuntimeError("Rollout gates failed for tracking quality/continuity")

    def _finalize_from_words_tracks(
        self,
        video_path: str,
        audio_path: str,
        youtube_url: str,
        words: list[dict],
        tracks: list[dict],
        tracking_metrics: dict | None = None,
    ) -> dict:
        """Finalize extraction from precomputed ASR words + tracking tracks."""
        metrics = dict(tracking_metrics) if isinstance(tracking_metrics, dict) else {}
        metrics["schema_pass_rate"] = self._tracking_contract_pass_rate(tracks)
        self._validate_tracking_contract(tracks)
        self._enforce_rollout_gates(metrics)

        _, track_to_dets = self._build_track_indexes(tracks)

        # Step 3: Global tracklet clustering
        print("[Phase 1] Step 3/4: Clustering tracklets into global IDs...")
        tracks = self._cluster_tracklets(
            video_path,
            tracks,
            track_to_dets=track_to_dets,
        )
        frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

        # Step 4: Speaker binding
        print("[Phase 1] Step 4/4: Running speaker binding...")
        speaker_bindings = self._run_speaker_binding(
            video_path,
            audio_path,
            tracks,
            words,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
        )
        geometry_type = (
            "mixed"
            if any(str(t.get("geometry_type", "")) == "obb" for t in tracks)
            else PHASE1_GEOMETRY_TYPE
        )

        phase_1_visual = {
            "source_video": youtube_url,
            "schema_version": PHASE1_SCHEMA_VERSION,
            "task_type": PHASE1_TASK_TYPE,
            "coordinate_space": PHASE1_COORDINATE_SPACE,
            "geometry_type": geometry_type,
            "class_taxonomy": PHASE1_CLASS_TAXONOMY,
            "tracking_metrics": metrics,
            "tracks": tracks,
        }

        phase_1_audio = {
            "source_audio": youtube_url,
            "words": words,
            "speaker_bindings": speaker_bindings,
        }

        print(f"[Phase 1] Complete — {len(words)} words, {len(tracks)} tracks")
        return {
            "status": "success",
            "phase_1_visual": phase_1_visual,
            "phase_1_audio": phase_1_audio,
            # Backward compatibility for older clients still expecting 1A keys.
            "phase_1a_visual": phase_1_visual,
            "phase_1a_audio": phase_1_audio,
        }

    @modal.method()
    def stage_video_for_tracking(self, video_bytes: bytes) -> dict:
        """Persist video bytes to volume, normalize codec, and return chunk plan."""
        import os
        import json
        import uuid

        job_id = uuid.uuid4().hex
        job_dir = f"/vol/clypt-chunks/jobs/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        raw_video_path = f"{job_dir}/video.mp4"
        with open(raw_video_path, "wb") as f:
            f.write(video_bytes)

        video_path = self._ensure_h264(raw_video_path)
        meta = self._probe_video_meta(video_path)
        if int(meta.get("total_frames", 0)) <= 0:
            raise RuntimeError("Could not stage video for tracking: no decodable frames")

        chunks = self._build_chunk_plan(int(meta["total_frames"]), float(meta["fps"]))
        manifest = {
            "job_id": job_id,
            "video_path": video_path,
            "meta": meta,
            "chunks": chunks,
        }
        with open(f"{job_dir}/manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        try:
            TRACKING_VOLUME.commit()
            TRACKING_VOLUME.reload()
        except Exception:
            pass

        print(
            f"[Phase 1] Staged tracking job {job_id[:8]}... "
            f"{meta['total_frames']} frames, {len(chunks)} chunks"
        )
        return manifest

    @modal.method()
    def run_asr_only(self, audio_wav_bytes: bytes) -> list[dict]:
        """Run ASR-only path for distributed fan-out workflow."""
        import os
        import uuid

        dl_dir = "/tmp/clypt"
        os.makedirs(dl_dir, exist_ok=True)
        audio_path = f"{dl_dir}/audio_only_{uuid.uuid4().hex}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_wav_bytes)
        try:
            return self._run_asr(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    @modal.method()
    def track_chunk_from_staged(
        self,
        job_id: str,
        video_path: str,
        meta: dict,
        chunk: dict,
    ) -> dict:
        """Track one chunk from a staged volume video in a separate GPU container."""
        import os

        _ = job_id
        TRACKING_VOLUME.reload()
        tracker_cfg = self._ensure_botsort_reid_yaml()
        chunk_dir = f"/vol/clypt-chunks/jobs/{job_id}/chunks"
        os.makedirs(chunk_dir, exist_ok=True)
        return self._track_single_chunk(video_path, meta, chunk, tracker_cfg, chunk_dir)

    @modal.method()
    def stitch_tracking_chunks(self, chunk_results: list[dict], fps: float) -> dict:
        """Stitch distributed chunk outputs into global track IDs."""
        tracks, metrics = self._stitch_chunk_tracks(chunk_results, fps=fps)
        return {"tracks": tracks, "tracking_metrics": metrics}

    @modal.method()
    def finalize_extraction(
        self,
        video_bytes: bytes,
        audio_wav_bytes: bytes,
        youtube_url: str,
        words: list[dict],
        tracks: list[dict],
        tracking_metrics: dict | None = None,
    ) -> dict:
        """Finalize from externally-fanned-out ASR/tracking outputs."""
        import os
        import uuid

        dl_dir = "/tmp/clypt"
        os.makedirs(dl_dir, exist_ok=True)
        suffix = uuid.uuid4().hex
        video_path = f"{dl_dir}/video_finalize_{suffix}.mp4"
        audio_path = f"{dl_dir}/audio_finalize_{suffix}.wav"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        with open(audio_path, "wb") as f:
            f.write(audio_wav_bytes)

        try:
            # Avoid mutating caller-owned objects in-place.
            words_local = [dict(w) for w in words]
            tracks_local = [dict(t) for t in tracks]
            return self._finalize_from_words_tracks(
                video_path=video_path,
                audio_path=audio_path,
                youtube_url=youtube_url,
                words=words_local,
                tracks=tracks_local,
                tracking_metrics=tracking_metrics if isinstance(tracking_metrics, dict) else {},
            )
        finally:
            h264_video_path = video_path.replace(".mp4", "_h264.mp4")
            for p in (video_path, audio_path, h264_video_path):
                if os.path.exists(p):
                    os.remove(p)

    @modal.method()
    def cleanup_tracking_job(self, job_id: str) -> None:
        """Remove staged chunk artifacts for a distributed tracking job."""
        import os
        import shutil

        job_dir = f"/vol/clypt-chunks/jobs/{job_id}"
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
        try:
            TRACKING_VOLUME.commit()
        except Exception:
            pass

    # ──────────────────────────────────────────
    # Main extraction method (called remotely)
    # ──────────────────────────────────────────
    @modal.method()
    def extract(
        self,
        video_bytes: bytes,
        audio_wav_bytes: bytes,
        youtube_url: str,
    ) -> dict:
        """Run the full Phase 1 extraction stack on pre-downloaded media.

        Args:
            video_bytes: Muxed MP4 video (for visual tracking).
            audio_wav_bytes: 16kHz mono WAV (for ASR).
            youtube_url: Original URL (for metadata only).

        Returns:
            dict with phase_1_visual and phase_1_audio payloads.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        dl_dir = "/tmp/clypt"
        os.makedirs(dl_dir, exist_ok=True)

        video_path = f"{dl_dir}/video.mp4"
        audio_path = f"{dl_dir}/audio_16k.wav"

        with open(video_path, "wb") as f:
            f.write(video_bytes)
        with open(audio_path, "wb") as f:
            f.write(audio_wav_bytes)

        video_mb = len(video_bytes) / 1e6
        audio_mb = len(audio_wav_bytes) / 1e6
        print(f"[Phase 1] Received video ({video_mb:.1f} MB) + audio ({audio_mb:.1f} MB)")

        try:
            # Step 1+2: Run ASR and tracking concurrently on separate modalities.
            print("[Phase 1] Step 1+2/4: Running Parakeet ASR + YOLO26 tracking concurrently...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                asr_future = pool.submit(self._run_asr, audio_path)
                track_future = pool.submit(self._run_tracking, video_path)
                words = asr_future.result()
                tracks, tracking_metrics = track_future.result()
            result = self._finalize_from_words_tracks(
                video_path=video_path,
                audio_path=audio_path,
                youtube_url=youtube_url,
                words=words,
                tracks=tracks,
                tracking_metrics=tracking_metrics if isinstance(tracking_metrics, dict) else {},
            )

            # Cleanup
            h264_video_path = video_path.replace(".mp4", "_h264.mp4")
            for p in (video_path, audio_path, h264_video_path):
                if os.path.exists(p):
                    os.remove(p)

            return result

        except Exception as e:
            print(f"[Phase 1] Error: {e}")
            return {"status": "error", "message": str(e)}
