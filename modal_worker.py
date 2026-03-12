"""
Clypt Phase 1 — Modal GPU Worker
=================================
Serverless GPU microservice that performs deterministic multimodal extraction:

  1. NVIDIA Parakeet-TDT → word-level ASR with timestamps + punctuation
  2. YOLOv11 + BoT-SORT  → dense person/face tracking with persistent IDs
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
MODEL_DEBUG_SECRET = modal.Secret.from_dict(
    {
        "CLYPT_MODEL_DEBUG": "1",
        "CLYPT_MODEL_DEBUG_EVERY": "10",
    }
)

ASR_MODEL_NAME = "nvidia/parakeet-tdt-1.1b"
TALKNET_MODEL_PATH = "/root/.cache/clypt/pretrain_TalkSet.model"
TALKNET_REPO_ROOT = "/root/talknet_asd"
YOLO_WEIGHTS_PATH = "yolo11s.pt"
YOLO_ENGINE_PATH = "/root/.cache/clypt/yolo11s.engine"


def download_asr_model():
    """Download Parakeet weights at image build time so they're cached."""
    import nemo.collections.asr as nemo_asr

    print(f"Downloading {ASR_MODEL_NAME} weights into container cache...")
    nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)


def download_yolo_model():
    """Download YOLO11 weights at image build time so they're cached."""
    from ultralytics import YOLO

    print("Downloading YOLO11s weights into container cache...")
    YOLO(YOLO_WEIGHTS_PATH)


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
        "python_speech_features",
        "pandas",
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
    .run_function(download_talknet_model)
    .run_function(download_insightface_model)
)


# ──────────────────────────────────────────────
# GPU Worker (class-based for VRAM persistence)
# ──────────────────────────────────────────────
@app.cls(image=clypt_image, gpu="H100", timeout=1800, secrets=[MODEL_DEBUG_SECRET])
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

        # --- Load YOLOv11 ---
        print("Loading YOLO11s into GPU VRAM...")
        self.yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        try:
            # Avoid runtime TensorRT export on cold start; only load an existing
            # prebuilt engine if it is already present.
            if os.path.exists(YOLO_ENGINE_PATH):
                print("Loading YOLO11s TensorRT engine...")
                self.yolo_model = YOLO(YOLO_ENGINE_PATH)
            else:
                print("Using PyTorch YOLO11s weights (no runtime TensorRT export).")
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

    def _run_tracking(self, video_path: str) -> list[dict]:
        """Run YOLO11 + BoT-SORT for dense person tracking with persistent IDs."""
        import cv2
        import time

        print("Running YOLO11s + BoT-SORT tracking inference...")

        # Ensure OpenCV can decode the video (AV1/VP9 often fail)
        video_path = self._ensure_h264(video_path)

        total_frames = 0
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        log_every_n_frames = 600
        started_at = time.time()

        # stream=True returns a generator, preventing RAM overload on long videos.
        # classes=[0] ensures we ONLY track people.
        results = self.yolo_model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            classes=[0],
            stream=True,
            verbose=False,
        )

        tracks = []
        n_boxes = 0
        for frame_idx, r in enumerate(results, start=1):
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xywh.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    cx, cy, w, h = box
                    tracks.append({
                        "frame_idx": frame_idx - 1,
                        "track_id": f"track_{track_id}",
                        "x_center": float(cx),
                        "y_center": float(cy),
                        "width": float(w),
                        "height": float(h),
                        "confidence": float(conf),
                    })
                    n_boxes += 1

            if frame_idx % log_every_n_frames == 0:
                elapsed = max(1e-6, time.time() - started_at)
                fps_eff = frame_idx / elapsed
                if total_frames > 0:
                    pct = (100.0 * frame_idx) / max(1, total_frames)
                    print(
                        "  YOLO progress: "
                        f"{frame_idx}/{total_frames} frames ({pct:.1f}%), "
                        f"{n_boxes} boxes, {fps_eff:.1f} fps"
                    )
                else:
                    print(
                        "  YOLO progress: "
                        f"{frame_idx} frames, {n_boxes} boxes, {fps_eff:.1f} fps"
                    )

        print(f"Tracking complete: {len(tracks)} bounding boxes across frames.")
        return tracks

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
            print("[Phase 1] Step 1+2/4: Running Parakeet ASR + YOLO11 tracking concurrently...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                asr_future = pool.submit(self._run_asr, audio_path)
                track_future = pool.submit(self._run_tracking, video_path)
                words = asr_future.result()
                tracks = track_future.result()
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

            # Cleanup
            h264_video_path = video_path.replace(".mp4", "_h264.mp4")
            for p in (video_path, audio_path, h264_video_path):
                if os.path.exists(p):
                    os.remove(p)

            phase_1_visual = {
                "source_video": youtube_url,
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

        except Exception as e:
            print(f"[Phase 1] Error: {e}")
            return {"status": "error", "message": str(e)}
