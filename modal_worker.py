"""
Clypt Phase 1A — Modal GPU Worker
===================================
Serverless GPU microservice that performs deterministic multimodal extraction:

  1. NVIDIA Parakeet-TDT → word-level ASR with timestamps + punctuation
  2. YOLOv11 + BoT-SORT  → dense person/face tracking with persistent IDs
  3. TalkNCE + LASER     → active speaker binding (audio-visual sync)

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

ASR_MODEL_NAME = "nvidia/parakeet-tdt-1.1b"
TALKNCE_LASER_MODEL_GDRIVE_ID = "1N8nFVybKXL7NFzJHMfo9x8FjJkUzEUc_"
TALKNCE_LASER_MODEL_PATH = "/root/.cache/clypt/talknce_laser.model"
VGGISH_WEIGHTS_URL = (
    "https://github.com/harritaylor/torchvggish/releases/download/"
    "v0.1/vggish-10086976.pth"
)


def download_asr_model():
    """Download Parakeet weights at image build time so they're cached."""
    import nemo.collections.asr as nemo_asr

    print(f"Downloading {ASR_MODEL_NAME} weights into container cache...")
    nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)


def download_yolo_model():
    """Download YOLO11 weights at image build time so they're cached."""
    from ultralytics import YOLO

    print("Downloading YOLO11s weights into container cache...")
    YOLO("yolo11s.pt")


def download_talknce_laser_model():
    """Cache TalkNCE+LASER checkpoint + VGGish weights at image build time."""
    import os
    import gdown
    import torch

    os.makedirs(os.path.dirname(TALKNCE_LASER_MODEL_PATH), exist_ok=True)
    if not os.path.exists(TALKNCE_LASER_MODEL_PATH):
        url = f"https://drive.google.com/uc?id={TALKNCE_LASER_MODEL_GDRIVE_ID}"
        print("Downloading TalkNCE+LASER checkpoint...")
        gdown.download(url, TALKNCE_LASER_MODEL_PATH, quiet=False)
    else:
        print("TalkNCE+LASER checkpoint already cached.")

    # LoCoNet's audio branch depends on these VGGish weights.
    torch.hub.load_state_dict_from_url(
        VGGISH_WEIGHTS_URL,
        model_dir="/root/.cache/torch/hub/checkpoints",
        progress=True,
    )


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
        "laser-asd",
        "gdown",
        "pandas",
        "tqdm",
        "matplotlib",
        "imageio",
        "Pillow",
        "resampy",
        "soundfile",
    )
    # Step 2: dlib + face_recognition (dlib compiles from C++, ~5 min first time)
    .run_commands("pip install setuptools dlib face_recognition")
    # Step 3: NeMo ASR (separate layer so dlib layer stays cached)
    .pip_install("nemo_toolkit[asr]")
    # Step 4: Ensure setuptools survives NeMo's dep resolution and pkg_resources is available
    # face_recognition_models still imports pkg_resources, removed in setuptools>=81
    .run_commands("pip install --force-reinstall 'setuptools<81'")
    .run_commands("python -c \"import setuptools, pkg_resources; print(setuptools.__version__)\"")
    # Step 5: Cache model weights at build time
    .run_function(download_asr_model)
    .run_function(download_yolo_model)
    .run_function(download_talknce_laser_model)
)


# ──────────────────────────────────────────────
# GPU Worker (class-based for VRAM persistence)
# ──────────────────────────────────────────────
@app.cls(image=clypt_image, gpu="H100", timeout=1800)
class ClyptWorker:

    @staticmethod
    def _build_laser_cfg():
        """Build minimal config object expected by LASER's LoCoNet class."""
        class ModelConfig:
            NUM_SPEAKERS = 3
            CLIP_LENGTH = 200
            AV = "speaker_temporal"
            AV_layers = 3
            ADJUST_ATTENTION = 0
            AUDIO_MODEL = "resnet18"
            VISUAL_MODEL = "resnet18"

        class DataConfig:
            numWorkers = 1
            dataPathAVA = "/tmp"

        class Config:
            DATA = DataConfig()
            MODEL = ModelConfig()

        return Config()

    @staticmethod
    def _load_talknce_laser_checkpoint(model, ckpt_path: str):
        """Load TalkNCE+LASER weights with key-prefix normalization."""
        import torch

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format: {type(state)}")

        normalized = {}
        for key, value in state.items():
            if key.startswith("model.module."):
                key = key[len("model.module."):]
            elif key.startswith("module."):
                key = key[len("module."):]
            normalized[key] = value

        missing, unexpected = model.load_state_dict(normalized, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "TalkNCE+LASER checkpoint mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

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

    def _laser_forward_scores(
        self,
        audio_t,
        visual_t,
        landmark_t,
        use_landmark: bool = False,
    ):
        """Batched LASER forward that supports B>1 safely.

        Returns per-frame speaking probabilities in [0, 1].
        """
        import torch

        b, s, t = visual_t.shape[:3]
        landmark_feature = self.laser_model.create_landmark_tensor(
            landmark_t, visual_t.dtype, visual_t.device
        )
        landmark_feature = self.laser_model.landmark_bottleneck(landmark_feature)
        if not use_landmark:
            landmark_feature = torch.zeros_like(landmark_feature)

        visual_flat = visual_t.reshape(b * s, *visual_t.shape[2:])

        audio_embed = self.laser_model.model.forward_audio_frontend(audio_t)
        visual_embed = self.laser_model.forward_visual_frontend(visual_flat, landmark_feature)
        # Match flatten order [b0s0,b0s1,...,b1s0,b1s1,...].
        audio_embed = audio_embed.repeat_interleave(s, dim=0)
        audio_embed, visual_embed = self.laser_model.model.forward_cross_attention(
            audio_embed, visual_embed
        )
        outs_av = self.laser_model.model.forward_audio_visual_backend(
            audio_embed, visual_embed, b, s
        )
        outs_av = outs_av.reshape(b, s, t, -1)[:, 0, :, :].reshape(b * t, -1)

        # Use calibrated class-1 probability rather than raw logit.
        av_logits = self.laser_model.lossAV.FC(outs_av)
        av_prob = torch.softmax(av_logits, dim=-1)[:, 1].reshape(b, t)
        return av_prob

    @modal.enter()
    def load_model(self):
        """Load Parakeet, YOLO, and TalkNCE+LASER into GPU VRAM."""
        import nemo.collections.asr as nemo_asr
        import torch
        from ultralytics import YOLO
        from omegaconf import open_dict
        from laser_asd.landmark_loconet import Loconet
        from laser_asd.torchvggish import vggish_input

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
        self.yolo_model = YOLO("yolo11s.pt")

        # --- Load TalkNCE+LASER ---
        self.laser_model = None
        self.laser_vggish_input = vggish_input
        self.laser_device = torch.device("cuda")

        try:
            print("Loading TalkNCE+LASER model into GPU VRAM...")
            self.laser_model = Loconet(self._build_laser_cfg(), n_channel=4, layer=1)
            self._load_talknce_laser_checkpoint(
                self.laser_model, TALKNCE_LASER_MODEL_PATH
            )
            self.laser_model = self.laser_model.to(self.laser_device)
            self.laser_model.eval()
            print("TalkNCE+LASER ready.")
        except Exception as e:
            # Keep worker alive; binding step falls back if this fails at runtime.
            self.laser_model = None
            print(
                "Warning: failed to load TalkNCE+LASER checkpoint "
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
                    "speaker_track_id": None,  # populated by TalkNCE later
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
            print(f"  Re-encoding {codec} → H.264 for OpenCV compatibility...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "23", "-an", h264_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            tracker="bytetrack.yaml",
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
    # Global tracklet clustering (face_recognition + DBSCAN)
    # ──────────────────────────────────────────
    def _cluster_tracklets(
        self,
        video_path: str,
        tracks: list[dict],
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        """Cluster fragmented BoT-SORT track IDs into global person IDs via face embeddings."""
        import cv2
        import numpy as np
        from sklearn.cluster import DBSCAN

        face_recognition = None
        try:
            import face_recognition as _face_recognition
            face_recognition = _face_recognition
        except BaseException as e:
            print(
                "  Warning: face_recognition unavailable "
                f"({type(e).__name__}: {e}). Falling back to histogram-only clustering."
            )

        if not tracks:
            return tracks

        # Keep clustering cost bounded while giving each tracklet multiple chances.
        max_frames_per_tracklet = 6
        max_ranked_candidates = 18
        target_face_encodings_per_tracklet = 2
        dbscan_eps = 0.5
        dbscan_min_samples = 1
        reassign_noise_max_dist = 0.75
        # Conservative tiny-cluster merge: avoid collapsing real speakers.
        tiny_cluster_min_tracklets = 2
        tiny_cluster_min_boxes = max(35, int(len(tracks) * 0.002))
        tiny_cluster_merge_max_dist = 0.72

        # Ensure we can read the video (use H.264 version if it exists)
        h264_path = video_path.replace(".mp4", "_h264.mp4")
        import os
        read_path = h264_path if os.path.exists(h264_path) else video_path

        if track_to_dets is None:
            _, track_to_dets = self._build_track_indexes(tracks)
        tracklets = track_to_dets

        unique_ids = sorted(tracklets.keys())
        print(f"Clustering {len(unique_ids)} fragmented track IDs...")

        embeddings = {}  # track_id → 128D face encoding
        fallback_ids = []  # track_ids where face encoding failed

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

        # Decode all sampled frames in one sequential pass.
        frame_map: dict[int, object] = {}
        cap = cv2.VideoCapture(read_path)
        if not cap.isOpened():
            print("  Warning: could not open video for clustering, skipping")
            return tracks
        try:
            sorted_needed = sorted(needed_frames)
            first_needed = sorted_needed[0]
            last_needed = sorted_needed[-1]
            needed_lookup = set(sorted_needed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)
            frame_idx = first_needed
            while frame_idx <= last_needed:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx in needed_lookup:
                    frame_map[frame_idx] = frame.copy()
                frame_idx += 1
        finally:
            cap.release()

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

                # Convert YOLO center coords to crop bounds
                cx, cy = float(det["x_center"]), float(det["y_center"])
                w, h = float(det["width"]), float(det["height"])
                fh, fw = frame.shape[:2]
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(fw, int(cx + w / 2))
                y2 = min(fh, int(cy + h / 2))

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                used_face_encoding = False
                if face_recognition is not None:
                    # Option 1: explicit face detections guide the embedding crop.
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    face_locs = face_recognition.face_locations(
                        rgb_crop,
                        number_of_times_to_upsample=0,
                        model="hog",
                    )
                    if face_locs:
                        best_loc = max(
                            face_locs,
                            key=lambda loc: max(0, loc[2] - loc[0]) * max(0, loc[1] - loc[3]),
                        )
                        encs = face_recognition.face_encodings(
                            rgb_crop,
                            known_face_locations=[best_loc],
                            num_jitters=1,
                        )
                        if encs:
                            face_vectors.append(encs[0])
                            used_face_encoding = True

                if not used_face_encoding:
                    hist = cv2.calcHist(
                        [crop], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256],
                    )
                    hist = cv2.normalize(hist, hist).flatten()
                    if len(hist) < 128:
                        hist = np.pad(hist, (0, 128 - len(hist)))
                    else:
                        hist = hist[:128]
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
                embeddings[tid] = np.zeros(128, dtype=np.float32)

        if not embeddings:
            print("  No embeddings extracted, skipping clustering")
            return tracks

        # Run DBSCAN on the collected embeddings
        tid_order = sorted(embeddings.keys())
        X = np.array([embeddings[tid] for tid in tid_order])

        # Separate face embeddings from histogram fallbacks for better clustering
        face_tids = [tid for tid in tid_order if tid not in fallback_ids]
        hist_tids = [tid for tid in tid_order if tid in fallback_ids]

        print(f"  Face encodings: {len(face_tids)}, histogram fallbacks: {len(hist_tids)}")

        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean").fit(X)
        labels = db.labels_.astype(int)

        raw_clusters = len(set(labels) - {-1})
        raw_noise = int((labels == -1).sum())
        print(f"  DBSCAN raw: {raw_clusters} clusters, {raw_noise} noise tracklets")

        if raw_clusters == 0:
            print("  DBSCAN produced only noise. Falling back to min_samples=1.")
            db = DBSCAN(eps=dbscan_eps, min_samples=1, metric="euclidean").fit(X)
            labels = db.labels_.astype(int)

        def _cluster_to_indices(current_labels: np.ndarray) -> dict[int, list[int]]:
            out: dict[int, list[int]] = {}
            for idx, lbl in enumerate(current_labels):
                if lbl < 0:
                    continue
                out.setdefault(int(lbl), []).append(idx)
            return out

        # Reassign DBSCAN noise points to nearest existing centroid when close enough.
        cluster_map = _cluster_to_indices(labels)
        reassigned_noise = 0
        if cluster_map:
            centroids = {
                lbl: np.mean(X[idxs], axis=0)
                for lbl, idxs in cluster_map.items()
            }
            for idx in np.where(labels == -1)[0]:
                vec = X[idx]
                best_label, best_dist = min(
                    (
                        (lbl, float(np.linalg.norm(vec - centroid)))
                        for lbl, centroid in centroids.items()
                    ),
                    key=lambda kv: kv[1],
                )
                if best_dist <= reassign_noise_max_dist:
                    labels[idx] = int(best_label)
                    reassigned_noise += 1
        if reassigned_noise:
            print(f"  Noise reassigned to nearest centroid: {reassigned_noise}")

        # Merge tiny, weakly-supported clusters into nearest stable cluster.
        cluster_map = _cluster_to_indices(labels)
        cluster_support = {
            lbl: {
                "tracklets": len(idxs),
                "boxes": sum(len(tracklets[tid_order[i]]) for i in idxs),
            }
            for lbl, idxs in cluster_map.items()
        }

        tiny_labels = {
            lbl for lbl, s in cluster_support.items()
            if s["tracklets"] < tiny_cluster_min_tracklets and s["boxes"] < tiny_cluster_min_boxes
        }
        stable_labels = [lbl for lbl in cluster_map.keys() if lbl not in tiny_labels]

        merged_tiny = 0
        if tiny_labels and stable_labels:
            stable_centroids = {
                lbl: np.mean(X[cluster_map[lbl]], axis=0)
                for lbl in stable_labels
            }
            for tiny_lbl in sorted(tiny_labels):
                tiny_idxs = cluster_map[tiny_lbl]
                tiny_centroid = np.mean(X[tiny_idxs], axis=0)
                best_label, best_dist = min(
                    (
                        (lbl, float(np.linalg.norm(tiny_centroid - centroid)))
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

        # Normalize labels to contiguous IDs for deterministic Global_Person naming.
        unique_final_labels = sorted(set(int(lbl) for lbl in labels))
        final_label_map = {old: new for new, old in enumerate(unique_final_labels)}
        labels = np.array([final_label_map[int(lbl)] for lbl in labels], dtype=int)

        # Build mapping: fragmented track_id → Global_Person_X
        id_map = {}
        for tid, label in zip(tid_order, labels):
            id_map[tid] = f"Global_Person_{int(label)}"

        n_clusters = len(set(labels))
        print(f"  DBSCAN clusters: {n_clusters} global persons "
              f"(from {len(unique_ids)} fragments)")

        # Apply mapping to all tracks
        for t in tracks:
            old_id = t["track_id"]
            if old_id in id_map:
                t["track_id"] = id_map[old_id]

        return tracks

    def _run_talknce_laser_binding(
        self,
        video_path: str,
        audio_wav_path: str,
        tracks: list[dict],
        words: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
    ) -> list[dict] | None:
        """Run true TalkNCE+LASER inference and map words to visual track IDs."""
        import cv2
        import os
        import numpy as np
        import soundfile as sf
        import torch
        from bisect import bisect_left
        from collections import Counter

        if self.laser_model is None:
            print("  TalkNCE+LASER unavailable; falling back to heuristic binder.")
            return None
        if not words or not tracks:
            return []

        h264_path = video_path.replace(".mp4", "_h264.mp4")
        read_path = h264_path if os.path.exists(h264_path) else video_path

        cap = cv2.VideoCapture(read_path)
        if not cap.isOpened():
            print("  Warning: could not open video for TalkNCE+LASER binding")
            return None

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0.0:
                fps = 25.0

            wav, sr = sf.read(audio_wav_path, dtype="float32")
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            wav = wav.astype(np.float32, copy=False)
            if sr <= 0:
                print("  Warning: invalid audio sample rate for TalkNCE+LASER binding")
                return None

            if frame_to_dets is None or track_to_dets is None:
                frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

            frame_cache: dict[int, object] = {}
            last_decode_idx = -2

            def _get_frame(frame_idx: int):
                nonlocal last_decode_idx
                if frame_idx in frame_cache:
                    return frame_cache[frame_idx]

                # Fast path: sequential decode avoids costly random seeks.
                if frame_idx == last_decode_idx + 1:
                    ok, frame = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                last_decode_idx = frame_idx

                frame_cache[frame_idx] = frame if ok and frame is not None else None
                if len(frame_cache) > 192:
                    for k in list(frame_cache.keys())[:48]:
                        frame_cache.pop(k, None)
                return frame_cache[frame_idx]

            def _extract_face_crop(frame, det: dict):
                fh, fw = frame.shape[:2]
                cx = float(det["x_center"])
                cy = float(det["y_center"])
                w = float(det["width"])
                h = float(det["height"])

                # Person box -> head-focused crop approximation.
                x1 = max(0, int(cx - 0.36 * w))
                x2 = min(fw, int(cx + 0.36 * w))
                y1 = max(0, int(cy - 0.62 * h))
                y2 = min(fh, int(cy + 0.05 * h))
                if x2 <= x1 or y2 <= y1:
                    return None

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

            asd_scores: dict[tuple[str, int], float] = {}
            scored_track_ids = set()
            chunk_size = 120
            min_chunk_frames = 20
            laser_batch_size = 6
            contiguous_frame_gap = 2
            word_match_max_gap = 4
            score_lookup_max_delta = 8
            min_laser_prob = 0.50

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
                best_by_frame: dict[int, dict] = {}
                for d in dets:
                    fi = int(d["frame_idx"])
                    old = best_by_frame.get(fi)
                    if old is None or float(d.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
                        best_by_frame[fi] = d
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

            def _flush_pending(t_len: int):
                nonlocal scored_chunks
                pending = pending_by_t.get(t_len, [])
                if not pending:
                    return

                visual_batch = np.stack([p[2] for p in pending], axis=0)
                audio_batch = np.stack([p[3] for p in pending], axis=0)

                visual_t = (
                    torch.from_numpy(visual_batch)
                    .float()
                    .to(self.laser_device)
                    .unsqueeze(1)
                    .repeat(1, 3, 1, 1, 1)
                )
                audio_t = (
                    torch.from_numpy(audio_batch)
                    .float()
                    .to(self.laser_device)
                    .unsqueeze(1)
                )
                landmark_t = torch.zeros(
                    (len(pending), 3, t_len, 82, 2),
                    dtype=torch.float32,
                    device=self.laser_device,
                )

                with torch.no_grad():
                    score_bt = self._laser_forward_scores(
                        audio_t,
                        visual_t,
                        landmark_t,
                        use_landmark=False,
                    )
                score_np = score_bt.detach().float().cpu().numpy()

                for i, (tid, valid_frames, _, _) in enumerate(pending):
                    for fi, sc in zip(valid_frames, score_np[i]):
                        asd_scores[(tid, fi)] = float(sc)
                    scored_track_ids.add(tid)
                    scored_chunks += 1

                pending_by_t[t_len] = []

            for tid, dets in track_to_dets.items():
                best_by_frame: dict[int, dict] = {}
                for d in dets:
                    fi = int(d["frame_idx"])
                    old = best_by_frame.get(fi)
                    if old is None or float(d.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
                        best_by_frame[fi] = d

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

                        crops = []
                        valid_frames = []
                        for fi in chunk_frames:
                            frame = _get_frame(fi)
                            if frame is None:
                                continue
                            crop = _extract_face_crop(frame, best_by_frame[fi])
                            if crop is None:
                                continue
                            crops.append(crop)
                            valid_frames.append(fi)

                        t = len(valid_frames)
                        if t < min_chunk_frames:
                            continue

                        start_idx = int((valid_frames[0] / fps) * sr)
                        end_idx = int(((valid_frames[-1] + 1) / fps) * sr)
                        start_idx = max(0, start_idx)
                        end_idx = min(len(wav), end_idx)
                        if end_idx - start_idx < int(0.2 * sr):
                            continue

                        wav_seg = wav[start_idx:end_idx]
                        if wav_seg.size == 0:
                            continue

                        visual_np = np.stack(
                            [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops], axis=0
                        )

                        audio_np = self.laser_vggish_input.waveform_to_examples(
                            wav_seg, sr, t, fps, return_tensor=False
                        )
                        pending_by_t.setdefault(t, []).append((tid, valid_frames, visual_np, audio_np))
                        if len(pending_by_t[t]) >= laser_batch_size:
                            _flush_pending(t)

                        if chunk_counter % 40 == 0:
                            print(
                                "  TalkNCE/LASER progress: "
                                f"{chunk_counter}/{max(total_chunks, 1)} chunks, "
                                f"scored_chunks={scored_chunks}, scored_tracks={len(scored_track_ids)}"
                            )

            for t_len in list(pending_by_t.keys()):
                _flush_pending(t_len)

            if not asd_scores:
                print("  TalkNCE+LASER produced no valid frame scores")
                return None

            score_vals = np.asarray(list(asd_scores.values()), dtype=np.float32)
            print(
                "  TalkNCE/LASER score stats: "
                f"min={float(np.min(score_vals)):.3f}, "
                f"p10={float(np.percentile(score_vals, 10)):.3f}, "
                f"p50={float(np.percentile(score_vals, 50)):.3f}, "
                f"p90={float(np.percentile(score_vals, 90)):.3f}, "
                f"max={float(np.max(score_vals)):.3f}"
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

                best_tid = None
                best_score = -1e9
                best_laser_prob = -1.0
                has_scored_candidate = False
                for tid, d in best_by_track.items():
                    s = _score_near(tid, fi)
                    if s is None:
                        continue
                    has_scored_candidate = True
                    conf = float(d.get("confidence", 0.0))
                    area = max(1.0, float(d.get("width", 1.0)) * float(d.get("height", 1.0)))
                    area_n = min(1.0, area / 50000.0)
                    total = (0.92 * float(s)) + (0.06 * conf) + (0.02 * area_n)
                    if total > best_score:
                        best_score = total
                        best_laser_prob = float(s)
                        best_tid = tid
                if has_scored_candidate:
                    words_with_scored_candidate += 1

                if best_tid is not None and best_laser_prob >= min_laser_prob:
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
                "  TalkNCE/LASER word matching: "
                f"with_frame={words_with_frame}/{len(words)}, "
                f"with_dets={words_with_dets}/{len(words)}, "
                f"with_scored_candidate={words_with_scored_candidate}/{len(words)}"
            )
            assigned_ratio = assigned / max(1, len(words))
            if assigned_ratio < 0.03:
                print(
                    "  TalkNCE/LASER assignment ratio too low "
                    f"({assigned}/{len(words)}={assigned_ratio:.1%}); falling back to heuristic binder."
                )
                return None

            print(
                "TalkNCE+LASER binding complete: "
                f"{assigned}/{len(words)} words assigned, "
                f"{len(scored_track_ids)} scored tracks, {len(bindings)} segments"
            )
            return bindings
        finally:
            cap.release()

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
        high-level objective as TalkNCE/LASER: align speech timing with the most
        likely active on-screen speaker track.
        """
        import cv2
        import math
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

        cap = cv2.VideoCapture(read_path)
        if not cap.isOpened():
            print("  Warning: could not open video for speaker binding")
            return []

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
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

        def _get_frame(frame_idx: int):
            if frame_idx in frame_cache:
                return frame_cache[frame_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame_cache[frame_idx] = None
            else:
                frame_cache[frame_idx] = frame
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

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb)
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

        cap.release()
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
        """Bind words to track IDs using TalkNCE+LASER, with heuristic fallback."""
        bindings = self._run_talknce_laser_binding(
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
        """Run the full Phase 1A extraction stack on pre-downloaded media.

        Args:
            video_bytes: Muxed MP4 video (for visual tracking).
            audio_wav_bytes: 16kHz mono WAV (for ASR).
            youtube_url: Original URL (for metadata only).

        Returns:
            dict with phase_1a_visual and phase_1a_audio payloads.
        """
        import os

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
        print(f"[Phase 1A] Received video ({video_mb:.1f} MB) + audio ({audio_mb:.1f} MB)")

        try:
            # Step 1: ASR
            print("[Phase 1A] Step 1/4: Running Parakeet ASR...")
            words = self._run_asr(audio_path)

            # Step 2: Visual tracking
            print("[Phase 1A] Step 2/4: Running YOLO11 + BoT-SORT tracking...")
            tracks = self._run_tracking(video_path)
            _, track_to_dets = self._build_track_indexes(tracks)

            # Step 3: Global tracklet clustering
            print("[Phase 1A] Step 3/4: Clustering tracklets into global IDs...")
            tracks = self._cluster_tracklets(
                video_path,
                tracks,
                track_to_dets=track_to_dets,
            )
            frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

            # Step 4: Speaker binding
            print("[Phase 1A] Step 4/4: Running speaker binding...")
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

            phase_1a_visual = {
                "source_video": youtube_url,
                "tracks": tracks,
            }

            phase_1a_audio = {
                "source_audio": youtube_url,
                "words": words,
                "speaker_bindings": speaker_bindings,
            }

            print(f"[Phase 1A] Complete — {len(words)} words, {len(tracks)} tracks")
            return {
                "status": "success",
                "phase_1a_visual": phase_1a_visual,
                "phase_1a_audio": phase_1a_audio,
            }

        except Exception as e:
            print(f"[Phase 1A] Error: {e}")
            return {"status": "error", "message": str(e)}
