"""
Clypt Phase 1 extraction worker definitions.

When legacy serverless wrapper support is explicitly enabled, this module also
exposes the old remote deployment wrappers. Otherwise the pure Python extraction
code can be imported and reused by the DigitalOcean worker path.
"""

import os
import subprocess
import time

try:
    if os.getenv("CLYPT_ENABLE_MODAL_SDK", "0") != "1":
        raise ImportError("Legacy serverless SDK disabled for the DO runtime path")
    import modal  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised implicitly by DO runtime import
    class _ShimMethod:
        def __init__(self, fn):
            self._fn = fn

        def __get__(self, instance, owner):
            if instance is None:
                return self
            return self._fn.__get__(instance, owner)

        def _get_raw_f(self):
            return self._fn

    def _shim_method(*_args, **_kwargs):
        def _decorator(fn):
            return _ShimMethod(fn)
        return _decorator

    class _ShimVolume:
        @staticmethod
        def from_name(*_args, **_kwargs):
            return _ShimVolume()

        def commit(self):
            return None

        def reload(self):
            return None

    class _ShimSecret:
        @staticmethod
        def from_dict(*_args, **_kwargs):
            return _ShimSecret()

    class _ShimImage:
        @staticmethod
        def debian_slim(*_args, **_kwargs):
            return _ShimImage()

        def apt_install(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def run_function(self, *_args, **_kwargs):
            return self

    class _ShimApp:
        def __init__(self, name: str):
            self.name = name

        def cls(self, **_kwargs):
            def _decorator(cls):
                cls._get_user_cls = classmethod(lambda klass: klass)
                return cls
            return _decorator

    class _ShimCls:
        @staticmethod
        def from_name(*_args, **_kwargs):
            raise RuntimeError("Legacy remote class lookup is unavailable without the optional serverless SDK")

    class _ShimModal:
        App = _ShimApp
        Volume = _ShimVolume
        Secret = _ShimSecret
        Image = _ShimImage
        Cls = _ShimCls
        method = staticmethod(_shim_method)
        enter = staticmethod(_shim_method)

    modal = _ShimModal()

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
LRASD_MODEL_PATH = "/root/.cache/clypt/finetuning_TalkSet.model"
LRASD_REPO_ROOT = "/root/lrasd"
YOLO_WEIGHTS_PATH = "yolo26s.pt"
YOLO_ONNX_PATH = "/root/.cache/clypt/yolo26s.onnx"
YOLO_ENGINE_PATH = "/root/.cache/clypt/yolo26s.engine"
YOLO_OPENVINO_DIR = "/root/.cache/clypt/yolo26s_openvino_model"
PHASE1_SCHEMA_VERSION = "2.0.0"
PHASE1_TASK_TYPE = "person_tracking"
PHASE1_COORDINATE_SPACE = "absolute_original_frame_xyxy"
PHASE1_GEOMETRY_TYPE = "aabb"
PHASE1_CLASS_TAXONOMY = {"0": "person"}


def _cluster_extraction_config() -> dict:
    """Shared extraction knobs for tracklet face embedding sampling."""
    return {
        "max_frames_per_tracklet": 6,
        "max_ranked_candidates": 18,
        "target_face_encodings_per_tracklet": 2,
        "cluster_face_min_det_score": 0.35,
        "cluster_face_min_side_px": 36.0,
        "cluster_face_min_rel_area": 0.035,
        "cluster_cross_merge_max_cos": 0.28,
        "cluster_cross_merge_max_overlap": 0.08,
        "cluster_cross_merge_max_sig": 2.2,
        "cluster_hist_attach_max_sig": 1.15,
    }


def _build_cluster_sample_plan(tracklets: dict[str, list[dict]], config: dict) -> tuple[dict[str, list[dict]], set[int]]:
    """Pick representative detections per tracklet for face embedding extraction."""
    sampled_by_tid: dict[str, list[dict]] = {}
    needed_frames: set[int] = set()
    max_frames_per_tracklet = int(config["max_frames_per_tracklet"])
    max_ranked_candidates = int(config["max_ranked_candidates"])

    for tid in sorted(tracklets.keys()):
        detections = tracklets.get(tid, [])
        if not detections:
            continue

        ranked_dets = sorted(
            detections,
            key=lambda d: (
                float(d.get("confidence", 0.0)),
                float(d.get("width", 0.0)) * float(d.get("height", 0.0)),
            ),
            reverse=True,
        )

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
            ordered = sorted(detections, key=lambda d: int(d.get("frame_idx", -1)))
            sampled_dets = [ordered[len(ordered) // 2]]

        sampled_by_tid[tid] = sampled_dets
        needed_frames.update(int(d.get("frame_idx", -1)) for d in sampled_dets if int(d.get("frame_idx", -1)) >= 0)

    return sampled_by_tid, needed_frames


def _extract_cluster_embeddings_subset(
    face_analyzer,
    read_path: str,
    sampled_by_tid_subset: dict[str, list[dict]],
    config: dict,
    log_prefix: str = "",
) -> dict:
    """Extract one embedding per tracklet for a subset of track IDs."""
    import cv2
    import numpy as np
    from decord import VideoReader, cpu

    if not sampled_by_tid_subset:
        return {
            "embeddings": {},
            "fallback_ids": [],
            "face_accept_count": 0,
            "face_reject_lowq_count": 0,
            "sampled_frames": 0,
            "tracklets_processed": 0,
        }

    target_face_encodings_per_tracklet = int(config["target_face_encodings_per_tracklet"])
    cluster_face_min_det_score = float(config["cluster_face_min_det_score"])
    cluster_face_min_side_px = float(config["cluster_face_min_side_px"])
    cluster_face_min_rel_area = float(config["cluster_face_min_rel_area"])

    needed_frames = sorted(
        {
            int(det.get("frame_idx", -1))
            for sampled_dets in sampled_by_tid_subset.values()
            for det in sampled_dets
            if int(det.get("frame_idx", -1)) >= 0
        }
    )
    if not needed_frames:
        return {
            "embeddings": {},
            "fallback_ids": [],
            "face_accept_count": 0,
            "face_reject_lowq_count": 0,
            "sampled_frames": 0,
            "tracklets_processed": 0,
        }

    vr = VideoReader(read_path, ctx=cpu(0))
    valid_needed = [fi for fi in needed_frames if 0 <= fi < len(vr)]
    if not valid_needed:
        return {
            "embeddings": {},
            "fallback_ids": [],
            "face_accept_count": 0,
            "face_reject_lowq_count": 0,
            "sampled_frames": 0,
            "tracklets_processed": 0,
        }

    batch = vr.get_batch(valid_needed).asnumpy()
    frame_map: dict[int, np.ndarray] = {fi: batch[i] for i, fi in enumerate(valid_needed)}

    embeddings: dict[str, np.ndarray] = {}
    fallback_ids: list[str] = []
    face_accept_count = 0
    face_reject_lowq_count = 0
    total_tids = len(sampled_by_tid_subset)

    for tid_idx, tid in enumerate(sorted(sampled_by_tid_subset.keys()), start=1):
        sampled_dets = sampled_by_tid_subset.get(tid, [])
        if not sampled_dets:
            continue

        face_vectors = []
        hist_vectors = []
        for det in sampled_dets:
            frame_idx = int(det.get("frame_idx", -1))
            frame = frame_map.get(frame_idx)
            if frame is None:
                continue

            cx = float(det.get("x_center", 0.0))
            cy = float(det.get("y_center", 0.0))
            w = float(det.get("width", 0.0))
            h = float(det.get("height", 0.0))
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
            if face_analyzer is not None:
                try:
                    faces = face_analyzer.get(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
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
                        rel_area = (face_w * face_h) / max(1.0, float(crop.shape[0] * crop.shape[1]))
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
                hist_vectors.append(hist.astype(np.float32))

            if len(face_vectors) >= target_face_encodings_per_tracklet:
                break

        if face_vectors:
            embeddings[tid] = np.mean(np.asarray(face_vectors, dtype=np.float32), axis=0).astype(np.float32)
        elif hist_vectors:
            embeddings[tid] = np.mean(np.asarray(hist_vectors, dtype=np.float32), axis=0).astype(np.float32)
            fallback_ids.append(tid)
        else:
            embeddings[tid] = np.zeros(512, dtype=np.float32)
            fallback_ids.append(tid)

        if tid_idx % 10 == 0 or tid_idx == total_tids:
            print(
                f"{log_prefix}Embedding progress: {tid_idx}/{total_tids} tracklets "
                f"(accepted={face_accept_count}, fallback={len(fallback_ids)})"
            )

    return {
        "embeddings": embeddings,
        "fallback_ids": fallback_ids,
        "face_accept_count": face_accept_count,
        "face_reject_lowq_count": face_reject_lowq_count,
        "sampled_frames": len(valid_needed),
        "tracklets_processed": total_tids,
    }


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


def download_lrasd_model():
    """Cache LR-ASD checkpoint + architecture files at image build time."""
    import os
    import urllib.request

    os.makedirs(os.path.dirname(LRASD_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.join(LRASD_REPO_ROOT, "model"), exist_ok=True)

    if not os.path.exists(LRASD_MODEL_PATH):
        print("Downloading LR-ASD TalkSet fine-tuned weights...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/weight/finetuning_TalkSet.model",
            LRASD_MODEL_PATH,
        )
    else:
        print("LR-ASD checkpoint already cached.")

    files = {
        os.path.join(LRASD_REPO_ROOT, "model", "Model.py"):
        "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/model/Model.py",
        os.path.join(LRASD_REPO_ROOT, "model", "Classifier.py"):
        "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/model/Classifier.py",
        os.path.join(LRASD_REPO_ROOT, "model", "Encoder.py"):
        "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/model/Encoder.py",
        os.path.join(LRASD_REPO_ROOT, "loss.py"):
        "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/loss.py",
    }
    for out_path, url in files.items():
        if not os.path.exists(out_path):
            print(f"Downloading LR-ASD source file: {os.path.basename(out_path)}")
            urllib.request.urlretrieve(url, out_path)

    init_file = os.path.join(LRASD_REPO_ROOT, "model", "__init__.py")
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
        "mediapipe",
        "pandas",
        "scipy",
        "tqdm",
        "matplotlib",
        "imageio",
        "Pillow",
        "resampy",
        "soundfile",
    )
    # Step 2: NeMo ASR
    .pip_install("nemo_toolkit[asr]")
    # Step 3: Cache model weights at build time
    .run_function(download_asr_model)
    .run_function(download_yolo_model)
    .run_function(prepare_yolo_onnx_tensorrt)
    .run_function(download_lrasd_model)
    .run_function(download_insightface_model)
)


# ──────────────────────────────────────────────
# GPU Worker (class-based for VRAM persistence)
# ──────────────────────────────────────────────
@app.cls(
    image=clypt_image,
    gpu="H100",
    timeout=3600,
    max_containers=4,
    min_containers=0,
    scaledown_window=120,
    enable_memory_snapshot=False,
    secrets=[MODEL_DEBUG_SECRET],
    volumes={"/vol/clypt-chunks": TRACKING_VOLUME},
)
class ClusterEmbeddingWorker:

    @modal.enter()
    def load_face_model(self):
        from insightface.app import FaceAnalysis

        self.face_analyzer = None
        try:
            print("Loading InsightFace for cluster embedding shard worker...")
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.15)
            print("Cluster shard InsightFace ready.")
        except Exception as e:
            print(
                "Warning: cluster shard InsightFace initialization failed "
                f"({type(e).__name__}: {e})"
            )

    @modal.method()
    def extract_cluster_embeddings_shard(
        self,
        video_path: str,
        sampled_by_tid_subset: dict[str, list[dict]],
        shard_idx: int,
        total_shards: int,
    ) -> dict:
        import time

        if video_path.startswith("/vol/"):
            try:
                TRACKING_VOLUME.reload()
            except Exception:
                pass

        config = _cluster_extraction_config()
        print(
            f"[cluster-shard {shard_idx}/{total_shards}] Starting "
            f"{len(sampled_by_tid_subset)} tracklets"
        )
        started = time.time()
        result = _extract_cluster_embeddings_subset(
            face_analyzer=self.face_analyzer,
            read_path=video_path,
            sampled_by_tid_subset=sampled_by_tid_subset,
            config=config,
            log_prefix=f"  [cluster-shard {shard_idx}/{total_shards}] ",
        )
        embeddings = {
            tid: vec.tolist()
            for tid, vec in result.get("embeddings", {}).items()
        }
        elapsed_s = float(time.time() - started)
        print(
            f"[cluster-shard {shard_idx}/{total_shards}] Complete "
            f"tracklets={result.get('tracklets_processed', 0)} "
            f"accepted={result.get('face_accept_count', 0)} "
            f"fallback={len(result.get('fallback_ids', []))} "
            f"elapsed={elapsed_s:.1f}s"
        )
        return {
            "embeddings": embeddings,
            "fallback_ids": list(result.get("fallback_ids", [])),
            "face_accept_count": int(result.get("face_accept_count", 0)),
            "face_reject_lowq_count": int(result.get("face_reject_lowq_count", 0)),
            "sampled_frames": int(result.get("sampled_frames", 0)),
            "tracklets_processed": int(result.get("tracklets_processed", 0)),
            "elapsed_s": elapsed_s,
        }


@app.cls(
    image=clypt_image,
    gpu="H100",
    timeout=3600,
    max_containers=4,
    min_containers=0,
    scaledown_window=120,
    enable_memory_snapshot=False,
    secrets=[MODEL_DEBUG_SECRET],
    volumes={"/vol/clypt-chunks": TRACKING_VOLUME},
)
class ClyptWorker:

    @staticmethod
    def _load_lrasd_checkpoint(model, loss_av, ckpt_path: str):
        """Load LR-ASD checkpoint into model + AV classifier head."""
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
                "LR-ASD checkpoint mismatch: "
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
    def _clusters_conflict_by_visibility(
        tracklets: dict[str, list[dict]],
        left_track_ids: list[str],
        right_track_ids: list[str],
    ) -> bool:
        """Return True when two candidate identities are clearly co-visible as distinct people."""
        left_by_frame: dict[int, dict] = {}
        right_by_frame: dict[int, dict] = {}

        for tid in left_track_ids:
            for det in tracklets.get(tid, []):
                frame_idx = int(det.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                prev = left_by_frame.get(frame_idx)
                if prev is None or float(det.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    left_by_frame[frame_idx] = det

        for tid in right_track_ids:
            for det in tracklets.get(tid, []):
                frame_idx = int(det.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                prev = right_by_frame.get(frame_idx)
                if prev is None or float(det.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    right_by_frame[frame_idx] = det

        for frame_idx in set(left_by_frame.keys()) & set(right_by_frame.keys()):
            left = left_by_frame[frame_idx]
            right = right_by_frame[frame_idx]
            avg_width = max(
                1.0,
                0.5 * (float(left.get("width", 1.0)) + float(right.get("width", 1.0))),
            )
            avg_height = max(
                1.0,
                0.5 * (float(left.get("height", 1.0)) + float(right.get("height", 1.0))),
            )
            dx = abs(float(left.get("x_center", 0.0)) - float(right.get("x_center", 0.0)))
            dy = abs(float(left.get("y_center", 0.0)) - float(right.get("y_center", 0.0)))
            if dx > (0.65 * avg_width) or dy > (0.45 * avg_height):
                return True
        return False

    @staticmethod
    def _clusters_have_compatible_seat_signature(
        tracklets: dict[str, list[dict]],
        left_track_ids: list[str],
        right_track_ids: list[str],
        *,
        max_signature_distance: float = 1.6,
    ) -> bool:
        import numpy as np

        def _signature(track_ids: list[str]) -> np.ndarray | None:
            sigs = []
            for tid in track_ids:
                dets = tracklets.get(tid, [])
                if not dets:
                    continue
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
                sigs.append(np.median(arr, axis=0))
            if not sigs:
                return None
            return np.median(np.stack(sigs, axis=0), axis=0)

        left_sig = _signature(left_track_ids)
        right_sig = _signature(right_track_ids)
        if left_sig is None or right_sig is None:
            return True

        sx = max(1.0, 0.5 * (left_sig[2] + right_sig[2]))
        sy = max(1.0, 0.5 * (left_sig[3] + right_sig[3]))
        dx = (left_sig[0] - right_sig[0]) / sx
        dy = (left_sig[1] - right_sig[1]) / sy
        dw = np.log(max(left_sig[2], 1.0) / max(right_sig[2], 1.0))
        dh = np.log(max(left_sig[3], 1.0) / max(right_sig[3], 1.0))
        sig_dist = float(dx * dx + dy * dy + 0.25 * (dw * dw + dh * dh))
        return sig_dist <= max_signature_distance

    @staticmethod
    def _tracklet_signature(
        tracklets: dict[str, list[dict]],
        track_ids: list[str],
    ):
        import numpy as np

        sigs = []
        for tid in track_ids:
            dets = tracklets.get(tid, [])
            if not dets:
                continue
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
            sigs.append(np.median(arr, axis=0))
        if not sigs:
            return None
        return np.median(np.stack(sigs, axis=0), axis=0)

    @staticmethod
    def _tracklet_signature_distance(left_sig, right_sig) -> float:
        import math

        if left_sig is None or right_sig is None:
            return 0.0

        sx = max(1.0, 0.5 * (float(left_sig[2]) + float(right_sig[2])))
        sy = max(1.0, 0.5 * (float(left_sig[3]) + float(right_sig[3])))
        dx = (float(left_sig[0]) - float(right_sig[0])) / sx
        dy = (float(left_sig[1]) - float(right_sig[1])) / sy
        dw = math.log(max(float(left_sig[2]), 1.0) / max(float(right_sig[2]), 1.0))
        dh = math.log(max(float(left_sig[3]), 1.0) / max(float(right_sig[3]), 1.0))
        return float(dx * dx + dy * dy + 0.25 * (dw * dw + dh * dh))

    def _repair_covisible_cluster_merges(
        self,
        tracklets: dict[str, list[dict]],
        label_by_tid: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, int]]:
        from collections import defaultdict

        grouped: dict[int, list[str]] = defaultdict(list)
        for tid, label in label_by_tid.items():
            grouped[int(label)].append(str(tid))

        next_label = (max(grouped.keys()) + 1) if grouped else 0
        repaired = dict(label_by_tid)
        repaired_cluster_count = 0
        repaired_tracklet_count = 0
        repaired_conflict_pair_count = 0

        def _track_sort_key(tid: str) -> tuple[int, int, str]:
            dets = tracklets.get(tid, [])
            first_frame = min((int(det.get("frame_idx", -1)) for det in dets), default=10**9)
            return (-len(dets), first_frame, tid)

        def _bucket_sort_key(bucket_track_ids: list[str]) -> tuple[int, int, str]:
            total_boxes = sum(len(tracklets.get(tid, [])) for tid in bucket_track_ids)
            first_frame = min(
                (
                    int(det.get("frame_idx", -1))
                    for tid in bucket_track_ids
                    for det in tracklets.get(tid, [])
                ),
                default=10**9,
            )
            return (-total_boxes, first_frame, min(bucket_track_ids))

        for label in sorted(grouped.keys()):
            tids = sorted(grouped[label], key=_track_sort_key)
            if len(tids) < 2:
                continue

            conflict_pairs: set[tuple[str, str]] = set()
            for idx, left_tid in enumerate(tids):
                for right_tid in tids[idx + 1 :]:
                    if self._clusters_conflict_by_visibility(tracklets, [left_tid], [right_tid]):
                        conflict_pairs.add((left_tid, right_tid))
                        conflict_pairs.add((right_tid, left_tid))

            if not conflict_pairs:
                continue

            buckets: list[list[str]] = []
            for tid in tids:
                tid_sig = self._tracklet_signature(tracklets, [tid])
                compatible_buckets: list[tuple[float, int]] = []
                for bucket_idx, bucket in enumerate(buckets):
                    if any((tid, other_tid) in conflict_pairs for other_tid in bucket):
                        continue
                    if not self._clusters_have_compatible_seat_signature(
                        tracklets,
                        bucket,
                        [tid],
                        max_signature_distance=2.4,
                    ):
                        continue
                    bucket_sig = self._tracklet_signature(tracklets, bucket)
                    sig_dist = self._tracklet_signature_distance(bucket_sig, tid_sig)
                    compatible_buckets.append((sig_dist, bucket_idx))

                if compatible_buckets:
                    _, chosen_bucket_idx = min(compatible_buckets, key=lambda item: (item[0], item[1]))
                    buckets[chosen_bucket_idx].append(tid)
                else:
                    buckets.append([tid])

            if len(buckets) <= 1:
                continue

            buckets.sort(key=_bucket_sort_key)
            repaired_cluster_count += 1
            repaired_tracklet_count += len(tids)
            repaired_conflict_pair_count += len(conflict_pairs) // 2

            for bucket_idx, bucket in enumerate(buckets):
                assigned_label = label if bucket_idx == 0 else next_label
                if bucket_idx > 0:
                    next_label += 1
                for tid in bucket:
                    repaired[tid] = assigned_label

        return repaired, {
            "repaired_cluster_count": repaired_cluster_count,
            "repaired_tracklet_count": repaired_tracklet_count,
            "repaired_conflict_pair_count": repaired_conflict_pair_count,
        }

    @staticmethod
    def _face_ledger_workers() -> int:
        try:
            requested = int(
                os.getenv(
                    "CLYPT_FACE_LEDGER_WORKERS",
                    str(ClyptWorker._face_pipeline_workers()),
                )
            )
        except Exception:
            requested = ClyptWorker._face_pipeline_workers()
        return max(1, requested)

    @staticmethod
    def _face_pipeline_workers() -> int:
        raw = os.getenv("CLYPT_FACE_PIPELINE_WORKERS", "").strip()
        if not raw:
            raw = os.getenv("CLYPT_FACE_LEDGER_WORKERS", "").strip()
        try:
            requested = int(raw) if raw else int(os.cpu_count() or 8)
        except Exception:
            requested = int(os.cpu_count() or 8)
        return max(1, min(32, requested))

    @staticmethod
    def _face_pipeline_segment_frames() -> int:
        raw = os.getenv("CLYPT_FACE_PIPELINE_SEGMENT_FRAMES", "").strip()
        if not raw:
            raw = os.getenv("CLYPT_FACE_LEDGER_SEGMENT_FRAMES", "240").strip()
        try:
            requested = int(raw)
        except Exception:
            requested = 240
        return max(1, requested)

    @staticmethod
    def _shared_analysis_proxy_enabled() -> bool:
        raw = os.getenv("CLYPT_SHARED_ANALYSIS_PROXY", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return os.getenv("CLYPT_ANALYSIS_PROXY_ENABLE", "0") == "1"

    @staticmethod
    def _analysis_proxy_max_long_edge() -> int:
        raw = os.getenv("CLYPT_ANALYSIS_PROXY_MAX_LONG_EDGE", "").strip()
        if not raw:
            raw = os.getenv("CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE", "1280").strip()
        try:
            requested = int(raw)
        except Exception:
            requested = 1280
        return max(0, requested)

    @staticmethod
    def _split_tracks_into_face_segments(
        tracks: list[dict],
        segment_frames: int,
    ) -> list[list[dict]]:
        from collections import defaultdict

        if not tracks:
            return []

        grouped: dict[int, list[dict]] = defaultdict(list)
        for det in tracks:
            frame_idx = int(det.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            segment_idx = frame_idx // max(1, segment_frames)
            grouped[segment_idx].append(det)

        return [grouped[idx] for idx in sorted(grouped.keys()) if grouped[idx]]

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

    @staticmethod
    def _normalize_scale_factor(scale: float) -> float:
        if abs(scale) <= 1e-6:
            return 1.0
        return float(scale)

    @staticmethod
    def _person_bbox_xyxy_from_det(det: dict) -> tuple[float, float, float, float] | None:
        if not isinstance(det, dict):
            return None
        if all(key in det for key in ("x1", "y1", "x2", "y2")):
            x1 = float(det.get("x1", 0.0))
            y1 = float(det.get("y1", 0.0))
            x2 = float(det.get("x2", x1))
            y2 = float(det.get("y2", y1))
        else:
            cx = float(det.get("x_center", 0.0))
            cy = float(det.get("y_center", 0.0))
            width = float(det.get("width", 0.0))
            height = float(det.get("height", 0.0))
            x1 = cx - (0.5 * width)
            y1 = cy - (0.5 * height)
            x2 = cx + (0.5 * width)
            y2 = cy + (0.5 * height)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _face_bbox_plausibility_score(
        self,
        det: dict,
        bbox_xyxy: tuple[float, float, float, float] | None,
    ) -> float:
        if bbox_xyxy is None:
            return 0.0

        person_bbox = self._person_bbox_xyxy_from_det(det)
        if person_bbox is None:
            return 0.0

        px1, py1, px2, py2 = person_bbox
        fx1, fy1, fx2, fy2 = [float(v) for v in bbox_xyxy]
        person_w = max(1.0, px2 - px1)
        person_h = max(1.0, py2 - py1)
        face_w = max(1.0, fx2 - fx1)
        face_h = max(1.0, fy2 - fy1)
        width_ratio = face_w / person_w
        height_ratio = face_h / person_h
        face_center_x = ((fx1 + fx2) * 0.5 - px1) / person_w
        face_center_y = ((fy1 + fy2) * 0.5 - py1) / person_h
        face_bottom_y = (fy2 - py1) / person_h
        aspect_ratio = face_w / max(face_h, 1.0)

        if not (0.08 <= width_ratio <= 0.58):
            return 0.0
        if not (0.08 <= height_ratio <= 0.52):
            return 0.0
        if not (0.12 <= face_center_x <= 0.88):
            return 0.0
        if not (0.05 <= face_center_y <= 0.46):
            return 0.0
        if face_bottom_y > 0.62:
            return 0.0
        if not (0.55 <= aspect_ratio <= 1.65):
            return 0.0

        center_x_bonus = 1.0 - min(1.0, abs(face_center_x - 0.5) / 0.38)
        center_y_bonus = 1.0 - min(1.0, abs(face_center_y - 0.28) / 0.24)
        size_bonus = 1.0 - min(1.0, abs(width_ratio - 0.24) / 0.20)
        return max(0.0, (0.35 * center_x_bonus) + (0.35 * center_y_bonus) + (0.30 * size_bonus))

    def _analyze_face_in_person_det(self, frame_rgb, det: dict):
        """Detect and analyze the strongest face inside a person bbox."""
        import cv2
        import numpy as np

        if self.face_analyzer is None or frame_rgb is None:
            return None

        fh, fw = frame_rgb.shape[:2]
        person_bbox = self._person_bbox_xyxy_from_det(det)
        if person_bbox is None:
            return None
        px1, py1, px2, py2 = person_bbox
        cx = 0.5 * (px1 + px2)
        cy = 0.5 * (py1 + py2)
        w = max(1.0, px2 - px1)
        h = max(1.0, py2 - py1)

        # Limit face search to the head/shoulder band to avoid elbow/torso false positives.
        x1 = max(0, int(np.floor(px1 - (0.10 * w))))
        x2 = min(fw, int(np.ceil(px2 + (0.10 * w))))
        y1 = max(0, int(np.floor(py1 - (0.04 * h))))
        y2 = min(fh, int(np.ceil(py1 + (0.58 * h))))
        if x2 <= x1 or y2 <= y1:
            return None

        roi_rgb = frame_rgb[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            return None

        faces = None
        if self.face_analyzer is not None:
            try:
                faces = self.face_analyzer.get(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
            except Exception:
                faces = None
        if not faces:
            return self._analyze_face_in_person_det_with_mediapipe(
                frame_rgb=frame_rgb,
                det=det,
                roi_rgb=roi_rgb,
                roi_origin=(x1, y1),
            )

        scored_faces = []
        for face in faces:
            fb = np.asarray(face.bbox, dtype=np.float32)
            fx1_raw, fy1_raw, fx2_raw, fy2_raw = fb.tolist()
            candidate_bbox = (
                float(x1 + fx1_raw),
                float(y1 + fy1_raw),
                float(x1 + fx2_raw),
                float(y1 + fy2_raw),
            )
            plausibility = self._face_bbox_plausibility_score(det, candidate_bbox)
            if plausibility <= 0.0:
                continue
            scored_faces.append(
                (
                    (0.70 * float(getattr(face, "det_score", 0.0)))
                    + (0.30 * plausibility),
                    face,
                )
            )
        if not scored_faces:
            return self._analyze_face_in_person_det_with_mediapipe(
                frame_rgb=frame_rgb,
                det=det,
                roi_rgb=roi_rgb,
                roi_origin=(x1, y1),
            )

        _, best_face = max(scored_faces, key=lambda item: item[0])
        fb = np.asarray(best_face.bbox, dtype=np.float32)
        fx1, fy1, fx2, fy2 = fb.tolist()
        fw_face = max(2.0, fx2 - fx1)
        fh_face = max(2.0, fy2 - fy1)
        fx1 = max(0, int(np.floor(fx1 - 0.10 * fw_face)))
        fy1 = max(0, int(np.floor(fy1 - 0.10 * fh_face)))
        fx2 = min(roi_rgb.shape[1], int(np.ceil(fx2 + 0.10 * fw_face)))
        fy2 = min(roi_rgb.shape[0], int(np.ceil(fy2 + 0.10 * fh_face)))
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        gx1 = x1 + fx1
        gy1 = y1 + fy1
        gx2 = x1 + fx2
        gy2 = y1 + fy2
        if gx2 <= gx1 or gy2 <= gy1:
            return None
        if self._face_bbox_plausibility_score(det, (float(gx1), float(gy1), float(gx2), float(gy2))) <= 0.0:
            return self._analyze_face_in_person_det_with_mediapipe(
                frame_rgb=frame_rgb,
                det=det,
                roi_rgb=roi_rgb,
                roi_origin=(x1, y1),
            )

        face_rgb = frame_rgb[gy1:gy2, gx1:gx2]
        if face_rgb.size == 0:
            return None
        face_rgb = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        anchor = {
            "x_offset": (float(gx1) - cx) / w,
            "y_offset": (float(gy1) - cy) / h,
            "w_ratio": (float(gx2 - gx1)) / w,
            "h_ratio": (float(gy2 - gy1)) / h,
        }
        embedding = getattr(best_face, "normed_embedding", None)
        embedding_vec = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=np.float32)
            if emb_arr.size > 0:
                embedding_vec = emb_arr

        return {
            "face_rgb": face_rgb,
            "anchor": anchor,
            "bbox_xyxy": (float(gx1), float(gy1), float(gx2), float(gy2)),
            "det_score": float(getattr(best_face, "det_score", 0.0)),
            "embedding": embedding_vec,
        }

    def _analyze_face_in_person_det_with_mediapipe(
        self,
        *,
        frame_rgb,
        det: dict,
        roi_rgb,
        roi_origin: tuple[int, int],
    ):
        import cv2
        import numpy as np

        detector = getattr(self, "mediapipe_face_detector", None)
        if detector is None or frame_rgb is None or roi_rgb is None:
            return None

        try:
            result = detector.process(roi_rgb)
        except Exception:
            return None
        detections = getattr(result, "detections", None) or []
        if not detections:
            return None

        best_det = max(
            detections,
            key=lambda item: float(getattr(item, "score", [0.0])[0]),
        )
        rel_bbox = getattr(getattr(best_det, "location_data", None), "relative_bounding_box", None)
        if rel_bbox is None:
            return None

        roi_h, roi_w = roi_rgb.shape[:2]
        ox, oy = roi_origin
        fx1 = max(0, int(np.floor(float(rel_bbox.xmin) * roi_w)))
        fy1 = max(0, int(np.floor(float(rel_bbox.ymin) * roi_h)))
        fx2 = min(roi_w, int(np.ceil((float(rel_bbox.xmin) + float(rel_bbox.width)) * roi_w)))
        fy2 = min(roi_h, int(np.ceil((float(rel_bbox.ymin) + float(rel_bbox.height)) * roi_h)))
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        gx1 = ox + fx1
        gy1 = oy + fy1
        gx2 = ox + fx2
        gy2 = oy + fy2
        if self._face_bbox_plausibility_score(det, (float(gx1), float(gy1), float(gx2), float(gy2))) <= 0.0:
            return None
        face_rgb = frame_rgb[gy1:gy2, gx1:gx2]
        if face_rgb.size == 0:
            return None
        face_rgb = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)

        cx = float(det.get("x_center", 0.0))
        cy = float(det.get("y_center", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))
        if w <= 1e-6 or h <= 1e-6:
            return None

        return {
            "face_rgb": face_rgb,
            "anchor": {
                "x_offset": (float(gx1) - cx) / w,
                "y_offset": (float(gy1) - cy) / h,
                "w_ratio": (float(gx2 - gx1)) / w,
                "h_ratio": (float(gy2 - gy1)) / h,
            },
            "bbox_xyxy": (float(gx1), float(gy1), float(gx2), float(gy2)),
            "det_score": float(getattr(best_det, "score", [0.0])[0]),
            "embedding": None,
        }

    def _detect_face_in_person_det(self, frame_rgb, det: dict):
        """Detect a face inside a person bbox and return (112x112 crop, relative anchor)."""
        analysis = self._analyze_face_in_person_det(frame_rgb, det)
        if analysis is None:
            return None, None
        return analysis["face_rgb"], analysis["anchor"]

    def _extract_hist_embedding_from_det(self, frame_rgb, det: dict):
        import cv2
        import numpy as np

        if frame_rgb is None:
            return None

        fh, fw = frame_rgb.shape[:2]
        x1 = max(0, min(fw - 1, int(round(float(det.get("x1", 0.0))))))
        y1 = max(0, min(fh - 1, int(round(float(det.get("y1", 0.0))))))
        x2 = max(x1 + 1, min(fw, int(round(float(det.get("x2", x1 + 1.0))))))
        y2 = max(y1 + 1, min(fh, int(round(float(det.get("y2", y1 + 1.0))))))
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if len(hist) < 512:
            hist = np.pad(hist, (0, 512 - len(hist)))
        else:
            hist = hist[:512]
        return hist.astype(np.float32)

    def _extract_track_identity_features_for_segment(
        self,
        *,
        video_path: str,
        segment_tracks: list[dict],
        fps: float,
        frame_width: int,
        frame_height: int,
        output_frame_width: int | None = None,
        output_frame_height: int | None = None,
        coord_scale_x: float = 1.0,
        coord_scale_y: float = 1.0,
    ) -> dict[str, dict]:
        import numpy as np
        from collections import defaultdict

        if not segment_tracks:
            return {}

        config = _cluster_extraction_config()
        cluster_face_min_det_score = float(config["cluster_face_min_det_score"])
        max_face_embeddings = int(config["target_face_encodings_per_tracklet"]) * 3

        frame_to_dets: dict[int, list[dict]] = defaultdict(list)
        for det in segment_tracks:
            frame_idx = int(det.get("frame_idx", -1))
            if frame_idx < 0:
                continue
            frame_to_dets[frame_idx].append(det)

        frame_map = {}
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            needed_frames = [fi for fi in sorted(frame_to_dets.keys()) if 0 <= fi < total_frames]
            if not needed_frames:
                return {}
            batch = vr.get_batch(needed_frames).asnumpy()
            frame_map = {fi: batch[idx] for idx, fi in enumerate(needed_frames)}
        except Exception:
            needed_frames = sorted(frame_to_dets.keys())
            frame_map = {
                fi: self._read_frame_rgb(video_path, fi)
                for fi in needed_frames
            }
        if not needed_frames:
            return {}

        features: dict[str, dict] = {}
        out_w = int(output_frame_width or frame_width)
        out_h = int(output_frame_height or frame_height)
        inv_scale_x = self._normalize_scale_factor(coord_scale_x)
        inv_scale_y = self._normalize_scale_factor(coord_scale_y)

        for frame_idx in needed_frames:
            frame_rgb = frame_map.get(frame_idx)
            if frame_rgb is None:
                continue
            best_by_track: dict[str, dict] = {}
            for det in frame_to_dets[frame_idx]:
                tid = str(det.get("track_id", ""))
                if not tid:
                    continue
                previous = best_by_track.get(tid)
                if previous is None or float(det.get("confidence", 0.0)) > float(
                    previous.get("confidence", 0.0)
                ):
                    best_by_track[tid] = det

            for tid, det in best_by_track.items():
                slot = features.setdefault(
                    tid,
                    {
                        "face_observations": [],
                        "face_embeddings": [],
                        "hist_embeddings": [],
                    },
                )
                analysis = self._analyze_face_in_person_det(frame_rgb, det)
                if analysis is not None:
                    gx1, gy1, gx2, gy2 = analysis["bbox_xyxy"]
                    bbox = self._normalize_bbox(
                        float(gx1) / inv_scale_x,
                        float(gy1) / inv_scale_y,
                        float(gx2) / inv_scale_x,
                        float(gy2) / inv_scale_y,
                        out_w,
                        out_h,
                    )
                    slot["face_observations"].append(
                        {
                            "frame_idx": int(frame_idx),
                            "time_ms": int(round((frame_idx / max(1e-6, fps)) * 1000.0)),
                            "bounding_box": bbox,
                            "track_id": tid,
                            "confidence": float(max(det.get("confidence", 0.0), analysis["det_score"])),
                            "quality": float(analysis["det_score"]),
                            "source": "face_detector",
                            "provenance": "insightface_roi",
                        }
                    )
                    embedding = analysis.get("embedding")
                    if (
                        embedding is not None
                        and float(analysis.get("det_score", 0.0)) >= cluster_face_min_det_score
                        and len(slot["face_embeddings"]) < max_face_embeddings
                    ):
                        slot["face_embeddings"].append(np.asarray(embedding, dtype=np.float32))
                        continue

                if len(slot["hist_embeddings"]) < max_face_embeddings:
                    hist = self._extract_hist_embedding_from_det(frame_rgb, det)
                    if hist is not None:
                        slot["hist_embeddings"].append(hist)

        finalized: dict[str, dict] = {}
        for tid, feature in features.items():
            face_embeddings = feature.get("face_embeddings", [])
            hist_embeddings = feature.get("hist_embeddings", [])
            if face_embeddings:
                emb = np.mean(np.stack(face_embeddings, axis=0), axis=0).astype(np.float32).tolist()
                source = "face"
                count = len(face_embeddings)
            elif hist_embeddings:
                emb = np.mean(np.stack(hist_embeddings, axis=0), axis=0).astype(np.float32).tolist()
                source = "histogram"
                count = len(hist_embeddings)
            else:
                emb = None
                source = "none"
                count = 0

            face_observations = sorted(
                feature.get("face_observations", []),
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            deduped_face_observations: list[dict] = []
            seen_frames: set[int] = set()
            for observation in face_observations:
                frame_idx = int(observation.get("frame_idx", -1))
                if frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                deduped_face_observations.append(observation)

            finalized[tid] = {
                "embedding": emb,
                "embedding_source": source,
                "embedding_count": count,
                "face_observations": deduped_face_observations,
                "face_observation_count": len(deduped_face_observations),
            }
        return finalized

    def _merge_track_identity_feature_sets(self, feature_maps: list[dict[str, dict]]) -> dict[str, dict]:
        import numpy as np

        merged: dict[str, dict] = {}
        for feature_map in feature_maps:
            for tid, feature in (feature_map or {}).items():
                slot = merged.setdefault(
                    tid,
                    {
                        "face_vectors": [],
                        "hist_vectors": [],
                        "face_observations": [],
                    },
                )
                embedding = feature.get("embedding")
                if embedding is not None:
                    emb_arr = np.asarray(embedding, dtype=np.float32)
                    if emb_arr.size > 0:
                        repeat = max(1, int(feature.get("embedding_count", 1)))
                        if str(feature.get("embedding_source", "none")) == "face":
                            slot["face_vectors"].extend([emb_arr] * repeat)
                        elif str(feature.get("embedding_source", "none")) == "histogram":
                            slot["hist_vectors"].extend([emb_arr] * repeat)
                slot["face_observations"].extend(feature.get("face_observations", []))

        finalized: dict[str, dict] = {}
        for tid, feature in merged.items():
            if feature["face_vectors"]:
                emb = np.mean(np.stack(feature["face_vectors"], axis=0), axis=0).astype(np.float32).tolist()
                source = "face"
                count = len(feature["face_vectors"])
            elif feature["hist_vectors"]:
                emb = np.mean(np.stack(feature["hist_vectors"], axis=0), axis=0).astype(np.float32).tolist()
                source = "histogram"
                count = len(feature["hist_vectors"])
            else:
                emb = None
                source = "none"
                count = 0

            observations = sorted(
                feature["face_observations"],
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            deduped: list[dict] = []
            seen_frames: set[int] = set()
            for observation in observations:
                frame_idx = int(observation.get("frame_idx", -1))
                if frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                deduped.append(observation)

            finalized[tid] = {
                "embedding": emb,
                "embedding_source": source,
                "embedding_count": count,
                "face_observations": deduped,
                "face_observation_count": len(deduped),
            }
        return finalized

    def _extract_track_identity_features_from_segments(
        self,
        *,
        video_path: str,
        fps: float,
        frame_width: int,
        frame_height: int,
        output_frame_width: int | None = None,
        output_frame_height: int | None = None,
        coord_scale_x: float = 1.0,
        coord_scale_y: float = 1.0,
        segments: list[list[dict]] | None = None,
        segment_futures=None,
    ) -> tuple[dict[str, dict], dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        segments = [segment for segment in (segments or []) if segment]
        workers = self._face_pipeline_workers()
        segment_results: list[dict[str, dict]] = []

        if segment_futures is not None:
            for future in as_completed(segment_futures):
                segment_results.append(future.result())
            mode = "staggered"
        elif not segments:
            mode = "disabled"
        elif workers <= 1 or len(segments) <= 1:
            segment_results = [
                self._extract_track_identity_features_for_segment(
                    video_path=video_path,
                    segment_tracks=segment,
                    fps=fps,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    output_frame_width=output_frame_width,
                    output_frame_height=output_frame_height,
                    coord_scale_x=coord_scale_x,
                    coord_scale_y=coord_scale_y,
                )
                for segment in segments
            ]
            mode = "serial"
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._extract_track_identity_features_for_segment,
                        video_path=video_path,
                        segment_tracks=segment,
                        fps=fps,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        output_frame_width=output_frame_width,
                        output_frame_height=output_frame_height,
                        coord_scale_x=coord_scale_x,
                        coord_scale_y=coord_scale_y,
                    )
                    for segment in segments
                ]
                for future in as_completed(futures):
                    segment_results.append(future.result())
            mode = "parallel"

        merged = self._merge_track_identity_feature_sets(segment_results)
        metrics = {
            "face_pipeline_mode": mode,
            "face_pipeline_worker_count": workers,
            "face_pipeline_segment_count": len(segments) if segment_futures is None else len(segment_futures),
            "face_pipeline_segments_processed": len(segment_results),
            "face_pipeline_track_count": len(merged),
        }
        return merged, metrics

    def _read_frame_rgb(self, video_path: str, frame_idx: int):
        """Load a single RGB frame for downstream face-ledger extraction."""
        import cv2

        if frame_idx < 0:
            return None

        cached_path = getattr(self, "_cached_frame_capture_path", None)
        cap = getattr(self, "_cached_frame_capture", None)
        if cap is None or cached_path != video_path or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self._cached_frame_capture = None
                self._cached_frame_capture_path = None
                return None
            self._cached_frame_capture = cap
            self._cached_frame_capture_path = video_path
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            self._cached_frame_capture = cap
            self._cached_frame_capture_path = video_path

    def _close_cached_frame_reader(self) -> None:
        cap = getattr(self, "_cached_frame_capture", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._cached_frame_capture = None
        self._cached_frame_capture_path = None

    @staticmethod
    def _normalize_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> dict:
        width = max(1, int(width))
        height = max(1, int(height))
        left = max(0.0, min(1.0, float(x1) / width))
        top = max(0.0, min(1.0, float(y1) / height))
        right = max(0.0, min(1.0, float(x2) / width))
        bottom = max(0.0, min(1.0, float(y2) / height))
        return {
            "left": left,
            "top": top,
            "right": max(left, right),
            "bottom": max(top, bottom),
        }

    def _anchor_to_face_bbox(
        self,
        det: dict,
        anchor: dict | None,
        frame_width: int,
        frame_height: int,
    ) -> dict | None:
        if not anchor:
            return None

        cx = float(det.get("x_center", 0.0))
        cy = float(det.get("y_center", 0.0))
        width = max(1.0, float(det.get("width", 1.0)))
        height = max(1.0, float(det.get("height", 1.0)))
        x1 = cx + (float(anchor.get("x_offset", 0.0)) * width)
        y1 = cy + (float(anchor.get("y_offset", 0.0)) * height)
        x2 = x1 + max(1.0, float(anchor.get("w_ratio", 0.0)) * width)
        y2 = y1 + max(1.0, float(anchor.get("h_ratio", 0.0)) * height)
        if x2 <= x1 or y2 <= y1:
            return None
        return self._normalize_bbox(x1, y1, x2, y2, frame_width, frame_height)

    def _build_visual_detection_ledgers(
        self,
        video_path: str,
        tracks: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
        track_identity_features: dict[str, dict] | None = None,
    ) -> tuple[list[dict], list[dict], dict]:
        """Build downstream person/face ledgers using true detector-derived face tracks."""
        started_at = time.perf_counter()
        meta = self._probe_video_meta(video_path)
        fps = float(meta.get("fps", 0.0) or 0.0) or 30.0
        frame_width = int(meta.get("width", 0) or 0)
        frame_height = int(meta.get("height", 0) or 0)

        if frame_to_dets is None or track_to_dets is None:
            frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

        person_detections = []
        for idx, tid in enumerate(sorted(track_to_dets.keys())):
            dets = sorted(
                track_to_dets.get(tid, []),
                key=lambda det: int(det.get("frame_idx", -1)),
            )
            if not dets:
                continue
            ts_objs = []
            for det in dets:
                frame_idx = int(det.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                ts_objs.append(
                    {
                        "time_ms": int(round((frame_idx / max(1e-6, fps)) * 1000.0)),
                        "bounding_box": self._normalize_bbox(
                            float(det.get("x1", 0.0)),
                            float(det.get("y1", 0.0)),
                            float(det.get("x2", 1.0)),
                            float(det.get("y2", 1.0)),
                            frame_width,
                            frame_height,
                        ),
                        "track_id": tid,
                        "confidence": float(det.get("confidence", 0.0)),
                        "source": "person_tracker",
                        "provenance": "yolo26_botsort",
                    }
                )
            if not ts_objs:
                continue
            person_detections.append(
                {
                    "confidence": float(
                        sum(float(obj.get("confidence", 0.0)) for obj in ts_objs) / max(1, len(ts_objs))
                    ),
                    "segment_start_ms": int(ts_objs[0]["time_ms"]),
                    "segment_end_ms": int(ts_objs[-1]["time_ms"]),
                    "person_track_index": idx,
                    "track_id": tid,
                    "source": "person_tracker",
                    "provenance": "yolo26_botsort",
                    "timestamped_objects": ts_objs,
                }
            )

        face_ts_by_track: dict[str, list[dict]] = {}
        precomputed_frames: set[int] = set()
        if track_identity_features:
            for tid, feature in track_identity_features.items():
                observations = feature.get("face_observations", [])
                if not observations:
                    continue
                deduped_by_frame: dict[int, dict] = {}
                for observation in observations:
                    frame_idx = int(observation.get("frame_idx", -1))
                    if frame_idx < 0:
                        continue
                    current = deduped_by_frame.get(frame_idx)
                    if current is None or float(observation.get("confidence", 0.0)) > float(
                        current.get("confidence", 0.0)
                    ):
                        deduped_by_frame[frame_idx] = {
                            "time_ms": int(
                                observation.get(
                                    "time_ms",
                                    round((frame_idx / max(1e-6, fps)) * 1000.0),
                                )
                            ),
                            "bounding_box": dict(observation.get("bounding_box", {})),
                            "track_id": tid,
                            "confidence": float(observation.get("confidence", 0.0)),
                            "source": str(observation.get("source", "face_detector")),
                            "provenance": observation.get("provenance", "insightface_roi"),
                        }
                if deduped_by_frame:
                    face_ts_by_track[tid] = [
                        deduped_by_frame[frame_idx]
                        for frame_idx in sorted(deduped_by_frame.keys())
                    ]
                    precomputed_frames.update(deduped_by_frame.keys())

        sampled_frames = len(precomputed_frames)
        missing_face_track_ids = {
            str(tid) for tid in track_to_dets.keys() if not face_ts_by_track.get(str(tid))
        }
        frame_items = []
        for frame_idx in sorted(frame_to_dets.keys()):
            dets = [
                det
                for det in frame_to_dets.get(frame_idx, [])
                if str(det.get("track_id", "")) in missing_face_track_ids
            ]
            if dets:
                frame_items.append((frame_idx, dets))
        face_worker_count = self._face_ledger_workers()
        segment_size = max(1, int(os.getenv("CLYPT_FACE_LEDGER_SEGMENT_FRAMES", "240")))
        segment_count = 0 if not frame_items else max(1, len(frame_items) // max(1, segment_size))

        def _consume_frame_items(items: list[tuple[int, list[dict]]]) -> tuple[dict[str, list[dict]], int]:
            local_face_ts_by_track: dict[str, list[dict]] = {}
            local_sampled_frames = 0
            if not items:
                return local_face_ts_by_track, local_sampled_frames
            try:
                from decord import VideoReader, cpu

                vr = VideoReader(video_path, ctx=cpu(0))
                valid_items = [(int(frame_idx), dets) for frame_idx, dets in items if 0 <= int(frame_idx) < len(vr)]
                if not valid_items:
                    return local_face_ts_by_track, local_sampled_frames
                frame_indices = [frame_idx for frame_idx, _ in valid_items]
                batch = vr.get_batch(frame_indices).asnumpy()
                frame_map = {frame_idx: batch[idx] for idx, frame_idx in enumerate(frame_indices)}
            except Exception:
                valid_items = [(int(frame_idx), dets) for frame_idx, dets in items]
                frame_map = {
                    frame_idx: self._read_frame_rgb(video_path, frame_idx)
                    for frame_idx, _ in valid_items
                }

            for frame_idx, dets in valid_items:
                frame_rgb = frame_map.get(frame_idx)
                if frame_rgb is None:
                    continue
                local_sampled_frames += 1
                best_by_track: dict[str, dict] = {}
                for det in dets:
                    tid = str(det.get("track_id", ""))
                    if not tid:
                        continue
                    previous = best_by_track.get(tid)
                    if previous is None or float(det.get("confidence", 0.0)) > float(
                        previous.get("confidence", 0.0)
                    ):
                        best_by_track[tid] = det

                for tid, det in best_by_track.items():
                    _, anchor = self._detect_face_in_person_det(frame_rgb, det)
                    bbox = self._anchor_to_face_bbox(det, anchor, frame_width, frame_height)
                    if bbox is None:
                        continue
                    local_face_ts_by_track.setdefault(tid, []).append(
                        {
                            "time_ms": int(round((frame_idx / max(1e-6, fps)) * 1000.0)),
                            "bounding_box": bbox,
                            "track_id": tid,
                            "confidence": float(det.get("confidence", 0.0)),
                            "source": "face_detector",
                            "provenance": "insightface_roi",
                        }
                    )
            return local_face_ts_by_track, local_sampled_frames

        if face_worker_count <= 1 or len(frame_items) <= segment_size:
            local_face_ts_by_track, local_sampled_frames = _consume_frame_items(frame_items)
            sampled_frames += local_sampled_frames
            for tid, ts_objs in local_face_ts_by_track.items():
                face_ts_by_track.setdefault(tid, []).extend(ts_objs)
            segment_count = max(segment_count, 1 if frame_items else 0)
        else:
            from concurrent.futures import ThreadPoolExecutor

            segments = [
                frame_items[idx : idx + segment_size]
                for idx in range(0, len(frame_items), segment_size)
            ]
            segment_count = max(segment_count, len(segments))
            with ThreadPoolExecutor(max_workers=face_worker_count) as pool:
                futures = [
                    pool.submit(_consume_frame_items, segment)
                    for segment in segments
                ]
                for fut in futures:
                    local_face_ts_by_track, local_sampled_frames = fut.result()
                    sampled_frames += local_sampled_frames
                    for tid, ts_objs in local_face_ts_by_track.items():
                        face_ts_by_track.setdefault(tid, []).extend(ts_objs)

        face_detections = []
        for idx, tid in enumerate(sorted(face_ts_by_track.keys())):
            ts_objs = sorted(face_ts_by_track[tid], key=lambda obj: int(obj.get("time_ms", 0)))
            if not ts_objs:
                continue
            face_detections.append(
                {
                    "confidence": float(
                        sum(float(obj.get("confidence", 0.0)) for obj in ts_objs) / max(1, len(ts_objs))
                    ),
                    "segment_start_ms": int(ts_objs[0]["time_ms"]),
                    "segment_end_ms": int(ts_objs[-1]["time_ms"]),
                    "face_track_index": idx,
                    "track_id": tid,
                    "source": "face_detector",
                    "provenance": "insightface_roi",
                    "timestamped_objects": ts_objs,
                }
            )

        metrics = {
            "face_detection_wallclock_s": round(time.perf_counter() - started_at, 3),
            "face_detection_frame_samples": sampled_frames,
            "face_detection_track_count": len(face_detections),
            "face_detection_segment_count": segment_count,
            "face_detection_worker_count": face_worker_count,
        }
        return face_detections, person_detections, metrics

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

    def _build_sparse_face_box_map(
        self,
        video_path: str,
        frame_to_dets: dict[int, list[dict]],
        track_to_dets: dict[str, list[dict]],
        fps: float,
        target_fps: float,
        relevant_frames: set[int] | None = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> dict[tuple[str, int], tuple[int, int, int, int, float, float]]:
        """Precompute per-track face boxes at sparse FPS and interpolate short gaps.

        Returns:
            {(track_id, frame_idx): (x1, y1, x2, y2, det_conf, quality)}
            quality=1.0 direct detector hit, <1.0 interpolated/propagated.
        """
        import os
        import cv2
        import numpy as np
        from bisect import bisect_left
        from collections import defaultdict
        from decord import VideoReader, cpu

        h264_path = video_path.replace(".mp4", "_h264.mp4")
        read_path = h264_path if os.path.exists(h264_path) else video_path

        try:
            vr = VideoReader(read_path, ctx=cpu(0))
        except Exception as e:
            print(
                "  Warning: sparse face precompute could not open video "
                f"({type(e).__name__}: {e})"
            )
            return {}

        total_frames = len(vr)
        if total_frames <= 0:
            return {}

        # Sparse sampling controls.
        sample_stride = max(1, int(round(max(1e-6, fps) / max(0.1, target_fps))))
        min_det_conf = float(os.getenv("CLYPT_ASD_FACE_MIN_DET_CONF", "0.25"))
        min_area_ratio = float(os.getenv("CLYPT_ASD_FACE_MIN_AREA_RATIO", "0.0004"))
        max_interp_gap = int(
            os.getenv(
                "CLYPT_ASD_FACE_MAX_INTERP_GAP",
                str(max(4, int(round(max(1.0, fps) * 1.0)))),
            )
        )

        sample_frames = set(range(0, total_frames, sample_stride))
        # Anchor endpoints for each track to improve interpolation continuity.
        for dets in track_to_dets.values():
            if not dets:
                continue
            sample_frames.add(int(dets[0]["frame_idx"]))
            sample_frames.add(int(dets[-1]["frame_idx"]))
        sample_frames = sorted(fi for fi in sample_frames if 0 <= fi < total_frames)
        # When caller provides relevant_frames (e.g. frames overlapping word
        # timestamps), skip sampling frames that can't contribute to binding.
        if relevant_frames is not None:
            sample_frames = [fi for fi in sample_frames if fi in relevant_frames]
            print(f"  Sparse precompute restricted to {len(sample_frames)} word-relevant frames")

        sampled: dict[str, dict[int, tuple[int, int, int, int, float, float]]] = defaultdict(dict)
        for fi in sample_frames:
            dets = frame_to_dets.get(fi, [])
            if not dets:
                continue
            try:
                frame = vr[fi].asnumpy()  # RGB
            except Exception:
                continue
            if frame is None or frame.size == 0:
                continue

            fh, fw = frame.shape[:2]
            frame_area = float(max(1, fh * fw))
            # Keep strongest detection per track at frame fi.
            best_by_track: dict[str, dict] = {}
            for d in dets:
                tid = str(d.get("track_id", ""))
                if not tid:
                    continue
                cur = best_by_track.get(tid)
                if cur is None or float(d.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                    best_by_track[tid] = d

            for tid, d in best_by_track.items():
                d_scaled = self._scale_detection_geometry(d, scale_x=scale_x, scale_y=scale_y)
                conf = float(d_scaled.get("confidence", 0.0))
                if conf < min_det_conf:
                    continue
                area = float(d_scaled.get("width", 0.0)) * float(d_scaled.get("height", 0.0))
                if area < (min_area_ratio * frame_area):
                    continue
                crop, anchor = self._detect_face_in_person_det(frame, d_scaled)
                if crop is None or anchor is None:
                    continue
                cx = float(d_scaled.get("x_center", 0.0))
                cy = float(d_scaled.get("y_center", 0.0))
                bw = float(d_scaled.get("width", 0.0))
                bh = float(d_scaled.get("height", 0.0))
                if bw <= 1e-6 or bh <= 1e-6:
                    continue
                x1 = int(round(cx + float(anchor["x_offset"]) * bw))
                y1 = int(round(cy + float(anchor["y_offset"]) * bh))
                ww = int(round(max(2.0, float(anchor["w_ratio"]) * bw)))
                hh = int(round(max(2.0, float(anchor["h_ratio"]) * bh)))
                x1 = max(0, min(fw - 1, x1))
                y1 = max(0, min(fh - 1, y1))
                x2 = max(x1 + 1, min(fw, x1 + ww))
                y2 = max(y1 + 1, min(fh, y1 + hh))
                sampled[tid][fi] = (x1, y1, x2, y2, conf, 1.0)

        # Interpolate sparse boxes onto dense track frames.
        out: dict[tuple[str, int], tuple[int, int, int, int, float, float]] = {}
        for tid, dets in track_to_dets.items():
            frames = sorted({int(d.get("frame_idx", -1)) for d in dets if int(d.get("frame_idx", -1)) >= 0})
            if not frames:
                continue
            s = sampled.get(tid, {})
            if not s:
                continue
            s_keys = sorted(s.keys())
            for fi in frames:
                key = (tid, fi)
                if fi in s:
                    out[key] = s[fi]
                    continue
                pos = bisect_left(s_keys, fi)
                left = s_keys[pos - 1] if pos > 0 else None
                right = s_keys[pos] if pos < len(s_keys) else None
                chosen = None
                if left is not None and right is not None:
                    gap = right - left
                    if gap > 0 and gap <= max_interp_gap:
                        a = (fi - left) / float(gap)
                        l = np.array(s[left][:4], dtype=np.float32)
                        r = np.array(s[right][:4], dtype=np.float32)
                        b = (1.0 - a) * l + a * r
                        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
                        # Clamp and sanitize via known frame dimensions from track box.
                        dref = next((d for d in dets if int(d.get("frame_idx", -1)) == fi), None)
                        if dref is not None:
                            cx = float(dref.get("x_center", 0.0))
                            cy = float(dref.get("y_center", 0.0))
                            bw = float(dref.get("width", 1.0))
                            bh = float(dref.get("height", 1.0))
                            x1 = int(round(max(0.0, min(x1, cx + 1.2 * bw))))
                            y1 = int(round(max(0.0, min(y1, cy + 1.2 * bh))))
                            x2 = int(round(max(x1 + 1.0, x2)))
                            y2 = int(round(max(y1 + 1.0, y2)))
                        conf = float((s[left][4] + s[right][4]) * 0.5)
                        chosen = (x1, y1, x2, y2, conf, 0.7)
                if chosen is None:
                    nearest = None
                    if left is not None:
                        nearest = left
                    if right is not None and (
                        nearest is None or abs(right - fi) < abs(nearest - fi)
                    ):
                        nearest = right
                    if nearest is not None and abs(nearest - fi) <= max(1, max_interp_gap // 2):
                        x1, y1, x2, y2, conf, _ = s[nearest]
                        chosen = (x1, y1, x2, y2, conf, 0.5)
                if chosen is not None:
                    out[key] = chosen

        return out

    @staticmethod
    def _normalized_bbox_to_xyxy_abs(bounding_box: dict, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
        if not isinstance(bounding_box, dict):
            return None
        width = max(1, int(frame_width))
        height = max(1, int(frame_height))
        left = int(round(float(bounding_box.get("left", 0.0)) * width))
        top = int(round(float(bounding_box.get("top", 0.0)) * height))
        right = int(round(float(bounding_box.get("right", 0.0)) * width))
        bottom = int(round(float(bounding_box.get("bottom", 0.0)) * height))
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _build_canonical_face_bbox_lookup(
        self,
        *,
        track_identity_features: dict[str, dict] | None,
        track_to_dets: dict[str, list[dict]],
        frame_width: int,
        frame_height: int,
        max_interp_gap: int = 12,
    ) -> dict[tuple[str, int], tuple[int, int, int, int, float, float]]:
        import numpy as np
        from bisect import bisect_left

        if not track_identity_features:
            return {}

        out: dict[tuple[str, int], tuple[int, int, int, int, float, float]] = {}
        for tid, dets in track_to_dets.items():
            feature = track_identity_features.get(str(tid))
            observations = list((feature or {}).get("face_observations", []))
            if not observations:
                continue
            obs_by_frame: dict[int, tuple[int, int, int, int, float, float]] = {}
            for observation in observations:
                frame_idx = int(observation.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                bbox = self._normalized_bbox_to_xyxy_abs(
                    observation.get("bounding_box", {}),
                    frame_width,
                    frame_height,
                )
                if bbox is None:
                    continue
                obs_by_frame[frame_idx] = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    float(observation.get("confidence", 0.0)),
                    float(observation.get("quality", observation.get("confidence", 0.0))),
                )
            if not obs_by_frame:
                continue

            obs_frames = sorted(obs_by_frame.keys())
            det_frames = sorted({int(det.get("frame_idx", -1)) for det in dets if int(det.get("frame_idx", -1)) >= 0})
            for frame_idx in det_frames:
                key = (str(tid), int(frame_idx))
                if frame_idx in obs_by_frame:
                    out[key] = obs_by_frame[frame_idx]
                    continue
                pos = bisect_left(obs_frames, frame_idx)
                left = obs_frames[pos - 1] if pos > 0 else None
                right = obs_frames[pos] if pos < len(obs_frames) else None
                if left is not None and right is not None:
                    gap = right - left
                    if gap > 0 and gap <= max_interp_gap:
                        alpha = (frame_idx - left) / float(gap)
                        left_box = np.array(obs_by_frame[left][:4], dtype=np.float32)
                        right_box = np.array(obs_by_frame[right][:4], dtype=np.float32)
                        interp = ((1.0 - alpha) * left_box) + (alpha * right_box)
                        conf = 0.5 * (obs_by_frame[left][4] + obs_by_frame[right][4])
                        quality = 0.5 * (obs_by_frame[left][5] + obs_by_frame[right][5]) * 0.8
                        out[key] = (
                            int(round(float(interp[0]))),
                            int(round(float(interp[1]))),
                            int(round(float(interp[2]))),
                            int(round(float(interp[3]))),
                            float(conf),
                            float(quality),
                        )
                        continue
                nearest = None
                if left is not None:
                    nearest = left
                if right is not None and (nearest is None or abs(right - frame_idx) < abs(nearest - frame_idx)):
                    nearest = right
                if nearest is not None and abs(nearest - frame_idx) <= max(1, max_interp_gap // 2):
                    x1, y1, x2, y2, conf, quality = obs_by_frame[nearest]
                    out[key] = (x1, y1, x2, y2, conf, float(quality) * 0.6)
        return out

    def _load_or_build_lrasd_audio_features(
        self,
        *,
        audio_wav_path: str,
        cache_path: str,
        fps: float,
    ):
        import numpy as np
        import python_speech_features
        from scipy.io import wavfile

        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                cached_audio = cached.get("audio_features")
                if cached_audio is not None:
                    return np.asarray(cached_audio, dtype=np.float32)
            except Exception:
                pass

        sr, wav = wavfile.read(audio_wav_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = wav.astype(np.int16, copy=False)
        if sr <= 0:
            return np.zeros((0, 13), dtype=np.float32)

        mfcc_winlen = 0.025 * 25.0 / max(fps, 1e-6)
        mfcc_winstep = 0.010 * 25.0 / max(fps, 1e-6)
        audio_features = python_speech_features.mfcc(
            wav,
            sr,
            numcep=13,
            winlen=mfcc_winlen,
            winstep=mfcc_winstep,
        )
        audio_features = np.asarray(audio_features, dtype=np.float32)
        try:
            np.savez_compressed(cache_path, audio_features=audio_features)
        except Exception:
            pass
        return audio_features

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

    def _lrasd_forward_scores(
        self,
        audio_t,
        visual_t,
    ):
        """Batched LR-ASD forward.

        Returns per-frame speaking probabilities in [0, 1].
        """
        import torch

        b, t = visual_t.shape[:2]
        self._lrasd_debug_calls = getattr(self, "_lrasd_debug_calls", 0) + 1
        debug_now = (
            getattr(self, "model_debug", False)
            and (
                self._lrasd_debug_calls
                % max(1, getattr(self, "model_debug_every", 20))
                == 1
            )
        )
        if debug_now:
            print("  [LR-ASD DEBUG] Input tensors:")
            print("   " + self._tensor_debug_stats("audio_t", audio_t))
            print("   " + self._tensor_debug_stats("visual_t", visual_t))

        audio_embed = self.lrasd_model.forward_audio_frontend(audio_t)
        visual_embed = self.lrasd_model.forward_visual_frontend(visual_t)
        if debug_now:
            print("   " + self._tensor_debug_stats("audio_embed", audio_embed))
            print("   " + self._tensor_debug_stats("visual_embed", visual_embed))

        outs_av = self.lrasd_model.forward_audio_visual_backend(audio_embed, visual_embed)
        if debug_now:
            print("   " + self._tensor_debug_stats("outs_av", outs_av))
        if outs_av.shape[0] != b * t:
            raise RuntimeError(
                f"LR-ASD output shape mismatch: outs_av={tuple(outs_av.shape)}, expected first dim {b*t}"
            )
        av_logits = self.lrasd_loss_av.FC(outs_av)
        av_prob = torch.softmax(av_logits, dim=-1)[:, 1].reshape(b, t)
        if debug_now:
            print("   " + self._tensor_debug_stats("av_logits", av_logits))
            print("   " + self._tensor_debug_stats("av_prob", av_prob))
            print(f"  [LR-ASD DEBUG] forward_call={self._lrasd_debug_calls} b={b} t={t}")
        return av_prob

    @modal.enter()
    def load_model(self):
        """Load Parakeet, YOLO, and LR-ASD into GPU VRAM."""
        import os
        import sys
        import nemo.collections.asr as nemo_asr
        import torch
        from insightface.app import FaceAnalysis
        from ultralytics import YOLO
        from omegaconf import open_dict

        self.model_debug = os.getenv("CLYPT_MODEL_DEBUG", "0") == "1"
        self.model_debug_every = int(os.getenv("CLYPT_MODEL_DEBUG_EVERY", "20"))
        self._lrasd_debug_calls = 0
        if self.model_debug:
            print(
                "Model debug logging enabled: "
                f"CLYPT_MODEL_DEBUG=1, every={self.model_debug_every} LR-ASD forwards"
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
        self.mediapipe_face_detector = None
        try:
            import mediapipe as mp

            self.mediapipe_face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.20,
            )
            print("MediaPipe face fallback ready.")
        except Exception as e:
            print(
                "Warning: MediaPipe face fallback unavailable "
                f"({type(e).__name__}: {e})"
            )

        # --- Load LR-ASD ---
        self.lrasd_model = None
        self.lrasd_loss_av = None
        try:
            print("Loading LR-ASD model into GPU VRAM...")
            if LRASD_REPO_ROOT not in sys.path:
                sys.path.insert(0, LRASD_REPO_ROOT)
            from model.Model import ASD_Model
            from loss import lossAV

            self.lrasd_model = ASD_Model()
            self.lrasd_loss_av = lossAV()
            self._load_lrasd_checkpoint(
                self.lrasd_model,
                self.lrasd_loss_av,
                LRASD_MODEL_PATH,
            )
            self.lrasd_model = self.lrasd_model.to(self.gpu_device)
            self.lrasd_model.eval()
            self.lrasd_loss_av = self.lrasd_loss_av.to(self.gpu_device)
            self.lrasd_loss_av.eval()
            print("LR-ASD ready.")
        except Exception as e:
            # Keep worker alive; binding step falls back if this fails at runtime.
            self.lrasd_model = None
            self.lrasd_loss_av = None
            print(
                "Warning: failed to load LR-ASD checkpoint "
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
                    "speaker_track_id": None,  # populated by speaker binding later
                })
        else:
            print("Warning: no timestamp dict with 'word' key found")

        print(f"ASR complete: {len(words)} words transcribed")
        return words

    # ──────────────────────────────────────────
    # Visual tracking (YOLOv26 + BoT-SORT)
    # ──────────────────────────────────────────
    def _ensure_h264(self, video_path: str) -> str:
        """Re-encode to H.264 if the video uses AV1 or another codec OpenCV can't decode."""
        started = time.perf_counter()
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
                        "-threads",
                        "0",
                        "-crf",
                        "23",
                        "-an",
                        h264_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elapsed = time.perf_counter() - started
            print(
                f"  Re-encoded: {os.path.getsize(h264_path) / 1e6:.1f} MB "
                f"in {elapsed:.1f}s"
            )
            return h264_path
        return video_path

    @staticmethod
    def _tracking_chunk_workers() -> int:
        try:
            requested = int(os.getenv("CLYPT_TRACK_CHUNK_WORKERS", "1"))
        except Exception:
            requested = 1
        return max(1, min(3, requested))

    def _get_tracking_model(self):
        model = getattr(self, "_shared_tracking_model", None)
        if model is not None:
            return model

        model = getattr(self, "yolo_model", None)
        if model is not None:
            self._shared_tracking_model = model
            return model

        model = self._build_tracking_model()
        self._shared_tracking_model = model
        self.yolo_model = model
        return model

    @staticmethod
    def _build_tracking_model():
        from ultralytics import YOLO

        model_path = YOLO_ENGINE_PATH if os.path.exists(YOLO_ENGINE_PATH) else YOLO_WEIGHTS_PATH
        return YOLO(model_path)

    def _select_tracking_mode(self) -> str:
        requested_mode = os.getenv("CLYPT_TRACKING_MODE", "auto").strip().lower()
        if requested_mode in {"direct", "chunked"}:
            return requested_mode
        if requested_mode == "shared_analysis_proxy":
            return requested_mode
        if requested_mode not in {"", "auto"}:
            print(
                f"  Warning: unknown CLYPT_TRACKING_MODE={requested_mode!r}; "
                "falling back to auto"
            )
        if self._shared_analysis_proxy_enabled():
            return "shared_analysis_proxy"
        return "direct" if self._tracking_chunk_workers() == 1 else "chunked"

    @staticmethod
    def _scale_detection_geometry(det: dict, scale_x: float, scale_y: float) -> dict:
        if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
            return det
        scaled = dict(det)
        for key, scale in (
            ("x1", scale_x),
            ("x2", scale_x),
            ("y1", scale_y),
            ("y2", scale_y),
        ):
            if key in scaled:
                scaled[key] = float(scaled[key]) * scale
        for key, scale in (
            ("x_center", scale_x),
            ("y_center", scale_y),
            ("width", scale_x),
            ("height", scale_y),
        ):
            if key in scaled:
                scaled[key] = float(scaled[key]) * scale
        return scaled

    def _prepare_analysis_video(self, video_path: str) -> dict:
        """Create one shared H.264/analysis-proxy path for tracking, faces, and LR-ASD."""
        prepared_video_path = self._ensure_h264(video_path)
        source_meta = self._probe_video_meta(prepared_video_path)
        source_width = int(source_meta.get("width", 0) or 0)
        source_height = int(source_meta.get("height", 0) or 0)
        analysis_video_path = prepared_video_path
        analysis_meta = dict(source_meta)
        scale_x = 1.0
        scale_y = 1.0

        if self._shared_analysis_proxy_enabled():
            max_long_edge = self._analysis_proxy_max_long_edge()
            long_edge = max(source_width, source_height)
            if source_width > 0 and source_height > 0 and max_long_edge > 0 and long_edge > max_long_edge:
                proxy_path = prepared_video_path.replace(".mp4", "_analysis_proxy.mp4")
                if not os.path.exists(proxy_path):
                    scale_expr = (
                        f"scale={max_long_edge}:-2"
                        if source_width >= source_height
                        else f"scale=-2:{max_long_edge}"
                    )
                    print(
                        "  Shared analysis proxy: "
                        f"{source_width}x{source_height} -> long_edge {max_long_edge}px"
                    )
                    try:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                prepared_video_path,
                                "-vf",
                                scale_expr,
                                "-c:v",
                                "h264_nvenc",
                                "-preset",
                                "p4",
                                "-cq",
                                "23",
                                "-an",
                                proxy_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                prepared_video_path,
                                "-vf",
                                scale_expr,
                                "-c:v",
                                "libx264",
                                "-preset",
                                "ultrafast",
                                "-threads",
                                "0",
                                "-crf",
                                "23",
                                "-an",
                                proxy_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                proxy_meta = self._probe_video_meta(proxy_path)
                proxy_width = int(proxy_meta.get("width", 0) or 0)
                proxy_height = int(proxy_meta.get("height", 0) or 0)
                if proxy_width > 0 and proxy_height > 0:
                    analysis_video_path = proxy_path
                    analysis_meta = proxy_meta
                    scale_x = proxy_width / max(1.0, float(source_width))
                    scale_y = proxy_height / max(1.0, float(source_height))
                    print(
                        "  Shared analysis proxy ready: "
                        f"{proxy_width}x{proxy_height} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})"
                    )

        return {
            "source_video_path": video_path,
            "prepared_video_path": prepared_video_path,
            "analysis_video_path": analysis_video_path,
            "source_meta": source_meta,
            "analysis_meta": analysis_meta,
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
        }

    def _prepare_speaker_binding_video(self, video_path: str) -> tuple[str, float, float]:
        """Compatibility wrapper around the shared analysis proxy."""
        if not self._shared_analysis_proxy_enabled():
            if os.getenv("CLYPT_SPEAKER_BINDING_PROXY_ENABLE", "1") != "1":
                return video_path, 1.0, 1.0
            max_long_edge = max(0, int(os.getenv("CLYPT_SPEAKER_BINDING_PROXY_MAX_LONG_EDGE", "1280")))
            if max_long_edge <= 0:
                return video_path, 1.0, 1.0
            meta = self._probe_video_meta(video_path)
            width = int(meta.get("width", 0) or 0)
            height = int(meta.get("height", 0) or 0)
            long_edge = max(width, height)
            if width <= 0 or height <= 0 or long_edge <= max_long_edge:
                return video_path, 1.0, 1.0

            proxy_path = video_path.replace(".mp4", "_speaker_proxy.mp4")
            if not os.path.exists(proxy_path):
                scale_expr = f"scale={max_long_edge}:-2" if width >= height else f"scale=-2:{max_long_edge}"
                print(f"  Speaker-binding proxy: {width}x{height} -> long_edge {max_long_edge}px")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            video_path,
                            "-vf",
                            scale_expr,
                            "-c:v",
                            "h264_nvenc",
                            "-preset",
                            "p4",
                            "-cq",
                            "23",
                            "-an",
                            proxy_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            video_path,
                            "-vf",
                            scale_expr,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "ultrafast",
                            "-crf",
                            "23",
                            "-an",
                            proxy_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            proxy_meta = self._probe_video_meta(proxy_path)
            proxy_width = int(proxy_meta.get("width", 0) or 0)
            proxy_height = int(proxy_meta.get("height", 0) or 0)
            if proxy_width <= 0 or proxy_height <= 0:
                return video_path, 1.0, 1.0
            scale_x = proxy_width / max(1.0, float(width))
            scale_y = proxy_height / max(1.0, float(height))
            print(
                "  Speaker-binding proxy ready: "
                f"{proxy_width}x{proxy_height} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})"
            )
            return proxy_path, scale_x, scale_y
        context = self._prepare_analysis_video(video_path)
        return (
            str(context["analysis_video_path"]),
            float(context["scale_x"]),
            float(context["scale_y"]),
        )

    def _select_speaker_binding_mode(
        self,
        video_path: str,
        tracks: list[dict],
        words: list[dict],
    ) -> str:
        """Choose the speaker-binding path for this clip."""
        eval_profile = os.getenv("CLYPT_PHASE1_EVAL_PROFILE", "").strip().lower()
        if eval_profile in {"podcast", "eval", "test"}:
            print(
                "  Speaker binding mode=lrasd (eval profile): "
                f"profile={eval_profile}"
            )
            return "lrasd"

        requested_mode = os.getenv("CLYPT_SPEAKER_BINDING_MODE", "auto").strip().lower()
        if requested_mode in {"heuristic", "lrasd"}:
            return requested_mode
        if requested_mode == "shared_analysis_proxy":
            return requested_mode
        if requested_mode not in {"", "auto"}:
            print(
                f"  Warning: unknown CLYPT_SPEAKER_BINDING_MODE={requested_mode!r}; "
                "falling back to auto"
            )
        if self._shared_analysis_proxy_enabled():
            return "shared_analysis_proxy"

        meta = self._probe_video_meta(video_path)
        width = int(meta.get("width", 0) or 0)
        height = int(meta.get("height", 0) or 0)
        duration_s = float(meta.get("duration_s", 0.0) or 0.0)
        long_edge = max(width, height)

        auto_max_duration_s = float(
            os.getenv("CLYPT_SPEAKER_BINDING_AUTO_MAX_DURATION_S", "180")
        )
        auto_max_long_edge = int(
            os.getenv("CLYPT_SPEAKER_BINDING_AUTO_MAX_LONG_EDGE", "1920")
        )
        auto_max_words = int(os.getenv("CLYPT_SPEAKER_BINDING_AUTO_MAX_WORDS", "450"))
        auto_max_tracks = int(os.getenv("CLYPT_SPEAKER_BINDING_AUTO_MAX_TRACKS", "12"))

        if (
            (duration_s > auto_max_duration_s and long_edge > auto_max_long_edge)
            or duration_s > (auto_max_duration_s * 1.5)
            or len(words) > auto_max_words
            or len(tracks) > auto_max_tracks
        ):
            print(
                "  Speaker binding mode=heuristic (auto): "
                f"duration_s={duration_s:.1f}, long_edge={long_edge}, "
                f"tracks={len(tracks)}, words={len(words)}"
            )
            return "heuristic"

        print(
            "  Speaker binding mode=lrasd (auto): "
            f"duration_s={duration_s:.1f}, long_edge={long_edge}, "
            f"tracks={len(tracks)}, words={len(words)}"
        )
        return "lrasd"

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
        output_meta: dict | None = None,
        coord_scale_x: float = 1.0,
        coord_scale_y: float = 1.0,
        model=None,
    ) -> dict:
        """Track one chunk independently and emit per-chunk NDJSON."""
        import json
        import os
        import subprocess
        import time

        fps = float(meta["fps"])
        width = int(meta["width"])
        height = int(meta["height"])
        output_meta = output_meta or meta
        output_width = int(output_meta.get("width", width) or width)
        output_height = int(output_meta.get("height", height) or height)
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

        if model is None:
            model = self._build_tracking_model()
        # QA fidelity mode: keep chunk tracking dense so downstream camera-follow
        # uses detector-driven boxes instead of propagated/ROI-refined boxes.
        stride = 1
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
        analysis_tracks = []
        for local_fi, r in enumerate(results):
            processed_frames += 1
            global_fi = start_f + local_fi
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
                analysis_tracks.append(dict(out))
                tracks.append(
                    self._scale_detection_geometry(
                        out,
                        scale_x=(1.0 / max(coord_scale_x, 1e-6)),
                        scale_y=(1.0 / max(coord_scale_y, 1e-6)),
                    )
                )

        track_identity_features, face_pipeline_metrics = self._extract_track_identity_features_from_segments(
            video_path=chunk_video_path,
            fps=fps,
            frame_width=width,
            frame_height=height,
            output_frame_width=output_width,
            output_frame_height=output_height,
            coord_scale_x=coord_scale_x,
            coord_scale_y=coord_scale_y,
            segments=self._split_tracks_into_face_segments(analysis_tracks, self._face_pipeline_segment_frames()),
        )
        emb_map = {
            tid: feature.get("embedding")
            for tid, feature in track_identity_features.items()
            if feature.get("embedding") is not None
        }
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
            "track_identity_features": track_identity_features,
            "face_pipeline_metrics": face_pipeline_metrics,
            "ndjson_path": ndjson_path,
        }

    def _run_tracking_direct(self, video_path: str, analysis_context: dict | None = None) -> tuple[list[dict], dict]:
        """Run YOLO + BoT-SORT as a single full-video stream."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        print("Running YOLO26s + BoT-SORT direct tracking inference...")
        analysis_context = analysis_context or self._prepare_analysis_video(video_path)
        tracking_video_path = str(analysis_context.get("analysis_video_path", video_path))
        meta = dict(analysis_context.get("analysis_meta", self._probe_video_meta(tracking_video_path)))
        source_meta = dict(analysis_context.get("source_meta", meta))
        scale_x = float(analysis_context.get("scale_x", 1.0) or 1.0)
        scale_y = float(analysis_context.get("scale_y", 1.0) or 1.0)
        source_width = int(source_meta.get("width", meta.get("width", 0)) or 0)
        source_height = int(source_meta.get("height", meta.get("height", 0)) or 0)
        fps = float(meta["fps"])
        total_frames = int(meta["total_frames"])
        width = int(meta["width"])
        height = int(meta["height"])
        if total_frames <= 0:
            print("  Warning: no frames found in video")
            return [], {}

        infer_imgsz = max(320, int(os.getenv("CLYPT_YOLO_IMGSZ", "640")))
        tracker_cfg = self._ensure_botsort_reid_yaml()
        model = self._get_tracking_model()
        lb_meta = self._compute_letterbox_meta(width, height, infer_imgsz, infer_imgsz)
        started = time.time()
        log_every_n_frames = 600
        face_pipeline_segment_frames = self._face_pipeline_segment_frames()
        face_pipeline_workers = self._face_pipeline_workers()
        pending_face_segment_tracks: list[dict] = []
        face_segment_futures = []
        face_segments_submitted = 0

        results = model.track(
            source=tracking_video_path,
            tracker=tracker_cfg,
            persist=True,
            classes=[0],
            stream=True,
            verbose=False,
            vid_stride=1,
            imgsz=infer_imgsz,
        )

        tracks = []
        n_boxes = 0
        with ThreadPoolExecutor(max_workers=face_pipeline_workers) as face_pool:
            for frame_idx, r in enumerate(results, start=1):
                if r.boxes is None or r.boxes.id is None:
                    if frame_idx % face_pipeline_segment_frames == 0 and pending_face_segment_tracks:
                        face_segment_futures.append(
                            face_pool.submit(
                                self._extract_track_identity_features_for_segment,
                                video_path=video_path,
                                segment_tracks=list(pending_face_segment_tracks),
                                fps=fps,
                                frame_width=width,
                                frame_height=height,
                            )
                        )
                        face_segments_submitted += 1
                        if face_segments_submitted == 1 or face_segments_submitted % 4 == 0:
                            print(
                                "  Face pipeline queued: "
                                f"{face_segments_submitted} segments, "
                                f"workers={face_pipeline_workers}"
                            )
                        pending_face_segment_tracks = []
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
                        "frame_idx": int(frame_idx - 1),
                        "local_frame_idx": int(frame_idx - 1),
                        "chunk_idx": 0,
                        "track_id": f"track_{int(tid_raw)}",
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
                    pending_face_segment_tracks.append(dict(out))
                    projected_out = self._scale_detection_geometry(
                        out,
                        scale_x=(1.0 / max(scale_x, 1e-6)),
                        scale_y=(1.0 / max(scale_y, 1e-6)),
                    )
                    px1, py1, px2, py2 = (
                        float(projected_out["x1"]),
                        float(projected_out["y1"]),
                        float(projected_out["x2"]),
                        float(projected_out["y2"]),
                    )
                    xcn, ycn, wn, hn = self._xyxy_abs_to_xywhn(
                        px1,
                        py1,
                        px2,
                        py2,
                        source_width or width,
                        source_height or height,
                    )
                    projected_out["bbox_norm_xywh"] = {
                        "x_center": float(xcn),
                        "y_center": float(ycn),
                        "width": float(wn),
                        "height": float(hn),
                    }
                    tracks.append(projected_out)
                    n_boxes += 1

                if frame_idx % face_pipeline_segment_frames == 0 and pending_face_segment_tracks:
                    face_segment_futures.append(
                        face_pool.submit(
                            self._extract_track_identity_features_for_segment,
                            video_path=tracking_video_path,
                            segment_tracks=list(pending_face_segment_tracks),
                            fps=fps,
                            frame_width=width,
                            frame_height=height,
                            output_frame_width=source_width or width,
                            output_frame_height=source_height or height,
                            coord_scale_x=scale_x,
                            coord_scale_y=scale_y,
                        )
                    )
                    face_segments_submitted += 1
                    if face_segments_submitted == 1 or face_segments_submitted % 4 == 0:
                        print(
                            "  Face pipeline queued: "
                            f"{face_segments_submitted} segments, "
                            f"workers={face_pipeline_workers}"
                        )
                    pending_face_segment_tracks = []

                if frame_idx % log_every_n_frames == 0:
                    elapsed = max(1e-6, time.time() - started)
                    fps_eff = frame_idx / elapsed
                    pct = (100.0 * frame_idx) / max(1, total_frames)
                    print(
                        "  YOLO progress: "
                        f"{frame_idx}/{total_frames} frames ({pct:.1f}%), "
                        f"{n_boxes} boxes, {fps_eff:.1f} fps"
                    )

            if pending_face_segment_tracks:
                face_segment_futures.append(
                    face_pool.submit(
                        self._extract_track_identity_features_for_segment,
                        video_path=tracking_video_path,
                        segment_tracks=list(pending_face_segment_tracks),
                        fps=fps,
                        frame_width=width,
                        frame_height=height,
                        output_frame_width=source_width or width,
                        output_frame_height=source_height or height,
                        coord_scale_x=scale_x,
                        coord_scale_y=scale_y,
                    )
                )
                face_segments_submitted += 1
                print(
                    "  Face pipeline queued: "
                    f"{face_segments_submitted} segments, "
                    f"workers={face_pipeline_workers}"
                )

        elapsed = max(1e-6, time.time() - started)
        eff_fps = float(total_frames / elapsed)
        metrics = {
            "tracking_wallclock_s": float(elapsed),
            "throughput_fps": float(eff_fps),
            "schema_pass_rate": 1.0,
            "track_fragmentation_rate": float(
                len({str(t.get("track_id")) for t in tracks}) / max(1.0, len(tracks) ** 0.5)
            ),
        }
        track_identity_features, face_pipeline_metrics = self._extract_track_identity_features_from_segments(
            video_path=tracking_video_path,
            fps=fps,
            frame_width=width,
            frame_height=height,
            output_frame_width=source_width or width,
            output_frame_height=source_height or height,
            coord_scale_x=scale_x,
            coord_scale_y=scale_y,
            segment_futures=face_segment_futures,
        )
        metrics.update(face_pipeline_metrics)
        metrics["analysis_context"] = analysis_context
        if track_identity_features:
            metrics["track_identity_features"] = track_identity_features
        print(
            "  Face pipeline complete: "
            f"{int(face_pipeline_metrics.get('face_pipeline_segments_processed', 0))}/"
            f"{int(face_pipeline_metrics.get('face_pipeline_segment_count', len(face_segment_futures)))} segments, "
            f"{int(face_pipeline_metrics.get('face_pipeline_track_count', len(track_identity_features or {})))} tracks"
        )
        print(
            "Tracking complete: "
            f"{len(tracks)} boxes across {total_frames} frames, "
            f"{eff_fps:.1f} effective fps"
        )
        return tracks, metrics

    def _run_tracking_chunked(self, video_path: str, analysis_context: dict | None = None) -> tuple[list[dict], dict]:
        """Run chunked YOLO + BoT-SORT tracking with stitching."""
        import os
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("Running YOLO26s + BoT-SORT (ReID/GMC) chunked tracking inference...")
        analysis_context = analysis_context or self._prepare_analysis_video(video_path)
        tracking_video_path = str(analysis_context.get("analysis_video_path", video_path))
        meta = dict(analysis_context.get("analysis_meta", self._probe_video_meta(tracking_video_path)))
        source_meta = dict(analysis_context.get("source_meta", meta))
        scale_x = float(analysis_context.get("scale_x", 1.0) or 1.0)
        scale_y = float(analysis_context.get("scale_y", 1.0) or 1.0)
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
        workers = self._tracking_chunk_workers()
        shared_model = self._get_tracking_model() if workers == 1 else None
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    self._track_single_chunk,
                    tracking_video_path,
                    meta,
                    chunk,
                    tracker_cfg,
                    chunk_dir,
                    source_meta,
                    scale_x,
                    scale_y,
                    shared_model,
                )
                for chunk in chunks
            ]
            for i, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                results.append(res)
                pct = (100.0 * i) / max(1, len(chunks))
                print(f"  Chunk progress: {i}/{len(chunks)} ({pct:.1f}%)")

        try:
            TRACKING_VOLUME.commit()
            TRACKING_VOLUME.reload()
        except Exception:
            pass

        stitched, metrics = self._stitch_chunk_tracks(results, fps=fps)
        elapsed = max(1e-6, time.time() - started)
        eff_fps = float(total_frames / elapsed)
        if isinstance(metrics, dict):
            metrics["tracking_mode"] = "chunked"
            metrics["tracking_wallclock_s"] = float(elapsed)
            metrics["throughput_fps"] = float(eff_fps)
            metrics["schema_pass_rate"] = 1.0
            metrics["analysis_context"] = analysis_context
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
        merged_feature_maps: list[dict[str, dict]] = []
        for chunk_data in chunk_results:
            cidx = int(chunk_data["chunk_idx"])
            local_features = chunk_data.get("track_identity_features", {}) or {}
            mapped_features: dict[str, dict] = {}
            for local_tid, feature in local_features.items():
                gid = local_to_global.get((cidx, str(local_tid)))
                if gid is None:
                    continue
                mapped_features[gid] = feature
            if mapped_features:
                merged_feature_maps.append(mapped_features)
        stitched_track_identity_features = self._merge_track_identity_feature_sets(merged_feature_maps)
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
        if stitched_track_identity_features:
            metrics["track_identity_features"] = stitched_track_identity_features
            metrics["face_pipeline_track_count"] = len(stitched_track_identity_features)
        _ = fragments
        return unified, metrics

    def _run_tracking(self, video_path: str) -> tuple[list[dict], dict]:
        """Run tracking using either direct or chunked execution."""
        mode = self._select_tracking_mode()
        print(f"Tracking mode={mode}")
        analysis_context = self._prepare_analysis_video(video_path)
        execution_mode = "direct" if mode in {"direct", "shared_analysis_proxy"} else "chunked"
        if execution_mode == "direct":
            try:
                tracks, metrics = self._run_tracking_direct(video_path, analysis_context=analysis_context)
            except TypeError:
                tracks, metrics = self._run_tracking_direct(video_path)
        else:
            try:
                tracks, metrics = self._run_tracking_chunked(video_path, analysis_context=analysis_context)
            except TypeError:
                tracks, metrics = self._run_tracking_chunked(video_path)
        metrics = dict(metrics or {})
        metrics.setdefault("tracking_mode", mode)
        return tracks, metrics

    # ──────────────────────────────────────────
    # Global tracklet clustering (InsightFace + DBSCAN)
    # ──────────────────────────────────────────
    def _cluster_tracklets(
        self,
        video_path: str,
        tracks: list[dict],
        track_to_dets: dict[str, list[dict]] | None = None,
        track_identity_features: dict[str, dict] | None = None,
    ) -> list[dict]:
        """Cluster fragmented BoT-SORT track IDs into global person IDs via GPU face embeddings."""
        import os
        import numpy as np
        from sklearn.cluster import DBSCAN

        if not tracks:
            self._last_clustering_metrics = {
                "cluster_visible_people_estimate": 0,
                "overfragmentation_proxy": 0.0,
                "accidental_merge_proxy": 0.0,
                "covisibility_conflict_rejections": 0,
                "histogram_attach_rejections": 0,
            }
            self._last_track_identity_features_after_clustering = (
                dict(track_identity_features) if isinstance(track_identity_features, dict) else None
            )
            return tracks

        extraction_cfg = _cluster_extraction_config()
        # ArcFace embeddings are unit-normalized; cosine distance is preferred.
        dbscan_eps = 0.44
        dbscan_min_samples = 1
        reassign_noise_max_dist = 0.45
        # Conservative tiny-cluster merge: avoid collapsing real speakers.
        tiny_cluster_min_tracklets = 3
        tiny_cluster_min_boxes = max(60, int(len(tracks) * 0.003))
        tiny_cluster_merge_max_dist = 0.40
        # Final de-fragmentation pass: only merge clusters that are highly
        # similar and nearly never co-visible.
        cross_cluster_merge_base_cos = float(extraction_cfg["cluster_cross_merge_max_cos"])
        cross_cluster_merge_max_overlap = float(extraction_cfg["cluster_cross_merge_max_overlap"])
        cross_cluster_merge_max_sig = float(extraction_cfg["cluster_cross_merge_max_sig"])
        histogram_attach_max_sig = float(extraction_cfg["cluster_hist_attach_max_sig"])

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
        print(
            "  Cluster visible-person estimate: "
            f"visible_people_est={visible_people_est}"
        )
        covisibility_conflict_rejections = 0
        histogram_attach_rejections = 0

        embeddings = {}  # track_id → 512D face/hist embedding
        fallback_ids = []  # track_ids where face encoding failed
        face_accept_count = 0
        face_reject_lowq_count = 0
        precomputed_identity_frames: set[int] = set()
        precomputed_feature_tracklets = 0

        if track_identity_features:
            for tid in unique_ids:
                feature = track_identity_features.get(tid)
                if not isinstance(feature, dict):
                    continue
                embedding = feature.get("embedding")
                if embedding is None:
                    continue
                try:
                    emb_arr = np.asarray(embedding, dtype=np.float32)
                except Exception:
                    continue
                if emb_arr.size == 0:
                    continue
                embeddings[tid] = emb_arr
                precomputed_feature_tracklets += 1
                source = str(feature.get("embedding_source", "none"))
                if source == "face":
                    face_accept_count += max(1, int(feature.get("embedding_count", 1)))
                else:
                    fallback_ids.append(tid)
                for observation in feature.get("face_observations", []):
                    frame_idx = int(observation.get("frame_idx", -1))
                    if frame_idx >= 0:
                        precomputed_identity_frames.add(frame_idx)

        sampled_by_tid, needed_frames = _build_cluster_sample_plan(tracklets, extraction_cfg)
        missing_sampled_by_tid = {
            tid: sampled_dets
            for tid, sampled_dets in sampled_by_tid.items()
            if tid not in embeddings
        }
        missing_needed_frames = {
            int(det.get("frame_idx", -1))
            for sampled_dets in missing_sampled_by_tid.values()
            for det in sampled_dets
            if int(det.get("frame_idx", -1)) >= 0
        }
        if not needed_frames and not embeddings:
            print("  No usable sampled frames for clustering")
            self._last_clustering_metrics = {
                "cluster_visible_people_estimate": visible_people_est,
                "overfragmentation_proxy": round(float(len(unique_ids) / max(1, visible_people_est)), 3),
                "accidental_merge_proxy": round(
                    float(max(0, visible_people_est - len(unique_ids)) / max(1, visible_people_est)),
                    3,
                ),
                "covisibility_conflict_rejections": 0,
                "histogram_attach_rejections": 0,
            }
            self._last_track_identity_features_after_clustering = (
                dict(track_identity_features) if isinstance(track_identity_features, dict) else None
            )
            return tracks

        if precomputed_feature_tracklets:
            print(
                "  Cluster identity features: "
                f"precomputed={precomputed_feature_tracklets}, "
                f"remaining={max(0, len(sampled_by_tid) - precomputed_feature_tracklets)}"
            )

        shard_workers = max(1, min(4, int(os.getenv("CLYPT_CLUSTER_SHARD_WORKERS", "4"))))
        min_shard_tracklets = max(8, int(os.getenv("CLYPT_CLUSTER_MIN_SHARD_TRACKLETS", "24")))
        can_shard_remotely = (
            shard_workers > 1
            and len(missing_sampled_by_tid) >= min_shard_tracklets
            and read_path.startswith("/vol/")
        )

        extraction_result = None
        if can_shard_remotely:
            shard_tid_order = sorted(missing_sampled_by_tid.keys())
            shard_count = min(shard_workers, len(shard_tid_order))
            shard_size = max(1, (len(shard_tid_order) + shard_count - 1) // shard_count)
            shards = []
            for shard_idx in range(shard_count):
                start = shard_idx * shard_size
                end = min(len(shard_tid_order), start + shard_size)
                shard_tids = shard_tid_order[start:end]
                if not shard_tids:
                    continue
                shards.append(
                    {
                        "shard_idx": shard_idx + 1,
                        "sampled_by_tid": {
                            tid: missing_sampled_by_tid[tid]
                            for tid in shard_tids
                            if tid in missing_sampled_by_tid
                        },
                    }
                )

            if len(shards) > 1:
                print(
                    "  Cluster embedding fan-out: "
                    f"{len(shards)} shards across up to {shard_workers} workers"
                )
                try:
                    cluster_worker = modal.Cls.from_name(app.name, "ClusterEmbeddingWorker")()
                    shard_handles = [
                        cluster_worker.extract_cluster_embeddings_shard.spawn(
                            video_path=read_path,
                            sampled_by_tid_subset=shard["sampled_by_tid"],
                            shard_idx=int(shard["shard_idx"]),
                            total_shards=len(shards),
                        )
                        for shard in shards
                    ]

                    merged_embeddings: dict[str, np.ndarray] = dict(embeddings)
                    merged_fallbacks: list[str] = list(fallback_ids)
                    sampled_frames_total = len(precomputed_identity_frames)
                    for idx, handle in enumerate(shard_handles, start=1):
                        shard_result = handle.get(timeout=None)
                        sampled_frames_total += int(shard_result.get("sampled_frames", 0))
                        face_accept_count += int(shard_result.get("face_accept_count", 0))
                        face_reject_lowq_count += int(shard_result.get("face_reject_lowq_count", 0))
                        merged_fallbacks.extend(str(tid) for tid in shard_result.get("fallback_ids", []))
                        for tid, vec in shard_result.get("embeddings", {}).items():
                            merged_embeddings[str(tid)] = np.asarray(vec, dtype=np.float32)
                        print(
                            "  Cluster embedding shard "
                            f"{idx}/{len(shard_handles)} collected: "
                            f"tracklets={int(shard_result.get('tracklets_processed', 0))}, "
                            f"accepted={int(shard_result.get('face_accept_count', 0))}, "
                            f"fallback={len(shard_result.get('fallback_ids', []))}, "
                            f"elapsed={float(shard_result.get('elapsed_s', 0.0)):.1f}s"
                        )

                    extraction_result = {
                        "embeddings": merged_embeddings,
                        "fallback_ids": merged_fallbacks,
                        "face_accept_count": face_accept_count,
                        "face_reject_lowq_count": face_reject_lowq_count,
                        "sampled_frames": sampled_frames_total,
                    }
                except Exception as e:
                    print(
                        "  Warning: cluster embedding fan-out failed; "
                        f"falling back to local extraction ({type(e).__name__}: {e})"
                    )

        if extraction_result is None:
            print(
                "  Cluster embedding mode: local "
                f"({len(missing_sampled_by_tid)} tracklets, {len(missing_needed_frames)} sampled frames)"
            )
            if missing_sampled_by_tid:
                extraction_result = _extract_cluster_embeddings_subset(
                    face_analyzer=self.face_analyzer,
                    read_path=read_path,
                    sampled_by_tid_subset=missing_sampled_by_tid,
                    config=extraction_cfg,
                    log_prefix="  [cluster-local] ",
                )
                extraction_result["embeddings"] = {
                    **embeddings,
                    **extraction_result.get("embeddings", {}),
                }
                extraction_result["fallback_ids"] = list(fallback_ids) + [
                    str(tid) for tid in extraction_result.get("fallback_ids", [])
                ]
                extraction_result["face_accept_count"] = int(extraction_result.get("face_accept_count", 0)) + face_accept_count
                extraction_result["face_reject_lowq_count"] = int(
                    extraction_result.get("face_reject_lowq_count", 0)
                ) + face_reject_lowq_count
                extraction_result["sampled_frames"] = int(extraction_result.get("sampled_frames", 0)) + len(
                    precomputed_identity_frames
                )
            else:
                extraction_result = {
                    "embeddings": dict(embeddings),
                    "fallback_ids": list(fallback_ids),
                    "face_accept_count": face_accept_count,
                    "face_reject_lowq_count": face_reject_lowq_count,
                    "sampled_frames": len(precomputed_identity_frames),
                    "tracklets_processed": precomputed_feature_tracklets,
                }

        embeddings = dict(extraction_result.get("embeddings", {}))
        fallback_ids = [str(tid) for tid in extraction_result.get("fallback_ids", [])]
        face_accept_count = int(extraction_result.get("face_accept_count", face_accept_count))
        face_reject_lowq_count = int(
            extraction_result.get("face_reject_lowq_count", face_reject_lowq_count)
        )

        if not embeddings:
            print("  No embeddings extracted, skipping clustering")
            self._last_clustering_metrics = {
                "cluster_visible_people_estimate": visible_people_est,
                "overfragmentation_proxy": round(float(len(unique_ids) / max(1, visible_people_est)), 3),
                "accidental_merge_proxy": round(
                    float(max(0, visible_people_est - len(unique_ids)) / max(1, visible_people_est)),
                    3,
                ),
                "covisibility_conflict_rejections": 0,
                "histogram_attach_rejections": 0,
            }
            self._last_track_identity_features_after_clustering = (
                dict(track_identity_features) if isinstance(track_identity_features, dict) else None
            )
            return tracks

        # Separate face embeddings from histogram fallbacks. We do NOT cluster both
        # together because they live in different feature spaces.
        tid_order_all = sorted(embeddings.keys())
        face_tids = [tid for tid in tid_order_all if tid not in fallback_ids]
        hist_tids = [tid for tid in tid_order_all if tid in fallback_ids]
        print(
            "  Face quality gate: "
            f"accepted={face_accept_count}, rejected_lowq={face_reject_lowq_count}, "
            f"min_det_score={float(extraction_cfg['cluster_face_min_det_score']):.2f}"
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
                        tids_a = [tid_order_face[i] for i in cluster_map[a]]
                        tids_b = [tid_order_face[i] for i in cluster_map[b]]
                        if self._clusters_conflict_by_visibility(tracklets, tids_a, tids_b):
                            covisibility_conflict_rejections += 1
                            continue
                        if not self._clusters_have_compatible_seat_signature(tracklets, tids_a, tids_b):
                            continue
                        cos_dist = _cosine_dist(cluster_centroids[a], cluster_centroids[b])
                        if cos_dist > cross_cluster_merge_base_cos:
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
                    f"{merged_cross} (cos_thresh={cross_cluster_merge_base_cos:.2f})"
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
                next_hist_label = n_face_clusters
                for tid in hist_tids:
                    sig = _track_signature(tid)
                    candidates = []
                    for lbl, centroid in cluster_sig_centroids.items():
                        cluster_tids = [cluster_tid for cluster_tid, cluster_lbl in face_label_by_tid.items() if cluster_lbl == lbl]
                        if self._clusters_conflict_by_visibility(tracklets, [tid], cluster_tids):
                            covisibility_conflict_rejections += 1
                            continue
                        sig_dist = _sig_dist(sig, centroid)
                        if sig_dist > histogram_attach_max_sig:
                            histogram_attach_rejections += 1
                            continue
                        if not self._clusters_have_compatible_seat_signature(
                            tracklets,
                            [tid],
                            cluster_tids,
                            max_signature_distance=histogram_attach_max_sig,
                        ):
                            histogram_attach_rejections += 1
                            continue
                        candidates.append((sig_dist, lbl))
                    if not candidates:
                        id_map[tid] = f"Global_Person_{int(next_hist_label)}"
                        next_hist_label += 1
                        continue
                    _, best_label = min(candidates, key=lambda kv: kv[0])
                    id_map[tid] = f"Global_Person_{int(best_label)}"
                    reassigned_hist += 1
                if reassigned_hist:
                    print(
                        "  Histogram tracklets attached to face clusters: "
                        f"{reassigned_hist} (max_sig={histogram_attach_max_sig:.2f})"
                    )
            else:
                # Worst-case fallback: keep them deterministic and separate.
                for i, tid in enumerate(sorted(hist_tids)):
                    id_map[tid] = f"Global_Person_{i}"

        label_by_tid = {
            tid: int(str(mapped_id).split("_")[-1])
            for tid, mapped_id in id_map.items()
            if str(mapped_id).startswith("Global_Person_")
        }
        label_by_tid, repair_metrics = self._repair_covisible_cluster_merges(tracklets, label_by_tid)
        id_map = {tid: f"Global_Person_{int(label)}" for tid, label in label_by_tid.items()}

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
        self._last_clustering_metrics = {
            "cluster_visible_people_estimate": visible_people_est,
            "overfragmentation_proxy": round(float(n_clusters / max(1, visible_people_est)), 3),
            "accidental_merge_proxy": round(
                float(max(0, visible_people_est - n_clusters) / max(1, visible_people_est)),
                3,
            ),
            "covisibility_conflict_rejections": covisibility_conflict_rejections,
            "histogram_attach_rejections": histogram_attach_rejections,
            "cluster_cross_merge_cos_threshold": cross_cluster_merge_base_cos,
            "cluster_hist_attach_max_sig": histogram_attach_max_sig,
            **repair_metrics,
        }
        if isinstance(track_identity_features, dict):
            remapped_feature_maps: list[dict[str, dict]] = []
            for old_tid, feature in track_identity_features.items():
                mapped_tid = id_map.get(old_tid, old_tid)
                remapped_feature_maps.append({mapped_tid: feature})
            self._last_track_identity_features_after_clustering = self._merge_track_identity_feature_sets(
                remapped_feature_maps
            )
        else:
            self._last_track_identity_features_after_clustering = None

        # Apply mapping to all tracks
        for t in tracks:
            old_id = t["track_id"]
            if old_id in id_map:
                t["track_id"] = id_map[old_id]

        return tracks

    def _run_lrasd_binding(
        self,
        video_path: str,
        audio_wav_path: str,
        tracks: list[dict],
        words: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
        track_identity_features: dict[str, dict] | None = None,
        analysis_context: dict | None = None,
        force_dense: bool = False,
    ) -> list[dict] | None:
        """Run LR-ASD inference and map words to visual track IDs."""
        import concurrent.futures as cf
        import cv2
        import os
        import numpy as np
        import torch
        from decord import VideoReader, cpu
        from bisect import bisect_left
        from collections import Counter

        if self.lrasd_model is None or self.lrasd_loss_av is None:
            print("  LR-ASD unavailable; falling back to heuristic binder.")
            return None
        if not words or not tracks:
            return []

        analysis_context = analysis_context or self._prepare_analysis_video(video_path)
        binding_video_path = str(analysis_context.get("analysis_video_path", video_path))
        binding_scale_x = float(analysis_context.get("scale_x", 1.0) or 1.0)
        binding_scale_y = float(analysis_context.get("scale_y", 1.0) or 1.0)
        binding_meta = dict(analysis_context.get("analysis_meta", self._probe_video_meta(binding_video_path)))

        try:
            vr = VideoReader(binding_video_path, ctx=cpu(0))
        except Exception as e:
            print(
                "  Warning: could not open video for LR-ASD binding "
                f"({type(e).__name__}: {e})"
            )
            return None

        fps = float(vr.get_avg_fps() or 0.0)
        if fps <= 0.0:
            fps = 25.0

        if frame_to_dets is None or track_to_dets is None:
            frame_to_dets, track_to_dets = self._build_track_indexes(tracks)

        canonical_face_boxes = self._build_canonical_face_bbox_lookup(
            track_identity_features=track_identity_features,
            track_to_dets=track_to_dets,
            frame_width=int(binding_meta.get("width", 0) or 0) or 1,
            frame_height=int(binding_meta.get("height", 0) or 0) or 1,
            max_interp_gap=max(6, int(round(fps * 0.4))),
        )
        needed_face_boxes = sum(
            1
            for dets in track_to_dets.values()
            for det in dets
            if int(det.get("frame_idx", -1)) >= 0
        )
        canonical_coverage = (
            float(len(canonical_face_boxes) / max(1, needed_face_boxes))
            if needed_face_boxes
            else 0.0
        )
        print(
            "  Canonical face stream: "
            f"{len(canonical_face_boxes)}/{needed_face_boxes}={canonical_coverage:.1%} coverage"
        )
        audio_feature_cache_path = binding_video_path.replace(".mp4", "_lrasd_features.npz")
        full_audio_features = self._load_or_build_lrasd_audio_features(
            audio_wav_path=audio_wav_path,
            cache_path=audio_feature_cache_path,
            fps=fps,
        )

        frame_cache: dict[int, object] = {}
        total_frames = len(vr)

        def _get_frame(frame_idx: int):
            if frame_idx in frame_cache:
                return frame_cache[frame_idx]
            if frame_idx < 0 or frame_idx >= total_frames:
                return None

            # Pre-fetch a short forward window in one batched decord read.
            fetch_end = min(frame_idx + 16, total_frames)
            fetch_indices = list(range(frame_idx, fetch_end))
            missing_indices = [fi for fi in fetch_indices if fi not in frame_cache]
            if missing_indices:
                try:
                    batch = vr.get_batch(missing_indices).asnumpy()  # RGB
                    for idx, frame_data in zip(missing_indices, batch):
                        frame_cache[idx] = frame_data
                except Exception:
                    frame_cache[frame_idx] = None

            # Evict old frames to keep memory bounded.
            if len(frame_cache) > 192:
                stale_keys = [k for k in frame_cache.keys() if k < frame_idx - 32]
                for k in stale_keys:
                    frame_cache.pop(k, None)
                # Fallback trim if stale-eviction was insufficient.
                if len(frame_cache) > 192:
                    for k in sorted(frame_cache.keys())[: max(0, len(frame_cache) - 192)]:
                        frame_cache.pop(k, None)

            return frame_cache.get(frame_idx)

        face_cache: dict[tuple[str, int], tuple[object, object]] = {}

        def _face_crop(tid: str, fi: int, det: dict):
            key = (tid, fi)
            if key in face_cache:
                return face_cache[key]
            frame = _get_frame(fi)
            if frame is None:
                face_cache[key] = (None, None)
                return face_cache[key]

            def _cache_and_return(crop_val, anchor_val):
                face_cache[key] = (crop_val, anchor_val)
                if len(face_cache) > 768:
                    for k in list(face_cache.keys())[:192]:
                        face_cache.pop(k, None)
                return face_cache[key]

            crop = None
            anchor = None
            fh, fw = frame.shape[:2]
            cx = float(det.get("x_center", 0.0))
            cy = float(det.get("y_center", 0.0))
            bw = float(det.get("width", 0.0))
            bh = float(det.get("height", 0.0))
            pb = canonical_face_boxes.get((tid, fi))
            if pb is not None:
                x1, y1, x2, y2, _, _ = pb
                x1 = max(0, min(fw - 1, int(x1)))
                y1 = max(0, min(fh - 1, int(y1)))
                x2 = max(x1 + 1, min(fw, int(x2)))
                y2 = max(y1 + 1, min(fh, int(y2)))
                face_crop = frame[y1:y2, x1:x2]
                if face_crop is not None and face_crop.size > 0:
                    crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                    if bw > 1e-6 and bh > 1e-6:
                        anchor = {
                            "x_offset": (float(x1) - cx) / bw,
                            "y_offset": (float(y1) - cy) / bh,
                            "w_ratio": float(x2 - x1) / bw,
                            "h_ratio": float(y2 - y1) / bh,
                        }

            if crop is None:
                det_crop, det_anchor = self._detect_face_in_person_det(frame, det)
                if det_crop is not None:
                    crop = det_crop
                    anchor = det_anchor

            return _cache_and_return(crop, anchor)

        asd_scores: dict[tuple[str, int], float] = {}
        scored_track_ids = set()
        chunk_size = 120
        min_chunk_frames = 20
        # Moderate batching + optional CPU/GPU overlap on a single GPU.
        lrasd_batch_size = max(4, min(64, int(os.getenv("CLYPT_LRASD_BATCH_SIZE", "32"))))
        enable_lrasd_overlap = os.getenv("CLYPT_LRASD_PIPELINE_OVERLAP", "1") == "1"
        max_inflight = max(1, min(4, int(os.getenv("CLYPT_LRASD_MAX_INFLIGHT", "4"))))
        if not enable_lrasd_overlap:
            max_inflight = 1
        contiguous_frame_gap = 4
        interpolation_gap = 5
        word_match_max_gap = 4
        score_lookup_max_delta = 8
        min_lrasd_prob = 0.15
        min_lrasd_assign_ratio = 0.15
        print(
            "  LR-ASD pipeline: "
            f"batch_size={lrasd_batch_size}, overlap={'on' if enable_lrasd_overlap else 'off'}, "
            f"max_inflight={max_inflight}"
        )

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
        prepared_chunks = 0
        pending_by_t: dict[int, list[tuple[str, list[int], np.ndarray, np.ndarray]]] = {}
        flush_counter = 0
        face_hits = 0
        face_misses = 0
        inflight_futures: list[cf.Future] = []
        infer_executor = cf.ThreadPoolExecutor(max_workers=1) if enable_lrasd_overlap else None

        def _score_pending_batch(
            pending: list[tuple[str, list[int], np.ndarray, np.ndarray]],
            t_len: int,
            flush_id: int,
        ) -> list[tuple[str, list[int], np.ndarray]]:
            """Run one LR-ASD batch and return aligned per-frame scores."""
            visual_batch = np.stack([p[2] for p in pending], axis=0)
            audio_batch = np.stack([p[3] for p in pending], axis=0)
            visual_t = torch.from_numpy(visual_batch).float().to(self.gpu_device)
            audio_t = torch.from_numpy(audio_batch).float().to(self.gpu_device)
            if self.model_debug and (
                flush_id % max(1, self.model_debug_every // 2) == 1
            ):
                print("  [LR-ASD DEBUG] _flush_pending tensors:")
                print("   " + self._tensor_debug_stats("audio_t", audio_t))
                print("   " + self._tensor_debug_stats("visual_t", visual_t))
                print(f"   batch={len(pending)} t_len={t_len}")

            with torch.no_grad():
                score_bt = self._lrasd_forward_scores(
                    audio_t,
                    visual_t,
                )
            score_np = score_bt.detach().float().cpu().numpy()

            rows: list[tuple[str, list[int], np.ndarray]] = []
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
                rows.append((tid, valid_frames, row))
            return rows

        def _commit_scored_rows(rows: list[tuple[str, list[int], np.ndarray]]):
            nonlocal scored_chunks
            for tid, valid_frames, row in rows:
                for fi, sc in zip(valid_frames, row):
                    asd_scores[(tid, fi)] = float(sc)
                scored_track_ids.add(tid)
                scored_chunks += 1

        def _drain_one(block: bool = False):
            if not inflight_futures:
                return
            if block or len(inflight_futures) >= max_inflight:
                fut = inflight_futures.pop(0)
                rows = fut.result()
                _commit_scored_rows(rows)

        def _flush_pending(t_len: int):
            nonlocal flush_counter
            pending = pending_by_t.get(t_len, [])
            if not pending:
                return
            flush_counter += 1
            pending_by_t[t_len] = []
            if infer_executor is None:
                rows = _score_pending_batch(pending, t_len, flush_counter)
                _commit_scored_rows(rows)
                return
            _drain_one(block=False)
            inflight_futures.append(
                infer_executor.submit(_score_pending_batch, pending, t_len, flush_counter)
            )
            _drain_one(block=False)

        def _queue_subchunk(tid, frames, crops):
            nonlocal prepared_chunks
            t = len(frames)
            if t < min_chunk_frames:
                return

            visual_np = np.stack([cv2.cvtColor(c, cv2.COLOR_RGB2GRAY) for c in crops], axis=0)
            target_audio_frames = int(round(t * 4))
            start_audio_frame = max(0, int(round(frames[0] * 4)))
            end_audio_frame = start_audio_frame + target_audio_frames
            audio_np = np.asarray(full_audio_features[start_audio_frame:end_audio_frame], dtype=np.float32)
            if audio_np.ndim != 2 or (audio_np.size > 0 and audio_np.shape[1] != 13):
                return
            if audio_np.shape[0] < target_audio_frames:
                if audio_np.shape[0] == 0:
                    return
                shortage = target_audio_frames - audio_np.shape[0]
                audio_np = np.pad(audio_np, ((0, shortage), (0, 0)), mode="edge")

            pending_by_t.setdefault(t, []).append((tid, frames, visual_np, audio_np))
            prepared_chunks += 1
            if len(pending_by_t[t]) >= lrasd_batch_size:
                _flush_pending(t)

        try:
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
                                det = self._scale_detection_geometry(
                                    det,
                                    scale_x=binding_scale_x,
                                    scale_y=binding_scale_y,
                                )
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
                                "  LR-ASD progress: "
                                f"prepared={prepared_chunks}, "
                                f"inferred={scored_chunks}, "
                                f"track_windows={chunk_counter}/{max(total_chunks, 1)}, "
                                f"scored_tracks={len(scored_track_ids)}"
                            )

            for t_len in list(pending_by_t.keys()):
                _flush_pending(t_len)
            while inflight_futures:
                _drain_one(block=True)
        finally:
            if infer_executor is not None:
                infer_executor.shutdown(wait=True, cancel_futures=False)

        if not asd_scores:
            print("  LR-ASD produced no valid frame scores")
            return None
        if self.model_debug:
            total_face_events = face_hits + face_misses
            hit_rate = (face_hits / total_face_events) if total_face_events > 0 else 0.0
            print(
                "  [LR-ASD DEBUG] face crop stats: "
                f"hits={face_hits}, misses={face_misses}, hit_rate={hit_rate:.1%}, "
                f"flushes={flush_counter}, scored_chunks={scored_chunks}"
            )

        score_vals = np.asarray(list(asd_scores.values()), dtype=np.float32)
        print(
            "  LR-ASD score stats: "
            f"min={float(np.min(score_vals)):.3f}, "
            f"p10={float(np.percentile(score_vals, 10)):.3f}, "
            f"p50={float(np.percentile(score_vals, 50)):.3f}, "
            f"p90={float(np.percentile(score_vals, 90)):.3f}, "
            f"max={float(np.max(score_vals)):.3f}"
        )
        score_spread = float(np.percentile(score_vals, 90) - np.percentile(score_vals, 50))
        min_assignment_margin = max(0.004, 0.18 * score_spread)
        print(
            "  LR-ASD assignment gates: "
            f"min_prob={min_lrasd_prob:.3f}, margin={min_assignment_margin:.4f}"
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
                    best_prob >= min_lrasd_prob
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
            "  LR-ASD word matching: "
            f"with_frame={words_with_frame}/{len(words)}, "
            f"with_dets={words_with_dets}/{len(words)}, "
            f"with_scored_candidate={words_with_scored_candidate}/{len(words)}"
        )
        assigned_ratio = assigned / max(1, len(words))
        print(f"  LR-ASD assignment ratio: {assigned}/{len(words)}={assigned_ratio:.1%}")
        if assigned_ratio < min_lrasd_assign_ratio or len(scored_track_ids) < 2:
            print(
                "  LR-ASD confidence too low for final binding "
                f"(assigned_ratio={assigned_ratio:.1%}, scored_tracks={len(scored_track_ids)}); "
                "falling back to heuristic binder."
            )
            return None

        print(
            "LR-ASD binding complete: "
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
        high-level objective as neural AV-ASD: align speech timing with the most
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
        track_identity_features: dict[str, dict] | None = None,
        analysis_context: dict | None = None,
    ) -> list[dict]:
        """Bind words to track IDs using LR-ASD, with heuristic fallback."""
        mode = self._select_speaker_binding_mode(video_path, tracks, words)
        speaker_metrics = {
            "speaker_binding_selected_mode": mode,
            "speaker_binding_resolved_mode": ("lrasd" if mode == "shared_analysis_proxy" else mode),
            "speaker_binding_fallback_used": False,
        }
        if mode in {"lrasd", "shared_analysis_proxy"}:
            lrasd_started_at = time.perf_counter()
            bindings = self._run_lrasd_binding(
                video_path=video_path,
                audio_wav_path=audio_wav_path,
                tracks=tracks,
                words=words,
                frame_to_dets=frame_to_dets,
                track_to_dets=track_to_dets,
                track_identity_features=track_identity_features,
                analysis_context=analysis_context,
            )
            speaker_metrics["lrasd_wallclock_s"] = round(
                time.perf_counter() - lrasd_started_at,
                3,
            )
            if bindings is not None:
                self._last_speaker_binding_metrics = speaker_metrics
                return bindings

            speaker_metrics["speaker_binding_fallback_used"] = True
            speaker_metrics["speaker_binding_resolved_mode"] = "heuristic"
        print("Running fallback speaker binding heuristic...")
        bindings = self._run_speaker_binding_heuristic(
            video_path,
            tracks,
            words,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
        )
        self._last_speaker_binding_metrics = speaker_metrics
        return bindings

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
        track_identity_features = metrics.pop("track_identity_features", None)
        analysis_context = metrics.get("analysis_context") if isinstance(metrics.get("analysis_context"), dict) else None
        metrics["schema_pass_rate"] = self._tracking_contract_pass_rate(tracks)
        self._validate_tracking_contract(tracks)
        self._enforce_rollout_gates(metrics)
        metrics["track_identity_feature_track_count"] = len(track_identity_features or {})
        metrics["identity_track_count_before_clustering"] = len(
            {str(track.get("track_id", "")) for track in tracks if str(track.get("track_id", ""))}
        )

        _, track_to_dets = self._build_track_indexes(tracks)

        # Step 3: Global tracklet clustering
        print("[Phase 1] Step 3/4: Clustering tracklets into global IDs...")
        cluster_started_at = time.perf_counter()
        tracks = self._cluster_tracklets(
            video_path,
            tracks,
            track_to_dets=track_to_dets,
            track_identity_features=track_identity_features,
        )
        cluster_elapsed_s = time.perf_counter() - cluster_started_at
        metrics["cluster_tracklets_wallclock_s"] = round(cluster_elapsed_s, 3)
        print(f"[Phase 1] Step 3/4 complete in {cluster_elapsed_s:.2f}s")
        last_clustering_metrics = getattr(self, "_last_clustering_metrics", None)
        if isinstance(last_clustering_metrics, dict):
            metrics.update(last_clustering_metrics)
        clustered_track_identity_features = (
            getattr(self, "_last_track_identity_features_after_clustering", None) or track_identity_features
        )
        frame_to_dets, track_to_dets = self._build_track_indexes(tracks)
        metrics["identity_track_count_after_clustering"] = len(
            {str(track.get("track_id", "")) for track in tracks if str(track.get("track_id", ""))}
        )
        face_detections, person_detections, face_metrics = self._build_visual_detection_ledgers(
            video_path=video_path,
            tracks=tracks,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
            track_identity_features=clustered_track_identity_features,
        )
        metrics.update(face_metrics)

        # Step 4: Speaker binding
        print("[Phase 1] Step 4/4: Running speaker binding...")
        speaker_binding_started_at = time.perf_counter()
        speaker_bindings = self._run_speaker_binding(
            video_path,
            audio_path,
            tracks,
            words,
            frame_to_dets=frame_to_dets,
            track_to_dets=track_to_dets,
            track_identity_features=clustered_track_identity_features,
            analysis_context=analysis_context,
        )
        speaker_binding_elapsed_s = time.perf_counter() - speaker_binding_started_at
        metrics["speaker_binding_wallclock_s"] = round(speaker_binding_elapsed_s, 3)
        last_binding_metrics = getattr(self, "_last_speaker_binding_metrics", None)
        if isinstance(last_binding_metrics, dict):
            metrics.update(last_binding_metrics)
        assigned_words = sum(1 for word in words if word.get("speaker_track_id"))
        metrics["speaker_binding_assignment_ratio"] = round(
            float(assigned_words / max(1, len(words))),
            3,
        )
        print(f"[Phase 1] Step 4/4 complete in {speaker_binding_elapsed_s:.2f}s")
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
            "face_detections": face_detections,
            "person_detections": person_detections,
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
        audio_wav_bytes: bytes,
        youtube_url: str,
        words: list[dict],
        tracks: list[dict],
        tracking_metrics: dict | None = None,
        job_id: str | None = None,
        video_bytes: bytes | None = None,
    ) -> dict:
        """Finalize from externally-fanned-out ASR/tracking outputs.

        When *job_id* is provided the video is read directly from the shared
        volume (``/vol/clypt-chunks/jobs/{job_id}/``) avoiding a full
        video-bytes RPC transfer.  Falls back to *video_bytes* when the
        volume path is unavailable.
        """
        import os
        import json
        import uuid

        dl_dir = "/tmp/clypt"
        os.makedirs(dl_dir, exist_ok=True)
        suffix = uuid.uuid4().hex

        # --- Video: prefer staged volume path over RPC bytes ---
        video_path = None
        _video_from_volume = False
        if job_id:
            TRACKING_VOLUME.reload()
            manifest_path = f"/vol/clypt-chunks/jobs/{job_id}/manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                vol_video = manifest.get("video_path", "")
                if vol_video and os.path.exists(vol_video):
                    video_path = vol_video
                    _video_from_volume = True
                    print(f"[finalize] Reusing staged video from volume ({job_id[:8]}...)")

        if video_path is None:
            if video_bytes is None:
                raise RuntimeError(
                    "finalize_extraction requires either job_id (volume) or video_bytes"
                )
            video_path = f"{dl_dir}/video_finalize_{suffix}.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            print("[finalize] Wrote video from RPC bytes (no volume path available)")

        # --- Audio: always from bytes (small payload) ---
        audio_path = f"{dl_dir}/audio_finalize_{suffix}.wav"
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
            # Only clean up files we created locally, not volume-staged ones.
            h264_video_path = video_path.replace(".mp4", "_h264.mp4")
            cleanup = [audio_path]
            if not _video_from_volume:
                cleanup.extend([video_path, h264_video_path])
            for p in cleanup:
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
            # Keep NeMo CUDA-graph decoding enabled by ensuring tracking does not
            # launch GPU work until ASR has fully completed.
            print("[Phase 1] Step 1+2/4: Running Parakeet ASR, then YOLO26 tracking, on the same GPU...")
            words = self._run_asr(audio_path)
            tracks, tracking_metrics = self._run_tracking(video_path)
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
            analysis_context = (
                tracking_metrics.get("analysis_context")
                if isinstance(tracking_metrics, dict)
                else None
            )
            analysis_video_path = None
            prepared_video_path = None
            if isinstance(analysis_context, dict):
                analysis_video_path = analysis_context.get("analysis_video_path")
                prepared_video_path = analysis_context.get("prepared_video_path")
            cleanup_paths = {
                video_path,
                audio_path,
                h264_video_path,
                str(prepared_video_path) if prepared_video_path else None,
                str(analysis_video_path) if analysis_video_path else None,
                (str(analysis_video_path).replace(".mp4", "_lrasd_features.npz") if analysis_video_path else None),
            }
            for p in cleanup_paths:
                if not p:
                    continue
                if os.path.exists(p):
                    os.remove(p)

            return result

        except Exception as e:
            print(f"[Phase 1] Error: {e}")
            return {"status": "error", "message": str(e)}
