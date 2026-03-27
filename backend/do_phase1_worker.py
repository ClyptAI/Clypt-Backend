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
    if os.getenv("CLYPT_ENABLE_LEGACY_SERVERLESS_SDK", "0") != "1":
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
    def _detection_iou(det_a: dict, det_b: dict) -> float:
        ax1 = float(det_a.get("x_center", 0.0)) - 0.5 * float(det_a.get("width", 0.0))
        ay1 = float(det_a.get("y_center", 0.0)) - 0.5 * float(det_a.get("height", 0.0))
        ax2 = ax1 + float(det_a.get("width", 0.0))
        ay2 = ay1 + float(det_a.get("height", 0.0))
        bx1 = float(det_b.get("x_center", 0.0)) - 0.5 * float(det_b.get("width", 0.0))
        by1 = float(det_b.get("y_center", 0.0)) - 0.5 * float(det_b.get("height", 0.0))
        bx2 = bx1 + float(det_b.get("width", 0.0))
        by2 = by1 + float(det_b.get("height", 0.0))
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = max(1.0, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return float(inter / union)

    @classmethod
    def _visibility_conflict_stats(
        cls,
        tracklets: dict[str, list[dict]],
        left_track_ids: list[str],
        right_track_ids: list[str],
    ) -> dict[str, float | bool | int]:
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

        overlap_frames = sorted(set(left_by_frame.keys()) & set(right_by_frame.keys()))
        conflict_frames = 0
        severe_conflict = False
        duplicate_like_frames = 0
        for frame_idx in overlap_frames:
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
            iou = cls._detection_iou(left, right)

            duplicate_like = (
                iou >= 0.14
                and dx <= (0.65 * avg_width)
                and dy <= (0.35 * avg_height)
            )
            if duplicate_like:
                duplicate_like_frames += 1
                continue

            separated = dx > (0.65 * avg_width) or dy > (0.45 * avg_height)
            if not separated or iou > 0.2:
                continue
            conflict_frames += 1
            if dx > (1.25 * avg_width) or dy > (0.9 * avg_height):
                severe_conflict = True

        duplicate_like = bool(overlap_frames) and duplicate_like_frames >= max(1, min(2, len(overlap_frames)))
        return {
            "overlap_frames": len(overlap_frames),
            "conflict_frames": conflict_frames,
            "severe_conflict": severe_conflict,
            "duplicate_like": duplicate_like,
        }

    @classmethod
    def _clusters_conflict_by_visibility(
        cls,
        tracklets: dict[str, list[dict]],
        left_track_ids: list[str],
        right_track_ids: list[str],
    ) -> bool:
        """Return True when two candidate identities are clearly co-visible as distinct people."""
        disable_covisibility = os.getenv("CLYPT_CLUSTER_DISABLE_COVISIBILITY", "").strip().lower()
        if disable_covisibility in {"1", "true", "yes", "on"}:
            return False
        stats = cls._visibility_conflict_stats(tracklets, left_track_ids, right_track_ids)
        if bool(stats.get("duplicate_like")):
            return False
        return int(stats.get("conflict_frames", 0)) >= 2 or bool(stats.get("severe_conflict"))

    @classmethod
    def _same_identity_frame_collision_metrics(
        cls,
        tracklets: dict[str, list[dict]],
        label_by_tid: dict[str, int],
    ) -> dict[str, int]:
        from collections import defaultdict

        grouped: dict[int, list[str]] = defaultdict(list)
        for tid, label in dict(label_by_tid or {}).items():
            grouped[int(label)].append(str(tid))

        collision_pairs = 0
        collision_frames: set[int] = set()
        labels_with_collisions = 0

        for tids in grouped.values():
            group_pair_count = 0
            for idx, left_tid in enumerate(sorted(tids)):
                for right_tid in sorted(tids)[idx + 1 :]:
                    stats = cls._visibility_conflict_stats(tracklets, [left_tid], [right_tid])
                    if bool(stats.get("duplicate_like")):
                        continue
                    if int(stats.get("conflict_frames", 0)) <= 0 and not bool(stats.get("severe_conflict")):
                        continue
                    group_pair_count += 1
                    collision_pairs += 1
                    left_frames = {
                        int(det.get("frame_idx", -1))
                        for det in tracklets.get(left_tid, [])
                        if int(det.get("frame_idx", -1)) >= 0
                    }
                    right_frames = {
                        int(det.get("frame_idx", -1))
                        for det in tracklets.get(right_tid, [])
                        if int(det.get("frame_idx", -1)) >= 0
                    }
                    collision_frames.update(left_frames & right_frames)
            if group_pair_count:
                labels_with_collisions += 1

        return {
            "same_identity_frame_collision_pairs": int(collision_pairs),
            "same_identity_frame_collision_frames": int(len(collision_frames)),
            "same_identity_labels_with_collisions": int(labels_with_collisions),
        }

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

    @staticmethod
    def _eligible_associated_track_ids(associated_counts: dict[str, int]) -> set[str]:
        import math

        filtered_counts = {
            str(tid): int(count)
            for tid, count in dict(associated_counts or {}).items()
            if str(tid) and int(count) > 0
        }
        if not filtered_counts:
            return set()

        dominant_track_id, dominant_count = max(
            filtered_counts.items(),
            key=lambda item: item[1],
        )
        total_count = max(1, sum(filtered_counts.values()))
        min_count = max(2, int(os.getenv("CLYPT_FACE_TRACK_MIN_ASSOC_COUNT", "2")))
        min_share = float(os.getenv("CLYPT_FACE_TRACK_MIN_ASSOC_SHARE", "0.20"))
        dominant_ratio = float(os.getenv("CLYPT_FACE_TRACK_MIN_DOMINANT_RATIO", "0.50"))
        min_count_from_dominant = max(1, int(math.ceil(float(dominant_count) * dominant_ratio)))
        eligible_track_ids = {
            tid
            for tid, count in filtered_counts.items()
            if count >= min_count
            and (count / total_count) >= min_share
            and count >= min_count_from_dominant
        }
        if not eligible_track_ids:
            return set()
        if os.getenv("CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return eligible_track_ids
        if dominant_track_id in eligible_track_ids:
            return {str(dominant_track_id)}
        return {str(max(eligible_track_ids, key=lambda tid: filtered_counts.get(tid, 0)))}

    @staticmethod
    def _face_observation_signature(observations: list[dict]):
        import numpy as np

        vectors = []
        for observation in observations or []:
            bbox = observation.get("bounding_box", {}) or {}
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            width = max(1.0, float(bbox.get("width", 1.0)))
            height = max(1.0, float(bbox.get("height", 1.0)))
            vectors.append(
                np.asarray(
                    [x + (0.5 * width), y + (0.5 * height), width, height],
                    dtype=np.float32,
                )
            )
        if not vectors:
            return np.zeros(4, dtype=np.float32)
        return np.median(np.stack(vectors, axis=0), axis=0).astype(np.float32)

    def _cluster_seed_track_ids_for_face_track(
        self,
        associated_counts: dict[str, int],
        tracklets: dict[str, list[dict]] | None = None,
    ) -> set[str]:
        filtered_counts = {
            str(tid): int(count)
            for tid, count in dict(associated_counts or {}).items()
            if str(tid) and int(count) > 0
        }
        if not filtered_counts:
            return set()

        dominant_tid, dominant_count = max(filtered_counts.items(), key=lambda item: item[1])
        selected: set[str] = set()
        primary_candidates = self._eligible_associated_track_ids(filtered_counts)
        if dominant_tid in primary_candidates:
            selected.add(str(dominant_tid))
        elif primary_candidates:
            selected.add(str(max(primary_candidates, key=lambda tid: filtered_counts.get(tid, 0))))
        else:
            return set()

        if not tracklets:
            return selected
        if os.getenv("CLYPT_FACE_TRACK_ALLOW_MULTI_ASSOC", "0").strip().lower() in {"1", "true", "yes", "on"}:
            return set(primary_candidates)

        secondary_min_ratio = float(os.getenv("CLYPT_FACE_TRACK_SECONDARY_MIN_RATIO", "0.60"))
        secondary_max_sig = float(os.getenv("CLYPT_FACE_TRACK_SECONDARY_MAX_SIG", "0.18"))
        for tid, count in sorted(filtered_counts.items(), key=lambda item: (-item[1], item[0])):
            tid = str(tid)
            if tid in selected:
                continue
            if dominant_count <= 0 or (count / float(dominant_count)) < secondary_min_ratio:
                continue
            if not self._clusters_have_compatible_seat_signature(
                tracklets,
                [tid],
                list(selected),
                max_signature_distance=secondary_max_sig,
            ):
                continue
            selected.add(tid)
        return selected

    @staticmethod
    def _track_boundary_gap_frames_for_tracklets(
        tracklets: dict[str, list[dict]],
        tid_a: str,
        tid_b: str,
    ) -> int:
        dets_a = tracklets.get(str(tid_a), [])
        dets_b = tracklets.get(str(tid_b), [])
        if not dets_a or not dets_b:
            return 1_000_000
        a_start = min(int(d.get("frame_idx", -1)) for d in dets_a)
        a_end = max(int(d.get("frame_idx", -1)) for d in dets_a)
        b_start = min(int(d.get("frame_idx", -1)) for d in dets_b)
        b_end = max(int(d.get("frame_idx", -1)) for d in dets_b)
        if a_end < b_start:
            return max(0, b_start - a_end)
        if b_end < a_start:
            return max(0, a_start - b_end)
        return 0

    def _choose_signature_attachment_label(
        self,
        *,
        tid: str,
        tracklets: dict[str, list[dict]],
        face_label_by_tid: dict[str, int],
        histogram_attach_max_sig: float,
    ) -> int | None:
        import os

        max_gap_frames = max(0, int(os.getenv("CLYPT_CLUSTER_ATTACH_MAX_GAP_FRAMES", "180")))
        score_gap_weight = float(os.getenv("CLYPT_CLUSTER_ATTACH_GAP_WEIGHT", "0.35"))
        ambiguity_margin = float(os.getenv("CLYPT_CLUSTER_ATTACH_AMBIGUITY_MARGIN", "0.05"))
        tid_sig = self._tracklet_signature(tracklets, [tid])

        candidates: list[tuple[float, float, int, int]] = []
        labels = sorted(set(int(lbl) for lbl in face_label_by_tid.values()))
        for label in labels:
            cluster_tids = [cluster_tid for cluster_tid, cluster_lbl in face_label_by_tid.items() if int(cluster_lbl) == label]
            if not cluster_tids:
                continue

            best_member = None
            for cluster_tid in cluster_tids:
                if not self._clusters_have_compatible_seat_signature(
                    tracklets,
                    [tid],
                    [cluster_tid],
                    max_signature_distance=histogram_attach_max_sig,
                ):
                    continue
                member_sig = self._tracklet_signature(tracklets, [cluster_tid])
                sig_dist = self._tracklet_signature_distance(tid_sig, member_sig)
                if sig_dist > histogram_attach_max_sig:
                    continue
                gap_frames = self._track_boundary_gap_frames_for_tracklets(tracklets, tid, cluster_tid)
                if gap_frames > max_gap_frames:
                    continue
                score = sig_dist + (
                    score_gap_weight * (gap_frames / max(1.0, float(max_gap_frames or 1)))
                )
                candidate = (score, sig_dist, gap_frames, int(label))
                if best_member is None or candidate < best_member:
                    best_member = candidate
            if best_member is not None:
                candidates.append(best_member)

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        best_score, _, _, best_label = candidates[0]
        if len(candidates) > 1 and (candidates[1][0] - best_score) < ambiguity_margin:
            return None
        return int(best_label)

    def _choose_signature_attachment_label_for_group(
        self,
        *,
        tids: list[str],
        tracklets: dict[str, list[dict]],
        face_label_by_tid: dict[str, int],
        histogram_attach_max_sig: float,
    ) -> int | None:
        import math
        import os

        if not tids:
            return None

        max_gap_frames = max(0, int(os.getenv("CLYPT_CLUSTER_ATTACH_MAX_GAP_FRAMES", "180")))
        score_gap_weight = float(os.getenv("CLYPT_CLUSTER_ATTACH_GAP_WEIGHT", "0.35"))
        ambiguity_margin = float(os.getenv("CLYPT_CLUSTER_ATTACH_AMBIGUITY_MARGIN", "0.05"))
        group_relax = float(os.getenv("CLYPT_CLUSTER_GROUP_ATTACH_SIG_RELAX", "1.25"))
        min_support_share = float(os.getenv("CLYPT_CLUSTER_GROUP_ATTACH_MIN_SUPPORT_SHARE", "0.5"))
        min_support_count = max(1, int(os.getenv("CLYPT_CLUSTER_GROUP_ATTACH_MIN_SUPPORT_COUNT", "1")))
        group_max_sig = histogram_attach_max_sig * max(1.0, group_relax)
        group_sig = self._tracklet_signature(tracklets, tids)

        candidates: list[tuple[float, float, int, int, int]] = []
        labels = sorted(set(int(lbl) for lbl in face_label_by_tid.values()))
        for label in labels:
            cluster_tids = [
                cluster_tid
                for cluster_tid, cluster_lbl in face_label_by_tid.items()
                if int(cluster_lbl) == label
            ]
            if not cluster_tids:
                continue
            if not self._clusters_have_compatible_seat_signature(
                tracklets,
                tids,
                cluster_tids,
                max_signature_distance=group_max_sig,
            ):
                continue

            cluster_sig = self._tracklet_signature(tracklets, cluster_tids)
            sig_dist = self._tracklet_signature_distance(group_sig, cluster_sig)
            if sig_dist > group_max_sig:
                continue

            compatible_members = 0
            min_gap_frames = 1_000_000
            for tid in tids:
                member_best_gap = None
                for cluster_tid in cluster_tids:
                    if not self._clusters_have_compatible_seat_signature(
                        tracklets,
                        [tid],
                        [cluster_tid],
                        max_signature_distance=group_max_sig,
                    ):
                        continue
                    gap_frames = self._track_boundary_gap_frames_for_tracklets(tracklets, tid, cluster_tid)
                    if gap_frames > max_gap_frames:
                        continue
                    member_best_gap = gap_frames if member_best_gap is None else min(member_best_gap, gap_frames)
                if member_best_gap is not None:
                    compatible_members += 1
                    min_gap_frames = min(min_gap_frames, int(member_best_gap))

            if compatible_members <= 0:
                continue
            required_members = max(
                min_support_count,
                int(math.ceil(float(len(tids)) * min_support_share)),
            )
            if compatible_members < required_members:
                continue
            if min_gap_frames > max_gap_frames:
                continue

            score = sig_dist + (
                score_gap_weight * (min_gap_frames / max(1.0, float(max_gap_frames or 1)))
            ) - (0.03 * compatible_members)
            candidates.append((score, sig_dist, min_gap_frames, -compatible_members, int(label)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        best_score, _, _, _, best_label = candidates[0]
        if len(candidates) > 1 and (candidates[1][0] - best_score) < ambiguity_margin:
            return None
        return int(best_label)

    def _cluster_signature_only_tracklets(
        self,
        *,
        track_ids: list[str],
        tracklets: dict[str, list[dict]],
        base_max_sig: float,
    ) -> list[list[str]]:
        import os

        long_gap_frames = max(0, int(os.getenv("CLYPT_CLUSTER_HIST_LONG_GAP_FRAMES", "180")))
        long_gap_max_sig = float(os.getenv("CLYPT_CLUSTER_HIST_LONG_GAP_MAX_SIG", "0.12"))

        ordered_track_ids = sorted(str(tid) for tid in track_ids if str(tid))
        parent = {tid: tid for tid in ordered_track_ids}

        def _find(tid: str) -> str:
            root = parent[tid]
            while root != parent[root]:
                root = parent[root]
            while tid != root:
                nxt = parent[tid]
                parent[tid] = root
                tid = nxt
            return root

        def _union(left_tid: str, right_tid: str) -> None:
            left_root = _find(left_tid)
            right_root = _find(right_tid)
            if left_root == right_root:
                return
            keep_root, drop_root = sorted([left_root, right_root])
            parent[drop_root] = keep_root

        for idx, left_tid in enumerate(ordered_track_ids):
            left_sig = self._tracklet_signature(tracklets, [left_tid])
            if left_sig is None:
                continue
            for right_tid in ordered_track_ids[idx + 1 :]:
                right_sig = self._tracklet_signature(tracklets, [right_tid])
                if right_sig is None:
                    continue
                gap_frames = self._track_boundary_gap_frames_for_tracklets(tracklets, left_tid, right_tid)
                max_sig = base_max_sig if gap_frames <= long_gap_frames else long_gap_max_sig
                sig_dist = self._tracklet_signature_distance(left_sig, right_sig)
                if sig_dist > max_sig:
                    continue
                _union(left_tid, right_tid)

        groups: dict[str, list[str]] = {}
        for tid in ordered_track_ids:
            groups.setdefault(_find(tid), []).append(tid)
        return [sorted(group) for _, group in sorted(groups.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)]

    def _repair_covisible_cluster_merges(
        self,
        tracklets: dict[str, list[dict]],
        label_by_tid: dict[str, int],
        *,
        anchored_tids: set[str] | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        from collections import defaultdict

        anchored_tids = {str(tid) for tid in (anchored_tids or set()) if str(tid)}
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
                    if not self._clusters_conflict_by_visibility(tracklets, [left_tid], [right_tid]):
                        continue
                    if anchored_tids and left_tid not in anchored_tids and right_tid not in anchored_tids:
                        continue
                    conflict_pairs.add((left_tid, right_tid))
                    conflict_pairs.add((right_tid, left_tid))

            if not conflict_pairs:
                continue

            buckets: list[list[str]] = []

            anchor_tids = [tid for tid in tids if tid in anchored_tids]
            non_anchor_tids = [tid for tid in tids if tid not in anchored_tids]

            def _best_bucket_for_tid(
                tid: str,
                bucket_candidates: list[list[str]],
                *,
                max_signature_distance: float,
                anchor_signature_override: float | None = None,
            ) -> int | None:
                tid_sig = self._tracklet_signature(tracklets, [tid])
                compatible_buckets: list[tuple[float, int]] = []
                for bucket_idx, bucket in enumerate(bucket_candidates):
                    has_anchor_member = any(other_tid in anchored_tids for other_tid in bucket)
                    if any((tid, other_tid) in conflict_pairs for other_tid in bucket):
                        if not has_anchor_member or anchor_signature_override is None:
                            continue
                        if not self._clusters_have_compatible_seat_signature(
                            tracklets,
                            bucket,
                            [tid],
                            max_signature_distance=anchor_signature_override,
                        ):
                            continue
                    if not self._clusters_have_compatible_seat_signature(
                        tracklets,
                        bucket,
                        [tid],
                        max_signature_distance=max_signature_distance,
                    ):
                        continue
                    bucket_sig = self._tracklet_signature(tracklets, bucket)
                    sig_dist = self._tracklet_signature_distance(bucket_sig, tid_sig)
                    compatible_buckets.append((sig_dist, bucket_idx))

                if compatible_buckets:
                    _, chosen_bucket_idx = min(compatible_buckets, key=lambda item: (item[0], item[1]))
                    return int(chosen_bucket_idx)
                return None

            if anchor_tids:
                for tid in anchor_tids:
                    chosen_bucket_idx = _best_bucket_for_tid(
                        tid,
                        buckets,
                        max_signature_distance=0.45,
                        anchor_signature_override=0.30,
                    )
                    if chosen_bucket_idx is None:
                        buckets.append([tid])
                    else:
                        buckets[chosen_bucket_idx].append(tid)
                for tid in non_anchor_tids:
                    chosen_bucket_idx = _best_bucket_for_tid(
                        tid,
                        buckets,
                        max_signature_distance=1.8,
                        anchor_signature_override=0.32,
                    )
                    if chosen_bucket_idx is None:
                        buckets.append([tid])
                    else:
                        buckets[chosen_bucket_idx].append(tid)
            else:
                for tid in tids:
                    chosen_bucket_idx = _best_bucket_for_tid(
                        tid,
                        buckets,
                        max_signature_distance=2.4,
                    )
                    if chosen_bucket_idx is None:
                        buckets.append([tid])
                    else:
                        buckets[chosen_bucket_idx].append(tid)

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
    def _should_skip_cluster_repair(
        *,
        face_cluster_count: int,
        clusters_before_repair: int,
        visible_people_est: int,
        anchored_track_count: int,
    ) -> bool:
        face_cluster_count = max(0, int(face_cluster_count))
        clusters_before_repair = max(0, int(clusters_before_repair))
        visible_people_est = max(0, int(visible_people_est))
        anchored_track_count = max(0, int(anchored_track_count))

        if face_cluster_count <= 0 or clusters_before_repair <= 0:
            return False
        if anchored_track_count < max(4, face_cluster_count):
            return False

        target_identity_count = max(face_cluster_count, visible_people_est)
        if face_cluster_count < max(1, visible_people_est - 1):
            return False
        return clusters_before_repair <= (target_identity_count + 1)

    def _build_stable_follow_bindings(
        self,
        *,
        bindings: list[dict],
        track_to_dets: dict[str, list[dict]] | None = None,
        track_identity_features: dict[str, dict] | None = None,
    ) -> list[dict]:
        min_follow_segment_ms = int(os.getenv("CLYPT_SPEAKER_FOLLOW_MIN_SEGMENT_MS", "900"))

        normalized: list[dict] = []
        for binding in list(bindings or []):
            track_id = str(binding.get("track_id", "") or "")
            if not track_id:
                continue
            start_time_ms = int(binding.get("start_time_ms", 0) or 0)
            end_time_ms = int(binding.get("end_time_ms", 0) or 0)
            if end_time_ms <= start_time_ms:
                continue
            word_count = int(binding.get("word_count", 0) or 0)
            if normalized and normalized[-1]["track_id"] == track_id and start_time_ms <= normalized[-1]["end_time_ms"]:
                normalized[-1]["end_time_ms"] = max(normalized[-1]["end_time_ms"], end_time_ms)
                normalized[-1]["word_count"] += word_count
                continue
            normalized.append(
                {
                    "track_id": track_id,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "word_count": word_count,
                }
            )

        if len(normalized) < 3:
            return normalized

        stabilized = list(normalized)
        changed = True
        while changed and len(stabilized) >= 3:
            changed = False
            next_segments: list[dict] = []
            idx = 0
            while idx < len(stabilized):
                if idx + 2 < len(stabilized):
                    left = stabilized[idx]
                    middle = stabilized[idx + 1]
                    right = stabilized[idx + 2]
                    middle_duration_ms = int(middle["end_time_ms"]) - int(middle["start_time_ms"])
                    if (
                        left["track_id"] == right["track_id"]
                        and middle_duration_ms <= min_follow_segment_ms
                    ):
                        next_segments.append(
                            {
                                "track_id": left["track_id"],
                                "start_time_ms": int(left["start_time_ms"]),
                                "end_time_ms": int(right["end_time_ms"]),
                                "word_count": int(left.get("word_count", 0))
                                + int(middle.get("word_count", 0))
                                + int(right.get("word_count", 0)),
                            }
                        )
                        idx += 3
                        changed = True
                        continue
                segment = stabilized[idx]
                if (
                    next_segments
                    and next_segments[-1]["track_id"] == segment["track_id"]
                    and int(segment["start_time_ms"]) <= int(next_segments[-1]["end_time_ms"])
                ):
                    next_segments[-1]["end_time_ms"] = max(
                        int(next_segments[-1]["end_time_ms"]),
                        int(segment["end_time_ms"]),
                    )
                    next_segments[-1]["word_count"] += int(segment.get("word_count", 0))
                else:
                    next_segments.append(dict(segment))
                idx += 1
            stabilized = next_segments
        return stabilized

    def _build_speaker_follow_bindings(self, bindings: list[dict]) -> list[dict]:
        return self._build_stable_follow_bindings(
            bindings=bindings,
            track_to_dets=None,
            track_identity_features=None,
        )

    @staticmethod
    def _local_clip_bindings_enabled() -> bool:
        return os.getenv("CLYPT_EXPERIMENT_LOCAL_CLIP_BINDINGS", "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _project_bindings_to_local_track_space(
        bindings: list[dict],
        track_id_remap: dict[str, str] | None,
    ) -> list[dict]:
        if not bindings:
            return []
        remap = {
            str(old_tid): str(new_tid)
            for old_tid, new_tid in dict(track_id_remap or {}).items()
            if str(old_tid) and str(new_tid)
        }
        if not remap:
            return [dict(binding) for binding in bindings]

        reverse_unique: dict[str, str] = {}
        ambiguous_targets: set[str] = set()
        for local_tid, global_tid in remap.items():
            prev = reverse_unique.get(global_tid)
            if prev is None:
                reverse_unique[global_tid] = local_tid
            elif prev != local_tid:
                ambiguous_targets.add(global_tid)

        for global_tid in ambiguous_targets:
            reverse_unique.pop(global_tid, None)

        projected: list[dict] = []
        for binding in bindings:
            global_tid = str(binding.get("track_id", "") or "")
            local_tid = reverse_unique.get(global_tid)
            if not local_tid:
                continue
            local_binding = dict(binding)
            local_binding["track_id"] = local_tid
            projected.append(local_binding)
        return projected

    @staticmethod
    def _build_bindings_from_word_track_field(
        words: list[dict],
        *,
        field_name: str,
    ) -> list[dict]:
        bindings: list[dict] = []
        cur = None
        for word in words:
            tid = word.get(field_name)
            ws = int(word.get("start_time_ms", 0) or 0)
            we = int(word.get("end_time_ms", 0) or 0)
            if not tid:
                continue
            if cur and cur["track_id"] == tid and ws <= cur["end_time_ms"] + 600:
                cur["end_time_ms"] = we
                cur["word_count"] += 1
            else:
                if cur:
                    bindings.append(cur)
                cur = {
                    "track_id": str(tid),
                    "start_time_ms": ws,
                    "end_time_ms": we,
                    "word_count": 1,
                }
        if cur:
            bindings.append(cur)
        return bindings

    @staticmethod
    def _bind_audio_turns_to_local_tracks(
        turns: list[dict],
        local_candidate_evidence: list[dict],
        *,
        ambiguity_margin: float = 0.05,
        support_tiebreak_margin: float = 0.1,
    ) -> list[dict]:
        """Aggregate local-track candidate evidence across each diarized turn."""
        from collections import defaultdict

        def _as_int_ms(value, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def _as_score(candidate: dict) -> float | None:
            for field_name in ("score", "total", "prob", "body_prior", "confidence"):
                value = candidate.get(field_name)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        normalized_evidence: list[dict] = []
        for evidence in local_candidate_evidence or []:
            start_time_ms = _as_int_ms(evidence.get("start_time_ms"), default=0)
            end_time_ms = _as_int_ms(evidence.get("end_time_ms"), default=start_time_ms)
            if end_time_ms < start_time_ms:
                start_time_ms, end_time_ms = end_time_ms, start_time_ms
            if end_time_ms <= start_time_ms:
                continue

            candidates = evidence.get("candidates")
            if not isinstance(candidates, list):
                candidates = [evidence]

            normalized_candidates: list[dict] = []
            for candidate in candidates:
                if not isinstance(candidate, dict) or bool(candidate.get("hard_reject", False)):
                    continue
                local_track_id = (
                    candidate.get("local_track_id")
                    or candidate.get("local_tid")
                    or candidate.get("track_id")
                )
                if local_track_id in (None, ""):
                    continue
                score = _as_score(candidate)
                if score is None:
                    continue
                normalized_candidates.append(
                    {
                        "local_track_id": str(local_track_id),
                        "score": float(score),
                    }
                )

            if normalized_candidates:
                normalized_evidence.append(
                    {
                        "start_time_ms": start_time_ms,
                        "end_time_ms": end_time_ms,
                        "candidates": normalized_candidates,
                    }
                )

        bindings: list[dict] = []
        for turn in turns or []:
            start_time_ms = _as_int_ms(turn.get("start_time_ms"), default=0)
            end_time_ms = _as_int_ms(turn.get("end_time_ms"), default=start_time_ms)
            if end_time_ms < start_time_ms:
                start_time_ms, end_time_ms = end_time_ms, start_time_ms
            turn_duration_ms = max(1, end_time_ms - start_time_ms)
            overlap_present = bool(turn.get("overlap", False))
            explicit_exclusive = turn.get("exclusive")
            high_ambiguity_turn = bool(
                overlap_present
                or explicit_exclusive is False
            )

            weighted_score_ms_by_track: dict[str, float] = defaultdict(float)
            support_ms_by_track: dict[str, int] = defaultdict(int)
            clean_weighted_score_ms_by_track: dict[str, float] = defaultdict(float)
            clean_support_ms_by_track: dict[str, int] = defaultdict(int)
            overlapping_evidence: list[dict] = []
            slice_boundaries_ms = {start_time_ms, end_time_ms}
            max_visible_candidates = 0

            for evidence in normalized_evidence:
                overlap_start_ms = max(start_time_ms, int(evidence["start_time_ms"]))
                overlap_end_ms = min(end_time_ms, int(evidence["end_time_ms"]))
                if overlap_end_ms <= overlap_start_ms:
                    continue
                overlapping_evidence.append(
                    {
                        "start_time_ms": overlap_start_ms,
                        "end_time_ms": overlap_end_ms,
                        "candidates": evidence["candidates"],
                    }
                )
                slice_boundaries_ms.add(overlap_start_ms)
                slice_boundaries_ms.add(overlap_end_ms)

            ordered_boundaries_ms = sorted(slice_boundaries_ms)
            for slice_start_ms, slice_end_ms in zip(ordered_boundaries_ms, ordered_boundaries_ms[1:]):
                slice_duration_ms = slice_end_ms - slice_start_ms
                if slice_duration_ms <= 0:
                    continue
                active_by_track: dict[str, list[float]] = defaultdict(list)
                for evidence in overlapping_evidence:
                    if int(evidence["end_time_ms"]) <= slice_start_ms or int(evidence["start_time_ms"]) >= slice_end_ms:
                        continue
                    for candidate in evidence["candidates"]:
                        active_by_track[str(candidate["local_track_id"])].append(float(candidate["score"]))
                visible_candidate_count = len(active_by_track)
                max_visible_candidates = max(max_visible_candidates, visible_candidate_count)

                for local_track_id, active_scores in active_by_track.items():
                    if not active_scores:
                        continue
                    avg_score = (sum(active_scores) / len(active_scores))
                    weighted_score_ms_by_track[local_track_id] += (avg_score * slice_duration_ms)
                    support_ms_by_track[local_track_id] += slice_duration_ms
                    if visible_candidate_count <= 2:
                        clean_weighted_score_ms_by_track[local_track_id] += (avg_score * slice_duration_ms)
                        clean_support_ms_by_track[local_track_id] += slice_duration_ms

            binding = {
                "speaker_id": str(turn.get("speaker_id", "") or ""),
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "local_track_id": None,
                "ambiguous": False,
                "winning_score": None,
                "winning_margin": None,
                "support_ratio": 0.0,
            }
            if max_visible_candidates > 2:
                binding["max_visible_candidates"] = int(max_visible_candidates)
            if not weighted_score_ms_by_track:
                if high_ambiguity_turn:
                    binding["ambiguous"] = True
                bindings.append(binding)
                continue

            ranked = sorted(
                (
                    (
                        float(weighted_score_ms / turn_duration_ms),
                        float(support_ms_by_track[local_track_id] / turn_duration_ms),
                        str(local_track_id),
                    )
                    for local_track_id, weighted_score_ms in weighted_score_ms_by_track.items()
                ),
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            best_score, best_support_ratio, best_local_track_id = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else None
            second_support_ratio = ranked[1][1] if len(ranked) > 1 else None
            winning_margin = (
                float(best_score)
                if second_score is None
                else float(best_score - second_score)
            )
            support_margin = (
                float(best_support_ratio)
                if second_support_ratio is None
                else float(best_support_ratio - second_support_ratio)
            )

            if high_ambiguity_turn:
                best_score *= 0.5
                winning_margin *= 0.5

            binding["winning_score"] = float(best_score)
            binding["winning_margin"] = float(winning_margin)
            binding["support_ratio"] = float(best_support_ratio)
            crowded_turn = max_visible_candidates > 2
            if crowded_turn:
                clean_ranked = sorted(
                    (
                        (
                            float(clean_weighted_score_ms_by_track[local_track_id] / clean_support_ms_by_track[local_track_id]),
                            float(clean_support_ms_by_track[local_track_id] / turn_duration_ms),
                            str(local_track_id),
                        )
                        for local_track_id in clean_support_ms_by_track.keys()
                        if int(clean_support_ms_by_track[local_track_id]) > 0
                    ),
                    key=lambda item: (item[0], item[1], item[2]),
                    reverse=True,
                )
                if clean_ranked:
                    clean_best_score, clean_best_support_ratio, clean_best_local_track_id = clean_ranked[0]
                    clean_best_support_ms = int(clean_support_ms_by_track.get(clean_best_local_track_id, 0))
                    clean_second_score = clean_ranked[1][0] if len(clean_ranked) > 1 else None
                    binding["clean_local_track_id"] = str(clean_best_local_track_id)
                    binding["clean_support_ms"] = clean_best_support_ms
                    binding["clean_support_ratio"] = float(clean_best_support_ratio)
                    binding["clean_winning_score"] = float(clean_best_score)
                    binding["clean_winning_margin"] = (
                        float(clean_best_score)
                        if clean_second_score is None
                        else float(clean_best_score - clean_second_score)
                    )

            if high_ambiguity_turn:
                binding["ambiguous"] = True
            elif (
                second_score is not None
                and winning_margin < float(ambiguity_margin)
                and support_margin < float(support_tiebreak_margin)
            ):
                binding["ambiguous"] = True
            elif best_score > 0.0 and best_support_ratio > 0.0:
                binding["local_track_id"] = str(best_local_track_id)

            bindings.append(binding)

        return bindings

    @staticmethod
    def _build_audio_speaker_local_track_map(
        turn_bindings: list[dict],
        *,
        min_turn_score: float = 0.75,
        min_turn_margin: float = 0.16,
        min_turn_support_ratio: float = 0.75,
        min_support_segments: int = 2,
        min_support_ms: int = 1500,
        min_speaker_dominance: float = 0.7,
        min_local_track_owner_support_ratio: float = 1.25,
        min_local_track_owner_confidence_margin: float = 0.05,
    ) -> list[dict]:
        """Build a soft audio-speaker -> local-track map from strong turn support only."""
        from collections import defaultdict

        def _as_int(value, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def _as_float(value) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _turn_confidence(score: float, support_ratio: float, margin: float) -> float:
            normalized_margin = max(0.0, min(1.0, margin / 0.35))
            confidence = (
                (0.45 * score)
                + (0.35 * support_ratio)
                + (0.20 * normalized_margin)
            )
            return max(0.0, min(1.0, confidence))

        speaker_track_support: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
        speaker_total_support_ms: dict[str, int] = defaultdict(int)

        for binding in turn_bindings or []:
            speaker_id = str(binding.get("speaker_id", "") or "")
            crowded_turn = _as_int(binding.get("max_visible_candidates"), default=0) > 2
            local_track_id = str(
                binding.get("clean_local_track_id") if crowded_turn else binding.get("local_track_id")
            ) or ""
            if not speaker_id or not local_track_id:
                continue
            if bool(binding.get("ambiguous", False)) and not crowded_turn:
                continue
            if bool(binding.get("ambiguous", False)) and crowded_turn and not str(binding.get("clean_local_track_id") or ""):
                continue

            start_time_ms = _as_int(binding.get("start_time_ms"), default=0)
            end_time_ms = _as_int(binding.get("end_time_ms"), default=start_time_ms)
            if end_time_ms < start_time_ms:
                start_time_ms, end_time_ms = end_time_ms, start_time_ms
            full_duration_ms = end_time_ms - start_time_ms
            if full_duration_ms <= 0:
                continue

            duration_ms = (
                _as_int(binding.get("clean_support_ms"), default=0)
                if crowded_turn
                else full_duration_ms
            )
            if duration_ms <= 0:
                continue

            winning_score = _as_float(
                binding.get("clean_winning_score") if crowded_turn else binding.get("winning_score")
            )
            winning_margin = _as_float(
                binding.get("clean_winning_margin") if crowded_turn else binding.get("winning_margin")
            )
            support_ratio = _as_float(
                binding.get("clean_support_ratio") if crowded_turn else binding.get("support_ratio")
            )
            if (
                winning_score is None
                or winning_margin is None
                or support_ratio is None
                or winning_score < min_turn_score
                or winning_margin < min_turn_margin
                or support_ratio < min_turn_support_ratio
            ):
                continue

            confidence = _turn_confidence(winning_score, support_ratio, winning_margin)
            if confidence < min_turn_score:
                continue

            aggregate = speaker_track_support.setdefault(speaker_id, {}).setdefault(
                local_track_id,
                {
                    "support_segments": 0,
                    "support_ms": 0,
                    "confidence_ms": 0.0,
                },
            )
            aggregate["support_segments"] = int(aggregate["support_segments"]) + 1
            aggregate["support_ms"] = int(aggregate["support_ms"]) + duration_ms
            aggregate["confidence_ms"] = float(aggregate["confidence_ms"]) + (confidence * duration_ms)
            speaker_total_support_ms[speaker_id] += duration_ms

        mappings: list[dict] = []
        for speaker_id, track_support in speaker_track_support.items():
            if not track_support:
                continue
            total_support_ms = int(speaker_total_support_ms.get(speaker_id, 0))
            if total_support_ms <= 0:
                continue

            ranked = sorted(
                (
                    (
                        int(stats["support_ms"]),
                        int(stats["support_segments"]),
                        float(stats["confidence_ms"]),
                        local_track_id,
                    )
                    for local_track_id, stats in track_support.items()
                ),
                key=lambda item: (item[0], item[1], item[2], item[3]),
                reverse=True,
            )
            best_support_ms, best_support_segments, best_confidence_ms, best_local_track_id = ranked[0]
            dominance = float(best_support_ms / total_support_ms)
            runner_up_support_ms = ranked[1][0] if len(ranked) > 1 else 0
            if (
                best_support_segments < min_support_segments
                or best_support_ms < min_support_ms
                or dominance < min_speaker_dominance
                or runner_up_support_ms >= best_support_ms
            ):
                continue

            average_confidence = float(best_confidence_ms / max(best_support_ms, 1))
            mapping_confidence = max(
                0.0,
                min(1.0, average_confidence * (0.65 + (0.35 * dominance))),
            )
            mappings.append(
                {
                    "speaker_id": speaker_id,
                    "local_track_id": best_local_track_id,
                    "support_segments": int(best_support_segments),
                    "support_ms": int(best_support_ms),
                    "confidence": round(mapping_confidence, 3),
                }
            )

        resolved_mappings: list[dict] = []
        local_track_claims: dict[str, list[dict]] = defaultdict(list)
        for mapping in mappings:
            local_track_claims[str(mapping["local_track_id"])].append(mapping)

        for claims in local_track_claims.values():
            if len(claims) == 1:
                resolved_mappings.append(claims[0])
                continue

            ranked_claims = sorted(
                claims,
                key=lambda item: (
                    int(item["support_ms"]),
                    float(item["confidence"]),
                    int(item["support_segments"]),
                    str(item["speaker_id"]),
                ),
                reverse=True,
            )
            best_claim = ranked_claims[0]
            runner_up_claim = ranked_claims[1]
            support_ratio = float(
                int(best_claim["support_ms"]) / max(1, int(runner_up_claim["support_ms"]))
            )
            confidence_margin = float(best_claim["confidence"]) - float(runner_up_claim["confidence"])
            if (
                support_ratio >= min_local_track_owner_support_ratio
                and confidence_margin >= min_local_track_owner_confidence_margin
            ):
                resolved_mappings.append(best_claim)
                continue

        resolved_mappings.sort(
            key=lambda item: (
                item["speaker_id"],
                -int(item["support_ms"]),
                -int(item["support_segments"]),
                item["local_track_id"],
            )
        )
        return resolved_mappings

    @staticmethod
    def _speaker_remap_collision_metrics(words: list[dict]) -> dict[str, int]:
        from collections import defaultdict

        global_to_locals: dict[str, set[str]] = defaultdict(set)
        for word in words or []:
            global_tid = str(word.get("speaker_track_id", "") or "")
            local_tid = str(word.get("speaker_local_track_id", "") or "")
            if not global_tid or not local_tid:
                continue
            global_to_locals[global_tid].add(local_tid)

        local_counts = [len(local_ids) for local_ids in global_to_locals.values() if local_ids]
        return {
            "speaker_binding_globals_with_multiple_local_ids": int(
                sum(1 for count in local_counts if count > 1)
            ),
            "speaker_binding_max_local_ids_per_global": int(max(local_counts) if local_counts else 0),
        }

    def _build_speaker_binding_track_quality(
        self,
        track_to_dets: dict[str, list[dict]],
        *,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, dict]:
        import numpy as np

        frame_area = max(1.0, float(max(1, frame_width) * max(1, frame_height)))
        out: dict[str, dict] = {}
        for tid, dets in (track_to_dets or {}).items():
            valid = [det for det in dets if float(det.get("width", 0.0)) > 1e-6 and float(det.get("height", 0.0)) > 1e-6]
            if not valid:
                out[str(tid)] = {
                    "track_quality": 0.0,
                    "median_area_norm": 0.0,
                    "p90_area_norm": 0.0,
                    "duplicate_frame_ratio": 0.0,
                    "geometry_spread": 1.0,
                }
                continue

            areas = np.asarray(
                [
                    (float(det.get("width", 0.0)) * float(det.get("height", 0.0))) / frame_area
                    for det in valid
                ],
                dtype=np.float32,
            )
            confidences = np.asarray(
                [float(det.get("confidence", 0.0)) for det in valid],
                dtype=np.float32,
            )
            by_frame: dict[int, list[dict]] = {}
            for det in valid:
                frame_idx = int(det.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                by_frame.setdefault(frame_idx, []).append(det)

            median_area = float(np.median(areas))
            p90_area = float(np.percentile(areas, 90))
            median_conf = float(np.median(confidences)) if len(confidences) else 0.0
            duplicate_frames = sum(1 for frame_dets in by_frame.values() if len(frame_dets) > 1)
            duplicate_ratio = float(duplicate_frames / max(1, len(by_frame)))
            geometry_spread = float(p90_area / max(median_area, 1e-6))

            huge_median_penalty = min(1.0, max(0.0, (median_area - 0.28) / 0.20))
            huge_p90_penalty = min(1.0, max(0.0, (p90_area - 0.48) / 0.20))
            duplicate_penalty = min(1.0, duplicate_ratio / 0.08)
            spread_penalty = min(1.0, max(0.0, (geometry_spread - 2.1) / 1.4))

            track_quality = 1.0
            track_quality -= 0.42 * huge_median_penalty
            track_quality -= 0.28 * huge_p90_penalty
            track_quality -= 0.22 * duplicate_penalty
            track_quality -= 0.14 * spread_penalty
            track_quality = max(0.0, min(1.0, track_quality))

            out[str(tid)] = {
                "track_quality": float(track_quality),
                "median_area_norm": median_area,
                "p90_area_norm": p90_area,
                "duplicate_frame_ratio": duplicate_ratio,
                "geometry_spread": geometry_spread,
                "median_confidence": median_conf,
            }
        return out

    def _score_speaker_binding_body_candidate(
        self,
        *,
        det: dict,
        frame_dets: list[dict],
        frame_width: int,
        frame_height: int,
        track_quality: float,
        motion_rank: float,
    ) -> dict:
        frame_area = max(1.0, float(max(1, frame_width) * max(1, frame_height)))
        width = max(0.0, float(det.get("width", 0.0)))
        height = max(0.0, float(det.get("height", 0.0)))
        area_norm = (width * height) / frame_area
        conf = max(0.0, min(1.0, float(det.get("confidence", 0.0))))
        x1 = float(det.get("x1", float(det.get("x_center", 0.0)) - (0.5 * width)))
        x2 = float(det.get("x2", float(det.get("x_center", 0.0)) + (0.5 * width)))
        y1 = float(det.get("y1", float(det.get("y_center", 0.0)) - (0.5 * height)))
        y2 = float(det.get("y2", float(det.get("y_center", 0.0)) + (0.5 * height)))

        tiny_penalty = 0.0
        if area_norm < 0.010:
            tiny_penalty = 1.0
        elif area_norm < 0.025:
            tiny_penalty = (0.025 - area_norm) / 0.015

        huge_penalty = 0.0
        if area_norm > 0.45:
            huge_penalty = 1.0
        elif area_norm > 0.30:
            huge_penalty = (area_norm - 0.30) / 0.15

        edge_margin_x = 0.03 * float(max(1, frame_width))
        edge_margin_y = 0.03 * float(max(1, frame_height))
        touches_edge = (
            x1 <= edge_margin_x
            or y1 <= edge_margin_y
            or x2 >= float(frame_width) - edge_margin_x
            or y2 >= float(frame_height) - edge_margin_y
        )
        edge_penalty = 0.0
        if touches_edge and area_norm < 0.05:
            edge_penalty = min(1.0, (0.05 - area_norm) / 0.05)

        duplicate_penalty = 0.0
        for peer in frame_dets:
            if peer is det:
                continue
            peer_conf = float(peer.get("confidence", 0.0))
            if peer_conf + 1e-6 < conf:
                continue
            iou = self._detection_iou(det, peer)
            if iou >= 0.75:
                duplicate_penalty = max(duplicate_penalty, min(1.0, (iou - 0.75) / 0.20))

        det_quality = 1.0
        det_quality -= 0.70 * tiny_penalty
        det_quality -= 0.62 * huge_penalty
        det_quality -= 0.28 * edge_penalty
        det_quality -= 0.24 * duplicate_penalty
        det_quality = max(0.0, min(1.0, det_quality))
        hard_reject = bool(
            area_norm < 0.012
            or (touches_edge and area_norm < 0.020)
        )

        body_prior = (
            0.48 * det_quality
            + 0.26 * max(0.0, min(1.0, float(track_quality)))
            + 0.16 * max(0.0, min(1.0, float(motion_rank)))
            + 0.10 * conf
        )
        return {
            "body_prior": float(max(0.0, min(1.0, body_prior))),
            "detection_quality": float(det_quality),
            "area_norm": float(area_norm),
            "touches_edge": bool(touches_edge),
            "duplicate_penalty": float(duplicate_penalty),
            "hard_reject": hard_reject,
        }

    @staticmethod
    def _requested_face_pipeline_workers() -> int:
        raw = os.getenv("CLYPT_FACE_PIPELINE_WORKERS", "").strip()
        if not raw:
            raw = os.getenv("CLYPT_FACE_LEDGER_WORKERS", "").strip()
        try:
            requested = int(raw) if raw else int(os.cpu_count() or 8)
        except Exception:
            requested = int(os.cpu_count() or 8)
        return max(1, min(32, requested))

    @staticmethod
    def _face_pipeline_gpu_worker_cap() -> int:
        raw = os.getenv("CLYPT_FACE_PIPELINE_GPU_WORKERS", "").strip()
        try:
            requested = int(raw) if raw else 1
        except Exception:
            requested = 1
        return max(1, min(4, requested))

    @staticmethod
    def _face_pipeline_start_frame() -> int:
        raw = os.getenv("CLYPT_FACE_PIPELINE_START_FRAME", "").strip()
        try:
            requested = int(raw) if raw else 600
        except Exception:
            requested = 600
        return max(0, requested)

    def _face_pipeline_uses_gpu(self) -> bool:
        if getattr(self, "face_detector", None) is None:
            return False
        if getattr(self, "face_recognizer", None) is None:
            return False
        return bool(getattr(self, "_face_runtime_ctx_id", 0) >= 0)

    def _face_pipeline_workers(self) -> int:
        requested = self._requested_face_pipeline_workers()
        if self._face_pipeline_uses_gpu():
            return min(requested, self._face_pipeline_gpu_worker_cap())
        return requested

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
    def _audio_diarization_config() -> dict:
        """Return the env-driven pyannote diarization config surface."""
        enabled_raw = os.getenv("CLYPT_AUDIO_DIARIZATION_ENABLE", "0").strip().lower()
        model_name = os.getenv(
            "CLYPT_AUDIO_DIARIZATION_MODEL",
            "pyannote/speaker-diarization-3.1",
        ).strip()
        if not model_name:
            model_name = "pyannote/speaker-diarization-3.1"
        raw_min_segment_ms = os.getenv("CLYPT_AUDIO_DIARIZATION_MIN_SEGMENT_MS", "400").strip()
        try:
            min_segment_ms = int(raw_min_segment_ms)
        except Exception:
            min_segment_ms = 400
        min_segment_ms = max(0, min_segment_ms)
        return {
            "enabled": enabled_raw in {"1", "true", "yes", "on"},
            "model_name": model_name,
            "min_segment_ms": min_segment_ms,
            "min_segment_s": min_segment_ms / 1000.0,
            "token_env_vars": ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
        }

    def _load_audio_diarization_pipeline(self):
        config = self._audio_diarization_config()
        if not config["enabled"]:
            return None

        cached_pipeline = getattr(self, "_audio_diarization_pipeline", None)
        if cached_pipeline is not None:
            return cached_pipeline

        try:
            from pyannote.audio import Pipeline
        except Exception as exc:
            print(f"[Phase 1] Audio diarization unavailable: {type(exc).__name__}: {exc}")
            return None

        token = None
        for env_var in config["token_env_vars"]:
            token = os.getenv(env_var, "").strip()
            if token:
                break

        try:
            if token:
                pipeline = Pipeline.from_pretrained(config["model_name"], use_auth_token=token)
            else:
                pipeline = Pipeline.from_pretrained(config["model_name"])
        except Exception as exc:
            print(f"[Phase 1] Audio diarization model load failed: {type(exc).__name__}: {exc}")
            return None

        self._audio_diarization_pipeline = pipeline
        return pipeline

    def _record_audio_diarization_metrics(
        self,
        *,
        enabled: bool,
        status: str,
        turn_count: int = 0,
    ) -> None:
        self._last_audio_diarization_metrics = {
            "audio_diarization_enabled": bool(enabled),
            "audio_diarization_fallback": bool(status != "ok"),
            "audio_diarization_status": str(status),
            "audio_diarization_turn_count": max(0, int(turn_count)),
        }

    @staticmethod
    def _coerce_optional_bool(value):
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    @staticmethod
    def _turn_value(source, key: str, default=None):
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def _serialize_audio_speaker_turn(self, turn) -> dict | None:
        speaker_id = None
        start_time_ms = None
        end_time_ms = None
        exclusive = None
        overlap = None
        confidence = None
        segment = None
        metadata_sources = [turn]

        if isinstance(turn, dict):
            turn = dict(turn)
            speaker_id = turn.pop("speaker_id", None) or turn.pop("speaker", None) or turn.pop("label", None)
            start_time_ms = turn.pop("start_time_ms", turn.pop("start_ms", None))
            end_time_ms = turn.pop("end_time_ms", turn.pop("end_ms", None))
            exclusive = turn.pop("exclusive", None)
            overlap = turn.pop("overlap", None)
            confidence = turn.pop("confidence", turn.pop("score", None))
            segment = turn.pop("segment", None)
            metadata_sources = [turn, segment]
        elif isinstance(turn, (tuple, list)):
            if len(turn) >= 3:
                segment = turn[0]
                speaker_id = turn[2]
            elif len(turn) == 2:
                segment = turn[0]
                speaker_id = turn[1]
            elif len(turn) == 1:
                segment = turn[0]
            if len(turn) >= 4:
                metadata_sources.append(turn[3])
        else:
            speaker_id = self._turn_value(turn, "speaker_id")
            if speaker_id is None:
                speaker_id = self._turn_value(turn, "speaker") or self._turn_value(turn, "label")
            start_time_ms = self._turn_value(turn, "start_time_ms", self._turn_value(turn, "start_ms"))
            end_time_ms = self._turn_value(turn, "end_time_ms", self._turn_value(turn, "end_ms"))
            exclusive = self._turn_value(turn, "exclusive")
            overlap = self._turn_value(turn, "overlap")
            confidence = self._turn_value(turn, "confidence", self._turn_value(turn, "score"))
            segment = self._turn_value(turn, "segment")
            metadata_sources = [turn, segment]

        if segment is not None and (start_time_ms is None or end_time_ms is None):
            start = getattr(segment, "start", None)
            end = getattr(segment, "end", None)
            if start is not None and end is not None:
                start_time_ms = int(round(float(start) * 1000.0))
                end_time_ms = int(round(float(end) * 1000.0))

        if exclusive is None:
            for source in metadata_sources:
                exclusive = self._turn_value(source, "exclusive")
                if exclusive is not None:
                    break

        if overlap is None:
            for source in metadata_sources:
                overlap = self._turn_value(source, "overlap")
                if overlap is not None:
                    break

        if confidence is None:
            for source in metadata_sources:
                confidence = self._turn_value(source, "confidence", self._turn_value(source, "score"))
                if confidence is not None:
                    break

        if speaker_id is None or start_time_ms is None or end_time_ms is None:
            return None

        try:
            start_time_ms = max(0, int(round(float(start_time_ms))))
            end_time_ms = max(start_time_ms, max(0, int(round(float(end_time_ms)))))
        except Exception:
            return None

        serialized = {
            "speaker_id": str(speaker_id),
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
        }

        maybe_bool = self._coerce_optional_bool(exclusive)
        if maybe_bool is not None:
            serialized["exclusive"] = maybe_bool

        maybe_bool = self._coerce_optional_bool(overlap)
        if maybe_bool is not None:
            serialized["overlap"] = maybe_bool

        if confidence is not None:
            try:
                serialized["confidence"] = float(confidence)
            except Exception:
                pass

        return serialized

    def _serialize_audio_speaker_turns(self, diarization) -> list[dict]:
        if diarization is None:
            return []

        if isinstance(diarization, dict) and "audio_speaker_turns" in diarization:
            raw_turns = list(diarization.get("audio_speaker_turns") or [])
        elif isinstance(diarization, list):
            raw_turns = list(diarization)
        else:
            itertracks = getattr(diarization, "itertracks", None)
            if callable(itertracks):
                try:
                    raw_turns = list(itertracks(yield_label=True))
                except TypeError:
                    raw_turns = list(itertracks())
            else:
                try:
                    raw_turns = list(diarization)
                except TypeError:
                    return []

        serialized = []
        for turn in raw_turns:
            normalized = self._serialize_audio_speaker_turn(turn)
            if normalized is not None:
                serialized.append(normalized)

        serialized.sort(key=lambda item: (item["start_time_ms"], item["end_time_ms"], item["speaker_id"]))
        return serialized

    def _run_audio_diarization(self, audio_path: str) -> list[dict]:
        config = self._audio_diarization_config()
        if not config["enabled"]:
            self._record_audio_diarization_metrics(
                enabled=False,
                status="disabled",
                turn_count=0,
            )
            return []

        pipeline = self._load_audio_diarization_pipeline()
        if pipeline is None:
            self._record_audio_diarization_metrics(
                enabled=True,
                status="unavailable",
                turn_count=0,
            )
            return []

        try:
            diarization = pipeline(audio_path)
        except Exception as exc:
            print(f"[Phase 1] Audio diarization execution failed: {type(exc).__name__}: {exc}")
            self._record_audio_diarization_metrics(
                enabled=True,
                status="error",
                turn_count=0,
            )
            return []

        turns = self._serialize_audio_speaker_turns(diarization)
        self._record_audio_diarization_metrics(
            enabled=True,
            status="ok",
            turn_count=len(turns),
        )
        return turns

    @staticmethod
    def _face_detector_input_size() -> tuple[int, int]:
        raw = os.getenv("CLYPT_FACE_DETECTOR_INPUT_SIZE", "").strip()
        if not raw:
            raw = os.getenv("CLYPT_FACE_DETECTOR_INPUT_LONG_EDGE", "960").strip()
        try:
            requested = int(raw)
        except Exception:
            requested = 960
        requested = max(256, requested)
        return (requested, requested)

    @staticmethod
    def _split_frame_items_into_segments(
        frame_to_dets: dict[int, list[dict]],
        segment_frames: int,
    ) -> list[list[tuple[int, list[dict]]]]:
        if not frame_to_dets:
            return []
        grouped: dict[int, list[tuple[int, list[dict]]]] = {}
        for frame_idx in sorted(int(fi) for fi in frame_to_dets.keys()):
            dets = list(frame_to_dets.get(frame_idx, []))
            if frame_idx < 0 or not dets:
                continue
            segment_idx = frame_idx // max(1, segment_frames)
            grouped.setdefault(segment_idx, []).append((frame_idx, dets))
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

    def _create_scrfd_detector(self):
        from insightface import model_zoo

        detector = model_zoo.get_model(
            str(getattr(self, "_face_detector_model_file", "")),
            providers=list(getattr(self, "_face_runtime_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])),
        )
        detector.prepare(
            ctx_id=int(getattr(self, "_face_runtime_ctx_id", 0)),
            input_size=tuple(getattr(self, "_face_detector_input_size", (640, 640))),
            det_thresh=float(getattr(self, "_face_runtime_det_thresh", 0.15)),
        )
        return detector

    def _create_arcface_recognizer(self):
        from insightface import model_zoo

        recognizer = model_zoo.get_model(
            str(getattr(self, "_face_recognizer_model_file", "")),
            providers=list(getattr(self, "_face_runtime_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])),
        )
        recognizer.prepare(ctx_id=int(getattr(self, "_face_runtime_ctx_id", 0)))
        return recognizer

    def _get_thread_face_runtime(self):
        import threading

        detector = getattr(self, "face_detector", None)
        recognizer = getattr(self, "face_recognizer", None)
        if detector is None or recognizer is None:
            return None, None

        main_thread_ident = getattr(self, "_face_runtime_main_thread_ident", None)
        if main_thread_ident is None or threading.get_ident() == main_thread_ident:
            return detector, recognizer

        local = getattr(self, "_face_runtime_local", None)
        if local is None:
            local = threading.local()
            self._face_runtime_local = local

        local_detector = getattr(local, "face_detector", None)
        local_recognizer = getattr(local, "face_recognizer", None)
        if local_detector is None or local_recognizer is None:
            local_detector = self._create_scrfd_detector()
            local_recognizer = self._create_arcface_recognizer()
            local.face_detector = local_detector
            local.face_recognizer = local_recognizer
        return local_detector, local_recognizer

    def _prewarm_face_runtime_for_current_thread(self) -> bool:
        import numpy as np
        from insightface.app.common import Face

        detector, recognizer = self._get_thread_face_runtime()
        if detector is None or recognizer is None:
            return False

        det_size = tuple(getattr(self, "_face_detector_input_size", (640, 640)))
        warmup_frame = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
        detector.detect(
            warmup_frame,
            input_size=det_size,
            max_num=1,
        )

        recognizer_input = np.zeros((112, 112, 3), dtype=np.uint8)
        warmup_face = Face(
            bbox=np.asarray([16.0, 16.0, 96.0, 96.0], dtype=np.float32),
            kps=np.asarray(
                [
                    [34.0, 42.0],
                    [78.0, 42.0],
                    [56.0, 60.0],
                    [40.0, 80.0],
                    [72.0, 80.0],
                ],
                dtype=np.float32,
            ),
            det_score=1.0,
        )
        recognizer.get(recognizer_input, warmup_face)
        return True

    def _prewarm_face_runtime_in_pool(self, face_pool, worker_count: int) -> None:
        from concurrent.futures import wait

        if face_pool is None or int(worker_count) <= 0:
            return
        futures = [
            face_pool.submit(self._prewarm_face_runtime_for_current_thread)
            for _ in range(max(1, int(worker_count)))
        ]
        wait(futures)
        for future in futures:
            future.result()

    def _detect_faces_full_frame(self, frame_rgb, *, max_faces: int = 0) -> list[dict]:
        import cv2
        import numpy as np
        from insightface.app.common import Face

        detector, recognizer = self._get_thread_face_runtime()
        if detector is None or recognizer is None or frame_rgb is None:
            return []

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            det, kpss = detector.detect(
                frame_bgr,
                input_size=tuple(getattr(self, "_face_detector_input_size", (640, 640))),
                max_num=int(max_faces or 0),
            )
        except Exception:
            return []
        if det is None or len(det) == 0:
            return []

        fh, fw = frame_rgb.shape[:2]
        out: list[dict] = []
        min_face_size = float(os.getenv("CLYPT_FULLFRAME_FACE_MIN_SIZE", "28"))
        for idx, row in enumerate(np.asarray(det)):
            if len(row) < 5:
                continue
            x1, y1, x2, y2, score = [float(v) for v in row[:5]]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if min(bw, bh) < min_face_size:
                continue
            x1 = max(0.0, min(float(max(0, fw - 1)), x1))
            y1 = max(0.0, min(float(max(0, fh - 1)), y1))
            x2 = max(x1 + 1.0, min(float(max(1, fw)), x2))
            y2 = max(y1 + 1.0, min(float(max(1, fh)), y2))
            kps = None if kpss is None or idx >= len(kpss) else np.asarray(kpss[idx], dtype=np.float32)
            embedding_vec = None
            if kps is not None and kps.size > 0:
                try:
                    face = Face(
                        bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
                        kps=kps,
                        det_score=float(score),
                    )
                    emb = recognizer.get(frame_bgr, face)
                    emb_arr = np.asarray(emb, dtype=np.float32)
                    if emb_arr.size > 0:
                        embedding_vec = emb_arr
                except Exception:
                    embedding_vec = None
            out.append(
                {
                    "bbox_xyxy": (float(x1), float(y1), float(x2), float(y2)),
                    "det_score": float(score),
                    "kps": None if kps is None else kps.astype(np.float32),
                    "embedding": embedding_vec,
                }
            )
        return out

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

    def _associate_faces_to_person_dets(
        self,
        face_detections: list[dict],
        person_dets: list[dict],
    ) -> list[str | None]:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        if not face_detections:
            return []
        if not person_dets:
            return [None for _ in face_detections]

        min_match_score = float(os.getenv("CLYPT_FACE_ASSOC_MIN_SCORE", "0.18"))
        score_matrix = np.full((len(face_detections), len(person_dets)), fill_value=-1.0, dtype=np.float32)

        for face_idx, face in enumerate(face_detections):
            fx1, fy1, fx2, fy2 = face.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0))
            fcx = 0.5 * (float(fx1) + float(fx2))
            fcy = 0.5 * (float(fy1) + float(fy2))
            for det_idx, det in enumerate(person_dets):
                tid = str(det.get("track_id", ""))
                if not tid:
                    continue
                px1 = float(det.get("x1", 0.0))
                py1 = float(det.get("y1", 0.0))
                px2 = float(det.get("x2", px1 + 1.0))
                py2 = float(det.get("y2", py1 + 1.0))
                pw = max(1.0, px2 - px1)
                ph = max(1.0, py2 - py1)
                if not (px1 - 0.05 * pw <= fcx <= px2 + 0.05 * pw):
                    continue
                if not (py1 - 0.08 * ph <= fcy <= py1 + 0.68 * ph):
                    continue
                head_box = (
                    px1,
                    max(0.0, py1 - 0.04 * ph),
                    px2,
                    py1 + (0.62 * ph),
                )
                iou = self._bbox_iou_xyxy((fx1, fy1, fx2, fy2), head_box)
                center_x_bonus = 1.0 - min(1.0, abs(fcx - (0.5 * (px1 + px2))) / max(1.0, 0.55 * pw))
                center_y_bonus = 1.0 - min(1.0, abs(fcy - (py1 + 0.26 * ph)) / max(1.0, 0.32 * ph))
                score = (0.55 * iou) + (0.25 * center_x_bonus) + (0.20 * center_y_bonus)
                if score >= min_match_score:
                    score_matrix[face_idx, det_idx] = float(score)

        assignments: list[str | None] = [None for _ in face_detections]
        if np.max(score_matrix) < 0:
            return assignments

        cost_matrix = np.where(score_matrix >= 0.0, 1.0 - score_matrix, 10.0).astype(np.float32)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for face_idx, det_idx in zip(row_ind, col_ind):
            if score_matrix[face_idx, det_idx] < min_match_score:
                continue
            tid = str(person_dets[det_idx].get("track_id", ""))
            assignments[face_idx] = tid or None
        return assignments

    def _extract_face_track_features_for_frame_segment(
        self,
        *,
        video_path: str,
        frame_items: list[tuple[int, list[dict]]],
        fps: float,
        frame_width: int,
        frame_height: int,
        output_frame_width: int | None = None,
        output_frame_height: int | None = None,
        coord_scale_x: float = 1.0,
        coord_scale_y: float = 1.0,
    ) -> dict:
        import numpy as np
        from collections import defaultdict
        from scipy.optimize import linear_sum_assignment

        frame_items = [(int(frame_idx), list(dets)) for frame_idx, dets in frame_items if dets]
        if not frame_items:
            return {"face_track_features": {}}

        try:
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, ctx=cpu(0))
            valid_items = [(frame_idx, dets) for frame_idx, dets in frame_items if 0 <= frame_idx < len(vr)]
            if not valid_items:
                return {"face_track_features": {}}
            frame_indices = [frame_idx for frame_idx, _ in valid_items]
            batch = vr.get_batch(frame_indices).asnumpy()
            frame_map = {frame_idx: batch[idx] for idx, frame_idx in enumerate(frame_indices)}
        except Exception:
            valid_items = list(frame_items)
            frame_map = {
                frame_idx: self._read_frame_rgb(video_path, frame_idx)
                for frame_idx, _ in valid_items
            }

        out_w = int(output_frame_width or frame_width or 1)
        out_h = int(output_frame_height or frame_height or 1)
        inv_scale_x = self._normalize_scale_factor(coord_scale_x)
        inv_scale_y = self._normalize_scale_factor(coord_scale_y)
        segment_start = int(valid_items[0][0]) if valid_items else 0
        max_gap = max(2, int(os.getenv("CLYPT_FACE_TRACK_MAX_GAP", "8")))
        match_cost_threshold = float(os.getenv("CLYPT_FACE_TRACK_MATCH_COST", "0.78"))
        next_local_track_id = 0
        active_tracks: dict[int, dict] = {}
        face_tracks: dict[int, dict] = {}

        def _center_dist(box_a, box_b) -> float:
            ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
            bx1, by1, bx2, by2 = [float(v) for v in box_b]
            acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
            bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
            aw = max(1.0, ax2 - ax1)
            ah = max(1.0, ay2 - ay1)
            bw = max(1.0, bx2 - bx1)
            bh = max(1.0, by2 - by1)
            norm = max(1.0, 0.5 * (aw + bw), 0.5 * (ah + bh))
            return float((((acx - bcx) ** 2) + ((acy - bcy) ** 2)) ** 0.5 / norm)

        def _emb_dist(emb_a, emb_b) -> float:
            if emb_a is None or emb_b is None:
                return 0.45
            return self._cosine_dist(np.asarray(emb_a, dtype=np.float32), np.asarray(emb_b, dtype=np.float32))

        for frame_idx, dets in valid_items:
            frame_rgb = frame_map.get(frame_idx)
            if frame_rgb is None:
                continue
            detections = list(self._detect_faces_full_frame(frame_rgb))
            assignments = self._associate_faces_to_person_dets(detections, dets)

            if not detections:
                # retire stale tracks even on empty frames
                stale_ids = [
                    local_id
                    for local_id, state in active_tracks.items()
                    if frame_idx - int(state.get("last_frame_idx", frame_idx)) > max_gap
                ]
                for local_id in stale_ids:
                    active_tracks.pop(local_id, None)
                continue

            active_candidates = [
                (local_id, state)
                for local_id, state in active_tracks.items()
                if frame_idx - int(state.get("last_frame_idx", frame_idx)) <= max_gap
            ]
            matched_detection_indices: set[int] = set()
            matched_track_ids: set[int] = set()
            if active_candidates and detections:
                cost = np.full((len(active_candidates), len(detections)), fill_value=10.0, dtype=np.float32)
                for row_idx, (local_id, state) in enumerate(active_candidates):
                    for col_idx, detection in enumerate(detections):
                        iou = self._bbox_iou_xyxy(state.get("last_bbox"), detection.get("bbox_xyxy"))
                        if iou <= 0.01 and _center_dist(state.get("last_bbox"), detection.get("bbox_xyxy")) > 1.5:
                            continue
                        score = (
                            0.50 * (1.0 - iou)
                            + 0.25 * _center_dist(state.get("last_bbox"), detection.get("bbox_xyxy"))
                            + 0.25 * _emb_dist(state.get("last_embedding"), detection.get("embedding"))
                        )
                        cost[row_idx, col_idx] = float(score)
                row_idx, col_idx = linear_sum_assignment(cost)
                for r_i, c_i in zip(row_idx, col_idx):
                    if float(cost[r_i, c_i]) > match_cost_threshold:
                        continue
                    local_id = int(active_candidates[r_i][0])
                    matched_detection_indices.add(int(c_i))
                    matched_track_ids.add(local_id)
                    detection = detections[c_i]
                    track = face_tracks.setdefault(
                        local_id,
                        {
                            "observations": [],
                            "embeddings": [],
                            "associated_track_counts": defaultdict(int),
                        },
                    )
                    obs = self._face_detection_observation(
                        detection=detection,
                        frame_idx=frame_idx,
                        fps=fps,
                        output_frame_width=out_w,
                        output_frame_height=out_h,
                        inv_scale_x=inv_scale_x,
                        inv_scale_y=inv_scale_y,
                        associated_track_id=assignments[c_i],
                    )
                    track["observations"].append(obs)
                    if obs.get("associated_track_id"):
                        track["associated_track_counts"][str(obs["associated_track_id"])] += 1
                    embedding = detection.get("embedding")
                    if embedding is not None:
                        track["embeddings"].append(np.asarray(embedding, dtype=np.float32))
                    active_tracks[local_id] = {
                        "last_bbox": tuple(detection.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0))),
                        "last_embedding": detection.get("embedding"),
                        "last_frame_idx": frame_idx,
                    }

            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detection_indices:
                    continue
                local_id = next_local_track_id
                next_local_track_id += 1
                track = face_tracks.setdefault(
                    local_id,
                    {
                        "observations": [],
                        "embeddings": [],
                        "associated_track_counts": defaultdict(int),
                    },
                )
                obs = self._face_detection_observation(
                    detection=detection,
                    frame_idx=frame_idx,
                    fps=fps,
                    output_frame_width=out_w,
                    output_frame_height=out_h,
                    inv_scale_x=inv_scale_x,
                    inv_scale_y=inv_scale_y,
                    associated_track_id=assignments[det_idx],
                )
                track["observations"].append(obs)
                if obs.get("associated_track_id"):
                    track["associated_track_counts"][str(obs["associated_track_id"])] += 1
                embedding = detection.get("embedding")
                if embedding is not None:
                    track["embeddings"].append(np.asarray(embedding, dtype=np.float32))
                active_tracks[local_id] = {
                    "last_bbox": tuple(detection.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0))),
                    "last_embedding": detection.get("embedding"),
                    "last_frame_idx": frame_idx,
                }

            stale_ids = [
                local_id
                for local_id, state in active_tracks.items()
                if frame_idx - int(state.get("last_frame_idx", frame_idx)) > max_gap
            ]
            for local_id in stale_ids:
                active_tracks.pop(local_id, None)

        finalized_face_tracks: dict[str, dict] = {}
        for local_id, feature in face_tracks.items():
            observations = sorted(
                feature.get("observations", []),
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            if not observations:
                continue
            embeddings = feature.get("embeddings", [])
            emb = None
            if embeddings:
                emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32).tolist()
            face_track_id = f"face_{segment_start}_{int(local_id)}"
            associated_counts = {
                str(tid): int(count)
                for tid, count in dict(feature.get("associated_track_counts", {})).items()
                if str(tid)
            }
            dominant_track_id = None
            if associated_counts:
                dominant_track_id = max(associated_counts.items(), key=lambda item: item[1])[0]
            finalized_face_tracks[face_track_id] = {
                "face_track_id": face_track_id,
                "embedding": emb,
                "embedding_source": "face" if emb is not None else "none",
                "embedding_count": len(embeddings),
                "face_observations": observations,
                "face_observation_count": len(observations),
                "associated_track_counts": associated_counts,
                "dominant_track_id": dominant_track_id,
            }
        return {"face_track_features": finalized_face_tracks}

    def _face_detection_observation(
        self,
        *,
        detection: dict,
        frame_idx: int,
        fps: float,
        output_frame_width: int,
        output_frame_height: int,
        inv_scale_x: float,
        inv_scale_y: float,
        associated_track_id: str | None,
    ) -> dict:
        gx1, gy1, gx2, gy2 = [float(v) for v in detection.get("bbox_xyxy", (0.0, 0.0, 1.0, 1.0))]
        bbox = self._normalize_bbox(
            float(gx1) / inv_scale_x,
            float(gy1) / inv_scale_y,
            float(gx2) / inv_scale_x,
            float(gy2) / inv_scale_y,
            int(output_frame_width),
            int(output_frame_height),
        )
        return {
            "frame_idx": int(frame_idx),
            "time_ms": int(round((frame_idx / max(1e-6, fps)) * 1000.0)),
            "bounding_box": bbox,
            "confidence": float(detection.get("det_score", 0.0)),
            "quality": float(detection.get("det_score", 0.0)),
            "source": str(detection.get("source", "face_detector")),
            "provenance": str(detection.get("provenance", "scrfd_fullframe")),
            "associated_track_id": associated_track_id,
        }

    def _merge_face_track_feature_sets(self, feature_maps: list[dict[str, dict]]) -> dict[str, dict]:
        import numpy as np
        from collections import defaultdict

        merged: dict[str, dict] = {}
        for feature_map in feature_maps:
            for face_track_id, feature in (feature_map or {}).items():
                slot = merged.setdefault(
                    str(face_track_id),
                    {
                        "embeddings": [],
                        "observations": [],
                        "associated_track_counts": defaultdict(int),
                    },
                )
                embedding = feature.get("embedding")
                if embedding is not None:
                    emb_arr = np.asarray(embedding, dtype=np.float32)
                    if emb_arr.size > 0:
                        repeat = max(1, int(feature.get("embedding_count", 1)))
                        slot["embeddings"].extend([emb_arr] * repeat)
                slot["observations"].extend(list(feature.get("face_observations", [])))
                for tid, count in dict(feature.get("associated_track_counts", {})).items():
                    if str(tid):
                        slot["associated_track_counts"][str(tid)] += int(count)

        stitch_max_gap = max(0, int(os.getenv("CLYPT_FACE_TRACK_STITCH_MAX_GAP_FRAMES", "72")))
        stitch_max_cos = float(os.getenv("CLYPT_FACE_TRACK_STITCH_MAX_COS", "0.24"))
        stitch_max_sig = float(os.getenv("CLYPT_FACE_TRACK_STITCH_MAX_SIG", "0.20"))
        stitch_same_track_bonus_gap = max(
            stitch_max_gap,
            int(os.getenv("CLYPT_FACE_TRACK_STITCH_SAME_TRACK_MAX_GAP_FRAMES", "144")),
        )

        metadata: dict[str, dict] = {}
        for face_track_id, feature in merged.items():
            observations = sorted(
                feature["observations"],
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            frames = [int(obs.get("frame_idx", -1)) for obs in observations if int(obs.get("frame_idx", -1)) >= 0]
            if not frames:
                continue
            emb = None
            if feature["embeddings"]:
                emb = np.mean(np.stack(feature["embeddings"], axis=0), axis=0).astype(np.float32)
            associated_counts = {
                str(tid): int(count)
                for tid, count in dict(feature["associated_track_counts"]).items()
                if str(tid)
            }
            dominant_track_id = None
            if associated_counts:
                dominant_track_id = max(associated_counts.items(), key=lambda item: item[1])[0]
            metadata[face_track_id] = {
                "start_frame": min(frames),
                "end_frame": max(frames),
                "embedding": emb,
                "signature": self._face_observation_signature(observations),
                "dominant_track_id": dominant_track_id,
            }

        parent = {face_track_id: face_track_id for face_track_id in merged.keys()}

        def _find(face_track_id: str) -> str:
            root = parent[face_track_id]
            while root != parent[root]:
                root = parent[root]
            while face_track_id != root:
                nxt = parent[face_track_id]
                parent[face_track_id] = root
                face_track_id = nxt
            return root

        def _union(left_id: str, right_id: str) -> None:
            left_root = _find(left_id)
            right_root = _find(right_id)
            if left_root == right_root:
                return
            left_meta = metadata.get(left_root, {})
            right_meta = metadata.get(right_root, {})
            left_start = int(left_meta.get("start_frame", 1_000_000_000))
            right_start = int(right_meta.get("start_frame", 1_000_000_000))
            keep_root, drop_root = (left_root, right_root) if left_start <= right_start else (right_root, left_root)
            parent[drop_root] = keep_root

        ordered_face_track_ids = sorted(
            metadata.keys(),
            key=lambda face_track_id: (
                int(metadata[face_track_id]["start_frame"]),
                int(metadata[face_track_id]["end_frame"]),
                str(face_track_id),
            ),
        )
        for idx, left_id in enumerate(ordered_face_track_ids):
            left_meta = metadata[left_id]
            for right_id in ordered_face_track_ids[idx + 1 :]:
                right_meta = metadata[right_id]
                gap_frames = int(right_meta["start_frame"]) - int(left_meta["end_frame"])
                dominant_left = str(left_meta.get("dominant_track_id") or "")
                dominant_right = str(right_meta.get("dominant_track_id") or "")
                same_dominant_track = bool(dominant_left and dominant_left == dominant_right)
                allowed_gap = stitch_same_track_bonus_gap if same_dominant_track else stitch_max_gap
                if gap_frames < 0:
                    continue
                if gap_frames > allowed_gap:
                    break
                left_emb = left_meta.get("embedding")
                right_emb = right_meta.get("embedding")
                if left_emb is None or right_emb is None:
                    continue
                cos_dist = self._cosine_dist(left_emb, right_emb)
                if cos_dist > stitch_max_cos:
                    continue
                sig_dist = self._tracklet_signature_distance(left_meta["signature"], right_meta["signature"])
                if sig_dist > stitch_max_sig:
                    continue
                if dominant_left and dominant_right and dominant_left != dominant_right:
                    continue
                _union(left_id, right_id)

        if metadata:
            stitched: dict[str, dict] = {}
            for face_track_id, feature in merged.items():
                root_id = _find(face_track_id)
                slot = stitched.setdefault(
                    root_id,
                    {
                        "embeddings": [],
                        "observations": [],
                        "associated_track_counts": defaultdict(int),
                    },
                )
                slot["embeddings"].extend(feature["embeddings"])
                slot["observations"].extend(feature["observations"])
                for tid, count in dict(feature["associated_track_counts"]).items():
                    if str(tid):
                        slot["associated_track_counts"][str(tid)] += int(count)
            merged = stitched

        finalized: dict[str, dict] = {}
        for face_track_id, feature in merged.items():
            emb = None
            if feature["embeddings"]:
                emb = np.mean(np.stack(feature["embeddings"], axis=0), axis=0).astype(np.float32).tolist()
            observations = sorted(
                feature["observations"],
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            associated_counts = {
                str(tid): int(count)
                for tid, count in dict(feature["associated_track_counts"]).items()
                if str(tid)
            }
            dominant_track_id = None
            if associated_counts:
                dominant_track_id = max(associated_counts.items(), key=lambda item: item[1])[0]
            finalized[face_track_id] = {
                "face_track_id": face_track_id,
                "embedding": emb,
                "embedding_source": "face" if emb is not None else "none",
                "embedding_count": len(feature["embeddings"]),
                "face_observations": observations,
                "face_observation_count": len(observations),
                "associated_track_counts": associated_counts,
                "dominant_track_id": dominant_track_id,
            }
        return finalized

    def _derive_track_identity_features_from_face_tracks(
        self,
        face_track_features: dict[str, dict],
    ) -> dict[str, dict]:
        import numpy as np

        per_track: dict[str, dict] = {}
        for _, feature in (face_track_features or {}).items():
            associated_counts = {
                str(tid): int(count)
                for tid, count in dict(feature.get("associated_track_counts", {})).items()
                if str(tid) and int(count) > 0
            }
            if not associated_counts:
                continue
            eligible_track_ids = self._eligible_associated_track_ids(associated_counts)

            embedding = feature.get("embedding")
            emb_arr = None
            if embedding is not None:
                emb_arr = np.asarray(embedding, dtype=np.float32)
                if emb_arr.size == 0:
                    emb_arr = None
            repeat = max(1, int(feature.get("embedding_count", 1)))
            face_track_id = str(feature.get("face_track_id", ""))
            for tid in sorted(eligible_track_ids):
                slot = per_track.setdefault(
                    tid,
                    {
                        "face_vectors": [],
                        "face_observations": [],
                        "face_track_ids": set(),
                    },
                )
                if emb_arr is not None:
                    weight = max(1, associated_counts.get(tid, 1))
                    slot["face_vectors"].extend([emb_arr] * max(repeat, weight))
                if face_track_id:
                    slot["face_track_ids"].add(face_track_id)
            for observation in feature.get("face_observations", []):
                associated_track_id = str(observation.get("associated_track_id", "") or "")
                if associated_track_id not in eligible_track_ids:
                    continue
                obs = dict(observation)
                obs["track_id"] = associated_track_id
                per_track[associated_track_id]["face_observations"].append(obs)

        finalized: dict[str, dict] = {}
        for tid, feature in per_track.items():
            emb = None
            if feature["face_vectors"]:
                emb = np.mean(np.stack(feature["face_vectors"], axis=0), axis=0).astype(np.float32).tolist()
            observations = sorted(
                feature["face_observations"],
                key=lambda obs: (int(obs.get("frame_idx", -1)), -float(obs.get("confidence", 0.0))),
            )
            deduped = []
            seen_frames = set()
            for obs in observations:
                frame_idx = int(obs.get("frame_idx", -1))
                if frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                deduped.append(obs)
            finalized[tid] = {
                "embedding": emb,
                "embedding_source": "face" if emb is not None else "none",
                "embedding_count": len(feature["face_vectors"]),
                "face_observations": deduped,
                "face_observation_count": len(deduped),
                "face_track_ids": sorted(feature["face_track_ids"]),
            }
        return finalized

    def _extract_face_track_features_from_segments(
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
        frame_segments: list[list[tuple[int, list[dict]]]] | None = None,
        segment_futures=None,
    ) -> tuple[dict[str, dict], dict[str, dict], dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        frame_segments = [segment for segment in (frame_segments or []) if segment]
        workers = self._face_pipeline_workers()
        segment_results: list[dict] = []

        if segment_futures is not None:
            for future in as_completed(segment_futures):
                segment_results.append(future.result())
            mode = "staggered"
        elif not frame_segments:
            mode = "disabled"
        elif workers <= 1 or len(frame_segments) <= 1:
            segment_results = [
                self._extract_face_track_features_for_frame_segment(
                    video_path=video_path,
                    frame_items=segment,
                    fps=fps,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    output_frame_width=output_frame_width,
                    output_frame_height=output_frame_height,
                    coord_scale_x=coord_scale_x,
                    coord_scale_y=coord_scale_y,
                )
                for segment in frame_segments
            ]
            mode = "serial"
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._extract_face_track_features_for_frame_segment,
                        video_path=video_path,
                        frame_items=segment,
                        fps=fps,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        output_frame_width=output_frame_width,
                        output_frame_height=output_frame_height,
                        coord_scale_x=coord_scale_x,
                        coord_scale_y=coord_scale_y,
                    )
                    for segment in frame_segments
                ]
                for future in as_completed(futures):
                    segment_results.append(future.result())
            mode = "parallel"

        face_track_features = self._merge_face_track_feature_sets(
            [result.get("face_track_features", {}) for result in segment_results]
        )
        track_identity_features = self._derive_track_identity_features_from_face_tracks(face_track_features)
        metrics = {
            "face_pipeline_mode": mode,
            "face_pipeline_worker_count": workers,
            "face_pipeline_gpu_enabled": self._face_pipeline_uses_gpu(),
            "face_pipeline_segment_count": len(frame_segments) if segment_futures is None else len(segment_futures),
            "face_pipeline_segments_processed": len(segment_results),
            "face_pipeline_track_count": len(track_identity_features),
            "face_track_count": len(face_track_features),
        }
        return track_identity_features, face_track_features, metrics

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

    def _build_visual_detection_ledgers(
        self,
        video_path: str,
        tracks: list[dict],
        frame_to_dets: dict[int, list[dict]] | None = None,
        track_to_dets: dict[str, list[dict]] | None = None,
        track_identity_features: dict[str, dict] | None = None,
    ) -> tuple[list[dict], list[dict], dict]:
        """Build downstream person/face ledgers from canonical face observations."""
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
                            "provenance": observation.get("provenance", "scrfd_fullframe"),
                        }
                if deduped_by_frame:
                    face_ts_by_track[tid] = [
                        deduped_by_frame[frame_idx]
                        for frame_idx in sorted(deduped_by_frame.keys())
                    ]
                    precomputed_frames.update(deduped_by_frame.keys())

        sampled_frames = len(precomputed_frames)
        segment_count = 0

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
                    "provenance": str(ts_objs[0].get("provenance", "scrfd_fullframe")),
                    "timestamped_objects": ts_objs,
                }
            )

        metrics = {
            "face_detection_wallclock_s": round(time.perf_counter() - started_at, 3),
            "face_detection_frame_samples": sampled_frames,
            "face_detection_track_count": len(face_detections),
            "face_detection_segment_count": segment_count,
            "face_detection_worker_count": 0,
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
        import threading
        import nemo.collections.asr as nemo_asr
        import torch
        from insightface import model_zoo
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

        self.gpu_device = torch.device("cuda")
        self._face_runtime_name = "buffalo_l"
        self._face_runtime_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._face_runtime_ctx_id = 0 if torch.cuda.is_available() else -1
        self._face_runtime_det_size = self.__class__._face_detector_input_size()
        self._face_runtime_det_thresh = 0.15
        self._face_detector_input_size = self.__class__._face_detector_input_size()
        self._face_model_root = os.path.expanduser("~/.insightface/models/buffalo_l")
        self._face_detector_model_file = os.path.join(self._face_model_root, "det_10g.onnx")
        self._face_recognizer_model_file = os.path.join(self._face_model_root, "w600k_r50.onnx")
        self._face_runtime_main_thread_ident = threading.get_ident()
        self._face_runtime_local = threading.local()
        self.face_detector = None
        self.face_recognizer = None
        try:
            print("Loading SCRFD face detector into GPU VRAM...")
            self.face_detector = model_zoo.get_model(
                self._face_detector_model_file,
                providers=self._face_runtime_providers,
            )
            self.face_detector.prepare(
                ctx_id=self._face_runtime_ctx_id,
                input_size=self._face_detector_input_size,
                det_thresh=self._face_runtime_det_thresh,
            )
            print("SCRFD face detector ready.")
        except Exception as e:
            print(
                "Warning: SCRFD detector initialization failed "
                f"({type(e).__name__}: {e})"
            )
        try:
            print("Loading ArcFace recognizer into GPU VRAM...")
            self.face_recognizer = model_zoo.get_model(
                self._face_recognizer_model_file,
                providers=self._face_runtime_providers,
            )
            self.face_recognizer.prepare(ctx_id=self._face_runtime_ctx_id)
            print("ArcFace recognizer ready.")
        except Exception as e:
            print(
                "Warning: ArcFace recognizer initialization failed "
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

        return YOLO(YOLO_WEIGHTS_PATH)

    @staticmethod
    def _yolo_device_arg():
        import torch

        return 0 if torch.cuda.is_available() else "cpu"

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
            device=self._yolo_device_arg(),
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

        frame_to_analysis_dets: dict[int, list[dict]] = {}
        for det in analysis_tracks:
            frame_to_analysis_dets.setdefault(int(det.get("frame_idx", -1)), []).append(det)
        track_identity_features, face_track_features, face_pipeline_metrics = self._extract_face_track_features_from_segments(
            video_path=chunk_video_path,
            fps=fps,
            frame_width=width,
            frame_height=height,
            output_frame_width=output_width,
            output_frame_height=output_height,
            coord_scale_x=coord_scale_x,
            coord_scale_y=coord_scale_y,
            frame_segments=self._split_frame_items_into_segments(
                frame_to_analysis_dets,
                self._face_pipeline_segment_frames(),
            ),
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
            "face_track_features": face_track_features,
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
        requested_face_pipeline_workers = self._requested_face_pipeline_workers()
        face_pipeline_workers = self._face_pipeline_workers()
        face_pipeline_start_frame = self._face_pipeline_start_frame()
        pending_face_segment_frames: list[tuple[int, list[dict]]] = []
        face_segment_futures = []
        face_segments_submitted = 0

        print(
            "  Face pipeline config: "
            f"requested_workers={requested_face_pipeline_workers}, "
            f"effective_workers={face_pipeline_workers}, "
            f"gpu={'on' if self._face_pipeline_uses_gpu() else 'off'}, "
            f"start_frame={face_pipeline_start_frame}, "
            f"segment_frames={face_pipeline_segment_frames}"
        )

        results = model.track(
            source=tracking_video_path,
            tracker=tracker_cfg,
            persist=True,
            classes=[0],
            stream=True,
            verbose=False,
            vid_stride=1,
            imgsz=infer_imgsz,
            device=self._yolo_device_arg(),
        )

        tracks = []
        n_boxes = 0
        with ThreadPoolExecutor(max_workers=face_pipeline_workers) as face_pool:
            if self._face_pipeline_uses_gpu():
                self._prewarm_face_runtime_in_pool(face_pool, face_pipeline_workers)
            for frame_idx, r in enumerate(results, start=1):
                frame_face_dets: list[dict] = []
                if r.boxes is None or r.boxes.id is None:
                    if (
                        frame_idx >= face_pipeline_start_frame
                        and frame_idx % face_pipeline_segment_frames == 0
                        and pending_face_segment_frames
                    ):
                        face_segment_futures.append(
                            face_pool.submit(
                                self._extract_face_track_features_for_frame_segment,
                                video_path=video_path,
                                frame_items=list(pending_face_segment_frames),
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
                        pending_face_segment_frames = []
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
                    frame_face_dets.append(dict(out))
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

                if frame_face_dets:
                    pending_face_segment_frames.append((int(frame_idx - 1), frame_face_dets))

                if (
                    frame_idx >= face_pipeline_start_frame
                    and frame_idx % face_pipeline_segment_frames == 0
                    and pending_face_segment_frames
                ):
                    face_segment_futures.append(
                        face_pool.submit(
                            self._extract_face_track_features_for_frame_segment,
                            video_path=tracking_video_path,
                            frame_items=list(pending_face_segment_frames),
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
                    pending_face_segment_frames = []

                if frame_idx % log_every_n_frames == 0:
                    elapsed = max(1e-6, time.time() - started)
                    fps_eff = frame_idx / elapsed
                    pct = (100.0 * frame_idx) / max(1, total_frames)
                    print(
                        "  YOLO progress: "
                        f"{frame_idx}/{total_frames} frames ({pct:.1f}%), "
                        f"{n_boxes} boxes, {fps_eff:.1f} fps"
                    )

            if pending_face_segment_frames:
                face_segment_futures.append(
                    face_pool.submit(
                        self._extract_face_track_features_for_frame_segment,
                        video_path=tracking_video_path,
                        frame_items=list(pending_face_segment_frames),
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
        track_identity_features, face_track_features, face_pipeline_metrics = self._extract_face_track_features_from_segments(
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
        if face_track_features:
            metrics["face_track_features"] = face_track_features
        print(
            "  Face pipeline complete: "
            f"{int(face_pipeline_metrics.get('face_pipeline_segments_processed', 0))}/"
            f"{int(face_pipeline_metrics.get('face_pipeline_segment_count', len(face_segment_futures)))} segments, "
            f"{int(face_pipeline_metrics.get('face_track_count', len(face_track_features or {})))} face tracks, "
            f"{int(face_pipeline_metrics.get('face_pipeline_track_count', len(track_identity_features or {})))} body-linked tracks"
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
        merged_face_track_feature_maps: list[dict[str, dict]] = []
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
            local_face_track_features = chunk_data.get("face_track_features", {}) or {}
            mapped_face_track_features: dict[str, dict] = {}
            for face_track_id, feature in local_face_track_features.items():
                if not isinstance(feature, dict):
                    continue
                mapped_feature = dict(feature)
                associated_counts = {}
                for local_tid, count in dict(feature.get("associated_track_counts", {})).items():
                    gid = local_to_global.get((cidx, str(local_tid)))
                    if not gid:
                        continue
                    associated_counts[gid] = associated_counts.get(gid, 0) + int(count)
                mapped_feature["associated_track_counts"] = associated_counts
                dominant_local_tid = str(feature.get("dominant_track_id", "") or "")
                mapped_feature["dominant_track_id"] = (
                    local_to_global.get((cidx, dominant_local_tid))
                    if dominant_local_tid
                    else None
                )
                mapped_face_track_features[str(face_track_id)] = mapped_feature
            if mapped_face_track_features:
                merged_face_track_feature_maps.append(mapped_face_track_features)
        stitched_track_identity_features = self._merge_track_identity_feature_sets(merged_feature_maps)
        stitched_face_track_features = self._merge_face_track_feature_sets(merged_face_track_feature_maps)
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
        if stitched_face_track_features:
            metrics["face_track_features"] = stitched_face_track_features
            metrics["face_track_count"] = len(stitched_face_track_features)
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
        face_track_features: dict[str, dict] | None = None,
    ) -> list[dict]:
        """Cluster fragmented BoT-SORT track IDs into global person IDs via GPU face embeddings."""
        import os
        import numpy as np
        from sklearn.cluster import DBSCAN

        self._last_cluster_id_map = None
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
        face_track_seeded_tracklets = 0
        face_track_gap_propagated_tracklets = 0
        face_track_raw_clusters = 0
        seed_label_by_tid: dict[str, int] = {}
        seed_embedding_by_tid: dict[str, np.ndarray] = {}

        embeddings = {}  # track_id → 512D face embedding
        fallback_ids = []  # track_ids using non-face fallback embeddings
        face_accept_count = 0
        face_reject_lowq_count = 0
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

        if isinstance(face_track_features, dict) and face_track_features:
            face_track_ids = sorted(
                [
                    str(face_track_id)
                    for face_track_id, feature in face_track_features.items()
                    if isinstance(feature, dict) and feature.get("embedding") is not None
                ]
            )
            if face_track_ids:
                X_face_tracks = np.array(
                    [
                        np.asarray(face_track_features[face_track_id]["embedding"], dtype=np.float32)
                        for face_track_id in face_track_ids
                    ],
                    dtype=np.float32,
                )
                face_track_labels = DBSCAN(
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples,
                    metric="cosine",
                ).fit(X_face_tracks).labels_.astype(int)
                if len(set(face_track_labels) - {-1}) == 0:
                    face_track_labels = np.arange(len(face_track_ids), dtype=int)
                face_track_raw_clusters = len(set(int(lbl) for lbl in face_track_labels if int(lbl) >= 0))
                print(
                    "  Face-track clustering: "
                    f"raw_clusters={face_track_raw_clusters}, tracklets={len(face_track_ids)}"
                )
                cluster_centroids: dict[int, list[np.ndarray]] = {}
                votes_by_tid: dict[str, dict[int, int]] = {}
                for idx, face_track_id in enumerate(face_track_ids):
                    label = int(face_track_labels[idx])
                    if label < 0:
                        continue
                    cluster_centroids.setdefault(label, []).append(X_face_tracks[idx])
                    feature = face_track_features.get(face_track_id, {})
                    eligible_track_ids = self._cluster_seed_track_ids_for_face_track(
                        dict(feature.get("associated_track_counts", {})),
                        tracklets=tracklets,
                    )
                    for tid, count in dict(feature.get("associated_track_counts", {})).items():
                        tid = str(tid)
                        if not tid or tid not in eligible_track_ids:
                            continue
                        votes_by_tid.setdefault(tid, {})
                        votes_by_tid[tid][label] = votes_by_tid[tid].get(label, 0) + int(count)
                centroid_by_label = {
                    int(label): np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
                    for label, vecs in cluster_centroids.items()
                    if vecs
                }
                for tid, vote_map in votes_by_tid.items():
                    if not vote_map:
                        continue
                    sorted_votes = sorted(vote_map.items(), key=lambda item: (-item[1], item[0]))
                    best_label, best_votes = sorted_votes[0]
                    if best_votes <= 0:
                        continue
                    total_votes = max(1, sum(int(votes) for _, votes in sorted_votes))
                    min_seed_share = float(os.getenv("CLYPT_FACE_TRACK_SEED_MIN_SHARE", "0.60"))
                    min_seed_margin = max(1, int(os.getenv("CLYPT_FACE_TRACK_SEED_MIN_MARGIN", "2")))
                    if (best_votes / total_votes) < min_seed_share:
                        continue
                    if len(sorted_votes) > 1 and (best_votes - int(sorted_votes[1][1])) < min_seed_margin:
                        continue
                    seed_label_by_tid[tid] = int(best_label)
                    if best_label in centroid_by_label:
                        seed_embedding_by_tid[tid] = centroid_by_label[best_label]
                face_track_seeded_tracklets = len(seed_label_by_tid)
                if face_track_seeded_tracklets:
                    print(
                        "  Face-track-first seeding: "
                        f"{face_track_seeded_tracklets} tracklets"
                    )

        if not embeddings and not seed_label_by_tid:
            print("  No usable face identity features for clustering")
            self._last_clustering_metrics = {
                "cluster_visible_people_estimate": visible_people_est,
                "overfragmentation_proxy": round(float(len(unique_ids) / max(1, visible_people_est)), 3),
                "accidental_merge_proxy": round(
                    float(max(0, visible_people_est - len(unique_ids)) / max(1, visible_people_est)),
                    3,
                ),
                "covisibility_conflict_rejections": 0,
                "histogram_attach_rejections": 0,
                "face_track_gap_propagated_tracklets": 0,
            }
            self._last_track_identity_features_after_clustering = (
                dict(track_identity_features) if isinstance(track_identity_features, dict) else None
            )
            return tracks

        if precomputed_feature_tracklets:
            print(
                "  Cluster identity features: "
                f"precomputed={precomputed_feature_tracklets}, "
                f"remaining={max(0, len(unique_ids) - precomputed_feature_tracklets - len(seed_label_by_tid))}"
            )

        for tid, emb in seed_embedding_by_tid.items():
            if tid not in embeddings:
                embeddings[tid] = np.asarray(emb, dtype=np.float32)
            if tid in fallback_ids:
                fallback_ids = [fallback_tid for fallback_tid in fallback_ids if fallback_tid != tid]

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

        def _track_boundary_gap_frames(tid_a: str, tid_b: str) -> int:
            dets_a = tracklets.get(tid_a, [])
            dets_b = tracklets.get(tid_b, [])
            if not dets_a or not dets_b:
                return 1_000_000
            a_start = min(int(d.get("frame_idx", -1)) for d in dets_a)
            a_end = max(int(d.get("frame_idx", -1)) for d in dets_a)
            b_start = min(int(d.get("frame_idx", -1)) for d in dets_b)
            b_end = max(int(d.get("frame_idx", -1)) for d in dets_b)
            if a_end < b_start:
                return max(0, b_start - a_end)
            if b_end < a_start:
                return max(0, a_start - b_end)
            return 0

        def _propagate_face_identity_across_short_gaps() -> int:
            max_gap_frames = max(0, int(os.getenv("CLYPT_FACE_TRACK_PROPAGATE_MAX_GAP_FRAMES", "48")))
            max_sig_dist = float(os.getenv("CLYPT_FACE_TRACK_PROPAGATE_MAX_SIG_DIST", "0.18"))
            ambiguity_margin = float(os.getenv("CLYPT_FACE_TRACK_PROPAGATE_AMBIGUITY_MARGIN", "0.08"))
            pending_tids = [tid for tid in unique_ids if tid not in seed_label_by_tid and tid not in embeddings]
            if not pending_tids or not seed_label_by_tid:
                return 0

            propagated = 0
            seed_signature_by_tid = {
                tid: _track_signature(tid)
                for tid in seed_label_by_tid.keys()
                if tid in tracklets
            }
            for tid in pending_tids:
                sig = _track_signature(tid)
                candidates: list[tuple[float, int, float, str]] = []
                for seeded_tid, seeded_label in seed_label_by_tid.items():
                    seeded_sig = seed_signature_by_tid.get(seeded_tid)
                    if seeded_sig is None:
                        continue
                    if self._clusters_conflict_by_visibility(tracklets, [tid], [seeded_tid]):
                        continue
                    if not self._clusters_have_compatible_seat_signature(
                        tracklets,
                        [tid],
                        [seeded_tid],
                        max_signature_distance=max_sig_dist,
                    ):
                        continue
                    gap_frames = _track_boundary_gap_frames(tid, seeded_tid)
                    if gap_frames > max_gap_frames:
                        continue
                    sig_dist = _sig_dist(sig, seeded_sig)
                    if sig_dist > max_sig_dist:
                        continue
                    score = (gap_frames / max(1.0, float(max_gap_frames or 1))) + sig_dist
                    candidates.append((score, gap_frames, sig_dist, seeded_tid))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                best_score, _, _, best_seeded_tid = candidates[0]
                best_label = seed_label_by_tid[best_seeded_tid]
                if len(candidates) > 1:
                    competing = [
                        cand
                        for cand in candidates[1:]
                        if seed_label_by_tid[cand[3]] != best_label
                    ]
                    if competing and (competing[0][0] - best_score) < ambiguity_margin:
                        continue
                seed_label_by_tid[tid] = int(best_label)
                source_embedding = seed_embedding_by_tid.get(best_seeded_tid)
                if source_embedding is None:
                    source_embedding = embeddings.get(best_seeded_tid)
                if source_embedding is not None:
                    seed_embedding_by_tid[tid] = np.asarray(source_embedding, dtype=np.float32)
                propagated += 1
            return propagated

        if face_track_seeded_tracklets:
            face_track_gap_propagated_tracklets = _propagate_face_identity_across_short_gaps()
            if face_track_gap_propagated_tracklets:
                print(
                    "  Face-track gap propagation: "
                    f"{face_track_gap_propagated_tracklets} tracklets"
                )

        # Separate face embeddings from signature-only fallbacks. Tracklets that
        # were seeded from face tracks still need to participate in the later
        # face-cluster cleanup pass; otherwise the raw face-track labels bypass
        # the stronger merge / dedupe logic entirely.
        tid_order_all = sorted(set(unique_ids) | set(embeddings.keys()) | set(seed_label_by_tid.keys()))
        face_tids = [tid for tid in tid_order_all if tid in embeddings and tid not in fallback_ids]
        hist_tids = [tid for tid in tid_order_all if tid not in face_tids]
        print(
            "  Face quality gate: "
            f"accepted={face_accept_count}, rejected_lowq={face_reject_lowq_count}, "
            f"min_det_score={float(extraction_cfg['cluster_face_min_det_score']):.2f}"
        )
        print(f"  Face encodings: {len(face_tids)}, signature fallbacks: {len(hist_tids)}")

        id_map: dict[str, str] = {}
        n_face_clusters = 0
        clusters_after_hist_attach = 0
        new_globals_from_unattached_hist = 0

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
            print(f"  Face clusters after refinement: {n_face_clusters}")

        # Attach histogram-only tracklets to nearest face cluster by spatial signature.
        if hist_tids:
            if n_face_clusters > 0:
                face_label_by_tid = {
                    tid: int(id_map[tid].split("_")[-1])
                    for tid in sorted(set(face_tids))
                    if tid in id_map
                }

                reassigned_hist = 0
                unattached_hist_group_count = 0
                next_hist_label = n_face_clusters
                hist_groups = self._cluster_signature_only_tracklets(
                    track_ids=hist_tids,
                    tracklets=tracklets,
                    base_max_sig=histogram_attach_max_sig,
                )
                for group in hist_groups:
                    selected_label = self._choose_signature_attachment_label_for_group(
                        tids=group,
                        tracklets=tracklets,
                        face_label_by_tid=face_label_by_tid,
                        histogram_attach_max_sig=histogram_attach_max_sig,
                    )
                    label_votes: dict[int, int] = {}
                    if selected_label is None:
                        for tid in group:
                            best_label = self._choose_signature_attachment_label(
                                tid=tid,
                                tracklets=tracklets,
                                face_label_by_tid=face_label_by_tid,
                                histogram_attach_max_sig=histogram_attach_max_sig,
                            )
                            if best_label is None:
                                continue
                            label_votes[int(best_label)] = label_votes.get(int(best_label), 0) + 1
                        if label_votes:
                            ranked_votes = sorted(label_votes.items(), key=lambda item: (-item[1], item[0]))
                            best_label, best_votes = ranked_votes[0]
                            if len(ranked_votes) == 1 or best_votes > ranked_votes[1][1]:
                                selected_label = int(best_label)
                    if selected_label is None:
                        assigned_label = int(next_hist_label)
                        next_hist_label += 1
                        new_globals_from_unattached_hist += 1
                        unattached_hist_group_count += 1
                        histogram_attach_rejections += len(group)
                        for tid in group:
                            id_map[tid] = f"Global_Person_{assigned_label}"
                        continue
                    for tid in group:
                        id_map[tid] = f"Global_Person_{int(selected_label)}"
                        reassigned_hist += 1
                if reassigned_hist:
                    print(
                        "  Histogram tracklets attached to face clusters: "
                        f"{reassigned_hist} (max_sig={histogram_attach_max_sig:.2f})"
                    )
                if new_globals_from_unattached_hist:
                    print(
                        "  Histogram tracklets left unattached: "
                        f"{new_globals_from_unattached_hist} "
                        f"({unattached_hist_group_count} groups)"
                    )
                clusters_after_hist_attach = len(set(id_map.values()))
            else:
                # Worst-case fallback: keep them deterministic and separate.
                for i, tid in enumerate(sorted(hist_tids)):
                    id_map[tid] = f"Global_Person_{i}"
                clusters_after_hist_attach = len(set(id_map.values()))
        else:
            clusters_after_hist_attach = len(set(id_map.values()))

        label_by_tid = {
            tid: int(str(mapped_id).split("_")[-1])
            for tid, mapped_id in id_map.items()
            if str(mapped_id).startswith("Global_Person_")
        }
        clusters_before_repair = len(set(label_by_tid.values()))
        collision_metrics_before_repair = self._same_identity_frame_collision_metrics(tracklets, label_by_tid)
        if self._should_skip_cluster_repair(
            face_cluster_count=n_face_clusters,
            clusters_before_repair=clusters_before_repair,
            visible_people_est=visible_people_est,
            anchored_track_count=len(set(face_tids)),
        ):
            repair_metrics = {
                "repaired_cluster_count": 0,
                "repaired_tracklet_count": 0,
                "repaired_conflict_pair_count": 0,
                "repair_skipped": 1,
            }
            clusters_after_repair = clusters_before_repair
            collision_metrics_after_repair = collision_metrics_before_repair
            print(
                "  Repair stage skipped: "
                f"face_refined={n_face_clusters}, "
                f"before_repair={clusters_before_repair}, "
                f"visible_people_est={visible_people_est}"
            )
        else:
            label_by_tid, repair_metrics = self._repair_covisible_cluster_merges(
                tracklets,
                label_by_tid,
                anchored_tids=set(face_tids),
            )
            clusters_after_repair = len(set(label_by_tid.values()))
            collision_metrics_after_repair = self._same_identity_frame_collision_metrics(tracklets, label_by_tid)
        id_map = {tid: f"Global_Person_{int(label)}" for tid, label in label_by_tid.items()}
        print(
            "  Cluster stages: "
            f"face_refined={n_face_clusters}, "
            f"after_hist={clusters_after_hist_attach}, "
            f"before_repair={clusters_before_repair}, "
            f"after_repair={clusters_after_repair}"
        )
        print(
            "  Same-identity frame collisions: "
            f"before={collision_metrics_before_repair['same_identity_frame_collision_pairs']} pairs/"
            f"{collision_metrics_before_repair['same_identity_frame_collision_frames']} frames, "
            f"after={collision_metrics_after_repair['same_identity_frame_collision_pairs']} pairs/"
            f"{collision_metrics_after_repair['same_identity_frame_collision_frames']} frames"
        )
        if repair_metrics.get("repaired_cluster_count", 0) or repair_metrics.get("repaired_tracklet_count", 0):
            print(
                "  Repair splits: "
                f"clusters={int(repair_metrics.get('repaired_cluster_count', 0))}, "
                f"tracklets={int(repair_metrics.get('repaired_tracklet_count', 0))}, "
                f"conflict_pairs={int(repair_metrics.get('repaired_conflict_pair_count', 0))}"
            )

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
            "face_track_raw_clusters": face_track_raw_clusters,
            "face_track_seeded_tracklets": face_track_seeded_tracklets,
            "face_track_gap_propagated_tracklets": face_track_gap_propagated_tracklets,
            "cluster_cross_merge_cos_threshold": cross_cluster_merge_base_cos,
            "cluster_hist_attach_max_sig": histogram_attach_max_sig,
            "same_identity_frame_collision_pairs_before_repair": collision_metrics_before_repair["same_identity_frame_collision_pairs"],
            "same_identity_frame_collision_frames_before_repair": collision_metrics_before_repair["same_identity_frame_collision_frames"],
            "same_identity_labels_with_collisions_before_repair": collision_metrics_before_repair["same_identity_labels_with_collisions"],
            "same_identity_frame_collision_pairs_after_repair": collision_metrics_after_repair["same_identity_frame_collision_pairs"],
            "same_identity_frame_collision_frames_after_repair": collision_metrics_after_repair["same_identity_frame_collision_frames"],
            "same_identity_labels_with_collisions_after_repair": collision_metrics_after_repair["same_identity_labels_with_collisions"],
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

        self._last_cluster_id_map = dict(id_map)

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
        track_id_remap: dict[str, str] | None = None,
        force_dense: bool = False,
    ) -> list[dict] | None:
        """Run LR-ASD inference and map words to visual track IDs."""
        self._last_speaker_candidate_debug = []
        self._last_audio_turn_bindings = []

        if self.lrasd_model is None or self.lrasd_loss_av is None:
            print("  LR-ASD unavailable; falling back to heuristic binder.")
            return None
        if not words or not tracks:
            return []

        import concurrent.futures as cf
        import cv2
        import os
        import numpy as np
        import torch
        from decord import VideoReader, cpu
        from bisect import bisect_left
        from collections import Counter

        protected_unknown_key = "_speaker_binding_protected_unknown"

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

        remap = {
            str(old_tid): str(new_tid)
            for old_tid, new_tid in dict(track_id_remap or {}).items()
            if str(old_tid) and str(new_tid)
        }

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
        lrasd_metrics = {
            "canonical_face_stream_boxes": int(len(canonical_face_boxes)),
            "canonical_face_stream_needed_boxes": int(needed_face_boxes),
            "canonical_face_stream_coverage": round(float(canonical_coverage), 4),
        }
        existing_binding_metrics = getattr(self, "_last_speaker_binding_metrics", None)
        if isinstance(existing_binding_metrics, dict):
            existing_binding_metrics.update(lrasd_metrics)
        else:
            self._last_speaker_binding_metrics = dict(lrasd_metrics)
        print(
            "  Canonical face stream: "
            f"{len(canonical_face_boxes)}/{needed_face_boxes}={canonical_coverage:.1%} coverage"
        )
        track_quality_by_tid = self._build_speaker_binding_track_quality(
            track_to_dets,
            frame_width=int(binding_meta.get("width", 0) or 0) or 1,
            frame_height=int(binding_meta.get("height", 0) or 0) or 1,
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
                if bw > 1e-6 and bh > 1e-6:
                    head_x1 = int(round(cx - (0.28 * bw)))
                    head_x2 = int(round(cx + (0.28 * bw)))
                    head_y1 = int(round(cy - (0.48 * bh)))
                    head_y2 = int(round(cy - (0.02 * bh)))
                    head_x1 = max(0, min(fw - 1, head_x1))
                    head_y1 = max(0, min(fh - 1, head_y1))
                    head_x2 = max(head_x1 + 1, min(fw, head_x2))
                    head_y2 = max(head_y1 + 1, min(fh, head_y2))
                    face_crop = frame[head_y1:head_y2, head_x1:head_x2]
                    if face_crop is not None and face_crop.size > 0:
                        crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                        anchor = {
                            "x_offset": (float(head_x1) - cx) / bw,
                            "y_offset": (float(head_y1) - cy) / bh,
                            "w_ratio": float(head_x2 - head_x1) / bw,
                            "h_ratio": float(head_y2 - head_y1) / bh,
                        }

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
        min_body_fallback_score = 0.62
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

        motion_score: dict[tuple[str, int], float] = {}
        for tid, dets in track_to_dets.items():
            ordered = sorted(
                dets,
                key=lambda det: int(det.get("frame_idx", -1)),
            )
            if not ordered:
                continue
            for idx, cur in enumerate(ordered):
                prev = ordered[idx - 1] if idx > 0 else cur
                nxt = ordered[idx + 1] if idx + 1 < len(ordered) else cur
                dx = abs(float(nxt.get("x_center", 0.0)) - float(prev.get("x_center", 0.0)))
                dy = abs(float(nxt.get("y_center", 0.0)) - float(prev.get("y_center", 0.0)))
                dh = abs(float(nxt.get("height", 0.0)) - float(prev.get("height", 0.0)))
                h = max(float(cur.get("height", 1.0)), 1.0)
                motion = (0.45 * dx + 0.8 * dy + 1.1 * dh) / h
                motion_score[(str(tid), int(cur.get("frame_idx", -1)))] = float(motion)

        audio_turn_bindings: list[dict] = []
        raw_audio_turns = self._serialize_audio_speaker_turns(
            (analysis_context or {}).get("audio_speaker_turns")
        )
        audio_speaker_turns = [dict(turn) for turn in raw_audio_turns]

        def _clear_word_assignment(word: dict):
            word["speaker_track_id"] = None
            word["speaker_tag"] = "unknown"
            word["speaker_local_track_id"] = None
            word["speaker_local_tag"] = "unknown"
            word.pop(protected_unknown_key, None)

        def _mark_protected_unknown(word: dict):
            _clear_word_assignment(word)
            word[protected_unknown_key] = True

        def _clear_protected_unknown_markers(existing_words: list[dict]):
            for word in existing_words:
                word.pop(protected_unknown_key, None)

        assigned = 0
        audio_prior_abstentions = 0
        words_with_frame = 0
        words_with_dets = 0
        words_with_scored_candidate = 0
        local_candidate_evidence: list[dict] = []
        word_candidate_rows: list[dict] = []
        for w in words:
            mid_ms = (int(w["start_time_ms"]) + int(w["end_time_ms"])) // 2
            target_fi = int(round((mid_ms / 1000.0) * fps))
            fi = _nearest_frame_idx(target_fi, max_gap=word_match_max_gap)
            if fi is None:
                _clear_word_assignment(w)
                continue
            words_with_frame += 1

            dets = frame_to_dets.get(fi, [])
            if not dets:
                _clear_word_assignment(w)
                continue
            words_with_dets += 1

            best_by_track: dict[str, dict] = {}
            for d in dets:
                tid = str(d["track_id"])
                cur = best_by_track.get(tid)
                if cur is None or float(d.get("confidence", 0.0)) > float(cur.get("confidence", 0.0)):
                    best_by_track[tid] = d

            peer_motion = {
                tid: float(motion_score.get((tid, fi), 0.0))
                for tid in best_by_track.keys()
            }
            max_motion = max(peer_motion.values()) if peer_motion else 0.0
            scored_candidates: list[dict] = []
            has_scored_candidate = False
            for tid, d in best_by_track.items():
                motion_rank = (
                    float(peer_motion.get(tid, 0.0) / max_motion)
                    if max_motion > 1e-6
                    else 0.0
                )
                track_quality = float(track_quality_by_tid.get(tid, {}).get("track_quality", 0.0))
                body_meta = self._score_speaker_binding_body_candidate(
                    det=d,
                    frame_dets=dets,
                    frame_width=int(binding_meta.get("width", 0) or 0) or 1,
                    frame_height=int(binding_meta.get("height", 0) or 0) or 1,
                    track_quality=track_quality,
                    motion_rank=motion_rank,
                )
                s = _score_near(tid, fi)
                has_face_box = (tid, fi) in canonical_face_boxes
                conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
                body_prior = float(body_meta["body_prior"])
                face_bonus = 1.0 if has_face_box else 0.0
                if s is not None:
                    has_scored_candidate = True
                    total = (
                        0.76 * float(s)
                        + 0.18 * body_prior
                        + 0.04 * face_bonus
                        + 0.02 * conf
                    )
                else:
                    total = (
                        0.72 * body_prior
                        + 0.18 * face_bonus
                        + 0.10 * conf
                    )
                mapped_tid = remap.get(tid, tid)
                scored_candidates.append(
                    {
                        "total": float(total),
                        "prob": None if s is None else float(s),
                        "local_tid": tid,
                        "track_id": mapped_tid,
                        "body_prior": body_prior,
                        "track_quality": track_quality,
                        "detection_quality": float(body_meta["detection_quality"]),
                        "confidence": conf,
                        "hard_reject": bool(body_meta.get("hard_reject", False)),
                    }
                )
            if has_scored_candidate:
                words_with_scored_candidate += 1

            if any(not candidate["hard_reject"] for candidate in scored_candidates):
                scored_candidates = [
                    candidate for candidate in scored_candidates
                    if not candidate["hard_reject"]
                ]

            scored_candidates.sort(
                key=lambda item: (
                    item["total"],
                    item["prob"] if item["prob"] is not None else -1.0,
                    item["body_prior"],
                    item["confidence"],
                ),
                reverse=True,
            )
            local_candidate_evidence.append(
                {
                    "start_time_ms": int(w["start_time_ms"]),
                    "end_time_ms": int(w["end_time_ms"]),
                    "candidates": [dict(candidate) for candidate in scored_candidates],
                }
            )
            word_candidate_rows.append(
                {
                    "word": w,
                    "scored_candidates": [dict(candidate) for candidate in scored_candidates],
                }
            )

        if audio_speaker_turns and local_candidate_evidence:
            audio_turn_bindings = self._bind_audio_turns_to_local_tracks(
                audio_speaker_turns,
                local_candidate_evidence,
            )
        self._last_audio_turn_bindings = [dict(binding) for binding in audio_turn_bindings]

        def _active_audio_turn(word: dict) -> dict | None:
            if not audio_speaker_turns:
                return None
            word_start_ms = int(word.get("start_time_ms", 0) or 0)
            word_end_ms = int(word.get("end_time_ms", word_start_ms) or word_start_ms)
            if word_end_ms < word_start_ms:
                word_start_ms, word_end_ms = word_end_ms, word_start_ms
            word_mid_ms = (word_start_ms + word_end_ms) // 2
            best_turn = None
            best_overlap_ms = 0
            for turn in audio_speaker_turns:
                turn_start_ms = int(turn.get("start_time_ms", 0) or 0)
                turn_end_ms = int(turn.get("end_time_ms", turn_start_ms) or turn_start_ms)
                if turn_end_ms < turn_start_ms:
                    turn_start_ms, turn_end_ms = turn_end_ms, turn_start_ms
                if turn_start_ms <= word_mid_ms <= turn_end_ms:
                    return turn
                overlap_ms = min(word_end_ms, turn_end_ms) - max(word_start_ms, turn_start_ms)
                if overlap_ms > best_overlap_ms:
                    best_overlap_ms = overlap_ms
                    best_turn = turn
            return best_turn if best_overlap_ms > 0 else None

        def _active_audio_turn_binding(
            word: dict,
            *,
            include_ambiguous: bool = False,
            require_local_track: bool = True,
        ) -> dict | None:
            if not audio_turn_bindings:
                return None
            word_start_ms = int(word.get("start_time_ms", 0) or 0)
            word_end_ms = int(word.get("end_time_ms", word_start_ms) or word_start_ms)
            if word_end_ms < word_start_ms:
                word_start_ms, word_end_ms = word_end_ms, word_start_ms
            word_mid_ms = (word_start_ms + word_end_ms) // 2
            best_binding = None
            best_overlap_ms = 0
            for binding in audio_turn_bindings:
                local_track_id = binding.get("local_track_id")
                if require_local_track and local_track_id in (None, ""):
                    continue
                if not include_ambiguous and bool(binding.get("ambiguous", False)):
                    continue
                binding_start_ms = int(binding.get("start_time_ms", 0) or 0)
                binding_end_ms = int(binding.get("end_time_ms", binding_start_ms) or binding_start_ms)
                if binding_end_ms < binding_start_ms:
                    binding_start_ms, binding_end_ms = binding_end_ms, binding_start_ms
                if binding_start_ms <= word_mid_ms <= binding_end_ms:
                    return binding
                overlap_ms = min(word_end_ms, binding_end_ms) - max(word_start_ms, binding_start_ms)
                if overlap_ms > best_overlap_ms:
                    best_overlap_ms = overlap_ms
                    best_binding = binding
            return best_binding if best_overlap_ms > 0 else None

        def _turn_is_high_ambiguity(turn: dict | None) -> bool:
            if not isinstance(turn, dict):
                return False
            if bool(turn.get("overlap", False)):
                return True
            return turn.get("exclusive") is False

        audio_prior_bonus = max(0.02, min(0.06, 4.0 * min_assignment_margin))
        audio_prior_margin_threshold = max(0.02, min(0.08, 6.0 * min_assignment_margin))
        strong_visual_margin_threshold = max(0.08, min(0.18, 10.0 * min_assignment_margin))
        strong_visual_prob_threshold = max(0.32, min_lrasd_prob + 0.12)

        def _second_best_total_for_local_track(
            candidates: list[dict],
            winning_local_tid: str,
        ) -> float | None:
            for candidate in candidates[1:]:
                if str(candidate.get("local_tid", "")) != winning_local_tid:
                    return float(candidate["total"])
            return None

        speaker_candidate_debug: list[dict] = []

        def _append_speaker_candidate_debug(
            *,
            word: dict,
            scored_candidates: list[dict],
            active_turn: dict | None,
            debug_turn_binding: dict | None,
            chosen_track_id: str | None,
            chosen_local_track_id: str | None,
            decision_source: str,
            ambiguous: bool,
            top_margin: float | None,
        ) -> None:
            active_audio_local_track_id = None
            if isinstance(debug_turn_binding, dict):
                raw_debug_local_track_id = (
                    debug_turn_binding.get("local_track_id")
                    or debug_turn_binding.get("clean_local_track_id")
                )
                if raw_debug_local_track_id not in (None, ""):
                    active_audio_local_track_id = str(raw_debug_local_track_id)
            speaker_candidate_debug.append(
                {
                    "word": str(word.get("text") or word.get("word") or ""),
                    "start_time_ms": int(word.get("start_time_ms", 0) or 0),
                    "end_time_ms": int(word.get("end_time_ms", 0) or 0),
                    "active_audio_speaker_id": (
                        str(active_turn.get("speaker_id"))
                        if isinstance(active_turn, dict) and active_turn.get("speaker_id") not in (None, "")
                        else None
                    ),
                    "active_audio_local_track_id": active_audio_local_track_id,
                    "chosen_track_id": chosen_track_id,
                    "chosen_local_track_id": chosen_local_track_id,
                    "decision_source": decision_source,
                    "ambiguous": bool(
                        ambiguous
                        or (
                            isinstance(debug_turn_binding, dict)
                            and debug_turn_binding.get("ambiguous", False)
                        )
                    ),
                    "top_1_top_2_margin": top_margin,
                    "candidates": [
                        {
                            "local_track_id": str(candidate.get("local_tid", "")),
                            "track_id": str(candidate.get("track_id", "")),
                            "blended_score": float(candidate["total"]),
                            "asd_probability": (
                                None
                                if candidate.get("prob") is None
                                else float(candidate["prob"])
                            ),
                            "body_prior": float(candidate["body_prior"]),
                            "detection_confidence": float(candidate["confidence"]),
                        }
                        for candidate in scored_candidates[:3]
                    ],
                }
            )

        for row in word_candidate_rows:
            w = row["word"]
            scored_candidates = [dict(candidate) for candidate in row["scored_candidates"]]
            active_turn = _active_audio_turn(w)
            debug_turn_binding = _active_audio_turn_binding(
                w,
                include_ambiguous=True,
                require_local_track=False,
            )
            active_turn_binding = _active_audio_turn_binding(w)
            debug_audio_local_track_id = None
            if isinstance(debug_turn_binding, dict):
                raw_debug_local_track_id = (
                    debug_turn_binding.get("local_track_id")
                    or debug_turn_binding.get("clean_local_track_id")
                )
                if raw_debug_local_track_id not in (None, ""):
                    debug_audio_local_track_id = str(raw_debug_local_track_id)
            preserve_unknown_for_high_ambiguity_turn = bool(
                active_turn is not None
                and debug_turn_binding is not None
                and _turn_is_high_ambiguity(active_turn)
            )
            if not scored_candidates:
                _append_speaker_candidate_debug(
                    word=w,
                    scored_candidates=[],
                    active_turn=active_turn,
                    debug_turn_binding=debug_turn_binding,
                    chosen_track_id=None,
                    chosen_local_track_id=None,
                    decision_source="unknown",
                    ambiguous=False,
                    top_margin=None,
                )
                _clear_word_assignment(w)
                continue
            if (
                active_turn is not None
                and debug_turn_binding is not None
                and (
                    debug_audio_local_track_id is None
                    or preserve_unknown_for_high_ambiguity_turn
                )
            ):
                audio_prior_abstentions += 1
                _append_speaker_candidate_debug(
                    word=w,
                    scored_candidates=scored_candidates,
                    active_turn=active_turn,
                    debug_turn_binding=debug_turn_binding,
                    chosen_track_id=None,
                    chosen_local_track_id=None,
                    decision_source="unknown",
                    ambiguous=bool(debug_turn_binding.get("ambiguous", False)),
                    top_margin=None,
                )
                _mark_protected_unknown(w)
                continue

            best_candidate = scored_candidates[0]
            best_total = float(best_candidate["total"])
            best_prob = best_candidate["prob"]
            best_body = float(best_candidate["body_prior"])
            second_total = _second_best_total_for_local_track(
                scored_candidates,
                str(best_candidate.get("local_tid", "")),
            )
            visual_margin = (
                float(best_total)
                if second_total is None
                else float(best_total - second_total)
            )
            strong_visual_winner = bool(
                best_prob is not None
                and float(best_prob) >= strong_visual_prob_threshold
                and (
                    second_total is None
                    or visual_margin >= strong_visual_margin_threshold
                    or best_body >= 0.80
                )
            )

            audio_prior_applied = False
            if active_turn_binding is not None and not strong_visual_winner:
                prior_local_tid = str(active_turn_binding.get("local_track_id", "") or "")
                prior_candidate = next(
                    (
                        candidate
                        for candidate in scored_candidates
                        if str(candidate.get("local_tid", "")) == prior_local_tid
                    ),
                    None,
                )
                if prior_candidate is None:
                    audio_prior_abstentions += 1
                    _append_speaker_candidate_debug(
                        word=w,
                        scored_candidates=scored_candidates,
                        active_turn=active_turn,
                        debug_turn_binding=debug_turn_binding,
                        chosen_track_id=None,
                        chosen_local_track_id=None,
                        decision_source="unknown",
                        ambiguous=False,
                        top_margin=visual_margin,
                    )
                    _mark_protected_unknown(w)
                    continue
                if second_total is not None and visual_margin <= audio_prior_margin_threshold:
                    prior_strength = max(
                        0.0,
                        min(
                            1.0,
                            float(active_turn_binding.get("support_ratio", 0.0) or 0.0)
                            + float(active_turn_binding.get("winning_margin", 0.0) or 0.0),
                        ),
                    )
                    prior_candidate["total"] = float(prior_candidate["total"]) + (
                        audio_prior_bonus * prior_strength
                    )
                    audio_prior_applied = True
                    scored_candidates.sort(
                        key=lambda item: (
                            item["total"],
                            item["prob"] if item["prob"] is not None else -1.0,
                            item["body_prior"],
                            item["confidence"],
                        ),
                        reverse=True,
                    )

            best_candidate = scored_candidates[0]
            best_total = float(best_candidate["total"])
            best_prob = best_candidate["prob"]
            best_tid = str(best_candidate["track_id"])
            best_body = float(best_candidate["body_prior"])
            second_total = _second_best_total_for_local_track(
                scored_candidates,
                str(best_candidate.get("local_tid", "")),
            )
            if best_prob is not None:
                confident_pick = bool(
                    float(best_prob) >= min_lrasd_prob
                    and (
                        second_total is None
                        or (best_total - second_total) >= min_assignment_margin
                        or best_body >= 0.80
                        or audio_prior_applied
                    )
                )
            else:
                confident_pick = bool(
                    best_body >= min_body_fallback_score
                    and (
                        second_total is None
                        or (best_total - second_total) >= (0.5 * min_assignment_margin)
                    )
                )

            top_margin = (
                None
                if second_total is None
                else float(best_total - second_total)
            )
            decision_source = (
                "audio_boosted_visual"
                if audio_prior_applied and confident_pick
                else "visual"
                if confident_pick
                else "unknown"
            )
            decision_ambiguous = bool(
                isinstance(active_turn_binding, dict)
                and active_turn_binding.get("ambiguous", False)
            )
            if not decision_ambiguous and not confident_pick and top_margin is not None:
                decision_ambiguous = bool(top_margin < min_assignment_margin)

            if best_tid is not None and confident_pick:
                w["speaker_track_id"] = best_tid
                w["speaker_tag"] = best_tid
                w["speaker_local_track_id"] = str(best_candidate["local_tid"])
                w["speaker_local_tag"] = str(best_candidate["local_tid"])
                assigned += 1
            else:
                _clear_word_assignment(w)

            _append_speaker_candidate_debug(
                word=w,
                scored_candidates=scored_candidates,
                active_turn=active_turn,
                debug_turn_binding=debug_turn_binding,
                chosen_track_id=str(best_tid) if confident_pick and best_tid not in (None, "") else None,
                chosen_local_track_id=(
                    str(best_candidate["local_tid"])
                    if confident_pick and best_candidate.get("local_tid") not in (None, "")
                    else None
                ),
                decision_source=decision_source,
                ambiguous=decision_ambiguous,
                top_margin=top_margin,
            )

        self._last_speaker_candidate_debug = [
            {
                **{
                    key: value
                    for key, value in entry.items()
                    if key != "candidates"
                },
                "candidates": [dict(candidate) for candidate in entry.get("candidates", [])],
            }
            for entry in speaker_candidate_debug
        ]

        # Smooth local flicker.
        seq = [w.get("speaker_track_id") for w in words]
        smoothed = seq[:]
        local_seq = [w.get("speaker_local_track_id") for w in words]
        local_smoothed = local_seq[:]
        win = 2
        for i in range(len(seq)):
            if bool(words[i].get(protected_unknown_key, False)):
                smoothed[i] = None
                local_smoothed[i] = None
                continue
            lo = max(0, i - win)
            hi = min(len(seq), i + win + 1)
            neigh = [t for t in seq[lo:hi] if t]
            if not neigh:
                local_neigh = [t for t in local_seq[lo:hi] if t]
                if local_neigh:
                    local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
                    if local_cnt >= 2:
                        local_smoothed[i] = local_major
                continue
            major, cnt = Counter(neigh).most_common(1)[0]
            if cnt >= 2:
                smoothed[i] = major
            local_neigh = [t for t in local_seq[lo:hi] if t]
            if local_neigh:
                local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
                if local_cnt >= 2:
                    local_smoothed[i] = local_major
        for w, tid, local_tid in zip(words, smoothed, local_smoothed):
            if bool(w.get(protected_unknown_key, False)):
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"
                w["speaker_local_track_id"] = None
                w["speaker_local_tag"] = "unknown"
                continue
            w["speaker_track_id"] = tid
            w["speaker_tag"] = tid or "unknown"
            w["speaker_local_track_id"] = local_tid
            w["speaker_local_tag"] = local_tid or "unknown"

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
            if audio_prior_abstentions > 0 and assigned == 0:
                print(
                    "  LR-ASD preserved unknown assignments for off-screen audio turns; "
                    "skipping heuristic fallback."
                )
                _clear_protected_unknown_markers(words)
                return []
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
        _clear_protected_unknown_markers(words)
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
        track_id_remap: dict[str, str] | None = None,
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
        remap = {
            str(old_tid): str(new_tid)
            for old_tid, new_tid in dict(track_id_remap or {}).items()
            if str(old_tid) and str(new_tid)
        }

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
            local_tid = _assign_word(w)
            tid = None
            if local_tid is not None:
                tid = remap.get(str(local_tid), str(local_tid))
            if tid is not None:
                w["speaker_track_id"] = tid
                # Compatibility for downstream phases that still read speaker_tag.
                w["speaker_tag"] = tid
                w["speaker_local_track_id"] = str(local_tid)
                w["speaker_local_tag"] = str(local_tid)
                assigned += 1
            else:
                w["speaker_track_id"] = None
                w["speaker_tag"] = "unknown"
                w["speaker_local_track_id"] = None
                w["speaker_local_tag"] = "unknown"

        # Smooth flicker: majority vote in a small local word window.
        track_seq = [w.get("speaker_track_id") for w in words]
        smoothed = track_seq[:]
        local_track_seq = [w.get("speaker_local_track_id") for w in words]
        local_smoothed = local_track_seq[:]
        win = 2
        for i in range(len(track_seq)):
            lo = max(0, i - win)
            hi = min(len(track_seq), i + win + 1)
            neigh = [t for t in track_seq[lo:hi] if t]
            if not neigh:
                local_neigh = [t for t in local_track_seq[lo:hi] if t]
                if local_neigh:
                    local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
                    if local_cnt >= 2:
                        local_smoothed[i] = local_major
                continue
            major, cnt = Counter(neigh).most_common(1)[0]
            if cnt >= 2:
                smoothed[i] = major
            local_neigh = [t for t in local_track_seq[lo:hi] if t]
            if local_neigh:
                local_major, local_cnt = Counter(local_neigh).most_common(1)[0]
                if local_cnt >= 2:
                    local_smoothed[i] = local_major

        for w, tid, local_tid in zip(words, smoothed, local_smoothed):
            w["speaker_track_id"] = tid
            w["speaker_tag"] = tid or "unknown"
            w["speaker_local_track_id"] = local_tid
            w["speaker_local_tag"] = local_tid or "unknown"

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
        track_id_remap: dict[str, str] | None = None,
    ) -> list[dict]:
        """Bind words to track IDs using LR-ASD, with heuristic fallback."""
        protected_unknown_key = "_speaker_binding_protected_unknown"
        self._last_audio_turn_bindings = []
        self._last_speaker_candidate_debug = []

        def _restore_protected_unknowns(existing_words: list[dict]) -> bool:
            restored_any = False
            for word in existing_words:
                if not bool(word.get(protected_unknown_key, False)):
                    continue
                word["speaker_track_id"] = None
                word["speaker_tag"] = "unknown"
                word["speaker_local_track_id"] = None
                word["speaker_local_tag"] = "unknown"
                restored_any = True
            return restored_any

        def _clear_protected_unknown_markers(existing_words: list[dict]):
            for word in existing_words:
                word.pop(protected_unknown_key, None)

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
                track_id_remap=track_id_remap,
            )
            speaker_metrics["lrasd_wallclock_s"] = round(
                time.perf_counter() - lrasd_started_at,
                3,
            )
            lrasd_aux_metrics = getattr(self, "_last_speaker_binding_metrics", None)
            if isinstance(lrasd_aux_metrics, dict):
                speaker_metrics.update(lrasd_aux_metrics)
            if bindings is not None:
                if _restore_protected_unknowns(words):
                    bindings = self._build_bindings_from_word_track_field(
                        words,
                        field_name="speaker_track_id",
                    )
                _clear_protected_unknown_markers(words)
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
            track_id_remap=track_id_remap,
        )
        if _restore_protected_unknowns(words):
            bindings = self._build_bindings_from_word_track_field(
                words,
                field_name="speaker_track_id",
            )
        _clear_protected_unknown_markers(words)
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
        face_track_features = metrics.pop("face_track_features", None)
        analysis_context = metrics.get("analysis_context") if isinstance(metrics.get("analysis_context"), dict) else None
        metrics["schema_pass_rate"] = self._tracking_contract_pass_rate(tracks)
        self._validate_tracking_contract(tracks)
        self._enforce_rollout_gates(metrics)
        metrics["track_identity_feature_track_count"] = len(track_identity_features or {})
        metrics["identity_track_count_before_clustering"] = len(
            {str(track.get("track_id", "")) for track in tracks if str(track.get("track_id", ""))}
        )

        precluster_tracks = [dict(track) for track in tracks]
        _, track_to_dets = self._build_track_indexes(tracks)
        precluster_frame_to_dets, precluster_track_to_dets = self._build_track_indexes(precluster_tracks)

        # Step 3: Global tracklet clustering
        print("[Phase 1] Step 3/4: Clustering tracklets into global IDs...")
        cluster_started_at = time.perf_counter()
        tracks = self._cluster_tracklets(
            video_path,
            tracks,
            track_to_dets=track_to_dets,
            track_identity_features=track_identity_features,
            face_track_features=face_track_features,
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
        cluster_id_remap = getattr(self, "_last_cluster_id_map", None)
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
        audio_speaker_turns = self._run_audio_diarization(audio_path)
        last_audio_diarization_metrics = getattr(self, "_last_audio_diarization_metrics", None)
        if isinstance(last_audio_diarization_metrics, dict):
            metrics.update(last_audio_diarization_metrics)
        speaker_binding_started_at = time.perf_counter()
        binding_analysis_context = dict(analysis_context) if isinstance(analysis_context, dict) else {}
        binding_analysis_context["audio_speaker_turns"] = audio_speaker_turns
        self._last_speaker_candidate_debug = []
        speaker_bindings = self._run_speaker_binding(
            video_path,
            audio_path,
            precluster_tracks,
            words,
            frame_to_dets=precluster_frame_to_dets,
            track_to_dets=precluster_track_to_dets,
            track_identity_features=track_identity_features,
            analysis_context=binding_analysis_context,
            track_id_remap=cluster_id_remap,
        )
        speaker_follow_bindings = self._build_speaker_follow_bindings(speaker_bindings)
        speaker_bindings_local: list[dict] = []
        speaker_follow_bindings_local: list[dict] = []
        audio_speaker_local_track_map: list[dict] = []
        if self._local_clip_bindings_enabled():
            speaker_bindings_local = self._build_bindings_from_word_track_field(
                words,
                field_name="speaker_local_track_id",
            )
            if not speaker_bindings_local:
                speaker_bindings_local = self._project_bindings_to_local_track_space(
                    speaker_bindings,
                    cluster_id_remap,
                )
            speaker_follow_bindings_local = self._build_speaker_follow_bindings(
                speaker_bindings_local
            )
            audio_speaker_local_track_map = self._build_audio_speaker_local_track_map(
                getattr(self, "_last_audio_turn_bindings", []),
            )
        speaker_binding_elapsed_s = time.perf_counter() - speaker_binding_started_at
        metrics["speaker_binding_wallclock_s"] = round(speaker_binding_elapsed_s, 3)
        metrics["speaker_follow_binding_segment_count"] = len(speaker_follow_bindings)
        metrics["speaker_binding_local_segment_count"] = len(speaker_bindings_local)
        metrics["speaker_follow_binding_local_segment_count"] = len(speaker_follow_bindings_local)
        last_binding_metrics = getattr(self, "_last_speaker_binding_metrics", None)
        if isinstance(last_binding_metrics, dict):
            metrics.update(last_binding_metrics)
        assigned_words = sum(1 for word in words if word.get("speaker_track_id"))
        metrics["speaker_binding_assignment_ratio"] = round(
            float(assigned_words / max(1, len(words))),
            3,
        )
        metrics.update(self._speaker_remap_collision_metrics(words))
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
        if self._local_clip_bindings_enabled():
            phase_1_visual["tracks_local"] = [dict(track) for track in precluster_tracks]

        phase_1_audio = {
            "source_audio": youtube_url,
            "words": words,
            "speaker_bindings": speaker_bindings,
            "audio_speaker_turns": audio_speaker_turns,
            "speaker_candidate_debug": getattr(self, "_last_speaker_candidate_debug", []),
            "speaker_follow_bindings": speaker_follow_bindings,
        }
        if self._local_clip_bindings_enabled():
            phase_1_audio["speaker_bindings_local"] = speaker_bindings_local
            phase_1_audio["speaker_follow_bindings_local"] = speaker_follow_bindings_local
            phase_1_audio["audio_speaker_local_track_map"] = audio_speaker_local_track_map

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
