from __future__ import annotations

import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .models import LongFormAsrOutputs, ShardAsrResult, ShardPlan
from .speaker_stitch import stitch_global_speakers


_DEFAULT_SINGLE_PASS_MAX_MINUTES = 40
_DEFAULT_TWO_SHARD_MAX_MINUTES = 80
_DEFAULT_FOUR_SHARD_MAX_MINUTES = 160


def plan_audio_shards(
    *,
    duration_s: float,
    single_pass_max_minutes: int = _DEFAULT_SINGLE_PASS_MAX_MINUTES,
    two_shard_max_minutes: int = _DEFAULT_TWO_SHARD_MAX_MINUTES,
    four_shard_max_minutes: int = _DEFAULT_FOUR_SHARD_MAX_MINUTES,
    max_shards: int = 4,
) -> list[ShardPlan]:
    if duration_s <= 0:
        raise ValueError("audio duration must be positive")
    if max_shards < 1 or max_shards > 4:
        raise ValueError("VIBEVOICE_LONGFORM_MAX_SHARDS must be between 1 and 4")
    if single_pass_max_minutes <= 0:
        raise ValueError("VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES must be positive")
    if two_shard_max_minutes < single_pass_max_minutes:
        raise ValueError(
            "VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES must be >= "
            "VIBEVOICE_LONGFORM_SINGLE_PASS_MAX_MINUTES"
        )
    if four_shard_max_minutes < two_shard_max_minutes:
        raise ValueError(
            "VIBEVOICE_LONGFORM_FOUR_SHARD_MAX_MINUTES must be >= "
            "VIBEVOICE_LONGFORM_TWO_SHARD_MAX_MINUTES"
        )

    if duration_s <= single_pass_max_minutes * 60:
        return [ShardPlan(index=0, shard_count=1, start_s=0.0, end_s=float(duration_s))]
    if duration_s <= two_shard_max_minutes * 60:
        if max_shards < 2:
            raise ValueError(
                "audio duration requires 2 shards but "
                f"VIBEVOICE_LONGFORM_MAX_SHARDS={max_shards}"
            )
        return _equal_shards(duration_s=duration_s, shard_count=2)
    if duration_s <= four_shard_max_minutes * 60:
        if max_shards < 4:
            raise ValueError(
                "audio duration requires 4 shards but "
                f"VIBEVOICE_LONGFORM_MAX_SHARDS={max_shards}"
            )
        return _equal_shards(duration_s=duration_s, shard_count=4)
    raise ValueError(
        f"audio duration exceeds the supported long-form limit of {four_shard_max_minutes} minutes"
    )


def _equal_shards(*, duration_s: float, shard_count: int) -> list[ShardPlan]:
    shard_len = float(duration_s) / float(shard_count)
    shards: list[ShardPlan] = []
    for index in range(shard_count):
        start_s = shard_len * index
        end_s = float(duration_s) if index == shard_count - 1 else shard_len * (index + 1)
        shards.append(
            ShardPlan(
                index=index,
                shard_count=shard_count,
                start_s=start_s,
                end_s=end_s,
            )
        )
    return shards


def merge_shard_turns(shard_results: list[ShardAsrResult]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for shard_result in sorted(shard_results, key=lambda result: result.plan.index):
        shard_offset = shard_result.plan.start_s
        for turn in shard_result.turns:
            start_s = float(turn["Start"])
            end_s = float(turn["End"])
            if end_s < start_s:
                raise ValueError(
                    f"invalid turn timing in shard {shard_result.plan.index}: end before start"
                )
            merged.append(
                {
                    "Speaker": int(turn["Speaker"]),
                    "Start": shard_offset + start_s,
                    "End": shard_offset + end_s,
                    "Content": str(turn["Content"]),
                }
            )
    merged.sort(key=lambda turn: (float(turn["Start"]), float(turn["End"])))
    return merged


__all__ = [
    "merge_shard_turns",
    "plan_audio_shards",
    "run_longform_vibevoice_asr",
]


def run_longform_vibevoice_asr(
    *,
    audio_path: str | Path,
    canonical_audio_gcs_uri: str,
    run_id: str,
    vibevoice_provider: Any,
    storage_client: Any,
    speaker_verifier: Any,
    duration_s: float,
    single_pass_max_minutes: int = _DEFAULT_SINGLE_PASS_MAX_MINUTES,
    two_shard_max_minutes: int = _DEFAULT_TWO_SHARD_MAX_MINUTES,
    four_shard_max_minutes: int = _DEFAULT_FOUR_SHARD_MAX_MINUTES,
    max_shards: int = 4,
    threshold: float = 0.85,
    representative_clip_min_s: float = 15.0,
    representative_clip_max_s: float = 30.0,
    extract_shard_audio: Any = None,
) -> LongFormAsrOutputs:
    source_audio_path = Path(audio_path)
    shards = plan_audio_shards(
        duration_s=duration_s,
        single_pass_max_minutes=single_pass_max_minutes,
        two_shard_max_minutes=two_shard_max_minutes,
        four_shard_max_minutes=four_shard_max_minutes,
        max_shards=max_shards,
    )
    if len(shards) <= 1:
        raise ValueError("run_longform_vibevoice_asr requires a multi-shard duration")

    stage_events: list[dict[str, Any]] = [
        {
            "stage_name": "vibevoice_longform_plan",
            "status": "succeeded",
            "duration_ms": 0.0,
            "metadata": {
                "canonical_audio_gcs_uri": canonical_audio_gcs_uri,
                "shard_count": len(shards),
                "duration_s": float(duration_s),
            },
            "error_payload": None,
        }
    ]

    shard_writer = extract_shard_audio or _extract_shard_audio
    shard_dir = Path(tempfile.mkdtemp(prefix=f"vibevoice-shards-{run_id}-", dir=str(source_audio_path.parent)))

    planned_shards: list[tuple[ShardPlan, Path, str]] = []
    for shard in shards:
        shard_name = f"shard_{shard.index:03d}_of_{shard.shard_count:03d}.wav"
        shard_audio_path = shard_dir / shard_name
        shard_writer(
            source_audio_path=source_audio_path,
            output_audio_path=shard_audio_path,
            start_s=shard.start_s,
            end_s=shard.end_s,
        )
        shard_object_name = f"phase1/{run_id}/vibevoice_shards/{shard_name}"
        shard_gcs_uri = storage_client.upload_file(
            local_path=shard_audio_path,
            object_name=shard_object_name,
        )
        planned_shards.append((shard, shard_audio_path, shard_gcs_uri))

    def _run_shard(plan: ShardPlan, shard_audio_path: Path, shard_gcs_uri: str) -> ShardAsrResult:
        t0 = time.perf_counter()
        turns = vibevoice_provider.run(
            audio_path=shard_audio_path,
            audio_gcs_uri=shard_gcs_uri,
        )
        stage_events.append(
            {
                "stage_name": "vibevoice_shard_asr",
                "status": "succeeded",
                "duration_ms": (time.perf_counter() - t0) * 1000.0,
                "metadata": {
                    "shard_index": plan.index,
                    "shard_count": plan.shard_count,
                    "start_s": plan.start_s,
                    "end_s": plan.end_s,
                    "duration_s": plan.duration_s,
                    "turn_count": len(turns),
                },
                "error_payload": None,
            }
        )
        return ShardAsrResult(
            plan=plan,
            turns=list(turns),
            audio_path=shard_audio_path,
            audio_gcs_uri=shard_gcs_uri,
            representative_clips=_extract_representative_clips(
                shard_audio_path=shard_audio_path,
                plan=plan,
                turns=list(turns),
                min_clip_s=representative_clip_min_s,
                max_clip_s=representative_clip_max_s,
                extract_shard_audio=shard_writer,
            ),
        )

    with ThreadPoolExecutor(max_workers=len(planned_shards)) as pool:
        futures = [
            pool.submit(_run_shard, shard, shard_audio_path, shard_gcs_uri)
            for shard, shard_audio_path, shard_gcs_uri in planned_shards
        ]
        shard_results = [future.result() for future in futures]

    t_stitch = time.perf_counter()
    stitched_results = stitch_global_speakers(
        shard_results,
        speaker_verifier,
        threshold=threshold,
    )
    stage_events.append(
        {
            "stage_name": "vibevoice_speaker_stitch",
            "status": "succeeded",
            "duration_ms": (time.perf_counter() - t_stitch) * 1000.0,
            "metadata": {
                "shard_count": len(stitched_results),
            },
            "error_payload": None,
        }
    )

    t_merge = time.perf_counter()
    merged_turns = merge_shard_turns(stitched_results)
    stage_events.append(
        {
            "stage_name": "vibevoice_longform_merge",
            "status": "succeeded",
            "duration_ms": (time.perf_counter() - t_merge) * 1000.0,
            "metadata": {
                "turn_count": len(merged_turns),
            },
            "error_payload": None,
        }
    )
    stage_events.sort(key=lambda event: (event["stage_name"] != "vibevoice_longform_plan", event["stage_name"]))
    return LongFormAsrOutputs(
        turns=merged_turns,
        stage_events=stage_events,
        shard_results=stitched_results,
    )


def _extract_shard_audio(
    *,
    source_audio_path: Path,
    output_audio_path: Path,
    start_s: float,
    end_s: float,
) -> None:
    duration_s = end_s - start_s
    if duration_s <= 0:
        raise ValueError("shard duration must be positive")
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-t",
            str(duration_s),
            "-i",
            str(source_audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _extract_representative_clips(
    *,
    shard_audio_path: Path,
    plan: ShardPlan,
    turns: list[dict[str, Any]],
    min_clip_s: float,
    max_clip_s: float,
    extract_shard_audio: Any,
) -> dict[int, Path]:
    clips: dict[int, Path] = {}
    clip_dir = shard_audio_path.parent / f"{shard_audio_path.stem}_speaker_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for speaker_id in _speaker_ids(turns):
        speaker_turns = [turn for turn in turns if int(turn["Speaker"]) == speaker_id]
        start_s, end_s = _representative_window(
            turns=speaker_turns,
            shard_duration_s=plan.duration_s,
            min_clip_s=min_clip_s,
            max_clip_s=max_clip_s,
        )
        clip_path = clip_dir / f"speaker_{speaker_id}.wav"
        extract_shard_audio(
            source_audio_path=shard_audio_path,
            output_audio_path=clip_path,
            start_s=start_s,
            end_s=end_s,
        )
        clips[speaker_id] = clip_path
    return clips


def _representative_window(
    *,
    turns: list[dict[str, Any]],
    shard_duration_s: float,
    min_clip_s: float,
    max_clip_s: float,
) -> tuple[float, float]:
    longest_turn = max(
        turns,
        key=lambda turn: float(turn["End"]) - float(turn["Start"]),
    )
    start_s = float(longest_turn["Start"])
    end_s = float(longest_turn["End"])
    duration_s = max(0.0, end_s - start_s)
    if duration_s > max_clip_s:
        return start_s, start_s + max_clip_s
    if duration_s >= min_clip_s:
        return start_s, end_s

    missing = min_clip_s - duration_s
    pad_left = missing / 2.0
    pad_right = missing - pad_left
    clip_start = max(0.0, start_s - pad_left)
    clip_end = min(shard_duration_s, end_s + pad_right)
    if clip_end - clip_start < min_clip_s:
        if clip_start == 0.0:
            clip_end = min(shard_duration_s, clip_start + min_clip_s)
        elif clip_end == shard_duration_s:
            clip_start = max(0.0, clip_end - min_clip_s)
    return clip_start, clip_end


def _speaker_ids(turns: list[dict[str, Any]]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for turn in turns:
        speaker_id = int(turn["Speaker"])
        if speaker_id in seen:
            continue
        seen.add(speaker_id)
        ordered.append(speaker_id)
    return ordered
