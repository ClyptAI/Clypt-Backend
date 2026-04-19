from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import ShardAsrResult


SpeakerKey = tuple[int, int]


def stitch_global_speakers(
    shard_results: list[ShardAsrResult],
    verifier: Any,
    *,
    threshold: float,
) -> list[ShardAsrResult]:
    normalized = sorted(shard_results, key=lambda result: result.plan.index)
    parent: dict[SpeakerKey, SpeakerKey] = {}

    def _find(key: SpeakerKey) -> SpeakerKey:
        root = parent.setdefault(key, key)
        if root != key:
            root = _find(root)
            parent[key] = root
        return root

    def _union(left: SpeakerKey, right: SpeakerKey) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for shard_result in normalized:
        for speaker_id in _speaker_ids(shard_result.turns):
            _find((shard_result.plan.index, speaker_id))

    for left_result, right_result in zip(normalized, normalized[1:]):
        candidates: list[tuple[float, SpeakerKey, SpeakerKey]] = []
        for left_speaker in _speaker_ids(left_result.turns):
            for right_speaker in _speaker_ids(right_result.turns):
                similarity = _similarity(
                    verifier,
                    left_result=left_result,
                    left_speaker=left_speaker,
                    right_result=right_result,
                    right_speaker=right_speaker,
                )
                if similarity >= threshold:
                    candidates.append(
                        (
                            similarity,
                            (left_result.plan.index, left_speaker),
                            (right_result.plan.index, right_speaker),
                        )
                    )

        matched_left: set[SpeakerKey] = set()
        matched_right: set[SpeakerKey] = set()
        for _score, left_key, right_key in sorted(candidates, reverse=True):
            if left_key in matched_left or right_key in matched_right:
                continue
            _union(left_key, right_key)
            matched_left.add(left_key)
            matched_right.add(right_key)

    first_seen_roots = _global_first_seen_roots(normalized, _find)
    global_id_by_root = {root: index for index, root in enumerate(first_seen_roots)}

    stitched: list[ShardAsrResult] = []
    for shard_result in normalized:
        rewritten_turns: list[dict[str, Any]] = []
        for turn in shard_result.turns:
            root = _find((shard_result.plan.index, int(turn["Speaker"])))
            rewritten_turns.append(
                {
                    **turn,
                    "Speaker": global_id_by_root[root],
                }
            )
        stitched.append(
            ShardAsrResult(
                plan=shard_result.plan,
                turns=rewritten_turns,
                audio_path=shard_result.audio_path,
                audio_gcs_uri=shard_result.audio_gcs_uri,
                representative_clips=dict(shard_result.representative_clips),
            )
        )
    return stitched


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


def _global_first_seen_roots(
    shard_results: list[ShardAsrResult],
    find_fn: Callable[[SpeakerKey], SpeakerKey],
) -> list[SpeakerKey]:
    first_seen: list[tuple[float, SpeakerKey]] = []
    seen_roots: set[SpeakerKey] = set()
    for shard_result in shard_results:
        shard_offset = shard_result.plan.start_s
        for turn in shard_result.turns:
            speaker_key = (shard_result.plan.index, int(turn["Speaker"]))
            root = find_fn(speaker_key)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            first_seen.append((shard_offset + float(turn["Start"]), root))
    first_seen.sort(key=lambda item: item[0])
    return [root for _timestamp, root in first_seen]


def _similarity(
    verifier: Any,
    *,
    left_result: ShardAsrResult,
    left_speaker: int,
    right_result: ShardAsrResult,
    right_speaker: int,
) -> float:
    left_clip = left_result.representative_clips.get(left_speaker)
    right_clip = right_result.representative_clips.get(right_speaker)
    if left_clip is not None and right_clip is not None and hasattr(verifier, "similarity_paths"):
        return float(verifier.similarity_paths(left_clip, right_clip))
    return float(
        verifier.similarity(
            (left_result.plan.index, left_speaker),
            (right_result.plan.index, right_speaker),
        )
    )


__all__ = [
    "stitch_global_speakers",
]
