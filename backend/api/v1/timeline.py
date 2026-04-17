"""Timeline bundle endpoint — assembles Phase 1 artifacts into TimelineBundle."""

from __future__ import annotations

import json
import string
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.pipeline.artifacts import V31RunPaths
from backend.repository.phase14_repository import Phase14Repository

from .deps import get_artifact_root, get_repo
from .schemas import (
    TimelineAudioEvent,
    TimelineBundle,
    TimelineEmotionSegment,
    TimelineShot,
    TimelineShotTracklets,
    TimelineSpeaker,
    TimelineSpeakerTurn,
)

router = APIRouter(prefix="/runs/{run_id}", tags=["timeline"])


def _load_artifact(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/timeline", response_model=TimelineBundle)
def get_timeline(
    run_id: str,
    repo: Phase14Repository = Depends(get_repo),
    artifact_root: Path = Depends(get_artifact_root),
) -> TimelineBundle:
    paths = V31RunPaths(run_id=run_id, root=artifact_root)

    # Load Phase 1 artifacts
    canonical_raw = _load_artifact(paths.canonical_timeline)
    emotion_raw = _load_artifact(paths.speech_emotion_timeline)
    audio_raw = _load_artifact(paths.audio_event_timeline)
    tracklet_raw = _load_artifact(paths.shot_tracklet_index)

    # Also get timeline turns from the repository as a fallback
    turns_from_repo = repo.list_timeline_turns(run_id=run_id)

    if canonical_raw is None and not turns_from_repo:
        raise HTTPException(
            status_code=404,
            detail="no timeline data found for this run",
        )

    # Build emotion lookup: turn_id -> emotion event
    emotion_by_turn: dict[str, dict] = {}
    if emotion_raw and isinstance(emotion_raw, dict):
        for ev in emotion_raw.get("events", []):
            emotion_by_turn[ev["turn_id"]] = ev

    # Build speakers from canonical timeline turns (or repo turns)
    speakers_map: dict[str, list[TimelineSpeakerTurn]] = defaultdict(list)

    if canonical_raw and isinstance(canonical_raw, dict):
        for turn in canonical_raw.get("turns", []):
            turn_id = turn["turn_id"]
            speaker_id = turn["speaker_id"]
            emotion = emotion_by_turn.get(turn_id, {})
            per_class = emotion.get("per_class_scores", {})
            primary_label = emotion.get("primary_emotion_label", "neutral")
            primary_score = emotion.get("primary_emotion_score", 0.0)

            # Build secondary emotions (top 3 excluding primary)
            secondary = sorted(
                [{"label": k, "score": v} for k, v in per_class.items() if k != primary_label],
                key=lambda x: x["score"],
                reverse=True,
            )[:3]

            speakers_map[speaker_id].append(TimelineSpeakerTurn(
                turn_id=turn_id,
                speaker_id=speaker_id,
                start_ms=turn["start_ms"],
                end_ms=turn["end_ms"],
                transcript_text=turn["transcript_text"],
                emotion_primary=primary_label,
                emotion_score=primary_score,
                emotion_secondary=secondary,
            ))
    else:
        # Fallback to repo turns
        for turn in turns_from_repo:
            emotion = emotion_by_turn.get(turn.turn_id, {})
            primary_label = emotion.get("primary_emotion_label", "neutral")
            primary_score = emotion.get("primary_emotion_score", 0.0)
            speakers_map[turn.speaker_id].append(TimelineSpeakerTurn(
                turn_id=turn.turn_id,
                speaker_id=turn.speaker_id,
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                transcript_text=turn.transcript_text,
                emotion_primary=primary_label,
                emotion_score=primary_score,
                emotion_secondary=[],
            ))

    speakers = [
        TimelineSpeaker(
            speaker_id=sid,
            display_name=f"Speaker {sid}",
            turns=sorted(turns, key=lambda t: t.start_ms),
        )
        for sid, turns in sorted(speakers_map.items())
    ]

    # Compute duration from all turns
    all_end_ms = [t.end_ms for speaker in speakers for t in speaker.turns]
    duration_ms = max(all_end_ms) if all_end_ms else 0

    # Build emotion segments from emotion timeline
    emotions: list[TimelineEmotionSegment] = []
    if emotion_raw and isinstance(emotion_raw, dict):
        # Create segments from turns with their primary emotion
        for ev in emotion_raw.get("events", []):
            turn_id = ev["turn_id"]
            # Find the turn's time range from canonical
            for speaker in speakers:
                for t in speaker.turns:
                    if t.turn_id == turn_id:
                        emotions.append(TimelineEmotionSegment(
                            start_ms=t.start_ms,
                            end_ms=t.end_ms,
                            label=ev["primary_emotion_label"],
                        ))
                        break

    # Build audio events
    audio_events: list[TimelineAudioEvent] = []
    if audio_raw and isinstance(audio_raw, dict):
        for ev in audio_raw.get("events", []):
            audio_events.append(TimelineAudioEvent(
                start_ms=ev["start_ms"],
                end_ms=ev["end_ms"],
                label=ev["event_label"],
                confidence=ev.get("confidence", 0.0) or 0.0,
            ))

    # Build shots and tracklets
    shots: list[TimelineShot] = []
    shot_tracklets: list[TimelineShotTracklets] = []
    if tracklet_raw and isinstance(tracklet_raw, dict):
        shots_map: dict[str, dict] = {}
        shot_tracklet_map: dict[str, list[str]] = defaultdict(list)
        letters = list(string.ascii_uppercase)
        tracklet_letter_idx = 0

        for td in tracklet_raw.get("tracklets", []):
            shot_id = td["shot_id"]
            if shot_id not in shots_map:
                shots_map[shot_id] = {
                    "shot_id": shot_id,
                    "start_ms": td["start_ms"],
                    "end_ms": td["end_ms"],
                }
            else:
                shots_map[shot_id]["start_ms"] = min(shots_map[shot_id]["start_ms"], td["start_ms"])
                shots_map[shot_id]["end_ms"] = max(shots_map[shot_id]["end_ms"], td["end_ms"])

            letter = letters[tracklet_letter_idx % len(letters)]
            tracklet_letter_idx += 1
            shot_tracklet_map[shot_id].append(letter)

        for shot_id, info in sorted(shots_map.items(), key=lambda x: x[1]["start_ms"]):
            shots.append(TimelineShot(**info))
            shot_tracklets.append(TimelineShotTracklets(
                shot_id=shot_id,
                start_ms=info["start_ms"],
                end_ms=info["end_ms"],
                tracklet_letters=shot_tracklet_map[shot_id],
            ))

    return TimelineBundle(
        duration_ms=duration_ms,
        shots=shots,
        shot_tracklets=shot_tracklets,
        speakers=speakers,
        emotions=emotions,
        audio_events=audio_events,
    )
