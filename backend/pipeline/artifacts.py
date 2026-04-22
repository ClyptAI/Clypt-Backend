from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class V31RunPaths:
    run_id: str
    root: Path

    @property
    def timeline_dir(self) -> Path:
        return self.root / self.run_id / "timeline"

    @property
    def semantics_dir(self) -> Path:
        return self.root / self.run_id / "semantics"

    @property
    def graph_dir(self) -> Path:
        return self.root / self.run_id / "graph"

    @property
    def candidates_dir(self) -> Path:
        return self.root / self.run_id / "candidates"

    @property
    def render_dir(self) -> Path:
        return self.root / self.run_id / "render"

    @property
    def captions_dir(self) -> Path:
        return self.render_dir / "captions"

    @property
    def canonical_timeline(self) -> Path:
        return self.timeline_dir / "canonical_timeline.json"

    @property
    def speech_emotion_timeline(self) -> Path:
        return self.timeline_dir / "speech_emotion_timeline.json"

    @property
    def audio_event_timeline(self) -> Path:
        return self.timeline_dir / "audio_event_timeline.json"

    @property
    def shot_tracklet_index(self) -> Path:
        return self.timeline_dir / "shot_tracklet_index.json"

    @property
    def tracklet_geometry(self) -> Path:
        return self.timeline_dir / "tracklet_geometry.json"

    @property
    def semantic_graph_nodes(self) -> Path:
        return self.semantics_dir / "semantic_graph_nodes.json"

    @property
    def semantic_graph_edges(self) -> Path:
        return self.graph_dir / "semantic_graph_edges.json"

    @property
    def clip_candidates(self) -> Path:
        return self.candidates_dir / "clip_candidates.json"

    @property
    def source_context(self) -> Path:
        return self.render_dir / "source_context.json"

    @property
    def caption_plan(self) -> Path:
        return self.render_dir / "caption_plan.json"

    @property
    def publish_metadata(self) -> Path:
        return self.render_dir / "publish_metadata.json"

    @property
    def render_plan(self) -> Path:
        return self.render_dir / "render_plan.json"

    @property
    def turn_neighborhoods_debug(self) -> Path:
        return self.semantics_dir / "turn_neighborhoods_debug.json"

    @property
    def merge_debug(self) -> Path:
        return self.semantics_dir / "merge_debug.json"

    @property
    def classification_debug(self) -> Path:
        return self.semantics_dir / "classification_debug.json"

    @property
    def retrieval_prompts_debug(self) -> Path:
        return self.candidates_dir / "retrieval_prompts_debug.json"

    @property
    def seed_nodes_debug(self) -> Path:
        return self.candidates_dir / "seed_nodes_debug.json"

    @property
    def local_subgraphs_debug(self) -> Path:
        return self.candidates_dir / "local_subgraphs_debug.json"

    @property
    def candidate_dedup_debug(self) -> Path:
        return self.candidates_dir / "candidate_dedup_debug.json"

    @property
    def phase_4_summary(self) -> Path:
        return self.candidates_dir / "phase_4_summary.json"

    def captions_ass(self, clip_id: str) -> Path:
        return self.captions_dir / f"{clip_id}.ass"

    def ensure_dirs(self) -> None:
        for path in (
            self.timeline_dir,
            self.semantics_dir,
            self.graph_dir,
            self.candidates_dir,
            self.render_dir,
            self.captions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "run_root": str(self.root / self.run_id),
            "canonical_timeline": str(self.canonical_timeline),
            "speech_emotion_timeline": str(self.speech_emotion_timeline),
            "audio_event_timeline": str(self.audio_event_timeline),
            "shot_tracklet_index": str(self.shot_tracklet_index),
            "tracklet_geometry": str(self.tracklet_geometry),
            "semantic_graph_nodes": str(self.semantic_graph_nodes),
            "semantic_graph_edges": str(self.semantic_graph_edges),
            "clip_candidates": str(self.clip_candidates),
            "source_context": str(self.source_context),
            "caption_plan": str(self.caption_plan),
            "publish_metadata": str(self.publish_metadata),
            "render_plan": str(self.render_plan),
            "captions_dir": str(self.captions_dir),
        }

    def existing_artifact_paths(self) -> dict[str, str]:
        artifact_paths: dict[str, str] = {}
        for key, raw_path in self.to_dict().items():
            path = Path(raw_path)
            if path.exists():
                artifact_paths[key] = str(path)
        if self.captions_dir.exists():
            for ass_path in sorted(self.captions_dir.glob("*.ass")):
                artifact_paths[f"captions_{ass_path.stem}.ass"] = str(ass_path)
        return artifact_paths


def build_run_paths(*, output_root: Path, run_id: str) -> V31RunPaths:
    paths = V31RunPaths(run_id=run_id, root=output_root)
    paths.ensure_dirs()
    return paths


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "V31RunPaths",
    "build_run_paths",
    "load_json",
    "save_json",
]
