"""In-memory repository seeded with realistic sample data for local dev.

Start with:  python -m backend.api.v1.app --dev
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from backend.repository.models import (
    ClipCandidateRecord,
    PhaseMetricRecord,
    PhaseSubstepRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    TimelineTurnRecord,
)
from backend.repository.phase14_repository import Phase14Repository

NOW = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)


class InMemoryDevRepository(Phase14Repository):
    """Fully functional in-memory repository for local development."""

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}
        self._nodes: dict[str, list[SemanticNodeRecord]] = {}
        self._edges: dict[str, list[SemanticEdgeRecord]] = {}
        self._candidates: dict[str, list[ClipCandidateRecord]] = {}
        self._turns: dict[str, list[TimelineTurnRecord]] = {}
        self._metrics: dict[str, list[PhaseMetricRecord]] = {}
        self._substeps: dict[str, list[PhaseSubstepRecord]] = {}

    # ── Runs ──────────────────────────────────────────────────────────────

    def upsert_run(self, record):
        self._runs[record.run_id] = record
        return record

    def get_run(self, run_id):
        return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return list(self._runs.values())

    def delete_run(self, *, run_id):
        self._runs.pop(run_id, None)
        self._nodes.pop(run_id, None)
        self._edges.pop(run_id, None)
        self._candidates.pop(run_id, None)
        self._turns.pop(run_id, None)
        self._metrics.pop(run_id, None)

    # ── Timeline ──────────────────────────────────────────────────────────

    def write_timeline_turns(self, *, run_id, turns):
        self._turns[run_id] = list(turns)

    def list_timeline_turns(self, *, run_id):
        return self._turns.get(run_id, [])

    # ── Nodes / Edges ─────────────────────────────────────────────────────

    def write_nodes(self, *, run_id, nodes):
        self._nodes[run_id] = list(nodes)

    def list_nodes(self, *, run_id):
        return self._nodes.get(run_id, [])

    def write_edges(self, *, run_id, edges):
        self._edges[run_id] = list(edges)

    def list_edges(self, *, run_id):
        return self._edges.get(run_id, [])

    # ── Candidates ────────────────────────────────────────────────────────

    def write_candidates(self, *, run_id, candidates):
        self._candidates[run_id] = list(candidates)

    def list_candidates(self, *, run_id):
        return self._candidates.get(run_id, [])

    # ── Phase metrics ─────────────────────────────────────────────────────

    def write_phase_metric(self, record):
        self._metrics.setdefault(record.run_id, []).append(record)
        return record

    def list_phase_metrics(self, *, run_id):
        return self._metrics.get(run_id, [])

    def write_phase_substeps(self, *, run_id, substeps):
        self._substeps.setdefault(run_id, []).extend(substeps)

    def list_phase_substeps(self, *, run_id, phase_name=None):
        items = self._substeps.get(run_id, [])
        if phase_name:
            return [s for s in items if s.phase_name == phase_name]
        return items

    # ── Stubs (not needed for frontend dev) ───────────────────────────────

    def write_external_signals(self, *, run_id, signals): pass
    def list_external_signals(self, *, run_id): return []
    def write_external_signal_clusters(self, *, run_id, clusters): pass
    def list_external_signal_clusters(self, *, run_id): return []
    def write_node_signal_links(self, *, run_id, links): pass
    def list_node_signal_links(self, *, run_id): return []
    def write_candidate_signal_links(self, *, run_id, links): pass
    def list_candidate_signal_links(self, *, run_id): return []
    def write_prompt_source_links(self, *, run_id, links): pass
    def list_prompt_source_links(self, *, run_id): return []
    def write_subgraph_provenance(self, *, run_id, provenance): pass
    def list_subgraph_provenance(self, *, run_id): return []
    def upsert_phase24_job(self, record): return record
    def get_phase24_job(self, run_id): return None
    def acquire_phase24_job_lease(self, **kwargs): return {}


def build_seeded_dev_repo() -> InMemoryDevRepository:
    """Create an in-memory repo pre-loaded with realistic sample data."""
    repo = InMemoryDevRepository()
    run_id = "run_dev_001"

    # ── Run ───────────────────────────────────────────────────────────────

    repo.upsert_run(RunRecord(
        run_id=run_id,
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status="PHASE24_DONE",
        created_at=NOW,
        updated_at=NOW,
        metadata={"display_name": "Joe Rogan x Flagrant - Dev Demo"},
    ))

    # ── Phase metrics ─────────────────────────────────────────────────────

    for phase_name, dur_ms in [("phase1", 45200), ("phase2", 18300), ("phase3", 12100), ("phase4", 9800)]:
        repo.write_phase_metric(PhaseMetricRecord(
            run_id=run_id,
            phase_name=phase_name,
            status="succeeded",
            started_at=NOW,
            duration_ms=dur_ms,
        ))

    # ── Turns (6 speakers, 18 turns) ─────────────────────────────────────

    speakers = ["spk_joe", "spk_andrew"]
    turn_data = [
        ("spk_joe",    0,     8500,  "So I've been thinking about this whole AI thing and it's kind of blowing my mind"),
        ("spk_andrew", 8500,  15200, "Dude it's insane, like the stuff they're doing with video now is unreal"),
        ("spk_joe",    15200, 24000, "Yeah like you can literally take a two hour podcast and it'll find the best clips automatically"),
        ("spk_andrew", 24000, 31500, "That's exactly what we're building here man, that's the whole point of Clypt"),
        ("spk_joe",    31500, 42000, "Wait so explain to me how the graph thing works because I saw it and it looked like a spider web"),
        ("spk_andrew", 42000, 55000, "OK so basically every time someone makes a claim or tells a story or has a reaction, that becomes a node"),
        ("spk_joe",    55000, 63000, "And then the lines between them are like the relationships"),
        ("spk_andrew", 63000, 78000, "Exactly, so if you make a claim and then three minutes later you contradict yourself, there's an edge connecting those two moments"),
        ("spk_joe",    78000, 89000, "That's wild because then you can find those callback moments that make great clips"),
        ("spk_andrew", 89000, 102000, "Right and that's what Phase 4 does, it uses the graph to find subgraphs that would make compelling short-form content"),
        ("spk_joe",    102000, 115000, "So it's not just finding loud moments or laugh moments"),
        ("spk_andrew", 115000, 130000, "No no no it understands the narrative structure, like a setup and payoff that are five minutes apart"),
        ("spk_joe",    130000, 145000, "Bro that's like having a producer who watched the whole thing and knows exactly where the good stuff is"),
        ("spk_andrew", 145000, 158000, "That's literally the pitch, an AI producer that understands conversation the way a human editor does"),
        ("spk_joe",    158000, 172000, "And the grounding thing, that's the part where it figures out who to show on screen"),
        ("spk_andrew", 172000, 190000, "Yeah so once you have a clip, you need to decide camera framing — who's the speaker, who's reacting, when to cut between them"),
        ("spk_joe",    190000, 205000, "Man this is going to change the game for content creators"),
        ("spk_andrew", 205000, 218000, "That's the goal, make professional-quality clips accessible to everyone"),
    ]

    turns = []
    for i, (spk, start, end, text) in enumerate(turn_data):
        turns.append(TimelineTurnRecord(
            run_id=run_id,
            turn_id=f"t_{i+1:03d}",
            speaker_id=spk,
            start_ms=start,
            end_ms=end,
            transcript_text=text,
        ))
    repo.write_timeline_turns(run_id=run_id, turns=turns)

    # ── Nodes (10 semantic nodes) ─────────────────────────────────────────

    node_specs = [
        ("node_001", "claim",              0,     24000,  ["t_001", "t_002", "t_003"], "AI can automatically find the best clips from long-form content", ["topic_pivot"]),
        ("node_002", "explanation",         24000, 55000,  ["t_004", "t_005", "t_006"], "Clypt builds a semantic graph where conversational moments become nodes", []),
        ("node_003", "example",             55000, 89000,  ["t_007", "t_008", "t_009"], "Graph edges capture relationships like contradictions and callbacks across the conversation", []),
        ("node_004", "explanation",         89000, 115000, ["t_010", "t_011"],          "Phase 4 finds compelling subgraphs rather than just isolated loud moments", []),
        ("node_005", "claim",              115000, 145000, ["t_012", "t_013"],          "The system understands narrative structure like setup-payoff pairs", ["high_resonance_candidate"]),
        ("node_006", "anecdote",           130000, 158000, ["t_013", "t_014"],          "It's like having an AI producer who watched the whole thing", ["callback_candidate"]),
        ("node_007", "explanation",        158000, 190000, ["t_015", "t_016"],          "Grounding determines camera framing — speaker vs reaction shots", []),
        ("node_008", "reaction_beat",      190000, 205000, ["t_017"],                   "This technology will change content creation", []),
        ("node_009", "claim",              205000, 218000, ["t_018"],                   "Goal is to make professional clips accessible to everyone", []),
        ("node_010", "setup_payoff",        0,     145000, ["t_001", "t_013"],          "Setup: AI is mind-blowing → Payoff: it's like having a producer", ["callback_candidate"]),
    ]

    nodes = []
    for nid, ntype, start, end, turn_ids, summary, flags in node_specs:
        nodes.append(SemanticNodeRecord(
            run_id=run_id,
            node_id=nid,
            node_type=ntype,
            start_ms=start,
            end_ms=end,
            source_turn_ids=turn_ids,
            transcript_text=summary,
            node_flags=flags,
            summary=summary,
            evidence={"emotion_labels": ["happy", "surprised"], "audio_events": ["laughter"]},
        ))
    repo.write_nodes(run_id=run_id, nodes=nodes)

    # ── Edges (12 semantic edges) ─────────────────────────────────────────

    edge_specs = [
        ("node_001", "node_002", "elaborates",       "Node 2 explains the technology behind node 1's claim", 0.91),
        ("node_002", "node_003", "supports",          "Example of graph edges supports the explanation of nodes", 0.88),
        ("node_003", "node_004", "elaborates",        "Phase 4 elaborates on how the graph is used", 0.85),
        ("node_004", "node_005", "supports",          "Narrative understanding supports the subgraph approach", 0.82),
        ("node_001", "node_005", "setup_for",         "Initial AI claim sets up the narrative structure claim", 0.79),
        ("node_005", "node_006", "payoff_of",         "Producer analogy is payoff of the narrative claim", 0.90),
        ("node_006", "node_007", "elaborates",        "Grounding explanation follows the producer analogy", 0.75),
        ("node_007", "node_008", "reaction_to",       "Reaction to grounding explanation", 0.83),
        ("node_008", "node_009", "supports",          "Game-changing claim supports accessibility goal", 0.80),
        ("node_001", "node_010", "setup_for",         "Opening claim is setup for the callback", 0.87),
        ("node_006", "node_010", "callback_to",       "Producer analogy calls back to initial amazement", 0.92),
        ("node_001", "node_008", "topic_recurrence",  "AI amazement theme recurs at the end", 0.70),
    ]

    edges = []
    for src, tgt, etype, rationale, conf in edge_specs:
        edges.append(SemanticEdgeRecord(
            run_id=run_id,
            source_node_id=src,
            target_node_id=tgt,
            edge_type=etype,
            rationale=rationale,
            confidence=conf,
        ))
    repo.write_edges(run_id=run_id, edges=edges)

    # ── Clip candidates (4 clips) ─────────────────────────────────────────

    clip_specs = [
        ("clip_001", ["node_001", "node_002", "node_003"], 0,      89000,  0.94, "Strong opening with claim + explanation + example arc"),
        ("clip_002", ["node_004", "node_005", "node_006"], 89000,  158000, 0.91, "Narrative structure explanation with producer analogy payoff"),
        ("clip_003", ["node_005", "node_010"],             0,      145000, 0.88, "Setup-payoff callback spanning the full conversation"),
        ("clip_004", ["node_007", "node_008", "node_009"], 158000, 218000, 0.82, "Grounding explanation with strong closer"),
    ]

    candidates = []
    for cid, nids, start, end, score, rationale in clip_specs:
        candidates.append(ClipCandidateRecord(
            run_id=run_id,
            clip_id=cid,
            node_ids=nids,
            start_ms=start,
            end_ms=end,
            score=score,
            rationale=rationale,
            seed_node_id=nids[0],
            score_breakdown={"virality": score * 0.9, "coherence": score * 0.95, "engagement": score * 0.85},
        ))
    repo.write_candidates(run_id=run_id, candidates=candidates)

    # ── Second run (to show list view) ────────────────────────────────────

    repo.upsert_run(RunRecord(
        run_id="run_dev_002",
        source_url="https://www.youtube.com/watch?v=example2",
        status="PHASE24_RUNNING",
        created_at=NOW,
        updated_at=NOW,
        metadata={"display_name": "Lex Fridman x Elon - Dev Demo"},
    ))

    return repo
