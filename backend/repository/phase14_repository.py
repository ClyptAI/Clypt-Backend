from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .models import (
    CandidateSignalLinkRecord,
    ClipCandidateRecord,
    ExternalSignalClusterRecord,
    ExternalSignalRecord,
    Phase24JobRecord,
    PhaseMetricRecord,
    PromptSourceLinkRecord,
    RunRecord,
    SemanticEdgeRecord,
    SemanticNodeRecord,
    NodeSignalLinkRecord,
    SubgraphProvenanceRecord,
    TimelineTurnRecord,
)


class Phase14Repository(ABC):
    @abstractmethod
    def upsert_run(self, record: RunRecord) -> RunRecord:
        raise NotImplementedError

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None:
        raise NotImplementedError

    @abstractmethod
    def write_timeline_turns(self, *, run_id: str, turns: Sequence[TimelineTurnRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_timeline_turns(self, *, run_id: str) -> list[TimelineTurnRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_nodes(self, *, run_id: str, nodes: Sequence[SemanticNodeRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_nodes(self, *, run_id: str) -> list[SemanticNodeRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_edges(self, *, run_id: str, edges: Sequence[SemanticEdgeRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_edges(self, *, run_id: str) -> list[SemanticEdgeRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_candidates(self, *, run_id: str, candidates: Sequence[ClipCandidateRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_candidates(self, *, run_id: str) -> list[ClipCandidateRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_external_signals(self, *, run_id: str, signals: Sequence[ExternalSignalRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_external_signals(self, *, run_id: str) -> list[ExternalSignalRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_external_signal_clusters(
        self, *, run_id: str, clusters: Sequence[ExternalSignalClusterRecord]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_external_signal_clusters(self, *, run_id: str) -> list[ExternalSignalClusterRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_node_signal_links(self, *, run_id: str, links: Sequence[NodeSignalLinkRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_node_signal_links(self, *, run_id: str) -> list[NodeSignalLinkRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_candidate_signal_links(
        self, *, run_id: str, links: Sequence[CandidateSignalLinkRecord]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_candidate_signal_links(self, *, run_id: str) -> list[CandidateSignalLinkRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_prompt_source_links(self, *, run_id: str, links: Sequence[PromptSourceLinkRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_prompt_source_links(self, *, run_id: str) -> list[PromptSourceLinkRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_subgraph_provenance(
        self, *, run_id: str, provenance: Sequence[SubgraphProvenanceRecord]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_subgraph_provenance(self, *, run_id: str) -> list[SubgraphProvenanceRecord]:
        raise NotImplementedError

    @abstractmethod
    def write_phase_metric(self, record: PhaseMetricRecord) -> PhaseMetricRecord:
        raise NotImplementedError

    @abstractmethod
    def list_phase_metrics(self, *, run_id: str) -> list[PhaseMetricRecord]:
        raise NotImplementedError

    @abstractmethod
    def upsert_phase24_job(self, record: Phase24JobRecord) -> Phase24JobRecord:
        raise NotImplementedError

    @abstractmethod
    def get_phase24_job(self, run_id: str) -> Phase24JobRecord | None:
        raise NotImplementedError

    @abstractmethod
    def acquire_phase24_job_lease(
        self,
        *,
        run_id: str,
        job_id: str,
        worker_name: str,
        attempt: int,
        query_version: str | None,
        running_timeout_s: int = 1800,
    ) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def delete_run(self, *, run_id: str) -> None:
        raise NotImplementedError


__all__ = [
    "Phase14Repository",
]
