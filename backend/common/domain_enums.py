from __future__ import annotations

from typing import Literal

SignalType = Literal["comment_top", "comment_reply", "trend_topic", "trend_query"]
SourcePlatform = Literal["youtube", "google_trends"]
ClusterType = Literal["comment", "trend"]
LinkType = Literal["direct", "inferred"]
PromptSourceType = Literal["general", "comment", "trend"]
RunStatus = Literal["PHASE1_DONE", "PHASE24_QUEUED", "PHASE24_RUNNING", "PHASE24_DONE", "FAILED"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]

__all__ = [
    "ClusterType",
    "JobStatus",
    "LinkType",
    "PromptSourceType",
    "RunStatus",
    "SignalType",
    "SourcePlatform",
]
