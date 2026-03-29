#!/usr/bin/env python3
"""
Structured models for Crowd Clip artifacts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CrowdComment(BaseModel):
    comment_id: str
    parent_comment_id: str | None = None
    is_reply: bool = False
    author_name: str = ""
    like_count: int = 0
    reply_count: int = 0
    published_at: str = ""
    updated_at: str = ""
    text: str


class CrowdReference(BaseModel):
    kind: Literal["explicit_timestamp", "quote_match", "keyword_overlap"]
    anchor_ms: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    matched_text: str = ""
    note: str = ""


class ResolvedCrowdComment(BaseModel):
    comment_id: str
    parent_comment_id: str | None = None
    is_reply: bool = False
    author_name: str = ""
    like_count: int = 0
    reply_count: int = 0
    published_at: str = ""
    text: str
    excitement_score: float = Field(ge=0.0, le=1.0)
    references: list[CrowdReference] = Field(default_factory=list)


class CrowdClipCandidate(BaseModel):
    rank: int = Field(ge=1)
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(ge=0)
    raw_score: float = Field(ge=0.0)
    final_score: float = Field(ge=0.0, le=100.0)
    anchor_start_ms: int = Field(ge=0)
    anchor_end_ms: int = Field(ge=0)
    explicit_timestamp_count: int = Field(ge=0)
    quote_match_count: int = Field(ge=0)
    keyword_overlap_count: int = Field(ge=0)
    total_reference_count: int = Field(ge=0)
    unique_comment_count: int = Field(ge=0)
    unique_author_count: int = Field(ge=0)
    total_like_count: int = Field(ge=0)
    total_reply_count: int = Field(ge=0)
    transcript_excerpt: str = ""
    justification: str = ""
    evidence_comment_ids: list[str] = Field(default_factory=list)
    sample_comments: list[str] = Field(default_factory=list)
    aligned_node_indices: list[int] = Field(default_factory=list)
    signal_breakdown: dict[str, float | int | str] = Field(default_factory=dict)
