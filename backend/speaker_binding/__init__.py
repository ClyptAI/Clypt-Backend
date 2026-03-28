from .identity_store import (
    AudioVisualMappingSummary,
    VisualIdentity,
    VisualIdentityEvidenceEdge,
    normalize_confidence,
    normalize_ordered_unique_ids,
    normalize_track_ids,
)
from .audio_visual_mapping import (
    AudioVisualMappingEvidence,
    build_audio_visual_mapping_summaries,
    learn_audio_visual_mappings,
)
from .assignment_engine import resolve_span_assignments
from .mouth_motion import choose_visual_speaking_candidate
from .project_words import project_words
from .visual_identity import build_visual_identities

__all__ = [
    "AudioVisualMappingSummary",
    "AudioVisualMappingEvidence",
    "VisualIdentity",
    "VisualIdentityEvidenceEdge",
    "build_visual_identities",
    "build_audio_visual_mapping_summaries",
    "choose_visual_speaking_candidate",
    "learn_audio_visual_mappings",
    "project_words",
    "resolve_span_assignments",
    "normalize_confidence",
    "normalize_ordered_unique_ids",
    "normalize_track_ids",
]
