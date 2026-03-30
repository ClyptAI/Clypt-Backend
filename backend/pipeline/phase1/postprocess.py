"""Post-clustering / ledger seams (thin hooks for staged orchestration)."""


def merge_stage_metrics(target: dict, stage: dict | None) -> dict:
    """Copy optional stage metrics into the main tracking metrics dict."""
    if isinstance(stage, dict) and stage:
        target.update(stage)
    return target


__all__ = ["merge_stage_metrics"]
