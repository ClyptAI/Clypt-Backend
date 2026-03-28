from backend.speaker_binding.metrics import (
    finalize_lrasd_pipeline_metrics,
    new_lrasd_pipeline_metrics,
)


def test_new_lrasd_pipeline_metrics_initializes_queue_and_throughput_counters():
    metrics = new_lrasd_pipeline_metrics()

    assert metrics == {
        "lrasd_prep_queue_depth": 0,
        "lrasd_prep_queue_depth_max": 0,
        "lrasd_prep_wallclock_s": 0.0,
        "lrasd_infer_wallclock_s": 0.0,
        "lrasd_spans_processed": 0,
        "lrasd_spans_per_sec": 0.0,
        "lrasd_easy_cascade_skipped_jobs": 0,
    }


def test_finalize_lrasd_pipeline_metrics_derives_throughput_from_wallclock():
    metrics = new_lrasd_pipeline_metrics()
    metrics["lrasd_spans_processed"] = 6
    metrics["lrasd_prep_wallclock_s"] = 1.5
    metrics["lrasd_infer_wallclock_s"] = 0.5
    metrics["lrasd_easy_cascade_skipped_jobs"] = 2

    finalized = finalize_lrasd_pipeline_metrics(metrics)

    assert finalized["lrasd_spans_per_sec"] == 3.0
    assert finalized["lrasd_easy_cascade_skipped_jobs"] == 2
