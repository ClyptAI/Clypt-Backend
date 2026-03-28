import threading
import time

from backend.speaker_binding.lrasd_runner import LrasdPrepPipeline, build_lrasd_span_jobs


def test_build_lrasd_span_jobs_includes_only_hard_spans_and_surviving_candidates():
    jobs, debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-1",
                "span_type": "single",
                "speaker_ids": ["SPEAKER_00"],
                "speaker_id": "SPEAKER_00",
                "overlap": False,
                "context_start_time_ms": 500,
                "context_end_time_ms": 1000,
                "source_turn_ids": ["turn-1"],
            }
        ],
        words=[
            {"text": "hard", "start_time_ms": 520, "end_time_ms": 640},
            {"text": "span", "start_time_ms": 660, "end_time_ms": 880},
        ],
        rank_candidates_fn=lambda span: [
            {
                "local_track_id": "survivor",
                "candidate_survives": True,
                "rank_score": 0.9,
            },
            {
                "local_track_id": "rejected",
                "candidate_survives": False,
                "rank_score": 0.8,
            },
        ],
    )

    assert jobs == [
        {
            "job_id": "scheduled-1:subspan-0",
            "span_id": "scheduled-1",
            "span_type": "single",
            "speaker_ids": ["SPEAKER_00"],
            "source_turn_ids": ["turn-1"],
            "overlap": False,
            "context_start_time_ms": 500,
            "context_end_time_ms": 1000,
            "start_time_ms": 520,
            "end_time_ms": 880,
            "selected_local_track_ids": ["survivor"],
            "candidate_rows": [
                {
                    "local_track_id": "survivor",
                    "candidate_survives": True,
                    "rank_score": 0.9,
                }
            ],
            "word_indices": [0, 1],
        }
    ]
    assert debug_rows == [
        {
            "span_id": "scheduled-1",
            "job_count": 1,
            "selected_local_track_ids": ["survivor"],
        }
    ]


def test_build_lrasd_span_jobs_preserves_overlap_multi_speaker_context():
    jobs, _debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-2",
                "span_type": "overlap",
                "speaker_ids": ["SPEAKER_00", "SPEAKER_01"],
                "overlap": True,
                "context_start_time_ms": 1000,
                "context_end_time_ms": 1600,
                "source_turn_ids": ["turn-2", "turn-3"],
            }
        ],
        words=[
            {"text": "both", "start_time_ms": 1040, "end_time_ms": 1200},
        ],
        rank_candidates_fn=lambda span: [
            {"local_track_id": "left", "candidate_survives": True, "rank_score": 0.81},
            {"local_track_id": "right", "candidate_survives": True, "rank_score": 0.79},
        ],
    )

    assert jobs[0]["span_type"] == "overlap"
    assert jobs[0]["speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
    assert jobs[0]["overlap"] is True
    assert jobs[0]["selected_local_track_ids"] == ["left", "right"]
    assert jobs[0]["context_start_time_ms"] == 1000
    assert jobs[0]["context_end_time_ms"] == 1600


def test_build_lrasd_span_jobs_reuses_contiguous_words_within_same_span():
    jobs, _debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-3",
                "span_type": "single",
                "speaker_ids": ["SPEAKER_02"],
                "speaker_id": "SPEAKER_02",
                "overlap": False,
                "context_start_time_ms": 0,
                "context_end_time_ms": 1200,
                "source_turn_ids": ["turn-4"],
            }
        ],
        words=[
            {"text": "one", "start_time_ms": 0, "end_time_ms": 120},
            {"text": "two", "start_time_ms": 200, "end_time_ms": 320},
            {"text": "three", "start_time_ms": 900, "end_time_ms": 1100},
        ],
        rank_candidates_fn=lambda span: [
            {"local_track_id": "speaker", "candidate_survives": True, "rank_score": 0.88},
        ],
    )

    assert [job["job_id"] for job in jobs] == [
        "scheduled-3:subspan-0",
        "scheduled-3:subspan-1",
    ]
    assert jobs[0]["start_time_ms"] == 0
    assert jobs[0]["end_time_ms"] == 320
    assert jobs[0]["word_indices"] == [0, 1]
    assert jobs[1]["start_time_ms"] == 900
    assert jobs[1]["end_time_ms"] == 1100
    assert jobs[1]["word_indices"] == [2]


def test_build_lrasd_span_jobs_skips_spans_with_no_surviving_candidates():
    jobs, debug_rows = build_lrasd_span_jobs(
        scheduled_hard_spans=[
            {
                "span_id": "scheduled-4",
                "span_type": "single",
                "speaker_ids": ["SPEAKER_03"],
                "speaker_id": "SPEAKER_03",
                "overlap": False,
                "context_start_time_ms": 100,
                "context_end_time_ms": 500,
                "source_turn_ids": ["turn-5"],
            }
        ],
        words=[
            {"text": "none", "start_time_ms": 120, "end_time_ms": 300},
        ],
        rank_candidates_fn=lambda span: [
            {"local_track_id": "filtered", "candidate_survives": False, "rank_score": 0.7},
        ],
    )

    assert jobs == []
    assert debug_rows == []


def test_lrasd_prep_pipeline_preserves_submission_order_when_prep_finishes_out_of_order():
    pipeline = LrasdPrepPipeline(
        prepare_fn=lambda spec: (time.sleep(spec["delay_s"]), spec["job_id"])[1],
        prep_workers=2,
        queue_size=2,
    )

    try:
        prepared = []
        prepared.extend(pipeline.submit({"job_id": "job-0", "delay_s": 0.06}))
        prepared.extend(pipeline.submit({"job_id": "job-1", "delay_s": 0.01}))
        prepared.extend(pipeline.submit({"job_id": "job-2", "delay_s": 0.03}))
        prepared.extend(pipeline.drain())
    finally:
        pipeline.close()

    assert prepared == ["job-0", "job-1", "job-2"]


def test_lrasd_prep_pipeline_applies_backpressure_when_queue_is_full():
    started: list[str] = []
    release_first = threading.Event()

    def _prepare(spec: dict) -> str:
        started.append(spec["job_id"])
        if spec["job_id"] == "job-0":
            release_first.wait(timeout=1.0)
        return spec["job_id"]

    pipeline = LrasdPrepPipeline(
        prepare_fn=_prepare,
        prep_workers=2,
        queue_size=2,
    )

    def _submit_all():
        prepared.extend(pipeline.submit({"job_id": "job-0"}))
        prepared.extend(pipeline.submit({"job_id": "job-1"}))
        prepared.extend(pipeline.submit({"job_id": "job-2"}))

    prepared: list[str] = []
    submit_thread = threading.Thread(target=_submit_all)
    submit_thread.start()
    time.sleep(0.1)

    assert started[:2] == ["job-0", "job-1"]
    assert len(started) == 2
    assert submit_thread.is_alive()

    release_first.set()
    submit_thread.join(timeout=1.0)
    assert not submit_thread.is_alive()

    try:
        prepared.extend(pipeline.drain())
    finally:
        pipeline.close()

    assert prepared == ["job-0", "job-1", "job-2"]
    assert pipeline.metrics["lrasd_prep_queue_depth_max"] >= 1


def test_lrasd_prep_pipeline_drain_does_not_redeliver_items_already_returned_by_submit():
    pipeline = LrasdPrepPipeline(
        prepare_fn=lambda spec: (time.sleep(spec["delay_s"]), spec["job_id"])[1],
        prep_workers=2,
        queue_size=1,
    )

    try:
        ready_from_submit = pipeline.submit({"job_id": "job-0", "delay_s": 0.0})
        remaining = pipeline.drain()
    finally:
        pipeline.close()

    assert ready_from_submit == ["job-0"]
    assert remaining == []


def test_lrasd_prep_pipeline_snapshot_reports_oldest_pending_age():
    release = threading.Event()

    def _prepare(_spec: dict) -> str:
        release.wait(timeout=1.0)
        return "done"

    pipeline = LrasdPrepPipeline(
        prepare_fn=_prepare,
        prep_workers=1,
        queue_size=4,
    )

    try:
        assert pipeline.submit({"job_id": "job-0"}) == []
        time.sleep(0.05)
        snapshot = pipeline.snapshot()
    finally:
        release.set()
        pipeline.close()

    assert snapshot["pending_count"] == 1
    assert snapshot["next_submit_seq"] == 1
    assert snapshot["next_emit_seq"] == 0
    assert snapshot["oldest_pending_seq"] == 0
    assert snapshot["oldest_pending_age_s"] >= 0.04
    assert snapshot["head_future_done"] is False
