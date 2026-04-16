from __future__ import annotations

import urllib.error

from backend.runtime.phase24_error_policy import Phase24FailureClass, classify_phase24_exception


def test_classify_phase24_exception_treats_connection_refused_as_fail_fast() -> None:
    failure_class = classify_phase24_exception(
        urllib.error.URLError("[Errno 111] Connection refused")
    )
    assert failure_class == Phase24FailureClass.FAIL_FAST


def test_classify_phase24_exception_keeps_503_as_transient() -> None:
    exc = urllib.error.HTTPError(
        url="http://127.0.0.1:8001/v1/chat/completions",
        code=503,
        msg="service unavailable",
        hdrs=None,
        fp=None,
    )
    failure_class = classify_phase24_exception(exc)
    assert failure_class == Phase24FailureClass.TRANSIENT
