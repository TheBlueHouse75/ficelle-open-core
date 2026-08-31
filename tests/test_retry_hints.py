from datetime import datetime, timezone

from ficelle.retry_hints import MAX_RETRY_SECONDS, RetryHint, retry_hint


def test_retry_hint_prefers_millisecond_aliases_case_insensitively():
    assert retry_hint({"X-MS-Retry-After-MS": "1501", "Retry-After": "30"}, None) == RetryHint(
        seconds=2,
        source="x-ms-retry-after-ms",
    )


def test_retry_hint_parses_delta_seconds_and_http_date():
    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    assert retry_hint({"Retry-After": "2.2"}, None, now=now) == RetryHint(3, "retry-after")
    assert retry_hint({"Retry-After": "Sun, 23 Aug 2026 12:01:00 GMT"}, None, now=now) == RetryHint(
        60,
        "retry-after",
    )


def test_retry_hint_parses_google_style_nested_body_duration():
    body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3.5s"}]}}
    assert retry_hint({}, body) == RetryHint(4, "body")


def test_retry_hint_rejects_bad_values_and_bounds_large_delays():
    assert retry_hint({"Retry-After": "never"}, "not json") is None
    assert retry_hint({"Retry-After": "100ms"}, None) is None
    assert retry_hint({"Retry-After": "1h"}, None) is None
    assert retry_hint({"Retry-After": "-3"}, None) is None
    assert retry_hint({}, {"retry_after": "999h"}) == RetryHint(MAX_RETRY_SECONDS, "body")


def test_retry_hint_bounds_recursive_and_list_work():
    too_deep = current = {}
    for _ in range(10):
        child = {}
        current["child"] = child
        current = child
    current["retryDelay"] = "10s"
    assert retry_hint({}, too_deep) is None
    assert retry_hint({}, [{"ignored": True}] * 32 + [{"retryAfter": 7}]) is None


def test_retry_hint_rejects_pathological_json_without_raising():
    assert retry_hint({}, '{"value":' + "9" * 5000 + "}") is None
    assert retry_hint({}, "[" * 20_000 + "]" * 20_000) is None
