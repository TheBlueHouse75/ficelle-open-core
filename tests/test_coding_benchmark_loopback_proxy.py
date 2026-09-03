from __future__ import annotations

import importlib.util
import json
from email.message import Message
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coding-benchmark-loopback-proxy.py"
SPEC = importlib.util.spec_from_file_location("ficelle_coding_benchmark_loopback_proxy", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def headers(**values: str) -> Message:
    result = Message()
    for name, value in values.items():
        result[name.replace("_", "-")] = value
    return result


def test_relay_requires_the_per_run_bearer_token():
    assert proxy.UPSTREAM_TIMEOUT_SECONDS == 900
    assert proxy._authorized("Bearer random-token", "random-token") is True
    assert proxy._authorized("Bearer another-token", "random-token") is False
    assert proxy._authorized(None, "random-token") is False


def test_relay_bounds_request_bodies_before_reading_them():
    assert proxy._request_content_length(headers(Content_Length="42")) == 42

    with pytest.raises(ValueError, match="too large"):
        proxy._request_content_length(
            headers(Content_Length=str(proxy.MAX_REQUEST_BODY_BYTES + 1))
        )
    with pytest.raises(ValueError, match="transfer encoding"):
        proxy._request_content_length(headers(Transfer_Encoding="chunked"))
    with pytest.raises(ValueError, match="invalid content length"):
        proxy._request_content_length(headers(Content_Length="not-a-number"))


def test_relay_makes_truncated_before_content_non_retryable_for_benchmark():
    body = json.dumps(
        {
            "error": {
                "type": "upstream_failure",
                "reasons": {"truncated_before_content": 1},
            }
        }
    ).encode()

    assert proxy._benchmark_response_status(502, body) == 422
    assert proxy._benchmark_response_status(503, body) == 503


def test_relay_makes_empty_model_response_non_retryable_for_benchmark():
    body = json.dumps(
        {
            "error": {
                "type": "upstream_failure",
                "reasons": {"empty_assistant_message": 1},
            }
        }
    ).encode()

    assert proxy._benchmark_response_status(502, body) == 422


def test_relay_keeps_real_or_mixed_provider_failures_retryable():
    provider_error = json.dumps(
        {"error": {"type": "upstream_failure", "reasons": {"timeout": 1}}}
    ).encode()
    mixed_error = json.dumps(
        {
            "error": {
                "type": "upstream_failure",
                "reasons": {"truncated_before_content": 1, "timeout": 1},
            }
        }
    ).encode()

    assert proxy._benchmark_response_status(502, provider_error) == 502
    assert proxy._benchmark_response_status(502, mixed_error) == 502
    assert proxy._benchmark_response_status(502, b"not-json") == 502


def test_relay_classifies_model_and_provider_failures_for_audit():
    model_failure = json.dumps(
        {"error": {"type": "upstream_failure", "reasons": {"truncated_before_content": 1}}}
    ).encode()
    provider_failure = json.dumps(
        {"error": {"type": "upstream_failure", "reasons": {"quota_exhausted": 1}}}
    ).encode()

    assert proxy._response_outcome(200, b"") == "success"
    assert proxy._response_outcome(502, model_failure) == "model_failure"
    assert proxy._response_outcome(429, provider_failure) == "provider_error"
    assert proxy._response_failure_reason(502, model_failure) == "truncated_before_content"
    assert proxy._response_failure_reason(429, provider_failure) is None


def test_relay_audit_extracts_only_the_completion_token_budget():
    body = json.dumps(
        {
            "model": "ficelle/openrouter/example",
            "messages": [{"role": "user", "content": "private prompt"}],
            "max_tokens": 24_000,
        }
    ).encode()

    assert proxy._request_completion_token_budget(body) == 24_000
    assert proxy._request_completion_token_budget(b"not-json") is None


def test_relay_accepts_only_frozen_task_ids_for_audit():
    assert (
        proxy._benchmark_task_id("go/exercises/practice/crypto-square")
        == "go/exercises/practice/crypto-square"
    )
    assert proxy._benchmark_task_id("../../secret") == ""
    assert proxy._benchmark_task_id(None) == ""
