from __future__ import annotations

import importlib.util
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
