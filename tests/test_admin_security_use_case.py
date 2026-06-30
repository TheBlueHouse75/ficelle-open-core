from __future__ import annotations

from ficelle.use_cases.admin_security import (
    admin_origin_allowed,
    admin_token_matches,
    is_loopback_origin,
)


def test_is_loopback_origin_accepts_only_http_loopback_hosts() -> None:
    assert is_loopback_origin("http://127.0.0.1:8646") is True
    assert is_loopback_origin("http://localhost:8646") is True
    assert is_loopback_origin("http://[::1]:8646") is True
    assert is_loopback_origin("https://127.0.0.1:8646") is False
    assert is_loopback_origin("http://evil.example:8646") is False
    assert is_loopback_origin("not a url") is False


def test_admin_origin_allowed_requires_matching_port_when_origin_is_present() -> None:
    assert admin_origin_allowed(None, 8646) is True
    assert admin_origin_allowed("http://127.0.0.1:8646", 8646) is True
    assert admin_origin_allowed("http://localhost:9999", 8646) is False
    assert admin_origin_allowed("https://evil.example", 8646) is False
    assert admin_origin_allowed("http://127.0.0.1", 8646) is False


def test_admin_token_matches_requires_present_exact_token() -> None:
    assert admin_token_matches("secret-token", "secret-token") is True
    assert admin_token_matches("", "secret-token") is False
    assert admin_token_matches(None, "secret-token") is False
    assert admin_token_matches("wrong-token", "secret-token") is False
