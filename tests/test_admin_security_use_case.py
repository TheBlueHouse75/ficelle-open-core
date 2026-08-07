from __future__ import annotations

from ficelle.use_cases.admin_security import (
    ADMIN_CONTENT_SECURITY_POLICY,
    ADMIN_HTML_SECURITY_HEADERS,
    admin_origin_allowed,
    admin_token_matches,
    is_loopback_origin,
    request_host_allowed,
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


def test_request_host_allowed_accepts_every_loopback_spelling() -> None:
    assert request_host_allowed("127.0.0.1:8646", 8646, "127.0.0.1") is True
    assert request_host_allowed("localhost:8646", 8646, "127.0.0.1") is True
    assert request_host_allowed("LocalHost:8646", 8646, "127.0.0.1") is True
    assert request_host_allowed("[::1]:8646", 8646, "127.0.0.1") is True
    assert request_host_allowed("::1", 8646, "127.0.0.1") is True
    # Port-less Host is legal and still names this server.
    assert request_host_allowed("127.0.0.1", 8646, "127.0.0.1") is True
    # curl/CLI clients may omit Host entirely; a browser never does.
    assert request_host_allowed(None, 8646, "127.0.0.1") is True
    assert request_host_allowed("", 8646, "127.0.0.1") is True
    # A fully-qualified spelling names the same host.
    assert request_host_allowed("localhost.:8646", 8646, "127.0.0.1") is True
    assert request_host_allowed("evil.example.:8646", 8646, "127.0.0.1") is False


def test_request_host_allowed_rejects_rebound_and_mismatched_hosts() -> None:
    # The DNS-rebinding case: attacker name resolving to 127.0.0.1.
    assert request_host_allowed("evil.example:8646", 8646, "127.0.0.1") is False
    assert request_host_allowed("evil.example", 8646, "127.0.0.1") is False
    # A loopback name on someone else's port is not this server.
    assert request_host_allowed("127.0.0.1:9999", 8646, "127.0.0.1") is False
    assert request_host_allowed("[::1]:9999", 8646, "127.0.0.1") is False
    # Malformed values fail closed rather than parsing loosely.
    assert request_host_allowed("[::1", 8646, "127.0.0.1") is False
    assert request_host_allowed("127.0.0.1:notaport", 8646, "127.0.0.1") is False


def test_request_host_allowed_honours_a_deliberate_non_loopback_bind() -> None:
    # The Tailscale end-to-end test bed binds a real hostname; it must stay reachable.
    assert request_host_allowed("mac-air.tailnet.ts.net:8646", 8646, "mac-air.tailnet.ts.net") is True
    # Loopback keeps working alongside it.
    assert request_host_allowed("127.0.0.1:8646", 8646, "mac-air.tailnet.ts.net") is True
    # A wildcard bind answers on every interface, so its own value proves nothing: an
    # attacker-controlled name that resolves here must still be refused.
    assert request_host_allowed("evil.example:8646", 8646, "0.0.0.0") is False
    assert request_host_allowed("127.0.0.1:8646", 8646, "0.0.0.0") is True


def test_client_url_host_is_always_accepted_by_the_host_allowlist() -> None:
    # These two must agree or the router 403s its own CLI: the allowlist refuses a wildcard
    # literal (a browser can reach http://0.0.0.0:<port>, so accepting it as a Host would
    # reopen the rebinding hole), so every builder of a URL *to* the router has to resolve a
    # wildcard bind to loopback first.
    from ficelle.url_security import connectable_host

    for bind in ("0.0.0.0", "::", "*", "", "127.0.0.1", "localhost", "mac-air.tailnet.ts.net"):
        host = connectable_host(bind)
        host_header = f"[{host}]:8646" if ":" in host else f"{host}:8646"
        assert request_host_allowed(host_header, 8646, bind) is True, bind


def test_admin_html_security_headers_deny_framing_and_inline_script() -> None:
    assert ADMIN_HTML_SECURITY_HEADERS["X-Frame-Options"] == "DENY"
    assert ADMIN_HTML_SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
    # script-src without 'unsafe-inline' is what actually denies an injected handler.
    assert "script-src 'self'" in ADMIN_CONTENT_SECURITY_POLICY
    assert "'unsafe-inline'" not in ADMIN_CONTENT_SECURITY_POLICY.split("style-src")[0]
    assert "frame-ancestors 'none'" in ADMIN_CONTENT_SECURITY_POLICY
    assert "object-src 'none'" in ADMIN_CONTENT_SECURITY_POLICY
