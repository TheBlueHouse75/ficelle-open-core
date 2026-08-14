from __future__ import annotations

import base64
import binascii
import hmac
from urllib.parse import urlparse


def is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    return parsed.scheme == "http" and (parsed.hostname or "") in {"127.0.0.1", "localhost", "::1"}


def admin_origin_allowed(origin: str | None, server_port: int) -> bool:
    if origin is None:
        return True
    if not is_loopback_origin(origin):
        return False
    try:
        origin_port = urlparse(origin).port
    except Exception:
        return False
    return origin_port == server_port


def admin_token_matches(presented: str | None, expected: str) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "ignore"), expected.encode("utf-8"))


def bearer_token_matches(authorization: str | None, expected: str) -> bool:
    if not authorization:
        return False
    scheme, separator, token = authorization.partition(" ")
    return bool(separator) and scheme.lower() == "bearer" and admin_token_matches(token.strip(), expected)


def basic_token_matches(
    authorization: str | None,
    expected: str,
    *,
    username: str = "ficelle",
) -> bool:
    if not authorization:
        return False
    scheme, separator, encoded = authorization.partition(" ")
    if not separator or scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeError):
        return False
    presented_username, separator, token = decoded.partition(":")
    return bool(separator) and presented_username == username and admin_token_matches(token, expected)


def _split_host_header(host_header: str) -> tuple[str, str] | None:
    host_header = host_header.strip()
    if not host_header:
        return None
    if host_header.startswith("["):
        closing = host_header.find("]")
        if closing == -1:
            return None
        hostname = host_header[1:closing]
        remainder = host_header[closing + 1 :]
        if remainder and not remainder.startswith(":"):
            return None
        port_text = remainder[1:] if remainder else ""
    elif host_header.count(":") > 1:
        hostname, port_text = host_header, ""
    else:
        hostname, _, port_text = host_header.partition(":")
    return hostname.lower().rstrip("."), port_text


def same_origin_allowed(origin: str | None, host_header: str | None, server_port: int) -> bool:
    """Accept non-browser callers and exact HTTP origins for this request Host and port."""
    if origin is None:
        return True
    if not host_header:
        return False
    parsed_host = _split_host_header(host_header)
    if parsed_host is None:
        return False
    request_hostname, request_port_text = parsed_host
    if request_port_text:
        try:
            if int(request_port_text) != server_port:
                return False
        except ValueError:
            return False
    try:
        parsed_origin = urlparse(origin)
        origin_port = parsed_origin.port
    except (TypeError, ValueError):
        return False
    return (
        parsed_origin.scheme == "http"
        and (parsed_origin.hostname or "").lower().rstrip(".") == request_hostname
        and origin_port == server_port
    )


# Native webview clients attach an app-scheme Origin to every request (Jan is Tauri and
# sends tauri://localhost) — a value browser web content can never carry: pages get
# http(s), sandboxed/file pages get the literal "null". Extension schemes
# (chrome-extension://, moz-extension://, …) are deliberately absent — extensions are
# browser-resident attack surface — and an unlisted native client stays diagnosable
# through the rejected-origin line in the error log.
NATIVE_WEBVIEW_ORIGIN_SCHEMES = frozenset({"tauri", "app", "vscode-webview"})


def inference_origin_allowed(origin: str | None, host_header: str | None, server_port: int) -> bool:
    """Same-origin rule for inference, widened to named app-scheme origins from native webviews.

    Only the chat inference endpoint uses this; admin mutations keep the strict
    `same_origin_allowed` rule, where a webview client has no business writing.
    """
    if origin is not None:
        try:
            if urlparse(origin).scheme in NATIVE_WEBVIEW_ORIGIN_SCHEMES:
                return True
        except (TypeError, ValueError):
            return False
    return same_origin_allowed(origin, host_header, server_port)


# Kept local rather than imported: use cases stay free of `ficelle.*` imports. The same set
# exists as `_LOOPBACK_HOSTS` in `ficelle/url_security.py` for the credential-transport check;
# both answer "is this the local machine", so change them together.
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})

# Bind values that keep the socket itself as the trust boundary. Anything else is reachable from
# the network, where an absent Origin no longer proves the caller is local.
LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "127.0.0.1/32"})


def bind_host_is_loopback(bind_host: str | None) -> bool:
    """Classify loopback bind spellings after DNS-style case/dot normalization."""
    normalized = str(bind_host or "127.0.0.1").strip().lower().rstrip(".")
    return normalized in LOOPBACK_BIND_HOSTS


# The admin page loads exactly one script (`<script type="module" src="/admin/static/app.js">`),
# so `script-src 'self'` costs nothing and is the directive that matters: it denies the injected
# `onerror=`/inline payload an unescaped upstream string would otherwise run. `style-src` has to
# tolerate inline because the dashboard markup carries ~23 `style="…"` attributes; a style
# injection cannot execute script, so keeping script strict is what buys the protection.
ADMIN_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)

# `frame-ancestors` is the modern control, but X-Frame-Options is still what some engines honour,
# and framing is the sharp risk here: a framed admin page sends its own loopback Origin AND
# attaches its own token, so it satisfies both existing write guards at once.
ADMIN_HTML_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": ADMIN_CONTENT_SECURITY_POLICY,
}


def request_host_allowed(host_header: str | None, server_port: int, bind_host: str) -> bool:
    """Reject a request whose ``Host`` is not this server, which is what closes DNS rebinding.

    The Origin guard already blocks cross-origin *writes*, but it never runs on GET, so a page
    on a short-TTL domain rebound to 127.0.0.1 could read the whole admin surface — including the
    admin token embedded in the page. Matching ``Host`` against what this server actually binds
    makes the rebound name fail before routing.

    A deliberate non-loopback bind (the Tailscale end-to-end test bed) stays reachable: its own
    host is allowed alongside loopback. Absent ``Host`` is allowed — HTTP/1.0 clients and the CLI
    omit it, and a browser never does.
    """
    if host_header is None:
        return True
    host_header = host_header.strip()
    if not host_header:
        return True
    parsed_host = _split_host_header(host_header)
    if parsed_host is None:
        return False
    hostname, port_text = parsed_host
    if port_text:
        try:
            if int(port_text) != server_port:
                return False
        except ValueError:
            return False
    # A fully-qualified form carries a trailing dot ("localhost."); it names the same host, so
    # normalise it rather than failing a legitimate client. An attacker's "evil.example." is
    # still absent from the allowlist either way.
    allowed = set(LOOPBACK_HOSTNAMES)
    bind_host = (bind_host or "").strip().lower().rstrip(".")
    # A wildcard bind answers on every interface, so its own name proves nothing; keep the
    # allowlist at loopback rather than accepting whatever name resolved to this machine.
    if bind_host and bind_host not in {"0.0.0.0", "::", "*"}:
        allowed.add(bind_host)
    return hostname in allowed
