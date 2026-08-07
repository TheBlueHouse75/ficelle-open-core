from __future__ import annotations

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


# Kept local rather than imported: use cases stay free of `ficelle.*` imports. The same set
# exists as `_LOOPBACK_HOSTS` in `ficelle/url_security.py` for the credential-transport check;
# both answer "is this the local machine", so change them together.
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})

# Bind values that keep the socket itself as the trust boundary. Anything else is reachable from
# the network, where an absent Origin no longer proves the caller is local.
LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "127.0.0.1/32"})

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
    # Split host:port without tripping on an IPv6 literal, which carries its own colons.
    if host_header.startswith("["):
        closing = host_header.find("]")
        if closing == -1:
            return False
        hostname = host_header[1:closing]
        remainder = host_header[closing + 1 :]
        if remainder and not remainder.startswith(":"):
            return False
        port_text = remainder[1:] if remainder else ""
    elif host_header.count(":") > 1:
        # A bare IPv6 literal with no brackets cannot carry a port.
        hostname, port_text = host_header, ""
    else:
        hostname, _, port_text = host_header.partition(":")
    if port_text:
        try:
            if int(port_text) != server_port:
                return False
        except ValueError:
            return False
    # A fully-qualified form carries a trailing dot ("localhost."); it names the same host, so
    # normalise it rather than failing a legitimate client. An attacker's "evil.example." is
    # still absent from the allowlist either way.
    hostname = hostname.lower().rstrip(".")
    allowed = set(LOOPBACK_HOSTNAMES)
    bind_host = (bind_host or "").strip().lower()
    # A wildcard bind answers on every interface, so its own name proves nothing; keep the
    # allowlist at loopback rather than accepting whatever name resolved to this machine.
    if bind_host and bind_host not in {"0.0.0.0", "::", "*"}:
        allowed.add(bind_host)
    return hostname in allowed
