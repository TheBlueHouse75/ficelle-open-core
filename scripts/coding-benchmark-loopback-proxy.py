#!/usr/bin/env python3
"""Expose a minimal authenticated Ficelle relay to an isolated benchmark network.

Ficelle intentionally rejects ``Host: host.docker.internal``. This relay owns the real token in a
separate container, forwards only model listing and chat completion to Docker's host gateway, and
replaces that header with the loopback authority expected by Ficelle. It is not a general-purpose
proxy.
"""

from __future__ import annotations

import argparse
import hmac
import http.client
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
ALLOWED_REQUESTS = frozenset({("GET", "/v1/models"), ("POST", "/v1/chat/completions")})
RELAY_TOKEN_ENV = "FICELLE_BENCHMARK_RELAY_TOKEN"
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024


def _request_content_length(headers: object) -> int:
    transfer_encoding = str(headers.get("Transfer-Encoding", "")).strip()  # type: ignore[attr-defined]
    if transfer_encoding:
        raise ValueError("transfer encoding is not supported")
    raw_length = str(headers.get("Content-Length", "0")).strip()  # type: ignore[attr-defined]
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("invalid content length") from exc
    if length < 0 or length > MAX_REQUEST_BODY_BYTES:
        raise ValueError("request body is too large")
    return length


def _authorized(authorization: str | None, relay_token: str) -> bool:
    return hmac.compare_digest(str(authorization or ""), f"Bearer {relay_token}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--target", required=True)
    root.add_argument("--host-header", default="127.0.0.1:8646")
    root.add_argument("--listen-host", default="127.0.0.1")
    root.add_argument("--serve-only", action="store_true")
    root.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    target = urlparse(args.target)
    if (
        target.scheme != "http"
        or target.hostname != "host.docker.internal"
        or target.path.rstrip("/") != "/v1"
        or target.query
        or target.fragment
        or target.port is None
    ):
        sys.stderr.write("coding-benchmark-loopback-proxy: invalid target\n")
        return 2
    if not args.serve_only and not command:
        sys.stderr.write("coding-benchmark-loopback-proxy: command is required\n")
        return 2
    owner_token = os.getenv("OPENAI_API_KEY", "").strip()
    relay_token = os.getenv(RELAY_TOKEN_ENV, "").strip()
    if not relay_token and not args.serve_only:
        relay_token = secrets.token_urlsafe(32)
    if not owner_token or owner_token == relay_token:
        sys.stderr.write("coding-benchmark-loopback-proxy: Ficelle owner token is required\n")
        return 2
    if not relay_token:
        sys.stderr.write("coding-benchmark-loopback-proxy: benchmark relay token is required\n")
        return 2

    class RelayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._relay()

        def do_POST(self) -> None:  # noqa: N802
            self._relay()

        def log_message(self, format: str, *values: object) -> None:
            return

        def _relay(self) -> None:
            if (self.command, urlparse(self.path).path) not in ALLOWED_REQUESTS:
                self.send_error(403, "benchmark relay path forbidden")
                return
            if not _authorized(self.headers.get("Authorization"), relay_token):
                self.send_error(401, "benchmark relay authorization required")
                return
            try:
                length = _request_content_length(self.headers)
            except ValueError as exc:
                self.send_error(413, str(exc))
                return
            body = self.rfile.read(length) if length else None
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
            }
            headers["Host"] = args.host_header
            headers["Authorization"] = f"Bearer {owner_token}"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            connection = http.client.HTTPConnection(target.hostname, target.port, timeout=900)
            response_started = False
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status, response.reason)
                for name, value in response.getheaders():
                    if name.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                        self.send_header(name, value)
                self.end_headers()
                response_started = True
                while chunk := response.read(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException) as exc:
                if not response_started and not self.wfile.closed:
                    self.send_error(502, "Ficelle relay failure")
                sys.stderr.write(f"coding-benchmark-loopback-proxy: relay failed: {exc}\n")
            finally:
                connection.close()

    server = ThreadingHTTPServer((args.listen_host, 8765), RelayHandler)
    if args.serve_only:
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment["OPENAI_API_BASE"] = "http://127.0.0.1:8765/v1"
    environment["OPENAI_API_KEY"] = relay_token
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
