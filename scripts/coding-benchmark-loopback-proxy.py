#!/usr/bin/env python3
"""Expose a minimal authenticated Ficelle relay to an isolated benchmark network.

Ficelle intentionally rejects ``Host: host.docker.internal``. This relay owns the real token in a
separate container, forwards only model listing and chat completion to Docker's host gateway, and
replaces that header with the loopback authority expected by Ficelle. It is not a general-purpose
proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
UPSTREAM_TIMEOUT_SECONDS = 900
AUDIT_SCHEMA_VERSION = 2
BENCHMARK_TASK_HEADER = "X-Ficelle-Benchmark-Task"


def _benchmark_response_status(status: int, body: bytes) -> int:
    """Stop Aider retrying a Ficelle model verdict as provider infrastructure."""
    if status != 502:
        return status
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status
    error = payload.get("error") if isinstance(payload, dict) else None
    reasons = error.get("reasons") if isinstance(error, dict) else None
    if reasons in (
        {"truncated_before_content": 1},
        {"empty_assistant_message": 1},
    ):
        return 422
    return status


def _response_outcome(status: int, body: bytes) -> str:
    if status < 400:
        return "success"
    if _benchmark_response_status(status, body) != status:
        return "model_failure"
    return "provider_error"


def _response_failure_reason(status: int, body: bytes) -> str | None:
    """Return the prompt-free reason needed to trigger the extended-budget diagnostic."""
    if _response_outcome(status, body) != "model_failure":
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    reasons = error.get("reasons") if isinstance(error, dict) else None
    if reasons == {"truncated_before_content": 1}:
        return "truncated_before_content"
    if reasons == {"empty_assistant_message": 1}:
        return "empty_assistant_message"
    return None


def _request_completion_token_budget(body: bytes | None) -> int | None:
    try:
        payload = json.loads(body or b"")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("max_tokens") if isinstance(payload, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _benchmark_task_id(value: str | None) -> str:
    task_id = str(value or "").strip()
    if not re.fullmatch(r"[a-z0-9-]+/exercises/practice/[a-z0-9-]+", task_id):
        return ""
    return task_id


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
    root.add_argument("--audit-log")
    root.add_argument("--upstream-timeout-seconds", type=int, default=UPSTREAM_TIMEOUT_SECONDS)
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
    if not 1 <= args.upstream_timeout_seconds <= 3600:
        sys.stderr.write("coding-benchmark-loopback-proxy: invalid upstream timeout\n")
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
    audit_path = Path(args.audit_log) if args.audit_log else None
    if audit_path is not None:
        if not audit_path.is_absolute() or not audit_path.parent.is_dir():
            sys.stderr.write("coding-benchmark-loopback-proxy: invalid audit log path\n")
            return 2
        audit_path.write_text("", encoding="utf-8")
    audit_lock = threading.Lock()
    audit_index = 0

    def record_audit(
        *,
        method: str,
        path: str,
        task_id: str,
        request_body: bytes | None,
        response_status: int,
        outcome: str,
        failure_reason: str | None,
    ) -> None:
        nonlocal audit_index
        if audit_path is None or (method, path) != ("POST", "/v1/chat/completions"):
            return
        request_fingerprint = hashlib.sha256(request_body or b"").hexdigest()
        with audit_lock:
            audit_index += 1
            event = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "event_index": audit_index,
                "request_fingerprint": request_fingerprint,
                "task_id": task_id,
                "response_status": response_status,
                "outcome": outcome,
                "failure_reason": failure_reason,
                "completion_token_budget": _request_completion_token_budget(request_body),
            }
            with audit_path.open("a", encoding="utf-8") as audit_log:
                audit_log.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
                audit_log.flush()

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
            task_id = _benchmark_task_id(self.headers.get(BENCHMARK_TASK_HEADER))
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower()
                not in HOP_BY_HOP_HEADERS
                | {"host", "content-length", BENCHMARK_TASK_HEADER.lower()}
            }
            headers["Host"] = args.host_header
            headers["Authorization"] = f"Bearer {owner_token}"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            connection = http.client.HTTPConnection(
                target.hostname,
                target.port,
                timeout=args.upstream_timeout_seconds,
            )
            response_started = False
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read() if response.status >= 400 else None
                outcome = _response_outcome(response.status, response_body or b"")
                failure_reason = _response_failure_reason(response.status, response_body or b"")
                response_status = (
                    _benchmark_response_status(response.status, response_body)
                    if response_body is not None
                    else response.status
                )
                record_audit(
                    method=self.command,
                    path=urlparse(self.path).path,
                    task_id=task_id,
                    request_body=body,
                    response_status=response.status,
                    outcome=outcome,
                    failure_reason=failure_reason,
                )
                self.send_response(
                    response_status,
                    response.reason if response_status == response.status else None,
                )
                for name, value in response.getheaders():
                    if name.lower() not in HOP_BY_HOP_HEADERS | {"content-length"}:
                        self.send_header(name, value)
                self.end_headers()
                response_started = True
                if response_body is not None:
                    self.wfile.write(response_body)
                    self.wfile.flush()
                else:
                    while chunk := response.read(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (OSError, http.client.HTTPException) as exc:
                record_audit(
                    method=self.command,
                    path=urlparse(self.path).path,
                    task_id=task_id,
                    request_body=body,
                    response_status=502,
                    outcome="provider_error",
                    failure_reason=None,
                )
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
