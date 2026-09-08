#!/usr/bin/env python3
"""Prove Ficelle's Anthropic surface with the real Claude Code client and a fake upstream.

The fake upstream makes this deterministic and free while Claude Code still performs an actual
local Bash tool call, returns its result, and consumes Ficelle's streamed Messages protocol.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any


EXPECTED_TOOL_RESULT = "ficelle-client-tool-ok"
EXPECTED_FINAL = "ficelle-claude-code-ok"
TOOL_CALL_ID = "toolu_ficelle_smoke"


class StreamingResponse:
    status_code = 200
    headers = {"Content-Type": "text/event-stream"}
    content = b""

    def __init__(self, wire: bytes) -> None:
        self._wire = wire

    def iter_content(self, chunk_size: int | None = None):
        del chunk_size
        boundaries = (13, 47, 113, 229, len(self._wire))
        offset = 0
        for boundary in boundaries:
            if boundary > offset:
                yield self._wire[offset:boundary]
                offset = boundary

    def close(self) -> None:
        return None


def openai_sse(events: list[dict[str, Any]]) -> bytes:
    frames = [
        b"data: " + json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n\n"
        for event in events
    ]
    return b"".join([*frames, b"data: [DONE]\n\n"])


def tool_call_response(tool_name: str) -> StreamingResponse:
    arguments = json.dumps({"command": f"printf {EXPECTED_TOOL_RESULT}"}, separators=(",", ":"))
    split = max(1, len(arguments) // 2)
    return StreamingResponse(
        openai_sse(
            [
                {
                    "id": "chatcmpl-tool",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": TOOL_CALL_ID,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": arguments[:split],
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": arguments[split:]},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 8},
                },
            ]
        )
    )


def final_response() -> StreamingResponse:
    return text_response(EXPECTED_FINAL, "chatcmpl-final")


def text_response(text: str, response_id: str) -> StreamingResponse:
    return StreamingResponse(
        openai_sse(
            [
                {
                    "id": response_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 55, "completion_tokens": 5},
                },
            ]
        )
    )


def main() -> int:
    claude = shutil.which("claude")
    if claude is None:
        print("SKIP: Claude Code executable `claude` is not installed.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="ficelle-claude-code-") as temporary:
        temporary_path = Path(temporary)
        os.environ["FICELLE_HOME"] = str(temporary_path / "ficelle-home")
        os.environ["FICELLE_RUNTIME_DIR"] = str(temporary_path / "ficelle-runtime")
        os.environ["HERMES_HOME"] = str(temporary_path / "hermes-home")

        from ficelle import router

        router.ensure_access_tokens()
        config = router.load_config()
        config["auto_benchmark_enabled"] = False
        config["anthropic_model_mapping"] = {
            "claude-sonnet-4-5": "ficelle/auto-fast"
        }
        selected = {
            "id": "ficelle/openrouter/claude-code-smoke:free",
            "source": "openrouter",
            "upstream_id": "claude-code-smoke:free",
            "name": "Claude Code protocol smoke",
            "context_length": 200_000,
            "supports_tools": True,
            "supports_structured_outputs": True,
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["tools", "tool_choice", "parallel_tool_calls"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "invokable": True,
        }
        profile = config["virtual_profiles"]["ficelle/auto-fast"]
        profile.update(
            {
                "mode": "manual_order",
                "models": [selected["id"]],
                "auto_tail": False,
            }
        )
        catalog = {"models": [selected]}
        routed_requests: list[dict[str, Any]] = []
        main_turns: list[dict[str, Any]] = []

        def invoke_model(_model, body, _config, **_kwargs):
            routed_requests.append(copy.deepcopy(body))
            tool_results = [message for message in body["messages"] if message.get("role") == "tool"]
            tool_names = [
                tool.get("function", {}).get("name")
                for tool in body.get("tools", [])
                if isinstance(tool, dict)
            ]
            if not tool_results:
                tool_name = next((name for name in tool_names if name == "Bash"), None)
                if tool_name is None:
                    return text_response("Ficelle protocol smoke", "chatcmpl-auxiliary")
                main_turns.append(copy.deepcopy(body))
                return tool_call_response(tool_name)
            main_turns.append(copy.deepcopy(body))
            result = tool_results[-1]
            if result.get("tool_call_id") != TOOL_CALL_ID:
                raise RuntimeError("Claude Code changed the tool-call id between turns")
            if EXPECTED_TOOL_RESULT not in str(result.get("content") or ""):
                raise RuntimeError("Claude Code did not return the Bash output")
            return final_response()

        router._load_or_refresh_catalog_for_effective_config = lambda _config: catalog
        router.invoke_model = invoke_model
        router.license_ops.is_entitled = lambda: True

        class SmokeRouterHandler(router.RouterHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return None

        server = router.ThreadingHTTPServer(("127.0.0.1", 0), SmokeRouterHandler)
        server.config = config
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            environment = os.environ.copy()
            environment["ANTHROPIC_BASE_URL"] = (
                f"http://127.0.0.1:{server.server_address[1]}"
            )
            environment["ANTHROPIC_API_KEY"] = router.api_token()
            environment.pop("ANTHROPIC_AUTH_TOKEN", None)
            command = [
                claude,
                "--bare",
                "--print",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--dangerously-skip-permissions",
                "--allowedTools",
                "Bash",
                "--model",
                "claude-sonnet-4-5",
                (
                    f"Use Bash exactly once to run `printf {EXPECTED_TOOL_RESULT}`. "
                    f"After the tool result, reply exactly: {EXPECTED_FINAL}"
                ),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=temporary_path,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                print((exc.stdout or "")[-4000:], file=sys.stderr)
                print((exc.stderr or "")[-4000:], file=sys.stderr)
                return 1
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        if completed.returncode != 0:
            print(completed.stdout[-4000:], file=sys.stderr)
            print(completed.stderr[-4000:], file=sys.stderr)
            if router.ROUTE_LOG_PATH.exists():
                print(router.ROUTE_LOG_PATH.read_text(encoding="utf-8")[-8000:], file=sys.stderr)
            return completed.returncode or 1
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            print(completed.stdout[-4000:], file=sys.stderr)
            return 1
        if result.get("result") != EXPECTED_FINAL:
            print(json.dumps(result, indent=2)[:4000], file=sys.stderr)
            return 1
        if len(main_turns) != 2:
            print(
                f"Expected two main routed turns, got {len(main_turns)} "
                f"({len(routed_requests)} including Claude Code auxiliary calls)",
                file=sys.stderr,
            )
            print(
                json.dumps(
                    [
                        [
                            {
                                "role": message.get("role"),
                                "tool_call_id": message.get("tool_call_id"),
                                "content": str(message.get("content"))[:200],
                            }
                            for message in request.get("messages", [])
                        ]
                        for request in routed_requests
                    ],
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        print(
            "PASS: Claude Code completed a streamed two-turn Ficelle session with one Bash tool call."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
