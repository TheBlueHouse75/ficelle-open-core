"""Translation between Anthropic Messages and Ficelle's OpenAI wire format.

The router deliberately keeps protocol translation at its edge.  This module only shapes
requests, responses, and SSE events; it does not select providers or make claims about the
Anthropic model names presented by a client.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import json
import re
from typing import Any, Mapping, Sequence


ANTHROPIC_MESSAGES_PATH = "/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
_SUPPORTED_REQUEST_FIELDS = {
    "cache_control",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_p",
}

# Keep this table explicit and operator-visible.  Prefix matching below only fills in date
# aliases from the same documented family; it never turns a provider model id into a Claude claim.
DEFAULT_ANTHROPIC_MODEL_MAPPING: dict[str, str] = {
    "claude-code": "ficelle/auto-coding",
    "claude-code/haiku*": "ficelle/auto-fast",
    "claude-code/sonnet*": "ficelle/auto-coding",
    "claude-code/opus*": "ficelle/auto-orchestrator",
    "*haiku*": "ficelle/auto-fast",
    "*sonnet*": "ficelle/auto-coding",
    "*opus*": "ficelle/auto-orchestrator",
    "*fable*": "ficelle/auto-orchestrator",
    "claude-code*": "ficelle/auto-coding",
    "claude*": "ficelle/auto-coding",
}


class AnthropicRequestError(ValueError):
    """A client-facing request error which can safely be rendered as Anthropic JSON."""


class AnthropicResponseError(RuntimeError):
    """The upstream response cannot be represented without inventing Anthropic data."""


@dataclass(frozen=True)
class AnthropicRequestTranslation:
    openai_body: dict[str, Any]
    requested_model: str
    routed_model: str
    ignored_fields: tuple[str, ...]
    stop_sequences: tuple[str, ...]


def _record_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    ignored: set[str],
    label: str,
) -> None:
    if any(field not in allowed for field in value):
        ignored.add(label)


def _json_body(value: Any) -> dict[str, Any]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = json.loads(bytes(value).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnthropicRequestError("request body must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise AnthropicRequestError("request body must be a JSON object")
    return dict(value)


def _text_from_blocks(blocks: Any, *, path: str, ignored: set[str]) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise AnthropicRequestError(f"{path} must be a string or content-block array")
    parts: list[str] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise AnthropicRequestError(f"{path}[{index}] must be an object")
        block_type = block.get("type")
        if block_type == "text":
            _record_unknown_fields(
                block,
                {"type", "text", "cache_control"},
                ignored,
                "system_block_fields",
            )
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            else:
                raise AnthropicRequestError(f"{path}[{index}].text must be a string")
        elif block_type in {"thinking", "redacted_thinking"}:
            ignored.add("thinking")
        else:
            raise AnthropicRequestError(f"unsupported content block type: {block_type!r}")
        if "cache_control" in block:
            ignored.add("cache_control")
    return "".join(parts)


def _image_part(block: Mapping[str, Any], path: str) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, Mapping):
        raise AnthropicRequestError(f"{path}.source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        if any(field not in {"type", "media_type", "data"} for field in source):
            raise AnthropicRequestError(f"{path}.source contains unsupported fields")
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise AnthropicRequestError(f"{path}.source base64 image is incomplete")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        if any(field not in {"type", "url"} for field in source):
            raise AnthropicRequestError(f"{path}.source contains unsupported fields")
        url = source.get("url")
        if not isinstance(url, str) or not url:
            raise AnthropicRequestError(f"{path}.source.url must be a URL")
    else:
        raise AnthropicRequestError(f"unsupported image source type: {source_type!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_content(content: Any, *, path: str, ignored: set[str]) -> Any:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        pieces: list[Any] = []
        has_rich_content = False
        for index, block in enumerate(content):
            block_path = f"{path}[{index}]"
            if not isinstance(block, Mapping):
                raise AnthropicRequestError(f"{block_path} must be an object")
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                _record_unknown_fields(
                    block,
                    {"type", "text", "cache_control"},
                    ignored,
                    "tool_result_block_fields",
                )
                pieces.append(str(block["text"]))
            elif block.get("type") == "image":
                _record_unknown_fields(
                    block,
                    {"type", "source", "cache_control"},
                    ignored,
                    "tool_result_block_fields",
                )
                pieces.append(_image_part(block, block_path))
                has_rich_content = True
            elif block.get("type") in {"thinking", "redacted_thinking"}:
                ignored.add("thinking")
            else:
                raise AnthropicRequestError(f"unsupported tool_result block type: {block.get('type')!r}")
            if "cache_control" in block:
                ignored.add("cache_control")
        if has_rich_content:
            return [
                {"type": "text", "text": piece} if isinstance(piece, str) else piece
                for piece in pieces
            ]
        return "".join(str(piece) for piece in pieces)
    # Anthropic allows JSON-ish tool result values in practice; avoid Python repr on the wire.
    try:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise AnthropicRequestError(f"{path} cannot be serialized") from exc


def _message(message: Any, index: int, ignored: set[str]) -> list[dict[str, Any]]:
    path = f"messages[{index}]"
    if not isinstance(message, Mapping):
        raise AnthropicRequestError(f"{path} must be an object")
    _record_unknown_fields(
        message,
        {"role", "content"},
        ignored,
        "message_fields",
    )
    role = message.get("role")
    content = message.get("content", "")
    if role not in {"user", "assistant"}:
        raise AnthropicRequestError(f"{path}.role must be user or assistant")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise AnthropicRequestError(f"{path}.content must be a string or content-block array")

    result: list[dict[str, Any]] = []
    regular: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block_index, block in enumerate(content):
        block_path = f"{path}.content[{block_index}]"
        if not isinstance(block, Mapping):
            raise AnthropicRequestError(f"{block_path} must be an object")
        kind = block.get("type")
        if kind == "text":
            _record_unknown_fields(
                block,
                {"type", "text", "cache_control"},
                ignored,
                "content_block_fields",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise AnthropicRequestError(f"{block_path}.text must be a string")
            regular.append({"type": "text", "text": text})
        elif kind == "image":
            _record_unknown_fields(
                block,
                {"type", "source", "cache_control"},
                ignored,
                "content_block_fields",
            )
            if role != "user":
                raise AnthropicRequestError("image blocks are only valid in user messages")
            regular.append(_image_part(block, block_path))
        elif kind == "thinking" or kind == "redacted_thinking":
            ignored.add("thinking")
        elif kind == "tool_use":
            _record_unknown_fields(
                block,
                {"type", "id", "name", "input", "cache_control"},
                ignored,
                "content_block_fields",
            )
            if role != "assistant":
                raise AnthropicRequestError("tool_use blocks are only valid in assistant messages")
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not tool_id or not isinstance(name, str) or not name:
                raise AnthropicRequestError(f"{block_path} tool_use requires id and name")
            arguments = block.get("input", {})
            if not isinstance(arguments, Mapping):
                raise AnthropicRequestError(f"{block_path}.input must be an object")
            try:
                argument_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise AnthropicRequestError(f"{block_path}.input is not JSON serializable") from exc
            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {"name": name, "arguments": argument_text},
            })
        elif kind == "tool_result":
            _record_unknown_fields(
                block,
                {"type", "tool_use_id", "content", "is_error", "cache_control"},
                ignored,
                "content_block_fields",
            )
            if role != "user":
                raise AnthropicRequestError("tool_result blocks are only valid in user messages")
            tool_id = block.get("tool_use_id")
            if not isinstance(tool_id, str) or not tool_id:
                raise AnthropicRequestError(f"{block_path}.tool_use_id is required")
            if "is_error" in block:
                if not isinstance(block["is_error"], bool):
                    raise AnthropicRequestError(f"{block_path}.is_error must be a boolean")
                ignored.add("tool_result.is_error")
            result.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": _tool_result_content(block.get("content", ""), path=block_path, ignored=ignored),
            })
        else:
            raise AnthropicRequestError(f"unsupported content block type: {kind!r}")
        if "cache_control" in block:
            ignored.add("cache_control")
    if regular or tool_calls:
        content_value: Any = regular
        if regular and all(part.get("type") == "text" for part in regular):
            content_value = "".join(str(part["text"]) for part in regular)
        elif not regular:
            content_value = None
        entry: dict[str, Any] = {"role": role, "content": content_value}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        # OpenAI requires each assistant tool call to be followed immediately by its tool
        # result. Anthropic permits a later user turn to carry both results and new text, so
        # split that turn with tool results first and the new user text afterwards.
        if role == "user" and result:
            result.append(entry)
        else:
            result.insert(0, entry)
    return result


def _mapped_model(model: str, mapping: Mapping[str, Any]) -> str:
    if model.startswith("ficelle/"):
        return model
    exact = mapping.get(model)
    if isinstance(exact, str) and exact:
        return exact
    for key, value in mapping.items():
        if (
            isinstance(key, str)
            and "*" in key
            and fnmatch.fnmatchcase(model.lower(), key.lower())
            and isinstance(value, str)
            and value
        ):
            return value
    raise AnthropicRequestError(
        f"model {model!r} has no Ficelle profile mapping; configure anthropic_model_mapping"
    )


def translate_anthropic_request(
    body: Any,
    model_mapping: Mapping[str, Any] | None = None,
) -> AnthropicRequestTranslation:
    """Translate one Anthropic Messages request into a chat-completions request."""
    source = _json_body(body)
    requested_model = source.get("model")
    if not isinstance(requested_model, str) or not requested_model:
        raise AnthropicRequestError("model is required")
    max_tokens = source.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise AnthropicRequestError("max_tokens is required and must be a positive integer")
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise AnthropicRequestError("messages must be an array")
    ignored: set[str] = set()
    unknown_fields = {field for field in source if field not in _SUPPORTED_REQUEST_FIELDS}
    if len(unknown_fields) > 32:
        raise AnthropicRequestError("request contains too many unsupported fields")
    if any(
        not isinstance(field, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", field)
        for field in unknown_fields
    ):
        raise AnthropicRequestError("unsupported field names must use safe ASCII identifiers")
    ignored.update(unknown_fields)
    metadata = source.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise AnthropicRequestError("metadata must be an object")
    if "thinking" in source:
        ignored.add("thinking")
    if "cache_control" in source:
        ignored.add("cache_control")
    translated_messages: list[dict[str, Any]] = []
    if "system" in source:
        system = source["system"]
        if isinstance(system, str):
            system_text = system
        else:
            system_text = _text_from_blocks(system, path="system", ignored=ignored)
        if system_text:
            translated_messages.append({"role": "system", "content": system_text})
    for index, message in enumerate(messages):
        translated_messages.extend(_message(message, index, ignored))
    stop_value = source.get("stop_sequences", [])
    if stop_value is None:
        stop_value = []
    if not isinstance(stop_value, list) or any(not isinstance(item, str) for item in stop_value):
        raise AnthropicRequestError("stop_sequences must be an array of strings")
    stop_sequences = tuple(stop_value)
    mapping = model_mapping if model_mapping is not None else DEFAULT_ANTHROPIC_MODEL_MAPPING
    if not isinstance(mapping, Mapping):
        raise AnthropicRequestError("anthropic_model_mapping must be an object")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value.startswith("ficelle/")
        for key, value in mapping.items()
    ):
        raise AnthropicRequestError(
            "anthropic_model_mapping entries must map non-empty names to Ficelle profiles"
        )
    routed_model = _mapped_model(requested_model, mapping)
    openai_body: dict[str, Any] = {
        "model": routed_model,
        "messages": translated_messages,
        "max_tokens": max_tokens,
    }
    for field in ("temperature", "top_p"):
        if field in source:
            value = source[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AnthropicRequestError(f"{field} must be a number")
            if not 0 <= value <= 1:
                raise AnthropicRequestError(f"{field} must be between 0 and 1")
            openai_body[field] = value
    if stop_sequences:
        openai_body["stop"] = list(stop_sequences)
    if source.get("stream") is not None:
        if not isinstance(source["stream"], bool):
            raise AnthropicRequestError("stream must be a boolean")
        openai_body["stream"] = source["stream"]
        # `stream_options` is not portable across OpenAI-compatible providers. Translate usage
        # whenever an upstream reports it, but never make an otherwise valid stream depend on this
        # optional extension.
    if metadata is not None:
        openai_body["metadata"] = dict(metadata)
    tool_choice = source.get("tool_choice")
    if tool_choice is not None:
        if not isinstance(tool_choice, Mapping):
            raise AnthropicRequestError("tool_choice must be an object")
        _record_unknown_fields(
            tool_choice,
            {"type", "name", "disable_parallel_tool_use"},
            ignored,
            "tool_choice_fields",
        )
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none"}:
            openai_body["tool_choice"] = choice_type
        elif choice_type == "any":
            openai_body["tool_choice"] = "required"
        elif choice_type == "tool" and isinstance(tool_choice.get("name"), str) and tool_choice["name"]:
            openai_body["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
        else:
            raise AnthropicRequestError("tool_choice must be auto, any, none, or a named tool")
        if "disable_parallel_tool_use" in tool_choice:
            disabled = tool_choice["disable_parallel_tool_use"]
            if not isinstance(disabled, bool):
                raise AnthropicRequestError("tool_choice.disable_parallel_tool_use must be a boolean")
            openai_body["parallel_tool_calls"] = not disabled
    tools = source.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise AnthropicRequestError("tools must be an array")
        openai_tools: list[dict[str, Any]] = []
        for index, tool in enumerate(tools):
            if (
                not isinstance(tool, Mapping)
                or not isinstance(tool.get("name"), str)
                or not tool["name"]
            ):
                raise AnthropicRequestError(f"tools[{index}] requires a name")
            _record_unknown_fields(
                tool,
                {"name", "description", "input_schema", "cache_control"},
                ignored,
                "tool_fields",
            )
            schema = tool.get("input_schema")
            if not isinstance(schema, Mapping):
                raise AnthropicRequestError(f"tools[{index}].input_schema must be an object")
            fn: dict[str, Any] = {"name": tool["name"], "parameters": dict(schema)}
            if isinstance(tool.get("description"), str):
                fn["description"] = tool["description"]
            openai_tools.append({"type": "function", "function": fn})
            if "cache_control" in tool:
                ignored.add("cache_control")
        openai_body["tools"] = openai_tools
    return AnthropicRequestTranslation(
        openai_body=openai_body,
        requested_model=requested_model,
        routed_model=routed_model,
        ignored_fields=tuple(sorted(ignored)),
        stop_sequences=stop_sequences,
    )


def _stop_reason(
    finish_reason: Any,
    text: str,
    stop_sequences: Sequence[str],
    explicit_stop_sequence: Any = None,
) -> tuple[str, str | None]:
    if finish_reason in {"length", "max_tokens"}:
        return "max_tokens", None
    if finish_reason in {"tool_calls", "function_call", "tool_use"}:
        return "tool_use", None
    if finish_reason in {"stop", "end_turn", None}:
        if (
            isinstance(explicit_stop_sequence, str)
            and explicit_stop_sequence in stop_sequences
        ):
            return "stop_sequence", explicit_stop_sequence
        for sequence in stop_sequences:
            if sequence and text.endswith(sequence):
                return "stop_sequence", sequence
        # OpenAI's standard wire format strips the matched text and reports only `stop`, so
        # a single configured sequence is the one case whose identity remains unambiguous.
        if finish_reason == "stop" and len(stop_sequences) == 1:
            return "stop_sequence", stop_sequences[0]
        return "end_turn", None
    return "end_turn", None


def _message_id(request_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "", request_id)
    return "msg_" + (clean or "request")


def translate_openai_response(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    routed_model: str,
    stop_sequences: Sequence[str] = (),
) -> dict[str, Any]:
    """Render a non-streamed chat completion as an Anthropic message."""
    if not isinstance(payload, Mapping):
        raise ValueError("upstream response must be an object")
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice, Mapping) else {}
    if not isinstance(message, Mapping):
        message = {}
    content: list[dict[str, Any]] = []
    message_content = message.get("content")
    refusal = message.get("refusal")
    text = ""
    if isinstance(message_content, str):
        if message_content:
            text = message_content
            content.append({"type": "text", "text": message_content})
    elif isinstance(message_content, list):
        for block in message_content:
            if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text += str(block["text"])
                content.append({"type": "text", "text": block["text"]})
            else:
                raise AnthropicResponseError("upstream response contains an unsupported content block")
    elif message_content is not None:
        raise AnthropicResponseError("upstream response content has an unsupported shape")
    if not text and isinstance(refusal, str) and refusal:
        text = refusal
        content.append({"type": "text", "text": refusal})
    if message.get("audio") is not None:
        raise AnthropicResponseError("upstream audio output cannot be represented as Anthropic content")
    if message.get("function_call") is not None:
        raise AnthropicResponseError(
            "legacy upstream function_call has no tool-call id to preserve"
        )
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise AnthropicResponseError("upstream tool call must be an object")
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise AnthropicResponseError("upstream tool call is missing its function")
            call_id = call.get("id")
            name = function.get("name")
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
            ):
                raise AnthropicResponseError("upstream tool call is missing its id or name")
            raw_args = function.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise AnthropicResponseError(
                        f"tool arguments for {call_id} are not valid JSON"
                    ) from exc
            else:
                arguments = raw_args
            if not isinstance(arguments, dict):
                raise AnthropicResponseError(
                    f"tool arguments for {call_id} must be a JSON object"
                )
            content.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
    elif tool_calls is not None:
        raise AnthropicResponseError("upstream tool_calls must be an array")
    reason, stop_sequence = _stop_reason(
        choice.get("finish_reason"),
        text,
        stop_sequences,
        choice.get("stop_sequence") or choice.get("matched_stop"),
    )
    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
    output: dict[str, Any] = {
        "id": _message_id(request_id),
        "type": "message",
        "role": "assistant",
        "model": routed_model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }
    return output


_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    408: "api_error",
    409: "api_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
    529: "overloaded_error",
}


def translate_openai_error(status: int, payload: Mapping[str, Any], request_id: str) -> dict[str, Any]:
    """Convert an already-safe Ficelle/OpenAI error into Anthropic's envelope."""
    source = payload.get("error") if isinstance(payload, Mapping) else None
    source = source if isinstance(source, Mapping) else {}
    message = source.get("message")
    safe_message = str(message)[:2000] if isinstance(message, str) and message else "request failed"
    error: dict[str, Any] = {
        "type": _ERROR_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error"),
        "message": safe_message,
    }
    safe_keys = ("code", "requested_model", "candidate_count", "attempt_count", "reasons", "last_error", "details", "actions")
    for key in safe_keys:
        if key in source:
            value = source[key]
            if key in {"code", "requested_model"} and isinstance(value, str):
                error[key] = value[:250]
            elif key in {"candidate_count", "attempt_count"} and isinstance(value, int) and not isinstance(value, bool):
                error[key] = value
            elif key in {"reasons", "last_error", "details", "actions"} and isinstance(value, (dict, list)):
                error[key] = value
    return {"type": "error", "error": error, "request_id": request_id}


def encode_anthropic_sse_event(event: str, data: Any) -> bytes:
    """Encode one Anthropic SSE event, accepting either structured data or a JSON string."""
    if isinstance(data, str):
        encoded = data
    else:
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n".encode("utf-8")


def anthropic_sse_error_event(code: str, detail: str | None = None, *, error_type: str = "api_error") -> bytes:
    """Encode a stream failure with a valid Anthropic error type and Ficelle diagnostic code."""
    return encode_anthropic_sse_event(
        "error",
        {"type": "error", "error": {"type": error_type, "message": detail or code, "ficelle_code": code}},
    )


def encode_anthropic_message_stream(message: Mapping[str, Any]) -> list[bytes]:
    """Encode a complete translated message as a valid Anthropic event stream."""
    usage = message.get("usage") if isinstance(message.get("usage"), Mapping) else {}
    output = [
        encode_anthropic_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message.get("id"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": message.get("model"),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": 0,
                    },
                },
            },
        ),
        encode_anthropic_sse_event("ping", {"type": "ping"}),
    ]
    content = message.get("content")
    if not isinstance(content, list):
        raise AnthropicResponseError("translated Anthropic content must be an array")
    for index, block in enumerate(content):
        if not isinstance(block, Mapping):
            raise AnthropicResponseError("translated Anthropic content block must be an object")
        block_type = block.get("type")
        if block_type == "text":
            public_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        elif block_type == "tool_use":
            public_block = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    block.get("input", {}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        else:
            raise AnthropicResponseError("translated Anthropic content block is unsupported")
        output.extend(
            [
                encode_anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": public_block,
                    },
                ),
                encode_anthropic_sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                ),
                encode_anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": index},
                ),
            ]
        )
    output.extend(
        [
            encode_anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": message.get("stop_reason"),
                        "stop_sequence": message.get("stop_sequence"),
                    },
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    },
                },
            ),
            encode_anthropic_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    return output


class OpenAIToAnthropicStreamTranslator:
    """Incrementally translate OpenAI SSE chunks, including fragmented tool arguments."""

    def __init__(self, request_id: str, routed_model: str, stop_sequences: Sequence[str] = ()) -> None:
        self.request_id = request_id
        self.routed_model = routed_model
        self.stop_sequences = tuple(stop_sequences)
        self._buffer = b""
        self._started = False
        self._closed = False
        self._failed = False
        self._error_code: str | None = None
        self._error_detail: str | None = None
        self._text = ""
        self._finish_reason: str | None = None
        self._usage: dict[str, Any] = {}
        self._explicit_stop_sequence: str | None = None
        self._text_index: int | None = None
        self._tools: dict[int, dict[str, Any]] = {}

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def error_detail(self) -> str | None:
        return self._error_detail

    @property
    def error_code(self) -> str | None:
        return self._error_code

    def _start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [
            encode_anthropic_sse_event("message_start", {
                "type": "message_start",
                "message": {
                    "id": _message_id(self.request_id),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.routed_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }),
            encode_anthropic_sse_event("ping", {"type": "ping"}),
        ]

    def _fail(
        self,
        detail: str,
        *,
        code: str = "invalid_tool_arguments",
    ) -> list[bytes]:
        self._failed = True
        self._error_code = code
        self._error_detail = detail[:2000]
        return [anthropic_sse_error_event(code, self._error_detail)]

    def _chunk(self, payload: Mapping[str, Any]) -> list[bytes]:
        output: list[bytes] = []
        choices = payload.get("choices")
        if isinstance(payload.get("usage"), Mapping):
            self._usage = dict(payload["usage"])
        if not isinstance(choices, list) or not choices:
            return output
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                message = choice.get("message")
                delta = message if isinstance(message, Mapping) else {}
            content = delta.get("content")
            if not content and isinstance(delta.get("refusal"), str):
                content = delta["refusal"]
            if isinstance(content, str) and content:
                if self._text_index is None:
                    self._text_index = 0
                    output.append(
                        encode_anthropic_sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": self._text_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                self._text += content
                output.append(
                    encode_anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._text_index,
                            "delta": {"type": "text_delta", "text": content},
                        },
                    )
                )
            elif content is not None and content != "":
                return output + self._fail(
                    "upstream stream contains unsupported content blocks",
                    code="unsupported_stream_content",
                )
            if delta.get("audio") is not None:
                return output + self._fail(
                    "upstream audio output cannot be represented as Anthropic content",
                    code="unsupported_stream_content",
                )
            if delta.get("function_call") is not None:
                return output + self._fail(
                    "legacy upstream function_call has no tool-call id to preserve",
                    code="unsupported_function_call",
                )
            calls = delta.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, Mapping):
                        continue
                    index = call.get("index", 0)
                    if isinstance(index, bool) or not isinstance(index, int):
                        index = 0
                    function = (
                        call.get("function")
                        if isinstance(call.get("function"), Mapping)
                        else {}
                    )
                    tool = self._tools.setdefault(
                        index,
                        {"id": "", "name": "", "argument_fragments": []},
                    )
                    if isinstance(call.get("id"), str) and call["id"]:
                        tool["id"] = call["id"]
                    if isinstance(function.get("name"), str) and function["name"]:
                        tool["name"] = function["name"]
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        tool["argument_fragments"].append(arguments)
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                if finish.strip().lower() == "error":
                    return output + self._fail(
                        "upstream response ended with an error finish reason",
                        code="upstream_finish_error",
                    )
                self._finish_reason = finish
            explicit_stop = choice.get("stop_sequence") or choice.get("matched_stop")
            if isinstance(explicit_stop, str):
                self._explicit_stop_sequence = explicit_stop
        return output

    def _done(self) -> list[bytes]:
        if self._closed or self._failed:
            return []
        for tool in self._tools.values():
            if not tool["id"] or not tool["name"]:
                return self._fail("tool call ended without an id and name")
            arguments = "".join(tool["argument_fragments"])
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return self._fail("tool arguments ended before a complete JSON object was received")
            if not isinstance(parsed, dict):
                return self._fail("tool arguments must be a JSON object")
        output: list[bytes] = []
        if self._text_index is not None:
            output.append(
                encode_anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._text_index},
                )
            )
        next_block_index = 1 if self._text_index is not None else 0
        for tool_offset, tool_index in enumerate(sorted(self._tools)):
            block_index = next_block_index + tool_offset
            tool = self._tools[tool_index]
            output.append(
                encode_anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool["id"],
                            "name": tool["name"],
                            "input": {},
                        },
                    },
                )
            )
            output.extend(
                encode_anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": fragment,
                        },
                    },
                )
                for fragment in tool["argument_fragments"]
            )
            output.append(
                encode_anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": block_index},
                )
            )
        reason, stop_sequence = _stop_reason(
            self._finish_reason,
            self._text,
            self.stop_sequences,
            self._explicit_stop_sequence,
        )
        usage = {
            "input_tokens": int(self._usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(self._usage.get("completion_tokens", 0) or 0),
        }
        output.extend([
            encode_anthropic_sse_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": stop_sequence}, "usage": usage}),
            encode_anthropic_sse_event("message_stop", {"type": "message_stop"}),
        ])
        self._closed = True
        return output

    def _event(self, raw: bytes) -> list[bytes]:
        lines = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        data = "\n".join(line[5:].lstrip() for line in lines if line.startswith("data:"))
        if not data:
            return []
        if data.strip() == "[DONE]":
            return self._done()
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return self._fail("upstream returned malformed SSE JSON", code="invalid_sse")
        if isinstance(payload, Mapping) and isinstance(payload.get("error"), Mapping):
            # The shared stream core owns upstream-error classification and emits exactly one
            # protocol-specific terminal frame after deciding whether fallback is still legal.
            # Mark the translator failed so a later [DONE] in this same transport chunk cannot
            # emit message_stop before that error frame reaches the client.
            self._failed = True
            self._error_code = "upstream_stream_error"
            error = payload["error"]
            self._error_detail = str(error.get("message") or "upstream stream failed")[:2000]
            return []
        if not isinstance(payload, Mapping):
            return []
        return self._chunk(payload)

    def feed(self, chunk: bytes) -> list[bytes]:
        if self._closed or self._failed:
            return []
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("stream chunks must be bytes")
        output: list[bytes] = []
        if chunk and not self._started:
            output.extend(self._start())
        self._buffer += bytes(chunk)
        while True:
            lf_boundary = self._buffer.find(b"\n\n")
            crlf_boundary = self._buffer.find(b"\r\n\r\n")
            candidates = [
                (boundary, width)
                for boundary, width in ((lf_boundary, 2), (crlf_boundary, 4))
                if boundary >= 0
            ]
            boundary, width = min(candidates, default=(-1, 0))
            if boundary < 0:
                break
            event = self._buffer[:boundary]
            self._buffer = self._buffer[boundary + width :]
            output.extend(self._event(event))
        return output
