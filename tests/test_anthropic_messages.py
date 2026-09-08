import json

import pytest

from ficelle.anthropic_messages import (
    ANTHROPIC_MESSAGES_PATH,
    ANTHROPIC_VERSION,
    DEFAULT_ANTHROPIC_MODEL_MAPPING,
    AnthropicRequestError,
    AnthropicResponseError,
    OpenAIToAnthropicStreamTranslator,
    anthropic_sse_error_event,
    encode_anthropic_sse_event,
    translate_anthropic_request,
    translate_openai_error,
    translate_openai_response,
)


def event_json(raw: bytes) -> tuple[str, dict]:
    lines = raw.decode().splitlines()
    return lines[0].split(": ", 1)[1], json.loads(lines[1][6:])


def test_constants_and_request_translation_cover_multimodal_tools_and_ignored_fields():
    assert ANTHROPIC_MESSAGES_PATH == "/v1/messages"
    assert ANTHROPIC_VERSION == "2023-06-01"
    assert DEFAULT_ANTHROPIC_MODEL_MAPPING["claude-code/haiku*"] == "ficelle/auto-fast"
    translated = translate_anthropic_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 128,
            "system": [{"type": "text", "text": "You are useful", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Read this"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                ]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "read", "input": {"path": "x"}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": [
                                {"type": "text", "text": "done"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.invalid/result.png",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                    "future_option": True,
                }
            ],
            "temperature": 0,
            "top_p": 0.5,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "u"},
            "thinking": {"type": "enabled"},
            "tool_choice": {"type": "tool", "name": "read", "disable_parallel_tool_use": True},
            "context_management": {"edits": []},
            "stream": True,
        }
    )
    assert translated.requested_model == "claude-sonnet-4-20250514"
    assert translated.routed_model == "ficelle/auto-coding"
    assert translated.stop_sequences == ("END",)
    assert set(translated.ignored_fields) == {
        "cache_control",
        "context_management",
        "thinking",
        "tool_fields",
    }
    body = translated.openai_body
    assert body["metadata"] == {"user_id": "u"}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["messages"][2]["tool_calls"][0]["id"] == "tool-1"
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "tool-1",
        "content": [
            {"type": "text", "text": "done"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/result.png"},
            },
        ],
    }
    assert body["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert body["tool_choice"] == {"type": "function", "function": {"name": "read"}}
    assert body["parallel_tool_calls"] is False
    assert "stream_options" not in body


def test_request_requires_positive_max_tokens():
    for value in (None, 0, -1, True):
        body = {"model": "claude-haiku", "messages": [], "max_tokens": value}
        if value is None:
            body.pop("max_tokens")
        try:
            translate_anthropic_request(body)
        except AnthropicRequestError:
            pass
        else:
            raise AssertionError("invalid max_tokens was accepted")


def test_request_requires_an_explicit_model_mapping():
    try:
        translate_anthropic_request(
            {"model": "vendor-model", "messages": [], "max_tokens": 16},
            {},
        )
    except AnthropicRequestError as exc:
        assert "anthropic_model_mapping" in str(exc)
    else:
        raise AssertionError("an unmapped provider model bypassed the Ficelle profile mapping")

    with pytest.raises(AnthropicRequestError, match="safe ASCII"):
        translate_anthropic_request(
            {
                "model": "claude-haiku",
                "messages": [],
                "max_tokens": 16,
                "unsafe field": True,
            }
        )


def test_response_translation_preserves_tool_id_stop_reason_and_usage():
    result = translate_openai_response(
        {
            "choices": [{
                "message": {
                    "content": "answer",
                    "tool_calls": [{"id": "tool-7", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        },
        request_id="request-7",
        routed_model="ficelle/auto-coding",
    )
    assert result["id"] == "msg_request-7"
    assert result["model"] == "ficelle/auto-coding"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][1] == {"type": "tool_use", "id": "tool-7", "name": "lookup", "input": {"q": "x"}}
    assert result["usage"] == {"input_tokens": 9, "output_tokens": 4}

    with pytest.raises(AnthropicResponseError, match="missing its id or name"):
        translate_openai_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            request_id="request-invalid",
            routed_model="ficelle/auto-coding",
        )


def test_response_translation_maps_each_supported_stop_reason():
    cases = [
        ("stop", "answer", (), "end_turn", None),
        ("length", "answer", (), "max_tokens", None),
        ("tool_calls", "", (), "tool_use", None),
        ("stop", "answer", ("<END>",), "stop_sequence", "<END>"),
    ]
    for finish_reason, text, sequences, expected_reason, expected_sequence in cases:
        result = translate_openai_response(
            {
                "choices": [
                    {
                        "message": {"content": text},
                        "finish_reason": finish_reason,
                    }
                ]
            },
            request_id="stops",
            routed_model="ficelle/auto-fast",
            stop_sequences=sequences,
        )
        assert (result["stop_reason"], result["stop_sequence"]) == (
            expected_reason,
            expected_sequence,
        )


def test_errors_and_sse_encoder_are_safe_anthropic_shapes():
    error = translate_openai_error(503, {"error": {"message": "temporarily unavailable", "details": [{"provider": "x"}], "raw": "secret"}}, "req")
    assert error["type"] == "error"
    assert error["error"]["type"] == "overloaded_error"
    assert error["request_id"] == "req"
    assert "raw" not in error["error"]
    assert event_json(encode_anthropic_sse_event("ping", {"type": "ping"}))[0] == "ping"
    assert event_json(anthropic_sse_error_event("api_error", "nope"))[1]["error"]["message"] == "nope"
    for status in (408, 409, 504):
        assert translate_openai_error(status, {}, "req")["error"]["type"] == "api_error"
    assert translate_openai_error(402, {}, "req")["error"]["type"] == "billing_error"
    assert translate_openai_error(529, {}, "req")["error"]["type"] == "overloaded_error"


def test_stream_translates_fragmented_text_and_usage_to_anthropic_sequence():
    translator = OpenAIToAnthropicStreamTranslator("stream-1", "ficelle/auto-fast")
    wire = (
        b'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\r\n\r\n'
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}\r\n\r\n'
        b'data: [DONE]\r\n\r\n'
    )
    events = []
    for byte in wire:
        events.extend(translator.feed(bytes([byte])))
    names = [event_json(item)[0] for item in events]
    assert names == ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert event_json(events[-2])[1]["usage"] == {"input_tokens": 2, "output_tokens": 3}
    assert not translator.failed


def test_stream_fragmented_tool_arguments_fail_without_message_stop():
    translator = OpenAIToAnthropicStreamTranslator("stream-tool", "ficelle/auto-coding")
    wire = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tool-1","function":{"name":"lookup","arguments":"{\\\"q\\\":"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"x\\\""}}]},"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    events = translator.feed(wire)
    names = [event_json(item)[0] for item in events]
    assert translator.failed
    assert "complete JSON" in (translator.error_detail or "")
    assert "message_stop" not in names


def test_stream_error_event_suppresses_a_later_done_completion():
    translator = OpenAIToAnthropicStreamTranslator("stream-error", "ficelle/auto-fast")
    events = translator.feed(
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        b'event: error\ndata: {"type":"error","error":{"type":"server_error","message":"boom"}}\n\n'
        b'data: [DONE]\n\n'
    )

    names = [event_json(item)[0] for item in events]
    assert translator.failed
    assert translator.error_code == "upstream_stream_error"
    assert "message_delta" not in names
    assert "message_stop" not in names


def test_stream_error_finish_reason_emits_only_a_terminal_error():
    translator = OpenAIToAnthropicStreamTranslator("finish-error", "ficelle/auto-fast")
    events = translator.feed(
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"error"}]}\n\n'
        b'data: [DONE]\n\n'
    )

    names = [event_json(item)[0] for item in events]
    assert translator.failed
    assert translator.error_code == "upstream_finish_error"
    assert names[-1] == "error"
    assert "message_delta" not in names
    assert "message_stop" not in names


def test_stream_closes_text_before_starting_a_tool_block():
    translator = OpenAIToAnthropicStreamTranslator("stream-mixed", "ficelle/auto-coding")
    events = translator.feed(
        b'data: {"choices":[{"delta":{"content":"checking"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tool-1","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    names = [event_json(item)[0] for item in events]
    first_stop = names.index("content_block_stop")
    tool_start = names.index("content_block_start", names.index("content_block_start") + 1)
    assert first_stop < tool_start
    assert names[-2:] == ["message_delta", "message_stop"]


def test_stream_buffers_interleaved_tool_calls_into_sequential_blocks():
    translator = OpenAIToAnthropicStreamTranslator("stream-tools", "ficelle/auto-coding")
    events = translator.feed(
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":0,"id":"tool-1","function":{"name":"first","arguments":"{\\"a\\":"}},'
        b'{"index":1,"id":"tool-2","function":{"name":"second","arguments":"{\\"b\\":"}}]}}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":['
        b'{"index":1,"function":{"arguments":"2}"}},'
        b'{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    decoded = [event_json(item) for item in events]
    starts = [
        payload["content_block"]
        for name, payload in decoded
        if name == "content_block_start"
    ]
    assert [(block["id"], block["name"]) for block in starts] == [
        ("tool-1", "first"),
        ("tool-2", "second"),
    ]
    deltas = [
        payload["delta"]["partial_json"]
        for name, payload in decoded
        if name == "content_block_delta"
    ]
    assert deltas == ['{"a":', "1}", '{"b":', "2}"]
    assert [name for name, _payload in decoded][-2:] == ["message_delta", "message_stop"]


def test_stream_accepts_a_whole_message_event_and_refusal_text():
    translator = OpenAIToAnthropicStreamTranslator("stream-message", "ficelle/auto-fast")
    events = translator.feed(
        b'data: {"choices":[{"message":{"content":"whole answer"},"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    decoded = [event_json(item) for item in events]
    text_deltas = [
        payload["delta"]["text"]
        for name, payload in decoded
        if name == "content_block_delta"
        and payload.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_deltas == ["whole answer"]
    assert [name for name, _payload in decoded][-2:] == ["message_delta", "message_stop"]

    refusal = translate_openai_response(
        {
            "choices": [
                {
                    "message": {"content": None, "refusal": "cannot comply"},
                    "finish_reason": "stop",
                }
            ]
        },
        request_id="refusal",
        routed_model="ficelle/auto-fast",
    )
    assert refusal["content"] == [{"type": "text", "text": "cannot comply"}]
