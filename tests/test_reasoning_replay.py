"""Replaying a thinking model's reasoning trace back into its next request."""

from __future__ import annotations

import json

from ficelle.failures import classify_failure
from ficelle.reasoning_replay import (
    MAX_TRACE_CHARS,
    ReasoningReplayStore,
    ReasoningStreamObserver,
    fallback_trace,
    observe_stream_chunks,
    record_completion_payload,
)


def tool_call(call_id: str = "call_1", name: str = "get_weather") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def body_with_replayed_call(call_id: str = "call_1", **assistant_extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call(call_id)], **assistant_extra},
            {"role": "tool", "tool_call_id": call_id, "content": "24C"},
        ]
    }


def completion(trace: str | list | None, call_id: str = "call_1") -> dict:
    message: dict = {"role": "assistant", "content": "", "tool_calls": [tool_call(call_id)]}
    if trace is not None:
        message["reasoning_content"] = trace
    return {"choices": [{"index": 0, "message": message}]}


def stream_frame(delta: dict) -> bytes:
    return b"data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}).encode() + b"\n\n"


def test_an_unseen_model_gets_its_body_back_untouched():
    """The default has to be inaction: most providers have never heard of the field.

    Identity, not equality — a rewrite that reached every request would be a fix for one
    upstream turning into a mystery on the other 228.
    """
    store = ReasoningReplayStore()
    body = body_with_replayed_call()

    assert store.replay_into_body("some/model", body, now=100.0) is body


def test_the_trace_a_model_emitted_comes_back_on_the_next_turn():
    store = ReasoningReplayStore()
    record_completion_payload(store, "zen/deepseek", completion("The user wants the weather."), now=100.0)

    shaped = store.replay_into_body("zen/deepseek", body_with_replayed_call(), now=150.0)

    assert shaped["messages"][1]["reasoning_content"] == "The user wants the weather."
    # The other messages are forwarded as they arrived.
    assert shaped["messages"][0] == {"role": "user", "content": "weather in Paris?"}
    assert shaped["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "24C"}


def test_the_caller_s_body_is_never_mutated():
    """Ficelle holds no copy of the caller's object; a rewrite must not reach back into it."""
    store = ReasoningReplayStore()
    record_completion_payload(store, "zen/deepseek", completion("thinking"), now=100.0)
    body = body_with_replayed_call()

    store.replay_into_body("zen/deepseek", body, now=100.0)

    assert "reasoning_content" not in body["messages"][1]


def test_an_unknown_tool_call_falls_back_for_a_model_known_to_think():
    """A conversation older than the cache still has to route.

    The stand-in names what the assistant did rather than inventing a thought: the upstream
    checks presence, and a plausible fake reasoning is a worse thing to feed a model than a
    factual line.
    """
    store = ReasoningReplayStore()
    record_completion_payload(store, "zen/deepseek", completion("earlier trace"), now=100.0)

    shaped = store.replay_into_body("zen/deepseek", body_with_replayed_call("call_unknown"), now=100.0)

    assert shaped["messages"][1]["reasoning_content"] == "Calling get_weather to answer the request."


def test_a_turn_handed_over_between_providers_carries_the_first_one_s_trace():
    """The dogfood failure, in one test.

    NVIDIA answered the turn, went into a 180s cooldown on a 529, and the next turn landed on
    OpenCode Zen — which rejected the history because the tool-call ids in it were NVIDIA's.
    Measured against the live provider: a foreign id with no trace is a 400, the same id with a
    trace is a 200. So the trace has to survive the handover, which is why the cache is keyed by
    tool-call id alone and never by model.
    """
    store = ReasoningReplayStore()
    record_completion_payload(store, "ficelle/nvidia/deepseek", completion("Paris weather needed.", "nvidia_call_7"), now=100.0)

    shaped = store.replay_into_body(
        "ficelle/opencode_zen/deepseek-v4-flash-free",
        body_with_replayed_call("nvidia_call_7"),
        now=110.0,
    )

    assert shaped["messages"][1]["reasoning_content"] == "Paris weather needed."


def test_a_message_that_still_carries_its_trace_is_left_alone():
    store = ReasoningReplayStore()
    record_completion_payload(store, "zen/deepseek", completion("cached"), now=100.0)
    body = body_with_replayed_call(reasoning_content="the client kept it")

    assert store.replay_into_body("zen/deepseek", body, now=100.0) is body


def test_an_assistant_turn_without_tool_calls_is_not_rewritten():
    """The upstream only demands the field where tool calls are replayed."""
    store = ReasoningReplayStore()
    record_completion_payload(store, "zen/deepseek", completion("cached"), now=100.0)
    body = {"messages": [{"role": "assistant", "content": "It is 24C."}]}

    assert store.replay_into_body("zen/deepseek", body, now=100.0) is body


def test_a_trace_outlives_neither_its_ttl_nor_the_cache_bound():
    store = ReasoningReplayStore(max_entries=2, ttl_seconds=10)
    record_completion_payload(store, "m", completion("first", "a"), now=100.0)

    assert store.trace_for(["a"], ["get_weather({})"], now=105.0) == "first"
    assert store.trace_for(["a"], ["get_weather({})"], now=200.0) == ""

    record_completion_payload(store, "m", completion("x", "b"), now=300.0)
    record_completion_payload(store, "m", completion("y", "c"), now=300.0)
    record_completion_payload(store, "m", completion("z", "d"), now=300.0)

    assert store.trace_for(["b"], ["get_weather({})"], now=300.0) == ""
    assert store.trace_for(["d"], ["get_weather({})"], now=300.0) == "z"


def test_a_completion_without_a_trace_teaches_nothing():
    """No trace means no thinking mode, so nothing authorizes a fallback for that model."""
    store = ReasoningReplayStore()
    record_completion_payload(store, "plain/model", completion(None), now=100.0)

    assert store.is_thinking_model("plain/model") is False
    assert store.replay_into_body("plain/model", body_with_replayed_call("x"), now=100.0) is not None
    assert "reasoning_content" not in store.replay_into_body("plain/model", body_with_replayed_call("x"), now=100.0)["messages"][1]


def test_a_block_list_trace_is_flattened():
    store = ReasoningReplayStore()
    record_completion_payload(store, "m", completion([{"text": "first"}, {"text": "second"}]), now=100.0)

    assert store.trace_for(["call_1"], ["get_weather({})"], now=100.0) == "first\nsecond"


def test_a_streamed_turn_is_learned_without_touching_the_bytes():
    """The client's stream is the product; observation rides along and changes nothing."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "zen/deepseek")
    frames = [
        stream_frame({"reasoning_content": "The user "}),
        stream_frame({"reasoning_content": "wants weather."}),
        stream_frame({"tool_calls": [{"index": 0, "id": "call_s", "function": {"name": "get_weather"}}]}),
        b"data: [DONE]\n\n",
    ]

    relayed = list(observe_stream_chunks(iter(frames), observer, now_fn=lambda: 100.0))

    assert relayed == frames
    assert store.trace_for(["call_s"], ["get_weather()"], now=100.0) == "The user wants weather."


def test_a_frame_split_across_chunks_is_still_read():
    """`iter_content` cuts on buffer boundaries, not on frames."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    whole = stream_frame({"reasoning_content": "abc"}) + stream_frame(
        {"tool_calls": [{"index": 0, "id": "call_split", "function": {"name": "f"}}]}
    )
    chunks = [whole[:20], whole[20:41], whole[41:]]

    relayed = list(observe_stream_chunks(iter(chunks), observer, now_fn=lambda: 100.0))

    assert b"".join(relayed) == whole
    assert store.trace_for(["call_split"], ["f()"], now=100.0) == "abc"


def test_streamed_deltas_keep_the_whitespace_between_tokens():
    """Deltas are token fragments; stripping each one welds the words together."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    frames = [
        stream_frame({"reasoning_content": "I need "}),
        stream_frame({"reasoning_content": "to check "}),
        stream_frame({"reasoning_content": "the weather."}),
        stream_frame({"tool_calls": [{"index": 0, "id": "call_w", "function": {"name": "f"}}]}),
    ]

    list(observe_stream_chunks(iter(frames), observer, now_fn=lambda: 100.0))

    assert store.trace_for(["call_w"], ["f()"], now=100.0) == "I need to check the weather."


def test_a_malformed_stream_costs_the_trace_and_nothing_else():
    """Committed bytes are already on the wire: observation must never raise into the relay."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    junk = [b"data: {not json\n\n", b"\xff\xfe\x00 binary", b"data: {\"choices\":[]}\n\n"]

    assert list(observe_stream_chunks(iter(junk), observer, now_fn=lambda: 100.0)) == junk
    assert store.is_thinking_model("m") is False


def test_a_stream_with_no_tool_call_teaches_nothing():
    """Without an id there is nothing to key the trace by, and no turn to replay it on."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")

    list(observe_stream_chunks(iter([stream_frame({"reasoning_content": "thought"})]), observer, now_fn=lambda: 100.0))

    assert store.is_thinking_model("m") is False


def test_fallback_names_every_distinct_call_once():
    assert fallback_trace([tool_call("a", "search"), tool_call("b", "search")]) == "Calling search to answer the request."
    assert fallback_trace([tool_call("a", "search"), tool_call("b", "fetch")]) == "Calling search, fetch to answer the request."
    assert fallback_trace([]) == "Continuing from the previous tool call."


def test_a_rejection_naming_a_non_standard_field_stays_retryable():
    """The evidence: NVIDIA had answered this very body seconds earlier.

    A 400 over `reasoning_content` is this upstream's private contract, not a verdict on the
    body — so it must not inherit `bad_upstream_request`'s terminal, no-failover handling.
    """
    zen_400 = (
        '{"error":{"type":"invalid_request_error","code":"invalid_request_error","message":'
        '"Error from provider (Console): Upstream request failed: [invalid_request_error] The '
        '`reasoning_content` in the thinking mode must be passed back to the API."}}'
    )

    assert classify_failure(400, zen_400) == "bad_upstream_contract"


def test_a_reused_tool_call_id_does_not_leak_another_conversation_s_trace():
    """Ids are unique in practice, not by rule — a client is free to send `call_1` every time.

    Handing conversation A's reasoning to conversation B would put one prompt's context into
    another's, so a matching id with mismatched calls is treated as a collision, not a hit.
    """
    store = ReasoningReplayStore()
    record_completion_payload(store, "m", completion("A: the user wants the weather", "call_1"), now=100.0)

    other_conversation = {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "read_file")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "..."},
        ]
    }
    shaped = store.replay_into_body("m", other_conversation, now=100.0)

    # Falls through to the stand-in rather than replaying an unrelated trace.
    assert shaped["messages"][0]["reasoning_content"] == "Calling read_file to answer the request."


def test_an_oversized_trace_is_capped_rather_than_cached_whole():
    store = ReasoningReplayStore()
    record_completion_payload(store, "m", completion("x" * 100_000), now=100.0)

    assert len(store.trace_for(["call_1"], ["get_weather({})"], now=100.0)) == MAX_TRACE_CHARS


def test_a_runaway_stream_stops_being_parsed_and_still_relays():
    """Observation gives up before it holds a live response's bytes a second time."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    flood = [b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"" + b"x" * 50_000 + b"\"}}]}\n\n"] * 3
    flood.append(stream_frame({"tool_calls": [{"index": 0, "id": "call_f", "function": {"name": "f"}}]}))

    assert list(observe_stream_chunks(iter(flood), observer, now_fn=lambda: 100.0)) == flood


def test_a_chunk_reaches_the_client_before_it_is_parsed():
    """Parsing must not sit in front of the first byte the user is waiting for."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    order = []
    frames = [stream_frame({"reasoning_content": "a"}), stream_frame({"reasoning_content": "b"})]

    class Watched:
        def __init__(self, inner): self.inner = inner
        def feed(self, chunk): order.append("parse"); self.inner.feed(chunk)
        def finish(self, *, now): self.inner.finish(now=now)

    for _chunk in observe_stream_chunks(iter(frames), Watched(observer), now_fn=lambda: 100.0):
        order.append("sent")

    assert order[0] == "sent"


def test_every_choice_is_learned_not_just_the_first():
    """`n > 1` is rare with tool calls but legal, and the client may continue from any of them."""
    store = ReasoningReplayStore()
    payload = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "tool_calls": [tool_call("c0")], "reasoning_content": "first"}},
            {"index": 1, "message": {"role": "assistant", "tool_calls": [tool_call("c1")], "reasoning_content": "second"}},
        ]
    }
    record_completion_payload(store, "m", payload, now=100.0)

    assert store.trace_for(["c0"], ["get_weather({})"], now=100.0) == "first"
    assert store.trace_for(["c1"], ["get_weather({})"], now=100.0) == "second"


def test_a_tool_name_split_across_deltas_is_reassembled():
    """`function.name` may be fragmented like `arguments` is.

    Treating each fragment as a whole name stored `get_`/`weather`, which then failed to match
    the `get_weather` the next request presents — the trace was silently dropped in favour of
    the fallback, with nothing on screen to say why.
    """
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    frames = [
        stream_frame({"reasoning_content": "checking"}),
        stream_frame({"tool_calls": [{"index": 0, "id": "call_frag", "function": {"name": "get_"}}]}),
        stream_frame({"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": '{"city":'}}]}),
        stream_frame({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
    ]

    list(observe_stream_chunks(iter(frames), observer, now_fn=lambda: 100.0))

    assert store.trace_for(["call_frag"], ['get_weather({"city":"Paris"})'], now=100.0) == "checking"


def test_two_streamed_choices_keep_their_own_reasoning():
    """`n > 1` streams every choice down one connection; one buffer would splice them."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")

    def frame(choice_index, delta):
        return b"data: " + json.dumps({"choices": [{"index": choice_index, "delta": delta}]}).encode() + b"\n\n"

    frames = [
        frame(0, {"reasoning_content": "first"}),
        frame(1, {"reasoning_content": "second"}),
        frame(0, {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "f"}}]}),
        frame(1, {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f"}}]}),
    ]

    list(observe_stream_chunks(iter(frames), observer, now_fn=lambda: 100.0))

    assert store.trace_for(["c0"], ["f()"], now=100.0) == "first"
    assert store.trace_for(["c1"], ["f()"], now=100.0) == "second"


def test_the_same_id_and_tool_with_different_arguments_is_a_collision():
    """`call_1` + `get_weather` is not enough to identify a turn; what it asked for is."""
    store = ReasoningReplayStore()
    record_completion_payload(store, "m", completion("Paris was asked about"), now=100.0)
    lyon = {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Lyon"}'}}]},
        ]
    }

    shaped = store.replay_into_body("m", lyon, now=100.0)

    assert shaped["messages"][0]["reasoning_content"] == "Calling get_weather to answer the request."


def test_one_giant_delta_cannot_outgrow_the_cap():
    """The cap has to bite on the way in: checking after appending holds the whole thing first."""
    store = ReasoningReplayStore()
    observer = ReasoningStreamObserver(store, "m")
    frames = [
        stream_frame({"reasoning_content": "y" * (MAX_TRACE_CHARS * 3)}),
        stream_frame({"tool_calls": [{"index": 0, "id": "call_g", "function": {"name": "f"}}]}),
    ]

    assert list(observe_stream_chunks(iter(frames), observer, now_fn=lambda: 100.0)) == frames
    assert len(store.trace_for(["call_g"], ["f()"], now=100.0)) <= MAX_TRACE_CHARS


def test_a_body_wrong_in_two_ways_at_once_stays_terminal():
    """A requirement marker aimed at a different field must not lend its retry to ours."""
    both = '{"error":{"message":"reasoning_content must be a string; required field: messages is missing"}}'

    assert classify_failure(400, both) == "bad_upstream_request"


def test_a_complaint_about_a_value_that_was_sent_stays_terminal():
    """The field name alone is not the signal: here `reasoning_content` WAS supplied and is
    wrong, so every candidate would reject it and a failover buys nothing."""
    assert classify_failure(400, '{"error":{"message":"Invalid request: reasoning_content must be a string"}}') == "bad_upstream_request"


def test_a_genuinely_malformed_body_stays_terminal():
    """The other 400 from the same provider — here the body really is wrong, and every
    candidate would say so."""
    orphan_tool_message = (
        '{"error":{"type":"invalid_request_error","message":"Messages with role \'tool\' must be '
        "a response to a preceding message with 'tool_calls'\"}}"
    )

    assert classify_failure(400, orphan_tool_message) == "bad_upstream_request"
