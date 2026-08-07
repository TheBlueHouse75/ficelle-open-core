"""Replay a thinking model's own reasoning trace back into its next request.

Some upstreams served behind a relay run their model in "thinking mode" and then refuse a
replayed tool call they cannot tie back to a turn of their own. DeepSeek V4 Flash via
OpenCode Zen is the case that forced this module:

    400 [invalid_request_error] The `reasoning_content` in the thinking mode must be
    passed back to the API.

Measured against the live provider, three attempts per case, the trigger is narrower than
that message suggests — and worse for a router:

    tool_call.id the provider issued itself,  no trace  -> 200
    tool_call.id from ANOTHER provider,       no trace  -> 400
    tool_call.id from another provider,       trace     -> 200

So the rejection is not about the field being absent. It fires when the history carries a
tool-call id the upstream did not mint, and supplying ``reasoning_content`` is what lets it
accept one anyway. That makes this a **failover** defect, not a client defect: it needs two
providers to appear, which is the one thing a router does. The turn that broke in dogfood
had been answered by NVIDIA, so the ids OpenCode Zen received on the next turn were
NVIDIA's. Every web search reproduced it.

``reasoning_content`` is not part of the OpenAI chat-completions schema, so no standard
client returns it — the field reaches the client and is dropped on the way back.

This is the one place Ficelle rewrites a request body, against the passthrough rule stated
in ``malformed_tool_call_detail``. The distinction that earns the exception: nothing here
corrects the caller. The body is valid OpenAI, and the only field added is one **Ficelle
itself relayed outward moments earlier** and the schema gave the client no way to return.
Restoring it is what keeps the client standard while the upstream stays satisfied — the
router absorbing an upstream quirk, which is the job.

Two levels, in order of fidelity:

- **Faithful** — the trace the model actually emitted, kept keyed by the ``tool_call`` ids
  it came with. Those ids are the one part of the exchange a client is obliged to send
  back (``tool`` messages reference them), so they identify the turn without Ficelle
  holding any conversation state. What those calls asked for — ``name(arguments)`` — is
  stored alongside and checked on the way out: ids are unique in practice, not by rule, and
  a client that numbers its own ``call_1`` twice must not be handed another conversation's
  reasoning.
- **Fallback** — when no trace is on file (a conversation older than this process, or a
  turn whose answering candidate never emitted one), a short factual line naming the calls
  that were made. Measured sufficient: a made-up trace on a foreign id was accepted three
  times out of three, so the upstream wants the field populated, not matched.

The fallback only ever fires for a model already observed emitting a trace, so a provider
that has no idea what ``reasoning_content`` is never receives one.

State is process-local and deliberately not persisted. A restart mid-conversation costs
one failed turn on that model, which the ``bad_upstream_contract`` failover then covers by
trying the next candidate; persisting a per-model flag would buy that one turn at the cost
of a state write on the hot path.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Iterable, Iterator

# Bounds for the trace cache. A trace is only useful until the client stops replaying the
# tool call it belongs to, so both are generous for a live conversation and irrelevant
# after it: entries expire long before they add up to anything.
MAX_CACHED_TRACES = 512
TRACE_TTL_SECONDS = 2 * 3600

# A trace is replayed as a message, so its size is the upstream's problem as much as ours. The cap
# bounds one entry (a model looping on itself), and with MAX_CACHED_TRACES bounds the cache as a
# whole — a count alone bounds nothing when each entry is unbounded.
MAX_TRACE_CHARS = 32_768

# Arguments enter the cache-hit check (see `_tool_call_signatures`), so they are bounded too —
# comparing whole payloads would let one large call inflate every entry that mentions it.
MAX_SIGNATURE_ARG_CHARS = 512

# The provider-side check is presence and non-emptiness, so the fallback aims at being
# truthful rather than convincing: it names what the assistant did on that turn and stops.
FALLBACK_WITHOUT_NAMES = "Continuing from the previous tool call."

REASONING_KEYS = ("reasoning_content", "reasoning")


def _clean_trace(value: Any, *, strip: bool = True) -> str:
    """Normalize a trace to a string, or "" — mirrors `extract_reasoning_text`.

    Providers send it either as a plain string or as a list of ``{"text": ...}`` blocks.
    Kept here rather than imported so this module stays free of the benchmark stack.

    ``strip=False`` is for streamed deltas, which are token fragments: stripping each one
    would silently weld the words of the reassembled trace together. The join in
    `ReasoningStreamObserver.finish` strips once, at the end, where it is safe.
    """
    if isinstance(value, str):
        return value.strip() if strip else value
    if isinstance(value, list):
        parts = [item["text"] for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)]
        joined = "\n".join(parts)
        return joined.strip() if strip else joined
    return ""


def _tool_call_ids(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, list):
        return []
    ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if isinstance(call_id, str) and call_id.strip():
            ids.append(call_id.strip())
    return ids


def _tool_call_names(tool_calls: Any) -> list[str]:
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _tool_call_signatures(tool_calls: Any) -> list[str]:
    """What each call actually asked for, as `name(arguments)`.

    This is what a cache hit is checked against, because the id alone cannot carry the check:
    ids are unique in practice but nothing in the schema says so, and a client numbering its
    own `call_1` per conversation is entirely legal. Including the arguments narrows a
    collision to the same tool called with the same input — where one turn's reasoning is a
    fair description of the other's — instead of any two turns that happen to share a name.
    """
    if not isinstance(tool_calls, list):
        return []
    signatures: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = function.get("arguments")
        raw = arguments if isinstance(arguments, str) else ""
        signatures.append(f"{name.strip()}({raw.strip()[:MAX_SIGNATURE_ARG_CHARS]})")
    return signatures


def fallback_trace(tool_calls: Any) -> str:
    """A short, factual stand-in naming the calls the assistant made on that turn."""
    names = _tool_call_names(tool_calls)
    if not names:
        return FALLBACK_WITHOUT_NAMES
    unique = list(dict.fromkeys(names))
    return "Calling " + ", ".join(unique) + " to answer the request."


class ReasoningReplayStore:
    """Traces seen on the way out, keyed by the tool-call ids that identify their turn.

    Keyed by tool-call id alone, not by model: a trace belongs to the conversation, not to
    whichever candidate produced it. When a turn is answered by one model and the next by
    another — routine after a cooldown — replaying the first model's trace is exactly what
    a client that had kept the field would send.

    Safe to share across the server's request threads.
    """

    def __init__(self, *, max_entries: int = MAX_CACHED_TRACES, ttl_seconds: float = TRACE_TTL_SECONDS) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, tuple[float, str, tuple[str, ...]]] = OrderedDict()
        self._thinking_models: set[str] = set()

    def observe(self, model_id: str, trace: str, tool_call_ids: list[str], call_signatures: list[str], *, now: float) -> None:
        """Record a trace a model emitted alongside tool calls.

        Both halves matter and neither implies the other: the trace is what a later turn
        replays, while the model id going into `_thinking_models` is what authorizes the
        fallback for that model at all.

        `call_signatures` is stored to be checked at replay, not to key the entry. Tool-call ids
        are only unique in practice — providers mint random ones, but nothing stops a client
        from sending `call_1` — and handing conversation A's reasoning to conversation B
        would leak context into a prompt as well as confuse the model.
        """
        cleaned = _clean_trace(trace)[:MAX_TRACE_CHARS]
        if not cleaned:
            return
        model_key = str(model_id or "").strip()
        names = tuple(sorted(set(call_signatures)))
        with self._lock:
            if model_key:
                self._thinking_models.add(model_key)
            for call_id in tool_call_ids:
                self._traces[call_id] = (now, cleaned, names)
                self._traces.move_to_end(call_id)
            self._evict(now)

    def _evict(self, now: float) -> None:
        """Drop expired then oldest entries. Caller holds the lock."""
        expired = [key for key, entry in self._traces.items() if now - entry[0] > self._ttl_seconds]
        for key in expired:
            self._traces.pop(key, None)
        while len(self._traces) > self._max_entries:
            self._traces.popitem(last=False)

    def trace_for(self, tool_call_ids: list[str], call_signatures: list[str], *, now: float) -> str:
        """The trace stored for these calls, or "" — including when the id matches but the
        calls do not, which reads as a reused id rather than the same turn."""
        wanted = tuple(sorted(set(call_signatures)))
        with self._lock:
            for call_id in tool_call_ids:
                entry = self._traces.get(call_id)
                if entry is None:
                    continue
                stored_at, trace, names = entry
                if now - stored_at > self._ttl_seconds:
                    self._traces.pop(call_id, None)
                    continue
                if names and wanted and names != wanted:
                    continue
                return trace
        return ""

    def is_thinking_model(self, model_id: str) -> bool:
        with self._lock:
            return str(model_id or "").strip() in self._thinking_models

    def replay_into_body(self, model_id: str, body: dict[str, Any], *, now: float) -> dict[str, Any]:
        """Return `body` with traces restored on assistant messages that lost theirs.

        Returns the object unchanged — same identity, no copy — whenever there is nothing
        to restore, which is every request to a model that has never emitted a trace. The
        rewrite is copy-on-write and reaches only the messages it fills in.
        """
        if not isinstance(body, dict):
            return body
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        known_thinking = self.is_thinking_model(model_id)
        rewritten: list[Any] | None = None
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            if any(_clean_trace(message.get(key)) for key in REASONING_KEYS):
                continue
            call_ids = _tool_call_ids(tool_calls)
            trace = self.trace_for(call_ids, _tool_call_signatures(tool_calls), now=now)
            if not trace:
                # No trace on file: only a model already known to think gets the stand-in.
                # Sending one to a provider that never asked for the field is how a fix for
                # one upstream becomes an outage on the others.
                if not known_thinking:
                    continue
                trace = fallback_trace(tool_calls)
            if rewritten is None:
                rewritten = list(messages)
            rewritten[index] = {**message, "reasoning_content": trace}

        if rewritten is None:
            return body
        return {**body, "messages": rewritten}


def record_completion_payload(store: ReasoningReplayStore, model_id: str, payload: Any, *, now: float) -> None:
    """Learn from a non-streaming completion. Never raises: observation is not the job.

    Every choice, not just the first: `n > 1` is rare with tool calls but legal, and the
    client is free to continue from any of them — a trace missed here is the original bug
    coming back on that turn.
    """
    try:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            tool_call_ids = _tool_call_ids(tool_calls)
            if not tool_call_ids:
                continue
            for key in REASONING_KEYS:
                trace = _clean_trace(message.get(key))
                if trace:
                    store.observe(model_id, trace, tool_call_ids, _tool_call_signatures(tool_calls), now=now)
                    break
    except Exception:
        return


class _StreamedChoice:
    """One choice's trace and tool calls, rebuilt from the deltas that carried them.

    Tool calls arrive spread across deltas and are stitched together by their ``index``, not
    by arrival order: `arguments` is fragmented by design, and `name` may be too. Appending
    each fragment as if it were a whole value produced `get_`/`weather` where the request
    would later present `get_weather`, and the cache-hit check then rejected a legitimate
    replay — the fallback covered it, so the only visible symptom was a trace quietly not
    being used.
    """

    def __init__(self) -> None:
        self.trace_parts: list[str] = []
        self.trace_chars = 0
        self.calls: dict[int, dict[str, str]] = {}

    def absorb_tool_calls(self, tool_calls: Any) -> None:
        if not isinstance(tool_calls, list):
            return
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            raw_index = tool_call.get("index")
            index = raw_index if isinstance(raw_index, int) else position
            call = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id.strip():
                call["id"] = call_id.strip()
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name:
                call["name"] += name
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments and len(call["arguments"]) < MAX_SIGNATURE_ARG_CHARS:
                call["arguments"] += arguments

    def resolved_calls(self) -> tuple[list[str], list[str]]:
        """The ids and `name(arguments)` signatures, in the provider's own call order."""
        call_ids: list[str] = []
        signatures: list[str] = []
        for index in sorted(self.calls):
            call = self.calls[index]
            if not call["id"] or not call["name"].strip():
                continue
            call_ids.append(call["id"])
            signatures.append(f"{call['name'].strip()}({call['arguments'].strip()[:MAX_SIGNATURE_ARG_CHARS]})")
        return call_ids, signatures


class ReasoningStreamObserver:
    """Reassembles the trace and tool-call ids from a relayed SSE stream.

    Watches bytes already on their way to the client and never touches them. A streamed
    turn is the common case — the client that triggered this whole module streams — so
    without this the faithful path would only ever work for non-streaming callers.

    Every entry point swallows its own exceptions. A malformed frame must cost the trace,
    never the response: the stream is live and already committed by the time it gets here.
    """

    def __init__(self, store: ReasoningReplayStore, model_id: str) -> None:
        self._store = store
        self._model_id = model_id
        self._buffer = b""
        # Per choice index, because `n > 1` streams every choice down one connection: one
        # shared buffer would splice unrelated reasonings together and hand the result to
        # whichever tool call finished last.
        self._choices: dict[int, _StreamedChoice] = {}
        self._done = False

    def _give_up(self) -> None:
        """Stop observing and drop what was held. Whatever the reason, the response is
        unaffected — only the trace is lost, and the fallback covers its absence."""
        self._done = True
        self._buffer = b""
        self._choices = {}

    def feed(self, chunk: bytes) -> None:
        if self._done:
            return
        try:
            self._buffer += chunk
            # Frames are newline-delimited; hold the trailing partial line for the next chunk.
            *lines, self._buffer = self._buffer.split(b"\n")
            for line in lines:
                self._consume_line(line)
            # A stream that never emits a newline must not grow the buffer without bound, and a
            # trace past the cap is no longer worth what it costs to hold. Giving up is always
            # the right trade against keeping a live response's bytes twice.
            if len(self._buffer) > MAX_TRACE_CHARS or any(c.trace_chars >= MAX_TRACE_CHARS for c in self._choices.values()):
                self._give_up()
        except Exception:
            self._give_up()

    def _consume_line(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return
        data = stripped[len(b"data:"):].strip()
        if not data or data == b"[DONE]":
            return
        try:
            event = json.loads(data.decode("utf-8"))
        except Exception:
            return
        if not isinstance(event, dict):
            return
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            index = choice.get("index")
            self._consume_delta(index if isinstance(index, int) else position, delta)

    def _consume_delta(self, choice_index: int, delta: dict) -> None:
        accumulator = self._choices.get(choice_index)
        if accumulator is None:
            accumulator = _StreamedChoice()
            self._choices[choice_index] = accumulator
        for key in REASONING_KEYS:
            part = _clean_trace(delta.get(key), strip=False)
            if part:
                # Bounded on the way in, not after: a single oversized event would otherwise be
                # held whole before the cap in `feed` ever gets to look at it.
                room = MAX_TRACE_CHARS - accumulator.trace_chars
                if room > 0:
                    accumulator.trace_parts.append(part[:room])
                accumulator.trace_chars += len(part)
                break
        accumulator.absorb_tool_calls(delta.get("tool_calls"))

    def finish(self, *, now: float) -> None:
        try:
            if self._buffer.strip():
                self._consume_line(self._buffer)
                self._buffer = b""
            for accumulator in self._choices.values():
                call_ids, signatures = accumulator.resolved_calls()
                if not call_ids:
                    continue
                # Joined without a separator: these are token-level deltas of one continuous
                # trace, not independent lines.
                trace = "".join(accumulator.trace_parts).strip()
                if trace:
                    self._store.observe(self._model_id, trace, call_ids, signatures, now=now)
        except Exception:
            return


def observe_stream_chunks(
    chunks: Iterable[bytes],
    observer: ReasoningStreamObserver,
    *,
    now_fn: Any,
) -> Iterator[bytes]:
    """Yield every chunk untouched, parsing it only once it is the client's problem.

    The chunk is yielded BEFORE it is fed to the observer: resuming the generator is what
    the writer does after it has written and flushed, so the parse lands between two client
    writes instead of ahead of one. Time to first byte is then exactly what it was without
    the observer — which matters most on the first chunk, the one a user watches for.

    Wrapping the iterator rather than the writer keeps `stream_chunks_to_writer` — which
    owns mid-stream failure classification — exactly as it was.
    """
    try:
        for chunk in chunks:
            yield chunk
            observer.feed(chunk)
    finally:
        observer.finish(now=now_fn())
