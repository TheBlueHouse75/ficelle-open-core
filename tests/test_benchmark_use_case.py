from __future__ import annotations

import json
from typing import Any

from ficelle.use_cases.benchmark import (
    ROUTE_REJECTION_VERDICT_TTL_SECONDS,
    BenchmarkEvidencePorts,
    BenchmarkMediaError,
    BenchmarkMediaPorts,
    BenchmarkProbeBuilder,
    BenchmarkRunner,
    ProbeRegistry,
    apply_canary_result_to_state,
    benchmark_audio_part,
    benchmark_file_data_url,
    benchmark_media_answer,
    benchmark_media_settings,
    benchmark_result_is_aged,
    benchmark_result_matches_current_test,
    benchmark_result_test_type_matches,
    build_compression_benchmark_source,
    build_long_context_benchmark_source,
    configured_benchmark_media_path,
    expected_probe_test_type,
    extract_message_text,
    finish_reason_is_truncation,
    extract_reasoning_text,
    extract_tool_calls,
    has_deliverable_message,
    json_fields_match,
    max_benchmark_media_bytes,
    pick_vision_fixture,
    canary_state_payload,
    canary_status_from_benchmark_result,
    mark_stale_benchmark_row,
    profile_ids_from_admin_body,
    record_benchmark_failure_in_state,
    record_benchmark_result_in_state,
    redacted_benchmark_row,
    route_rejection_deserves_a_retry_soon,
    stale_score_decay_factor,
    tool_call_arguments,
    validate_probe_response,
    vision_benchmark_fixtures,
)
from ficelle.use_cases.capability_discovery import (
    ROUTE_REJECTION_VERDICT,
    VERDICT_BASIS_KEY,
    VERDICT_STREAK_KEY,
    CapabilityDiscoveryJob,
    ProviderProbePacer,
    clear_background_job_error_in_state,
    consecutive_verdict_count,
    discovery_eligible_models,
    model_due_capabilities,
    models_needing_discovery,
    probeable_capability_profiles,
    record_background_job_error_in_state,
)


def tool_call_payload(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                }
            }
        ]
    }


class FakeResponse:
    def __init__(self, text: str, *, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": self.text}}]}


def media_ports(
    *,
    env: dict[str, str] | None = None,
    default_paths: dict[str, Any] | None = None,
    fixtures: Any = None,
    now_seconds: float = 0.0,
) -> BenchmarkMediaPorts:
    return BenchmarkMediaPorts(
        safe_int=lambda value, default: int(value) if value is not None else default,
        env_get=lambda key: (env or {}).get(key),
        default_media_path=lambda kind: default_paths[kind],
        load_json=lambda _path, default: fixtures if fixtures is not None else default,
        vision_fixtures_path=default_paths.get("fixtures") if default_paths else None,
        now_seconds=lambda: now_seconds,
    )


def evidence_ports(now_seconds: float = 200.0, ttl_seconds: int = 100) -> BenchmarkEvidencePorts:
    def parse_timestamp(value: str | None) -> float | None:
        return float(value) if value is not None else None

    return BenchmarkEvidencePorts(
        expected_test_type=lambda profile_id: {"ficelle/auto-tools": "tool_call"}.get(profile_id, "exact_text"),
        active_ttl_seconds=lambda: ttl_seconds,
        evidence_timestamp_value=lambda row: row.get("ran_at") if isinstance(row, dict) else None,
        parse_timestamp=parse_timestamp,
        now_seconds=lambda: now_seconds,
    )


def make_probe_builder() -> BenchmarkProbeBuilder:
    return BenchmarkProbeBuilder(
        canonical_virtual_model_id=lambda profile_id: profile_id.split("@", 1)[0],
        compression_source=lambda _config: "large compression source",
        long_context_source=lambda _config: (
            "long source long-needle-alpha-4731 "
            "long-needle-bravo-8052 long-needle-charlie-2649"
        ),
        pick_vision_fixture=lambda: ("37", "data:image/png;base64,abc"),
        benchmark_file_data_url=lambda _config, kind, mime: f"data:{mime};base64,{kind}",
        benchmark_audio_part=lambda _config: {
            "type": "input_audio",
            "input_audio": {"format": "m4a", "data": "audio"},
        },
        benchmark_media_answer=lambda _config, _kind, default: default,
        video_default_answer="73",
        audio_default_answer="orange",
    )


def test_benchmark_probe_builder_creates_representative_specialized_probes():
    builder = make_probe_builder()

    orchestrator_body, orchestrator_type, orchestrator_expected = builder.build("ficelle/auto-orchestrator")
    tools_body, tools_type, tools_expected = builder.build("ficelle/auto-tools@openrouter")
    json_body, json_type, json_expected = builder.build("ficelle/auto-json")

    assert orchestrator_type == "orchestration"
    assert orchestrator_expected == "route_task:summarize:A7"
    assert {tool["function"]["name"] for tool in orchestrator_body["tools"]} >= {
        "search_docs",
        "route_task",
        "finish",
    }
    assert tools_type == "tool_call"
    assert tools_expected == "ficelle_probe:alpha-ok"
    assert {tool["function"]["name"] for tool in tools_body["tools"]} >= {
        "get_weather",
        "ficelle_probe",
    }
    assert json_type == "json"
    assert json_body["response_format"] == {"type": "json_object"}
    assert json_expected.startswith('{"name": "Maya Lopez"')


def test_benchmark_probe_builder_uses_runtime_media_ports():
    builder = make_probe_builder()

    vision_body, vision_type, vision_expected = builder.build("ficelle/auto-vision")
    video_body, video_type, video_expected = builder.build("ficelle/auto-video")
    audio_body, audio_type, audio_expected = builder.build("ficelle/auto-audio")

    assert vision_type == "vision"
    assert vision_expected == "37"
    assert vision_body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
    assert video_type == "video"
    assert video_expected == "73"
    video_url = video_body["messages"][0]["content"][1]["video_url"]["url"]

    assert video_url == "data:video/mp4;base64,video"
    assert audio_type == "audio"
    assert audio_expected == "orange"
    assert audio_body["messages"][0]["content"][1]["input_audio"]["data"] == "audio"


def test_benchmark_media_helpers_resolve_config_env_and_packaged_defaults(tmp_path):
    audio_path = tmp_path / "audio.m4a"
    env_audio_path = tmp_path / "env.mp3"
    default_video_path = tmp_path / "video.mp4"
    audio_path.write_bytes(b"abc")
    env_audio_path.write_bytes(b"env")
    default_video_path.write_bytes(b"video")
    ports = media_ports(
        env={"FICELLE_BENCHMARK_AUDIO_PATH": str(env_audio_path)},
        default_paths={"audio": audio_path, "video": default_video_path, "fixtures": tmp_path / "fixtures.json"},
    )

    assert benchmark_media_settings({"benchmark_media": {"audio_path": str(audio_path)}}) == {
        "audio_path": str(audio_path)
    }
    assert configured_benchmark_media_path({"benchmark_media": {"audio_path": str(audio_path)}}, "audio", ports=ports) == env_audio_path
    assert configured_benchmark_media_path({}, "video", ports=ports) == default_video_path
    assert max_benchmark_media_bytes({"benchmark_media": {"max_bytes": "7"}}, ports=ports) == 7


def test_benchmark_file_data_url_and_audio_part_validate_media_files(tmp_path):
    audio_path = tmp_path / "voice.m4a"
    audio_path.write_bytes(b"abc")
    ports = media_ports(default_paths={"audio": audio_path, "video": tmp_path / "missing.mp4", "fixtures": tmp_path / "fixtures.json"})

    data_url = benchmark_file_data_url({}, "audio", "audio/mpeg", ports=ports)
    assert data_url.startswith("data:audio/")
    assert data_url.endswith(";base64,YWJj")
    assert benchmark_audio_part({}, ports=ports) == {
        "type": "input_audio",
        "input_audio": {"data": "YWJj", "format": "m4a"},
    }
    assert benchmark_media_answer({"benchmark_media": {"audio_answer": " orange "}}, "audio", "fallback") == "orange"

    try:
        benchmark_file_data_url({"benchmark_media": {"max_bytes": 1}}, "audio", "audio/mpeg", ports=ports)
    except BenchmarkMediaError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("oversized media should fail")


def test_vision_benchmark_fixtures_filters_data_urls_and_rotates_by_time(tmp_path):
    ports = media_ports(
        default_paths={"audio": tmp_path / "audio.mp3", "video": tmp_path / "video.mp4", "fixtures": tmp_path / "fixtures.json"},
        fixtures={"2": "data:image/png;base64,two", "1": "data:image/png;base64,one", "bad": "http://example.test"},
        now_seconds=3.0,
    )

    assert vision_benchmark_fixtures(ports=ports) == {
        "1": "data:image/png;base64,one",
        "2": "data:image/png;base64,two",
    }
    assert pick_vision_fixture(ports=ports) == ("2", "data:image/png;base64,two")
    assert pick_vision_fixture(0, ports=ports) == ("1", "data:image/png;base64,one")


def test_record_benchmark_result_in_state_redacts_and_repairs_invalid_shape():
    state = {"benchmark_results": "bad-shape"}

    record_benchmark_result_in_state(
        state,
        "ficelle/auto-json",
        {"id": "model-a"},
        {"status": "pass", "secret": "token"},
        cooldown_key=lambda model: f"key:{model['id']}",
        redact_sensitive_json=lambda result: {**result, "secret": "[redacted]"},
    )

    assert state["benchmark_results"] == {
        "key:model-a": {
            "ficelle/auto-json": {"status": "pass", "secret": "[redacted]"},
        }
    }


def test_record_benchmark_failure_in_state_updates_stats_and_model_error():
    state: dict[str, Any] = {}
    calls: list[tuple[Any, ...]] = []

    record_benchmark_failure_in_state(
        state,
        {"id": "model-a"},
        "benchmark_failed",
        detail="semantic miss",
        profile_id="ficelle/auto-fast",
        update_failure_stats=lambda current_state, model, reason: calls.append(
            ("stats", current_state is state, model["id"], reason)
        ),
        record_model_error=lambda current_state, model, reason, detail, **kwargs: calls.append(
            ("error", current_state is state, model["id"], reason, detail, kwargs.get("profile_id"))
        ),
    )

    assert calls == [
        ("stats", True, "model-a", "benchmark_failed"),
        ("error", True, "model-a", "benchmark_failed", "semantic miss", "ficelle/auto-fast"),
    ]


def test_canary_helpers_build_state_from_result_rows_not_requested_profiles():
    result = {
        "ran_at": "2026-06-22T00:00:00+00:00",
        "summary": {"failed": "1"},
        "results": [
            {"profile_id": "ficelle/auto-fast", "status": "pass"},
            {"profile_id": "ficelle/auto-tools", "status": "fail"},
            {"profile_id": "", "status": "fail"},
            "bad-row",
        ],
    }
    state: dict[str, Any] = {}

    status = canary_status_from_benchmark_result(
        result,
        safe_int=lambda value, default: int(value) if value is not None else default,
    )
    apply_canary_result_to_state(state, result, status)

    assert status == "fail"
    assert state["canary"] == canary_state_payload(result, "fail")
    assert state["canary"]["profiles"] == ["ficelle/auto-fast", "ficelle/auto-tools"]
    assert state["canary"]["failed_profiles"] == ["ficelle/auto-tools", ""]


def test_profile_ids_from_admin_body_accepts_legacy_key_and_rejects_non_strings():
    assert profile_ids_from_admin_body({"profiles": None}) == []
    assert profile_ids_from_admin_body({"profiles": "ficelle/auto-fast"}) == ["ficelle/auto-fast"]
    assert profile_ids_from_admin_body({"profile_ids": ["ficelle/auto-json"]}) == ["ficelle/auto-json"]
    try:
        profile_ids_from_admin_body({"profiles": [{"api_key": "plainsecret12345"}]})
    except ValueError as exc:
        assert "profiles must contain only strings" in str(exc)
    else:
        raise AssertionError("non-string profile items should fail")


def test_benchmark_evidence_helpers_match_test_type_and_ttl():
    ports = evidence_ports(now_seconds=200.0, ttl_seconds=100)
    fresh = {"status": "pass", "test_type": "tool_call", "ran_at": "150"}
    stale = {"status": "pass", "test_type": "tool_call", "ran_at": "50"}
    legacy_exact = {"status": "pass", "ran_at": "150"}

    assert benchmark_result_test_type_matches("ficelle/auto-tools", fresh, ports=ports) is True
    assert benchmark_result_test_type_matches("ficelle/auto-tools", legacy_exact, ports=ports) is False
    assert benchmark_result_matches_current_test("ficelle/auto-tools", fresh, ports=ports) is True
    assert benchmark_result_is_aged(stale, ports=ports) is True
    assert benchmark_result_matches_current_test("ficelle/auto-tools", stale, ports=ports) is False
    assert stale_score_decay_factor("ficelle/auto-tools", stale, ports=ports) == 0.5
    assert stale_score_decay_factor("ficelle/auto-tools", {"test_type": "wrong", "ran_at": "50"}, ports=ports) == 0.0


def test_route_rejection_verdicts_age_out_faster_than_capability_verdicts():
    """"The route said no" is not the same evidence as "the model answered and got it wrong".

    Both are `failed`, but only the second is a property of the model. A 404 can be an account-level
    condition — OpenRouter's "No endpoints found matching your data policy" matches no marker — so
    capping it keeps a profile from staying gated for the full multi-day TTL after the cause is fixed.
    """
    long_ttl = ROUTE_REJECTION_VERDICT_TTL_SECONDS * 24
    just_past_the_short_ttl = ROUTE_REJECTION_VERDICT_TTL_SECONDS + 60
    ports = evidence_ports(now_seconds=just_past_the_short_ttl, ttl_seconds=long_ttl)

    answered = {"status": "fail", "test_type": "tool_call", "ran_at": "0"}
    route = {**answered, VERDICT_BASIS_KEY: ROUTE_REJECTION_VERDICT}

    assert benchmark_result_is_aged(answered, ports=ports) is False  # still trusted under the long TTL
    assert benchmark_result_is_aged(route, ports=ports) is True  # capped, so it is re-probed
    assert benchmark_result_is_aged({**route, VERDICT_STREAK_KEY: 2}, ports=ports) is False

    # The cap only ever shortens: a TTL configured below it still wins.
    short_ports = evidence_ports(now_seconds=100.0, ttl_seconds=10)
    assert benchmark_result_is_aged(route, ports=short_ports) is True
    assert benchmark_result_is_aged({**route, "ran_at": "99"}, ports=short_ports) is False


def test_route_rejection_streak_resets_and_repairs_invalid_state():
    current = {VERDICT_BASIS_KEY: ROUTE_REJECTION_VERDICT, "test_type": "audio"}

    assert consecutive_verdict_count(None, current) == 1
    assert consecutive_verdict_count(dict(current), current) == 2  # pre-streak persisted row
    assert consecutive_verdict_count({**current, VERDICT_STREAK_KEY: 1}, current) == 2
    assert consecutive_verdict_count({**current, VERDICT_STREAK_KEY: 2}, current) == 3
    assert consecutive_verdict_count(
        {**current, VERDICT_STREAK_KEY: 7},
        {**current, "test_type": "audio-v2"},
    ) == 1
    assert consecutive_verdict_count(
        {**current, VERDICT_STREAK_KEY: 7},
        {**current, VERDICT_BASIS_KEY: "model_answer"},
    ) == 1

    for corrupt in ("2", -1, 1.5, True, 10**100):
        row = {**current, VERDICT_STREAK_KEY: corrupt}
        assert route_rejection_deserves_a_retry_soon(row) is True
        assert consecutive_verdict_count(row, current) == 2

    assert route_rejection_deserves_a_retry_soon(current) is True  # backward-compatible row
    assert route_rejection_deserves_a_retry_soon({**current, VERDICT_STREAK_KEY: 1}) is True
    assert route_rejection_deserves_a_retry_soon({**current, VERDICT_STREAK_KEY: 2}) is False


def test_mark_and_redact_stale_benchmark_rows_preserve_contract():
    ports = evidence_ports(now_seconds=200.0, ttl_seconds=100)

    stale = mark_stale_benchmark_row(
        "ficelle/auto-tools",
        {"status": "pass", "test_type": "tool_call", "ran_at": "50"},
        ports=ports,
    )
    mismatch = mark_stale_benchmark_row(
        "ficelle/auto-tools",
        {"status": "pass", "test_type": "json", "ran_at": "150"},
        ports=ports,
    )
    safe = redacted_benchmark_row(
        "ficelle/auto-tools",
        {"status": "pass", "test_type": "tool_call", "ran_at": "50", "secret": "token"},
        ports=ports,
        redact_sensitive_json=lambda row: {**row, "secret": "[redacted]"},
    )

    assert stale["status"] == "stale"
    assert stale["original_status"] == "pass"
    assert stale["expected_test_type"] == "tool_call"
    assert "older than" in stale["message"]
    assert mismatch["message"] == "stale benchmark result: expected tool_call, stored json"
    assert safe["secret"] == "[redacted]"
    assert safe["status"] == "stale"


def test_benchmark_probe_builder_keeps_generic_and_volume_probes():
    builder = make_probe_builder()

    long_body, long_type, long_expected = builder.build("ficelle/auto-long")
    compression_body, compression_type, compression_expected = builder.build("ficelle/auto-compression")
    multimodal_body, multimodal_type, multimodal_expected = builder.build("ficelle/auto-multimodal")

    assert long_type == "long_context"
    assert long_expected == "long-needle-alpha-4731|long-needle-bravo-8052|long-needle-charlie-2649"
    assert "long-needle-alpha-4731" in long_body["messages"][0]["content"]
    assert compression_type == "compression"
    assert compression_expected == "4200|Lena Park|KX-19"
    assert compression_body["messages"][0]["content"] == "large compression source"
    assert multimodal_type == "exact_text"
    assert multimodal_expected == "multimodal-router-ok"
    assert multimodal_body["messages"][0]["content"] == "Reply exactly: multimodal-router-ok"


def test_benchmark_volume_sources_preserve_needles_and_clamp_operator_sizes():
    compression = build_compression_benchmark_source(
        {"benchmark_compression_chars": 100_000_000},
        safe_int=lambda value, default: int(value) if value is not None else default,
        default_chars=18_000,
        min_chars=2_000,
        max_chars=20_000,
    )
    long_context = build_long_context_benchmark_source(
        {"benchmark_long_context_tokens": 10_000_000},
        safe_int=lambda value, default: int(value) if value is not None else default,
        default_tokens=64_000,
        min_tokens=1_000,
        max_tokens=2_000,
        chars_per_token=4,
    )

    assert "4200 euros" in compression
    assert "Lena Park" in compression
    assert "KX-19" in compression
    assert len(compression) < 25_000
    for needle in (
        "long-needle-alpha-4731",
        "long-needle-bravo-8052",
        "long-needle-charlie-2649",
    ):
        assert needle in long_context
    assert len(long_context) < 12_000


def test_probeable_capability_profiles_skips_unavailable_probe_bodies():
    def benchmark_body(profile_id: str, _config: dict[str, Any] | None):
        if profile_id == "ficelle/auto-video":
            raise FileNotFoundError("missing video fixture")
        return {"messages": []}, "exact_text", "ok"

    profiles = probeable_capability_profiles(
        {},
        profile_ids=["ficelle/auto-json", "ficelle/auto-video", "ficelle/auto-tools"],
        benchmark_body=benchmark_body,
    )

    assert profiles == ["ficelle/auto-json", "ficelle/auto-tools"]


def test_background_job_error_helpers_redact_and_clear_state_section():
    state: dict[str, Any] = {"auto_benchmark": {"last_run_at": "before"}}

    record_background_job_error_in_state(
        state,
        "auto_benchmark",
        RuntimeError("provider token sk-test-secret failed"),
        state_key="auto_benchmark",
        safe_detail=lambda value: str(value).replace("sk-test-secret", "[redacted]"),
        now_iso=lambda: "2026-06-22T00:00:00+00:00",
    )

    last_error = state["auto_benchmark"]["last_error"]
    assert last_error["job"] == "auto_benchmark"
    assert last_error["type"] == "RuntimeError"
    assert "sk-test-secret" not in str(last_error)
    assert last_error["seen_at"] == "2026-06-22T00:00:00+00:00"
    assert clear_background_job_error_in_state(state, state_key="auto_benchmark") is True
    assert "last_error" not in state["auto_benchmark"]
    assert clear_background_job_error_in_state(state, state_key="auto_benchmark") is False


def test_models_needing_discovery_filters_cooldowns_quarantine_and_fresh_verdicts():
    ready = {"id": "ready", "invokable": True}
    cooled = {"id": "cooled", "invokable": True}
    quarantined = {"id": "quarantined", "invokable": True}
    complete = {"id": "complete", "invokable": True}
    not_invokable = {"id": "not-invokable", "invokable": False}
    profiles = ["ficelle/auto-json", "ficelle/auto-tools"]
    state = {"fresh": True}

    def model_on_cooldown(model: dict[str, Any], _state: dict[str, Any]) -> tuple[bool, str | None]:
        blocked = model["id"] == "cooled"
        return blocked, "rate_limited" if blocked else None

    def model_is_quarantined(model: dict[str, Any], _state: dict[str, Any]) -> bool:
        return model["id"] == "quarantined"

    def verified_capability_row(profile_id: str, model: dict[str, Any], _state: dict[str, Any]) -> dict[str, Any]:
        if model["id"] == "ready" and profile_id == "ficelle/auto-json":
            return {"status": "verified"}
        if model["id"] == "complete":
            return {"status": "verified"}
        return {}

    eligible = discovery_eligible_models(
        {"models": [ready, cooled, quarantined, complete, not_invokable, "bad-row"]},
        state,
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
    )
    due = model_due_capabilities(
        ready,
        profiles,
        state,
        verified_capability_row=verified_capability_row,
        catalog_denies_capability=lambda _profile_id, _model: False,
    )
    needing = models_needing_discovery(
        {"models": [ready, cooled, quarantined, complete, not_invokable]},
        state,
        profiles,
        model_on_cooldown=model_on_cooldown,
        model_is_quarantined=model_is_quarantined,
        verified_capability_row=verified_capability_row,
        catalog_denies_capability=lambda _profile_id, _model: False,
    )

    assert [model["id"] for model in eligible] == ["ready", "complete"]
    assert due == ["ficelle/auto-tools"]
    assert [model["id"] for model in needing] == ["ready"]


def test_benchmark_runner_falls_back_to_next_candidate_after_semantic_failure():
    events: list[tuple[str, str, str | None]] = []
    candidates = [
        {"id": "model-a", "upstream_id": "a", "source": "test"},
        {"id": "model-b", "upstream_id": "b", "source": "test"},
    ]

    def safe_detail(value: Any, limit: int | None = None) -> str:
        text = str(value)
        return text[:limit] if limit is not None else text

    runner = BenchmarkRunner(
        load_or_refresh_catalog=lambda _config: {"models": candidates},
        select_models=lambda _profile_id, _catalog, _config, *, purpose: candidates,
        order_benchmark_candidates=lambda _profile_id, models: models,
        benchmark_body=lambda _profile_id, _config: ({}, "exact_text", "ok"),
        invoke_model=lambda model, _body, _config: FakeResponse("ok" if model["id"] == "model-b" else "wrong"),
        classify_failure=lambda _status_code, _text, _model: "unavailable",
        set_cooldown=lambda model, reason, _config, **_kwargs: events.append(("cooldown", model["id"], reason)),
        record_benchmark_failure=lambda model, reason, **_kwargs: events.append(("failure", model["id"], reason)),
        record_benchmark_result=lambda _profile_id, model, result: events.append(
            ("result", model["id"], result["status"])
        ),
        record_success=lambda model, _latency, **_kwargs: events.append(("success", model["id"], None)),
        record_verified_capability=lambda _profile_id, model, result: events.append(
            ("verified", model["id"], result["status"])
        ),
        record_capability_discrepancy=lambda _profile_id, model, passed: events.append(
            ("discrepancy", model["id"], str(passed))
        ),
        extract_message_text=lambda payload: payload["choices"][0]["message"]["content"],
        safe_detail=safe_detail,
        redact_sensitive_json=dict,
        now_iso=lambda: "2026-06-21T00:00:00Z",
        pacer=ProviderProbePacer(pause=lambda _seconds: None),
        route_blocking_reasons=frozenset({"billing", "quota", "auth"}),
    )

    result = runner.benchmark_profile("ficelle/auto-fast", {})

    assert result["status"] == "pass"
    assert result["model_id"] == "model-b"
    assert ("failure", "model-a", "benchmark_failed") in events
    assert ("success", "model-b", None) in events


def test_benchmark_runner_aggregates_run_summary_and_default_profiles():
    candidates = [{"id": "model-a", "upstream_id": "a", "source": "test"}]

    runner = BenchmarkRunner(
        load_or_refresh_catalog=lambda _config: {"models": candidates},
        select_models=lambda _profile_id, _catalog, _config, *, purpose: candidates,
        order_benchmark_candidates=lambda _profile_id, models: models,
        benchmark_body=lambda _profile_id, _config: ({}, "exact_text", "ok"),
        invoke_model=lambda _model, _body, _config: FakeResponse("ok"),
        classify_failure=lambda _status_code, _text, _model: "unavailable",
        set_cooldown=lambda *_args, **_kwargs: None,
        record_benchmark_failure=lambda *_args, **_kwargs: None,
        record_benchmark_result=lambda *_args, **_kwargs: None,
        record_success=lambda *_args, **_kwargs: None,
        record_verified_capability=lambda *_args, **_kwargs: None,
        record_capability_discrepancy=lambda *_args, **_kwargs: None,
        extract_message_text=lambda payload: payload["choices"][0]["message"]["content"],
        safe_detail=lambda value, limit=None: str(value)[:limit] if limit is not None else str(value),
        redact_sensitive_json=dict,
        now_iso=lambda: "2026-06-21T00:00:00Z",
        pacer=ProviderProbePacer(pause=lambda _seconds: None),
        route_blocking_reasons=frozenset(),
    )

    result = runner.run_benchmarks(
        ["ficelle/unknown"],
        {},
        known_profile_ids={"ficelle/auto-fast"},
        default_profile_ids=["ficelle/auto-fast"],
        safe_int=lambda value, default: int(value) if value is not None else default,
        all_candidates=True,
    )

    assert result["summary"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "candidates": {"tested": 1, "passed": 1, "failed": 0},
    }
    assert result["results"][0]["profile_id"] == "ficelle/auto-fast"


def test_capability_discovery_records_shape_rejection_without_cooldown():
    events: list[tuple[str, str, str | None]] = []
    model = {"id": "model-a", "upstream_id": "a", "source": "test"}

    job = CapabilityDiscoveryJob(
        benchmark_body=lambda _profile_id, _config: ({}, "tool_call", "ficelle_probe:ok"),
        invoke_model=lambda _model, _body, _config: FakeResponse("shape rejected", status_code=422),
        classify_failure=lambda _status_code, _text, _model: "bad_request",
        set_cooldown=lambda cooled_model, reason, _config, **_kwargs: events.append(
            ("cooldown", cooled_model["id"], reason)
        ),
        record_benchmark_result=lambda _profile_id, result_model, result: events.append(
            ("result", result_model["id"], result["status"])
        ),
        record_verified_capability=lambda _profile_id, result_model, result: events.append(
            ("verified", result_model["id"], result["status"])
        ),
        record_success=lambda result_model, _latency, **_kwargs: events.append(("success", result_model["id"], None)),
        extract_message_text=lambda payload: payload["choices"][0]["message"]["content"],
        safe_detail=lambda value, *_args: str(value),
        now_iso=lambda: "2026-06-21T00:00:00Z",
        route_blocking_reasons=frozenset({"billing", "quota", "auth"}),
        pacer=ProviderProbePacer(pause=lambda _seconds: None),
        registry=ProbeRegistry(),
    )

    status = job.probe_model_capability(model, "ficelle/auto-tools", {})

    assert status == "failed"
    assert ("result", "model-a", "fail") in events
    assert ("verified", "model-a", "fail") in events
    assert not [event for event in events if event[0] == "cooldown"]


def test_expected_probe_test_type_uses_canonical_profile_lookup():
    test_types = {"ficelle/auto-json": "json"}

    assert expected_probe_test_type(
        "ficelle/auto-json@openrouter",
        test_types,
        canonicalize=lambda profile_id: profile_id.split("@", 1)[0],
    ) == "json"
    assert expected_probe_test_type("ficelle/auto-unknown", test_types) == "exact_text"


def test_probe_registry_validates_typed_json_fields():
    expected = json.dumps({"name": "Maya Lopez", "age": 34, "active": True, "tags": ["chess", "hiking"], "phone": None})

    ok, _message = validate_probe_response(
        "json",
        expected,
        '{"name": "Maya Lopez", "age": 34, "active": true, "tags": ["hiking", "chess"]}',
    )
    bad_type, type_message = validate_probe_response(
        "json",
        expected,
        '{"name": "Maya Lopez", "age": "34", "active": true, "tags": ["hiking", "chess"]}',
    )

    assert ok is True
    assert bad_type is False
    assert "expected number" in type_message


def test_probe_registry_validates_tool_call_and_orchestration_shapes():
    tool_ok, _message = validate_probe_response(
        "tool_call",
        "ficelle_probe:tool-canary-ok",
        "",
        tool_call_payload("ficelle_probe", {"status": "tool-canary-ok"}),
    )
    orchestration_ok, _message = validate_probe_response(
        "orchestration",
        "route_task:summarize:A7",
        "",
        tool_call_payload("route_task", {"category": "summarize", "ticket_id": "#A7"}),
    )
    orchestration_bad, bad_message = validate_probe_response(
        "orchestration",
        "route_task:summarize:A7",
        "",
        tool_call_payload("route_task", {"category": "code", "ticket_id": "#A7"}),
    )

    assert tool_ok is True
    assert orchestration_ok is True
    assert orchestration_bad is False
    assert "category" in bad_message


def test_probe_registry_requires_reasoning_trace_and_answer_marker():
    payload = {"choices": [{"message": {"reasoning_content": [{"text": "trace"}]}}]}

    ok, _message = validate_probe_response("reasoning", "final-answer", "The final-answer is here.", payload)
    missing_trace, trace_message = validate_probe_response("reasoning", "final-answer", "The final-answer is here.", {})
    wrong_answer, answer_message = validate_probe_response("reasoning", "final-answer", "not it", payload)

    assert ok is True
    assert missing_trace is False
    assert "no reasoning trace" in trace_message
    assert wrong_answer is False
    assert "answer marker" in answer_message


def test_probe_registry_validates_compression_word_budget_and_facts():
    registry = ProbeRegistry(compression_max_words=6)
    expected = "budget 4200|Lena Park|KX-19"

    ok, _message = registry.validate_response("compression", expected, "budget 4200 Lena Park KX-19")
    too_long, length_message = registry.validate_response(
        "compression",
        expected,
        "budget 4200 Lena Park KX-19 extra word",
    )
    missing_fact, missing_message = registry.validate_response("compression", expected, "budget 4200 KX-19")

    assert ok is True
    assert too_long is False
    assert "summary not bounded" in length_message
    assert missing_fact is False
    assert "Lena Park" in missing_message


def test_probe_registry_support_helpers_match_legacy_shapes():
    call = {"function": {"arguments": '{"status":"ok"}'}}

    assert tool_call_arguments(call) == {"status": "ok"}
    assert json_fields_match({"tags": ["a", "b"], "phone": None}, {"tags": ["B", "A"]}) == (True, "fields match")
    assert extract_message_text({"choices": [{"message": {"content": [{"text": " alpha "}, {"text": " beta"}]}}]}) == "alpha \n beta"
    assert extract_reasoning_text({"choices": [{"message": {"reasoning": "  trace  "}}]}) == "trace"
    assert extract_reasoning_text({"choices": [{"message": {"reasoning_content": [{"text": "a"}, {"text": "b"}]}}]}) == "a\nb"
    assert extract_tool_calls(tool_call_payload("ficelle_probe", {"status": "ok"})) == [
        {"type": "function", "function": {"name": "ficelle_probe", "arguments": {"status": "ok"}}}
    ]
    assert has_deliverable_message({"choices": [{"message": {"content": None, "reasoning": "thinking"}}]}) is False
    assert has_deliverable_message({"choices": [{"message": {"content": "hello"}}]}) is True
    assert has_deliverable_message({"choices": [{"message": {"tool_calls": [{"function": {"name": "x"}}]}}]}) is True


def test_finish_reason_is_truncation_accepts_every_provider_spelling():
    """One concept, several spellings: matching only OpenAI's `length` would half-fix the bug.

    Mistral is a configured provider and its enum is stop|length|model_length|error|tool_calls, so
    an exact `== "length"` would still cool a Mistral model that merely ran out of budget.
    """

    def payload(finish_reason: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": ""}, "finish_reason": finish_reason}]}

    for spelling in ("length", "model_length", "max_tokens", "MAX_TOKENS", " Length ", "max_output_tokens"):
        assert finish_reason_is_truncation(payload(spelling)) is True, spelling
    for spelling in ("stop", "tool_calls", "error", "content_filter", "", None):
        assert finish_reason_is_truncation(payload(spelling)) is False, spelling
    # Malformed upstream shapes must read as "not truncated", never raise.
    for broken in ({}, {"choices": None}, {"choices": []}, {"choices": [None]}, {"choices": ["x"]}, [], None, "x"):
        assert finish_reason_is_truncation(broken) is False, broken
