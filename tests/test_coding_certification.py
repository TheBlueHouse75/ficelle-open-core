from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ficelle import coding_certification


def benchmark(name: str) -> dict:
    return {
        "name": name,
        "suite_version": "2026-08",
        "harness_repository": "https://github.com/example/harness",
        "harness_commit": "a1b2c3d4" * 5,
        "task_count": 20,
        "pass_at_1": 0.75,
        "settings_fingerprint": "sha256:" + "a" * 64,
        "run_record_fingerprint": "sha256:" + "b" * 64,
        "official_result_fingerprint": "sha256:" + "c" * 64,
        "command_fingerprint": "sha256:" + "d" * 64,
        "evidence_kind": "central_run",
    }


def manifest(*, certifications: list[dict] | None = None, priors: list[dict] | None = None) -> dict:
    now = datetime.now(UTC)
    return {
        "schema_version": 1,
        "manifest_id": "fixture-1",
        "generated_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(days=30)).isoformat(),
        "policy_version": "coding-v1",
        "certifications": certifications or [],
        "priors": priors or [],
    }


def certification(provider: str = "openrouter", upstream_model_id: str = "acme/code") -> dict:
    return {
        "provider": provider,
        "upstream_model_id": upstream_model_id,
        "quality_score": 82.5,
        "certified_at": datetime.now(UTC).isoformat(),
        "compatibility_canary_version": "coding-compatibility-v1",
        "benchmarks": [benchmark(name) for name in sorted(coding_certification.REQUIRED_BENCHMARKS)],
    }


def signed_envelope(monkeypatch: pytest.MonkeyPatch, payload: dict) -> dict:
    private_key = Ed25519PrivateKey.generate()
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    monkeypatch.setitem(
        coding_certification.PUBLIC_KEYS_B64,
        "test-key",
        base64.b64encode(public).decode("ascii"),
    )
    signature = private_key.sign(coding_certification.canonical_json(payload))
    return {
        "key_id": "test-key",
        "manifest": payload,
        "signature": base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii"),
    }


def test_signed_manifest_verifies_and_matches_exact_provider_identity(monkeypatch):
    payload = manifest(certifications=[certification()])
    envelope = signed_envelope(monkeypatch, payload)

    verified = coding_certification.verify_envelope(envelope)

    exact = {"source": "openrouter", "upstream_id": "acme/code"}
    wrong_provider = {"source": "nous", "upstream_id": "acme/code"}
    wrong_model = {"source": "openrouter", "upstream_id": "acme/code-v2"}
    assert coding_certification.certification_for_model(exact, verified)["quality_score"] == 82.5
    assert coding_certification.certification_for_model(wrong_provider, verified) is None
    assert coding_certification.certification_for_model(wrong_model, verified) is None


def test_public_prior_never_becomes_a_route_certification(monkeypatch):
    prior = {
        "source_url": "https://example.test/leaderboard.json",
        "benchmark": "aider-polyglot",
        "observed_at": datetime.now(UTC).isoformat(),
        "provider": "openrouter",
        "upstream_model_id": "acme/code",
        "score": 99,
        "suite_version": "2026-08",
        "harness_commit": "a1b2c3d4" * 5,
        "evidence_kind": "prior",
    }
    verified = coding_certification.verify_envelope(signed_envelope(monkeypatch, manifest(priors=[prior])))

    assert coding_certification.certification_index(verified) == {}
    assert coding_certification.quality_score(
        {"source": "openrouter", "upstream_id": "acme/code"}, verified
    ) == 0


def test_tampered_unknown_and_expired_envelopes_fail_closed(monkeypatch):
    payload = manifest(certifications=[certification()])
    envelope = signed_envelope(monkeypatch, payload)
    envelope["manifest"]["certifications"][0]["quality_score"] = 100
    with pytest.raises(coding_certification.CodingCertificationError, match="signature"):
        coding_certification.verify_envelope(envelope)

    envelope["key_id"] = "unknown"
    with pytest.raises(coding_certification.CodingCertificationError, match="unknown"):
        coding_certification.verify_envelope(envelope)

    expired = manifest()
    expired["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with pytest.raises(coding_certification.CodingCertificationError, match="expired"):
        coding_certification.verify_envelope(signed_envelope(monkeypatch, expired))


def test_strict_parser_rejects_duplicate_keys_and_non_finite_numbers():
    with pytest.raises(coding_certification.CodingCertificationError, match="duplicate"):
        coding_certification.strict_json_loads('{"key_id":"a","key_id":"b"}')
    with pytest.raises(coding_certification.CodingCertificationError, match="non-finite"):
        coding_certification.strict_json_loads('{"score":NaN}')


def test_verifier_rejects_uncanonicalizable_manifest_as_a_domain_error():
    deeply_nested: object = 0
    for _ in range(2_000):
        deeply_nested = [deeply_nested]
    with pytest.raises(coding_certification.CodingCertificationError, match="canonicalized"):
        coding_certification.verify_envelope(
            {
                "key_id": coding_certification.KEY_ID,
                "manifest": {"nested": deeply_nested},
                "signature": "AA",
            }
        )

    with pytest.raises(coding_certification.CodingCertificationError, match="canonicalized"):
        coding_certification.verify_envelope(
            {
                "key_id": coding_certification.KEY_ID,
                "manifest": {"text": "\ud800"},
                "signature": "AA",
            }
        )


def test_publisher_validation_requires_the_exact_policy_benchmarks():
    incomplete = certification()
    incomplete["benchmarks"] = incomplete["benchmarks"][:-1]

    with pytest.raises(coding_certification.CodingCertificationError, match="exact required"):
        coding_certification.validate_manifest(
            manifest(certifications=[incomplete]),
            require_complete_policy=True,
        )


def test_bad_refresh_keeps_a_still_valid_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "coding.json"
    status_path = tmp_path / "status.json"
    envelope = signed_envelope(monkeypatch, manifest(certifications=[certification()]))
    from ficelle.json_store import atomic_write_json

    atomic_write_json(cache_path, envelope)
    monkeypatch.setattr(
        coding_certification,
        "fetch_envelope",
        lambda _url=None: (_ for _ in ()).throw(
            coding_certification.CodingCertificationError("remote signature invalid")
        ),
    )

    status = coding_certification.refresh_cache(cache_path, status_path)

    assert status["status"] == "valid_cache"
    assert status["certification_count"] == 1
    assert coding_certification.load_cached_envelope(cache_path)[0] == envelope


def test_public_status_recognizes_a_valid_cache_without_a_status_file(monkeypatch, tmp_path):
    cache_path = tmp_path / "coding.json"
    envelope = signed_envelope(monkeypatch, manifest())
    from ficelle.json_store import atomic_write_json

    atomic_write_json(cache_path, envelope)

    status = coding_certification.public_status(cache_path, tmp_path / "missing-status.json")

    assert status["status"] == "valid_cache"
    assert status["manifest_id"] == "fixture-1"


def test_refresh_loop_retries_after_a_local_write_failure(monkeypatch, tmp_path):
    calls = []

    class StopAfterOneWait:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            calls.append(delay)
            self.stopped = True

    monkeypatch.setattr(
        coding_certification,
        "refresh_cache",
        lambda *_args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    coding_certification.refresh_loop(
        tmp_path / "cache.json",
        tmp_path / "status.json",
        StopAfterOneWait(),
    )

    assert calls == [coding_certification.MANIFEST_ERROR_RETRY_SECONDS]


def test_refresh_loop_retries_quickly_when_remote_refresh_uses_valid_cache(monkeypatch, tmp_path):
    calls = []

    class StopAfterOneWait:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            calls.append(delay)
            self.stopped = True

    monkeypatch.setattr(
        coding_certification,
        "refresh_cache",
        lambda *_args: {"status": "valid_cache"},
    )

    coding_certification.refresh_loop(
        tmp_path / "cache.json",
        tmp_path / "status.json",
        StopAfterOneWait(),
    )

    assert calls == [coding_certification.MANIFEST_ERROR_RETRY_SECONDS]


def test_in_memory_manifest_stops_routing_when_signed_expiry_passes(monkeypatch, tmp_path):
    cache_path = tmp_path / "coding.json"
    now = datetime.now(UTC)
    envelope = signed_envelope(monkeypatch, manifest())
    from ficelle.json_store import atomic_write_json

    atomic_write_json(cache_path, envelope)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is not None else cls.current.replace(tzinfo=None)

    Clock.current = now
    monkeypatch.setattr(coding_certification, "datetime", Clock)
    assert coding_certification.cached_manifest(cache_path) is not None

    Clock.current = now + timedelta(days=31)

    assert coding_certification.cached_manifest(cache_path) is None
