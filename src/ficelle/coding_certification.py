"""Signed central evidence for the ``ficelle/auto-coding`` routing lane.

Public benchmark results are deliberately represented as priors in the envelope but are never
indexed by the route helpers below. Only an exact, unexpired Ficelle certification can make a
provider deployment eligible.
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ficelle.json_store import atomic_write_json, load_json


CODING_PROFILE_ID = "ficelle/auto-coding"
DEFAULT_MANIFEST_URL = "https://install.ficelle.ai/api/coding/certifications"
MANIFEST_ENV = "FICELLE_CODING_CERTIFICATION_URL"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_REFRESH_SECONDS = 6 * 3600
MANIFEST_ERROR_RETRY_SECONDS = 15 * 60
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_CLOCK_SKEW_SECONDS = 5 * 60
MAX_VALIDITY_SECONDS = 90 * 24 * 3600
KEY_ID = "ficelle-coding-2026-01"
PUBLIC_KEYS_B64 = {
    KEY_ID: "YQI1JvTL9awSPj/GoD0j5iwNG5TSLGYrpUYUvMBM4kc=",
}
REQUIRED_BENCHMARKS = frozenset({"aider-polyglot", "swe-rebench", "terminal-bench-2.1"})
POLICY_VERSION = "coding-v1"
COMPATIBILITY_CANARY_VERSION = "coding-compatibility-v1"


class CodingCertificationError(ValueError):
    """A manifest could not be trusted or used."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodingCertificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes | str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CodingCertificationError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CodingCertificationError("certification manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CodingCertificationError("certification envelope must be an object")
    return value


def canonical_json(value: Mapping[str, Any]) -> bytes:
    def reject_non_finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise CodingCertificationError("manifest contains a non-finite number")
        if isinstance(item, dict):
            for nested in item.values():
                reject_non_finite(nested)
        elif isinstance(item, list):
            for nested in item:
                reject_non_finite(nested)

    try:
        reject_non_finite(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except CodingCertificationError:
        raise
    except (RecursionError, UnicodeError, TypeError, ValueError, OverflowError) as exc:
        raise CodingCertificationError("certification manifest cannot be canonicalized") from exc


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CodingCertificationError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CodingCertificationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CodingCertificationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _required_text(row: Mapping[str, Any], field: str, *, limit: int = 300) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise CodingCertificationError(f"{field} must be a non-empty string")
    return value.strip()


def _bounded_score(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodingCertificationError(f"{field} must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise CodingCertificationError(f"{field} must be between 0 and 100")
    return score


def _validate_benchmark(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CodingCertificationError("benchmark rows must be objects")
    name = _required_text(row, "name", limit=80)
    _required_text(row, "suite_version", limit=120)
    harness_repository = _required_text(row, "harness_repository", limit=300)
    parsed_repository = urlparse(harness_repository)
    if parsed_repository.scheme != "https" or not parsed_repository.netloc:
        raise CodingCertificationError("harness_repository must use HTTPS")
    commit = _required_text(row, "harness_commit", limit=64)
    if not all(character in "0123456789abcdefABCDEF" for character in commit) or len(commit) != 40:
        raise CodingCertificationError("harness_commit must be a full 40-character git commit")
    task_count = row.get("task_count")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0:
        raise CodingCertificationError("task_count must be a positive integer")
    pass_at_1 = row.get("pass_at_1")
    if isinstance(pass_at_1, bool) or not isinstance(pass_at_1, (int, float)):
        raise CodingCertificationError("pass_at_1 must be a number")
    if not math.isfinite(float(pass_at_1)) or not 0 <= float(pass_at_1) <= 1:
        raise CodingCertificationError("pass_at_1 must be between 0 and 1")
    fingerprint = _required_text(row, "settings_fingerprint", limit=128)
    for field, value in (
        ("settings_fingerprint", fingerprint),
        ("run_record_fingerprint", _required_text(row, "run_record_fingerprint", limit=128)),
        ("official_result_fingerprint", _required_text(row, "official_result_fingerprint", limit=128)),
        ("command_fingerprint", _required_text(row, "command_fingerprint", limit=128)),
    ):
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise CodingCertificationError(f"{field} must be a SHA-256 digest")
    if row.get("evidence_kind") != "central_run":
        raise CodingCertificationError("certifying benchmark evidence_kind must be central_run")
    result = dict(row)
    result["name"] = name
    return result


def validate_manifest(
    manifest: Any,
    *,
    now: datetime | None = None,
    require_complete_policy: bool = False,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CodingCertificationError("manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CodingCertificationError("unsupported certification manifest schema")
    _required_text(manifest, "manifest_id", limit=120)
    if _required_text(manifest, "policy_version", limit=80) != POLICY_VERSION:
        raise CodingCertificationError("unsupported coding certification policy")
    generated_at = _parse_timestamp(manifest.get("generated_at"), "generated_at")
    expires_at = _parse_timestamp(manifest.get("expires_at"), "expires_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if generated_at.timestamp() > current.timestamp() + MAX_CLOCK_SKEW_SECONDS:
        raise CodingCertificationError("manifest generated_at is too far in the future")
    if expires_at <= current:
        raise CodingCertificationError("certification manifest is expired")
    if expires_at <= generated_at:
        raise CodingCertificationError("expires_at must be after generated_at")
    if (expires_at - generated_at).total_seconds() > MAX_VALIDITY_SECONDS:
        raise CodingCertificationError("certification manifest validity exceeds 90 days")

    raw_certifications = manifest.get("certifications")
    raw_priors = manifest.get("priors")
    if not isinstance(raw_certifications, list) or not isinstance(raw_priors, list):
        raise CodingCertificationError("certifications and priors must be arrays")
    identities: set[tuple[str, str]] = set()
    certifications: list[dict[str, Any]] = []
    for raw in raw_certifications:
        if not isinstance(raw, dict):
            raise CodingCertificationError("certification rows must be objects")
        provider = _required_text(raw, "provider", limit=80).lower()
        upstream_model_id = _required_text(raw, "upstream_model_id", limit=240)
        identity = (provider, upstream_model_id)
        if identity in identities:
            raise CodingCertificationError("duplicate provider/model certification")
        identities.add(identity)
        _bounded_score(raw, "quality_score")
        certified_at = _parse_timestamp(raw.get("certified_at"), "certified_at")
        if certified_at.timestamp() > generated_at.timestamp() + MAX_CLOCK_SKEW_SECONDS:
            raise CodingCertificationError("certified_at cannot be after manifest generation")
        if _required_text(raw, "compatibility_canary_version", limit=80) != COMPATIBILITY_CANARY_VERSION:
            raise CodingCertificationError("unsupported compatibility canary version")
        raw_benchmarks = raw.get("benchmarks")
        if not isinstance(raw_benchmarks, list):
            raise CodingCertificationError("benchmarks must be an array")
        benchmarks = [_validate_benchmark(item) for item in raw_benchmarks]
        benchmark_names = {item["name"] for item in benchmarks}
        if len(benchmark_names) != len(benchmarks):
            raise CodingCertificationError("duplicate benchmark in certification")
        if require_complete_policy and benchmark_names != REQUIRED_BENCHMARKS:
            raise CodingCertificationError("certification does not contain the exact required benchmark set")
        row = dict(raw)
        row.update({"provider": provider, "upstream_model_id": upstream_model_id, "benchmarks": benchmarks})
        certifications.append(row)
    for prior in raw_priors:
        if not isinstance(prior, dict):
            raise CodingCertificationError("prior rows must be objects")
        source_url = _required_text(prior, "source_url", limit=500)
        parsed_source = urlparse(source_url)
        if parsed_source.scheme != "https" or not parsed_source.netloc:
            raise CodingCertificationError("prior source_url must use HTTPS")
        _required_text(prior, "benchmark", limit=80)
        _parse_timestamp(_required_text(prior, "observed_at", limit=80), "observed_at")
        _required_text(prior, "provider", limit=80)
        _required_text(prior, "upstream_model_id", limit=240)
        _required_text(prior, "suite_version", limit=120)
        prior_commit = _required_text(prior, "harness_commit", limit=40)
        if len(prior_commit) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in prior_commit
        ):
            raise CodingCertificationError("prior harness_commit must be a full git commit")
        _bounded_score(prior, "score")
        if prior.get("evidence_kind") != "prior":
            raise CodingCertificationError("public benchmark evidence_kind must be prior")
    result = dict(manifest)
    result["certifications"] = certifications
    return result


def verify_envelope(envelope: Any, *, now: datetime | None = None) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise CodingCertificationError("certification envelope must be an object")
    key_id = _required_text(envelope, "key_id", limit=80)
    public_b64 = PUBLIC_KEYS_B64.get(key_id)
    if public_b64 is None:
        raise CodingCertificationError("unknown certification signing key")
    manifest = envelope.get("manifest")
    signature_text = envelope.get("signature")
    if not isinstance(signature_text, str) or not signature_text:
        raise CodingCertificationError("certification signature is missing")
    try:
        signature = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64, validate=True))
        public_key.verify(signature, canonical_json(manifest))
    except CodingCertificationError:
        raise
    except (ValueError, InvalidSignature) as exc:
        raise CodingCertificationError("certification signature is invalid") from exc
    return validate_manifest(manifest, now=now, require_complete_policy=True)


def certification_identity(model: Mapping[str, Any]) -> tuple[str, str]:
    return str(model.get("source") or "").lower(), str(model.get("upstream_id") or "")


def certification_index(manifest: Mapping[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return {}
    rows = manifest.get("certifications")
    if not isinstance(rows, list):
        return {}
    return {
        (str(row.get("provider") or "").lower(), str(row.get("upstream_model_id") or "")): dict(row)
        for row in rows
        if isinstance(row, dict)
    }


def certification_for_model(model: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return certification_index(manifest).get(certification_identity(model))


def manifest_url() -> str:
    return os.getenv(MANIFEST_ENV, DEFAULT_MANIFEST_URL).strip()


def _open_manifest(request: urllib.request.Request, timeout: float) -> Any:
    class SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self,
            redirected_request: urllib.request.Request,
            file_handle: Any,
            code: int,
            message: str,
            headers: Any,
            new_url: str,
        ) -> urllib.request.Request | None:
            if urlparse(new_url).scheme != "https":
                raise CodingCertificationError("refusing insecure certification redirect")
            return super().redirect_request(
                redirected_request, file_handle, code, message, headers, new_url
            )

    return urllib.request.build_opener(SecureRedirectHandler()).open(request, timeout=timeout)


def fetch_envelope(
    url: str | None = None,
    *,
    opener: Callable[[urllib.request.Request, float], Any] | None = None,
) -> dict[str, Any]:
    target = url or manifest_url()
    parsed = urlparse(target)
    if parsed.scheme != "https" or not parsed.netloc:
        raise CodingCertificationError("certification manifest URL must use HTTPS")
    request = urllib.request.Request(
        target,
        headers={"Accept": "application/json", "User-Agent": "Ficelle coding-certification"},
    )
    try:
        with (opener or _open_manifest)(request, 20) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else target
            if urlparse(final_url).scheme != "https":
                raise CodingCertificationError("certification response URL must use HTTPS")
            body = response.read(MAX_MANIFEST_BYTES + 1)
    except Exception as exc:
        raise CodingCertificationError(f"certification fetch failed ({type(exc).__name__})") from exc
    if len(body) > MAX_MANIFEST_BYTES:
        raise CodingCertificationError("certification manifest is too large")
    envelope = strict_json_loads(body)
    verify_envelope(envelope)
    return envelope


def load_cached_envelope(path: Path, *, now: datetime | None = None) -> tuple[dict[str, Any] | None, str | None]:
    raw = load_json(path, None)
    try:
        if not isinstance(raw, dict):
            raise CodingCertificationError("certification cache is missing")
        verify_envelope(raw, now=now)
        return raw, None
    except CodingCertificationError as exc:
        return None, str(exc)


def refresh_cache(cache_path: Path, status_path: Path, *, url: str | None = None) -> dict[str, Any]:
    checked_at = datetime.now(UTC).isoformat()
    try:
        envelope = fetch_envelope(url)
        manifest = verify_envelope(envelope)
        atomic_write_json(cache_path, envelope)
        status = {
            "status": "valid",
            "checked_at": checked_at,
            "manifest_id": manifest["manifest_id"],
            "generated_at": manifest["generated_at"],
            "expires_at": manifest["expires_at"],
            "certification_count": len(manifest["certifications"]),
            "prior_count": len(manifest["priors"]),
            "message": "",
        }
    except CodingCertificationError as exc:
        cached, cached_error = load_cached_envelope(cache_path)
        status = {
            "status": "valid_cache" if cached is not None else "unavailable",
            "checked_at": checked_at,
            "manifest_id": ((cached or {}).get("manifest") or {}).get("manifest_id"),
            "generated_at": ((cached or {}).get("manifest") or {}).get("generated_at"),
            "expires_at": ((cached or {}).get("manifest") or {}).get("expires_at"),
            "certification_count": len((((cached or {}).get("manifest") or {}).get("certifications") or [])),
            "prior_count": len((((cached or {}).get("manifest") or {}).get("priors") or [])),
            "message": str(exc if cached is not None else cached_error or exc),
        }
    atomic_write_json(status_path, status)
    return status


def public_status(cache_path: Path, status_path: Path) -> dict[str, Any]:
    cached, error = load_cached_envelope(cache_path)
    stored = load_json(status_path, {})
    status = dict(stored) if isinstance(stored, dict) else {}
    manifest = cached.get("manifest") if cached is not None else None
    status.update(
        {
            "status": (
                status.get("status")
                if cached is not None and status.get("status") in {"valid", "valid_cache"}
                else "valid_cache" if cached is not None else "unavailable"
            ),
            "manifest_id": manifest.get("manifest_id") if isinstance(manifest, dict) else None,
            "generated_at": manifest.get("generated_at") if isinstance(manifest, dict) else None,
            "expires_at": manifest.get("expires_at") if isinstance(manifest, dict) else None,
            "certification_count": len(manifest.get("certifications") or []) if isinstance(manifest, dict) else 0,
            "prior_count": len(manifest.get("priors") or []) if isinstance(manifest, dict) else 0,
        }
    )
    if cached is None:
        status["message"] = error or status.get("message") or "certification manifest unavailable"
    return status


def refresh_loop(cache_path: Path, status_path: Path, stop: threading.Event | None = None) -> None:
    stopper = stop or threading.Event()
    while not stopper.is_set():
        try:
            status = refresh_cache(cache_path, status_path)
        except OSError:
            status = {"status": "unavailable"}
        delay = MANIFEST_REFRESH_SECONDS if status["status"] == "valid" else MANIFEST_ERROR_RETRY_SECONDS
        stopper.wait(delay)


_MANIFEST_CACHE_LOCK = threading.Lock()
_MANIFEST_CACHE_PATH: Path | None = None
_MANIFEST_CACHE_MTIME_NS: int | None = None
_MANIFEST_CACHE_VALUE: dict[str, Any] | None = None
_MANIFEST_CACHE_EXPIRES_AT: datetime | None = None


def cached_manifest(cache_path: Path) -> dict[str, Any] | None:
    global _MANIFEST_CACHE_PATH, _MANIFEST_CACHE_MTIME_NS, _MANIFEST_CACHE_VALUE, _MANIFEST_CACHE_EXPIRES_AT
    try:
        mtime_ns = cache_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    with _MANIFEST_CACHE_LOCK:
        if (
            cache_path == _MANIFEST_CACHE_PATH
            and mtime_ns == _MANIFEST_CACHE_MTIME_NS
            and _MANIFEST_CACHE_EXPIRES_AT is not None
            and datetime.now(UTC) < _MANIFEST_CACHE_EXPIRES_AT
        ):
            return _MANIFEST_CACHE_VALUE
        envelope, _error = load_cached_envelope(cache_path)
        manifest = envelope.get("manifest") if envelope is not None else None
        value = manifest if isinstance(manifest, dict) else None
        _MANIFEST_CACHE_PATH = cache_path
        _MANIFEST_CACHE_MTIME_NS = mtime_ns
        _MANIFEST_CACHE_VALUE = value
        _MANIFEST_CACHE_EXPIRES_AT = (
            _parse_timestamp(value.get("expires_at"), "expires_at") if value is not None else None
        )
        return value


def quality_score(model: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> float:
    row = certification_for_model(model, manifest)
    return float(row.get("quality_score") or 0.0) if row else 0.0
