"""Verify offline-signed Ficelle Pro release artifacts before installation.

The embedded public key is independent from the delivery service and the customer licence. The
matching private key stays offline, so TLS or service compromise cannot mint an approved wheel.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ficelle import __version__


SCHEMA_VERSION = 1
RELEASE_KEY_ID = "ficelle-release-2026-09"
RELEASE_PUBLIC_KEYS = {
    RELEASE_KEY_ID: "jcmR+/3HQvCxN5Xql50ytVHz8gN6JjcsXW9fBbDEZ1g=",
}
MAX_ATTESTATION_BYTES = 16 * 1024
_ARTIFACT_FIELDS = frozenset(
    {"distribution", "version", "core_version", "filename", "sha256", "size"}
)


class ArtifactIntegrityError(ValueError):
    """An artifact attestation is invalid or does not describe the supplied wheel."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactIntegrityError(f"duplicate attestation key: {key}")
        result[key] = value
    return result


def parse_attestation(raw: bytes | str) -> dict[str, Any]:
    try:
        payload = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        envelope = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ArtifactIntegrityError(f"non-finite attestation number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("release attestation is invalid JSON") from exc
    if not isinstance(envelope, dict):
        raise ArtifactIntegrityError("release attestation must be an object")
    return envelope


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64decode(value: str, *, field: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"invalid {field}") from exc


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_identity(wheel: Path) -> tuple[str, str, str]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    except (OSError, UnicodeError, zipfile.BadZipFile, StopIteration) as exc:
        raise ArtifactIntegrityError("wheel metadata could not be verified") from exc
    distribution = _normalized_distribution(str(metadata.get("Name") or "").strip())
    version = str(metadata.get("Version") or "").strip()
    requirements = {
        requirement.replace(" ", "").lower()
        for requirement in metadata.get_all("Requires-Dist", [])
    }
    core_requirement = next(
        (
            requirement
            for requirement in requirements
            if requirement.startswith("ficelle-router==")
        ),
        "",
    )
    core_version = core_requirement.removeprefix("ficelle-router==")
    if distribution != "ficelle-pro" or not version or not core_version:
        raise ArtifactIntegrityError("artifact is not a compatible Ficelle Pro wheel")
    return distribution, version, core_version


def _signed_message(key_id: str, artifact: Mapping[str, Any]) -> bytes:
    return _canonical_json(
        {"schema_version": SCHEMA_VERSION, "key_id": key_id, "artifact": dict(artifact)}
    )


def verify_artifact_attestation(
    wheel: Path,
    envelope: Any,
    *,
    expected_core_version: str = __version__,
    public_keys: Mapping[str, str] = RELEASE_PUBLIC_KEYS,
) -> dict[str, Any]:
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "key_id",
        "artifact",
        "signature",
    }:
        raise ArtifactIntegrityError("invalid release attestation shape")
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactIntegrityError("unsupported release attestation schema")
    key_id = envelope.get("key_id")
    artifact = envelope.get("artifact")
    signature = envelope.get("signature")
    if not isinstance(key_id, str) or key_id not in public_keys:
        raise ArtifactIntegrityError("unknown release signing key")
    if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
        raise ArtifactIntegrityError("invalid attested artifact shape")
    if not isinstance(signature, str):
        raise ArtifactIntegrityError("invalid release signature")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            _b64decode(public_keys[key_id], field="release public key")
        )
        public_key.verify(
            _b64decode(signature, field="release signature"),
            _signed_message(key_id, artifact),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ArtifactIntegrityError("invalid release signature") from exc

    distribution, version, core_version = _wheel_identity(wheel)
    size = artifact.get("size")
    expected = {
        "distribution": distribution,
        "version": version,
        "core_version": core_version,
        "filename": wheel.name,
        "sha256": _sha256_file(wheel),
        "size": wheel.stat().st_size,
    }
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or artifact != expected:
        raise ArtifactIntegrityError("release attestation does not match the wheel")
    if core_version != expected_core_version:
        raise ArtifactIntegrityError(
            f"Pro wheel requires Ficelle Core {core_version}, not {expected_core_version}"
        )
    return dict(artifact)


def verify_attestation_file(wheel: Path, attestation: Path) -> dict[str, Any]:
    try:
        if attestation.stat().st_size > MAX_ATTESTATION_BYTES:
            raise ArtifactIntegrityError("release attestation exceeds the size limit")
        envelope = parse_attestation(attestation.read_bytes())
    except OSError as exc:
        raise ArtifactIntegrityError("release attestation is unavailable") from exc
    return verify_artifact_attestation(wheel, envelope)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a Ficelle Pro release attestation.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    verify = subcommands.add_parser("verify", help="Verify a wheel and its signed attestation.")
    verify.add_argument("wheel", type=Path)
    verify.add_argument("attestation", type=Path)
    options = parser.parse_args(argv)
    try:
        verify_attestation_file(options.wheel, options.attestation)
    except ArtifactIntegrityError as exc:
        parser.error(str(exc))
    print(f"release attestation verified: {options.wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
