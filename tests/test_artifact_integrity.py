from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ficelle import __version__, artifact_integrity


def _wheel(path: Path, *, content: bytes = b"payload") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"ficelle_pro-{__version__}.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: ficelle-pro\n"
            f"Version: {__version__}\n"
            f"Requires-Dist: ficelle-router=={__version__}\n",
        )
        archive.writestr("ficelle_pro/payload.bin", content)
    return path


def _attestation(wheel: Path) -> tuple[dict[str, object], dict[str, str]]:
    private_key = Ed25519PrivateKey.generate()
    public_key = base64.b64encode(
        private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")
    artifact = {
        "distribution": "ficelle-pro",
        "version": __version__,
        "core_version": __version__,
        "filename": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "size": wheel.stat().st_size,
    }
    unsigned = {
        "schema_version": artifact_integrity.SCHEMA_VERSION,
        "key_id": "test-release",
        "artifact": artifact,
    }
    message = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return {
        **unsigned,
        "signature": base64.b64encode(private_key.sign(message)).decode("ascii"),
    }, {"test-release": public_key}


def test_signed_release_attestation_accepts_the_exact_wheel(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path / f"ficelle_pro-{__version__}-py3-none-any.whl")
    envelope, public_keys = _attestation(wheel)

    artifact = artifact_integrity.verify_artifact_attestation(
        wheel,
        envelope,
        public_keys=public_keys,
    )

    assert artifact["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_attestation_rejects_tampered_wheel_and_signature(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path / f"ficelle_pro-{__version__}-py3-none-any.whl")
    envelope, public_keys = _attestation(wheel)
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(artifact_integrity.ArtifactIntegrityError, match="does not match"):
        artifact_integrity.verify_artifact_attestation(
            wheel,
            envelope,
            public_keys=public_keys,
        )

    wheel = _wheel(wheel)
    envelope, public_keys = _attestation(wheel)
    envelope["artifact"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(artifact_integrity.ArtifactIntegrityError, match="signature"):
        artifact_integrity.verify_artifact_attestation(
            wheel,
            envelope,
            public_keys=public_keys,
        )


def test_attestation_parser_rejects_duplicate_keys() -> None:
    with pytest.raises(artifact_integrity.ArtifactIntegrityError, match="duplicate"):
        artifact_integrity.parse_attestation('{"schema_version":1,"schema_version":1}')
