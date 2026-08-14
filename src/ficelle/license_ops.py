"""Shared license runtime facade for both callers: the CLI (`ficelle license`) and the admin
daemon (`/admin/license`).

The license *domain* — signing, Ed25519 verification, the on-disk cache, and the
license-service HTTP calls — lives in the closed pack ``ficelle_pro.licensing`` and is never
imported by the open core except through the one guarded seam below (open-core extraction R6).
This core module owns the *runtime* concerns the pack must not know about — the Ficelle home
paths, the opaque machine fingerprint, the configured service URL — and the
activate/refresh/deactivate/status orchestration that the CLI and the admin endpoint share, so
neither caller has to import the closed pack directly. Both reach Pro through this facade and
catch only the two exceptions it raises.
"""
from __future__ import annotations

import os
import platform
import uuid
from pathlib import Path
from typing import Any

from ficelle.runtime_paths import RuntimePaths
from ficelle.url_security import uses_secure_http_transport

_RUNTIME_PATHS = RuntimePaths.from_env()
ENTITLEMENT_PATH = _RUNTIME_PATHS.router_dir / "entitlement.token"
MACHINE_ID_PATH = _RUNTIME_PATHS.router_dir / "machine-id"
DEFAULT_LICENSE_SERVICE_URL = "https://install.ficelle.ai"


class LicenseNotInstalled(Exception):
    """Raised when the closed pack (and thus the license client) is absent — the free core."""


class LicenseOperationError(Exception):
    """An activate/refresh/deactivate operation failed (service rejected/unreachable, or the
    returned entitlement failed verification). Carries a human-readable, key-free message."""


def service_url() -> str:
    return os.getenv("FICELLE_LICENSE_SERVICE_URL", DEFAULT_LICENSE_SERVICE_URL).strip()


def validated_service_url() -> str:
    """Return the service URL only when license credentials can be sent safely."""
    url = service_url()
    if uses_secure_http_transport(url):
        return url
    raise LicenseOperationError("license service URL must use HTTPS (HTTP is allowed only on loopback)")


def _licensing() -> Any:
    """Return the closed-pack ``licensing`` module, or raise LicenseNotInstalled (core-only).

    The guarded import is the open core's one-directional dependency on ``ficelle_pro`` and
    doubles as the "is this the paid build?" gate.
    """
    try:
        from ficelle_pro import licensing
    except ModuleNotFoundError as exc:
        if exc.name not in {"ficelle_pro", "ficelle_pro.licensing"}:
            raise
        raise LicenseNotInstalled() from exc
    return licensing


def ensure_installed() -> None:
    """Raise LicenseNotInstalled when running the free core; no-op on a Pro build."""
    _licensing()


def load_or_create_machine_fingerprint() -> str:
    """Stable, opaque machine id persisted under the runtime home (no raw hardware serial).

    Stable while the runtime home persists; a home wipe yields a new id and a re-activation,
    which the service absorbs with a generous activation limit + manual reset.
    """
    machine_id_read_path = _RUNTIME_PATHS.read_path(MACHINE_ID_PATH)
    try:
        existing = machine_id_read_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    fingerprint = uuid.uuid4().hex
    MACHINE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate = MACHINE_ID_PATH.with_name(f".{MACHINE_ID_PATH.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(fingerprint)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(candidate, MACHINE_ID_PATH)
            return fingerprint
        except FileExistsError:
            winner = MACHINE_ID_PATH.read_text(encoding="utf-8").strip()
            if winner:
                return winner
            raise LicenseOperationError("machine identity could not be initialized")
    finally:
        candidate.unlink(missing_ok=True)


def _entitlement_read_path() -> Path:
    return _RUNTIME_PATHS.read_path(ENTITLEMENT_PATH)


def cached_entitlement_token() -> str | None:
    """Return the locally cached signed entitlement for an update service handoff.

    The token is never included in status payloads or logs. It is distinct from the user's
    license key and is only consumed by the update client when a release manifest explicitly
    requests ``authorization: entitlement`` over the configured HTTPS license service origin.
    """
    try:
        token = _entitlement_read_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def cached_entitlement() -> Any:
    """The verified, cached Entitlement, or None when unlicensed. Raises LicenseNotInstalled."""
    licensing = _licensing()
    return licensing.load_cached_entitlement(_entitlement_read_path())


def is_entitled(now: float | None = None) -> bool:
    """Whether the installed Pro pack may serve paid features at ``now``.

    A development pack without an embedded public key keeps the existing pack-presence behavior.
    Production builds embed the key and therefore require a valid cached entitlement.
    """
    try:
        licensing = _licensing()
    except (LicenseNotInstalled, ImportError):
        # A partially installed or broken optional pack must disable paid routing, not
        # take down the open-core router that remains usable without it.
        return False
    if not licensing.LICENSE_PUBLIC_KEY_B64:
        return True
    entitlement = licensing.load_cached_entitlement(_entitlement_read_path())
    return entitlement is not None and entitlement.is_entitled(now)


def status_dict(entitlement: Any) -> dict[str, Any]:
    """Redacted, display-safe entitlement snapshot for `license status --json` and /admin/license.

    Never includes the license key — the cache holds only the signed entitlement, not the key —
    so this is safe to log and to render in the admin UI. ``issued_at`` is the service's
    last-signed time (the page shows it as "last refresh").
    """
    if entitlement is None:
        return {"licensed": False}
    return {
        "licensed": entitlement.is_entitled(),
        "product": entitlement.product,
        "plan": entitlement.plan,
        "status": entitlement.status,
        "features": list(entitlement.features),
        "trial_end": entitlement.trial_end,
        "period_end": entitlement.period_end,
        "grace_deadline": entitlement.grace_deadline,
        "issued_at": entitlement.issued_at,
        "expires_at": entitlement.expires_at,
        "offline_grace": entitlement.in_offline_grace(),
    }


def current_status() -> dict[str, Any]:
    """status_dict for the currently cached entitlement. Raises LicenseNotInstalled (core-only)."""
    return status_dict(cached_entitlement())


def _redact_key(message: str, license_key: str) -> str:
    """Scrub the submitted license key out of a service error message.

    The message surfaces in a 400 admin response and on CLI stdout; a service that echoes the
    submitted key in its rejection ("unknown license_key SK-…") would otherwise leak it. The
    generic error redactor does not match license-key shapes, so scrub the exact key here — the
    one place that has it — before the message escapes the facade.
    """
    return message.replace(license_key, "<redacted>") if license_key else message


def activate(license_key: str) -> Any:
    """Activate against the service, verify the returned entitlement, and cache it.

    Returns the verified Entitlement. Raises LicenseNotInstalled (core-only) or
    LicenseOperationError (service rejected, unreachable, or the token failed verification).
    """
    licensing = _licensing()
    try:
        token, entitlement = licensing.activate_with_service(
            validated_service_url(),
            license_key,
            load_or_create_machine_fingerprint(),
            platform=platform.platform(),
        )
    except licensing.LicenseError as exc:
        raise LicenseOperationError(_redact_key(str(exc), license_key)) from exc
    licensing.store_entitlement_token(ENTITLEMENT_PATH, token)
    return entitlement


def refresh() -> Any:
    """Re-fetch the entitlement for this activated machine, verify, and cache it.

    Returns the verified Entitlement. Raises LicenseNotInstalled, or LicenseOperationError
    (nothing cached to refresh, service rejected/unreachable, or verification failed).
    """
    licensing = _licensing()
    cached = licensing.load_cached_entitlement(_entitlement_read_path())
    if cached is None:
        raise LicenseOperationError(
            "no active license to refresh; run `ficelle license activate` and use the hidden prompt"
        )
    try:
        token, entitlement = licensing.refresh_with_service(
            validated_service_url(), load_or_create_machine_fingerprint(), cached.machine_activation_id
        )
    except licensing.LicenseError as exc:
        raise LicenseOperationError(str(exc)) from exc
    licensing.store_entitlement_token(ENTITLEMENT_PATH, token)
    return entitlement


def deactivate() -> bool:
    """Release the activation slot on the service (best-effort), then always clear the local cache.

    Returns True when the service call succeeded (or nothing was cached), False when it failed
    but the local token was cleared anyway — deactivation must never leave a machine stuck with
    a license it cannot drop. Raises LicenseNotInstalled (core-only).
    """
    licensing = _licensing()
    cached = licensing.load_cached_entitlement(_entitlement_read_path())
    service_ok = True
    if cached is not None:
        try:
            licensing.deactivate_with_service(
                validated_service_url(), load_or_create_machine_fingerprint(), cached.machine_activation_id
            )
        except (licensing.LicenseError, LicenseOperationError):
            service_ok = False
    # An empty canonical cache is an explicit deactivation tombstone: read_path()
    # must not resurrect an untouched legacy token after the local cache is cleared.
    licensing.store_entitlement_token(ENTITLEMENT_PATH, "")
    return service_ok
