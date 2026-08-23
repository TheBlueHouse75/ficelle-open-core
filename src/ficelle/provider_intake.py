"""Machine-readable provider intake records and automation readiness."""
from __future__ import annotations

from datetime import date
from dataclasses import asdict, dataclass, fields
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
from urllib.parse import parse_qsl, urlparse


class IntakeValidationError(ValueError):
    """Raised when an intake record cannot be trusted for automation."""


PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
AUTH_ENV_PATTERN = re.compile(r"^(?:none|[A-Z][A-Z0-9_]*)$")
MODEL_ID_PATTERN = re.compile(r"^[^\s\x00-\x1f\x7f]+$")
# `date.fromisoformat` also accepts basic (``20260821``) and week (``2026-W34-5``) forms,
# which do not sort as calendar dates: mixing them silently picks the wrong `last_reviewed`.
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# A name that is not a plain YAML scalar is emitted double-quoted instead; see _yaml_scalar.
YAML_PLAIN_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/-]*$")
SOURCE_CLASSES = {"x_claim", "official_docs", "directory_claim", "runtime_probe", "manual_note"}
ENUM_FIELDS = {
    "endpoint_shape": {
        "openai_v1",
        "openai_compatible_nonstandard_path",
        "anthropic_messages",
        "new_api_panel",
        "non_openai",
        "unknown",
    },
    "served_model_identity_status": {"exact", "documented_alias", "drift", "unknown"},
    "free_mechanism": {
        "strict_zero_catalog",
        "recurring_free_quota",
        "free_model",
        "signup_credit",
        "trial_credit",
        "paid_topup_unlock",
        "no_published_cap",
        "public_shared_key",
        "grey_market_relay",
        "unknown",
    },
    "billing_cap_behavior": {"fails_closed", "auto_bills", "credit_balance", "unknown"},
    "account_key_posture": {
        "not_required",
        "no_card",
        "card_required",
        "deposit_required",
        "real_name_required",
        "discord_checkin",
        "unknown",
    },
    "commercial_use_posture": {
        "allowed",
        "non_commercial",
        "training_or_retention_caveat",
        "unknown",
    },
    "tool_call_status": {"verified", "claimed", "unsupported", "unknown"},
    "context_status": {"verified_128k_plus", "claimed_128k_plus", "below_floor", "unknown"},
    "status": {
        "new_signal",
        "official_docs_found",
        "needs_account_check",
        "sheet_ready_candidate",
        "do_not_use",
        "closed",
    },
    "adapter_fit": {
        "generic_config_only",
        "generic_config_with_allowlist",
        "generic_config_with_catalog_pricing",
        "generic_plus_normalizer",
        "provider_specific_runtime",
        "not_integrable",
        "unknown",
    },
}


# Provider-sheet vocabulary (docs/providers/_TEMPLATE.md): strict_zero_catalog, free_quota,
# free_model, trial_credit, paid, relay. The intake PRD (R3) forbids collapsing signup/trial
# credits and paid top-up unlocks into one label, so each keeps its own sheet class. `relay`
# stays the fallback for the relay-shaped and still-unresolved mechanisms because it is the
# least free-promoting value the sheet vocabulary offers.
PROVIDER_CLASS_BY_FREE_MECHANISM = {
    "strict_zero_catalog": "strict_zero_catalog",
    "recurring_free_quota": "free_quota",
    "free_model": "free_model",
}
# Evidence classes that satisfy PRD R4; `x_claim`, `directory_claim` and `manual_note` are
# discovery only and must not carry a record into a generated integration scaffold.
OFFICIAL_EVIDENCE_CLASSES = {"official_docs", "runtime_probe"}
APPEND_ONLY_LIST_FIELDS = ("aliases", "source_urls", "official_urls", "model_ids")
CANONICAL_LIST_FIELDS = (*APPEND_ONLY_LIST_FIELDS, "blockers")
DISCOVERY_STATUS_RANK = {
    "new_signal": 0,
    "official_docs_found": 1,
    "needs_account_check": 2,
    "sheet_ready_candidate": 3,
}
TERMINAL_STATUSES = {"do_not_use", "closed"}

# NEL, LS and PS are line breaks for YAML 1.1 readers of the generated front matter.
LINE_BREAK_CHARACTERS = "\r\n\x85\u2028\u2029"


def _has_unsafe_characters(value: str) -> bool:
    """Whitespace or control characters that must never reach a URL, path, or front-matter value."""
    return any(character.isspace() or ord(character) < 32 or character == "\x7f" for character in value)


def _is_secret_like_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(key).lower())
    return normalized in {
        "apikey",
        "token",
        "secret",
        "password",
        "credential",
        "key",
        "auth",
        "authorization",
    } or normalized.endswith(
        ("apikey", "token", "secret", "password", "credential")
    )


def _is_http_url(value: Any) -> bool:
    # urlparse strips ASCII newlines and tabs before parsing, so a URL carrying them would
    # validate here and still inject extra lines into the generated sheet.
    if not isinstance(value, str) or _has_unsafe_characters(value):
        return False
    try:
        parsed = urlparse(value)
        # `https://user:sk-live-...@host/v1` is a credential, and the record, the generated
        # sheet and the scaffold all copy this string verbatim into repository files.
        credentialed = bool(parsed.username or parsed.password)
        parameter_keys = [key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)]
        parameter_keys.extend(key for key, _value in parse_qsl(parsed.fragment, keep_blank_values=True))
        credentialed = credentialed or any(_is_secret_like_key(key) for key in parameter_keys)
    except ValueError:
        return False
    return not credentialed and parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_safe_endpoint_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and ".." not in value.split("/")
        and not _has_unsafe_characters(value)
    )


def _is_safe_single_line_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not any(
            character in LINE_BREAK_CHARACTERS or ord(character) < 32 or character == "\x7f"
            for character in value
        )
    )


def _yaml_scalar(value: str) -> str:
    """Render a front-matter value YAML reads back as this exact string.

    Provider names come from third-party claims, so `TokenRouter: the free one`, `[a, b]` or
    `*anchor` would otherwise break or retype the whole front-matter block. Anything outside a
    conservative plain-scalar shape is emitted as a double-quoted scalar, which JSON encoding
    produces exactly (control characters are already rejected upstream).
    """
    if YAML_PLAIN_SAFE_PATTERN.fullmatch(value) and not value.endswith(" "):
        return value
    return json.dumps(value, ensure_ascii=False)


def _find_secret_like_field(value: Any, path: str = "record") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            # Separators are dropped rather than normalized so `apiKey`, `x-api-key` and
            # `api_key` collapse onto the same token.
            if _is_secret_like_key(key):
                return f"{path}.{key}"
            found = _find_secret_like_field(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_secret_like_field(item, f"{path}[{index}]")
            if found:
                return found
    return None


@dataclass(frozen=True)
class IntakeEvidence:
    source_class: str
    url: str
    observed_at: str
    claim: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IntakeEvidence":
        if not isinstance(raw, dict):
            raise IntakeValidationError("evidence entry must be a JSON object")
        known_fields = {field.name for field in fields(cls)}
        missing_fields = sorted(known_fields - set(raw))
        if missing_fields:
            raise IntakeValidationError(f"evidence missing required field: {missing_fields[0]}")
        unknown_fields = sorted(set(raw) - known_fields)
        if unknown_fields:
            raise IntakeValidationError(f"evidence unknown field: {unknown_fields[0]}")
        if raw.get("source_class") not in SOURCE_CLASSES:
            raise IntakeValidationError("evidence source_class has an unsupported value")
        if not _is_http_url(raw.get("url")):
            raise IntakeValidationError("evidence url must be an HTTP(S) URL")
        if not _is_safe_single_line_text(raw.get("claim")):
            raise IntakeValidationError("evidence claim must be non-empty single-line text")
        observed_at = raw.get("observed_at")
        if not isinstance(observed_at, str) or not ISO_DATE_PATTERN.fullmatch(observed_at):
            raise IntakeValidationError("evidence observed_at must use YYYY-MM-DD")
        try:
            date.fromisoformat(observed_at)
        except ValueError:
            raise IntakeValidationError("evidence observed_at must use YYYY-MM-DD") from None
        return cls(**raw)


@dataclass(frozen=True)
class ProviderIntakeRecord:
    schema_version: int
    provider_id: str
    name: str
    aliases: list[str]
    source_urls: list[str]
    official_urls: list[str]
    claimed_base_url: str
    endpoint_shape: str
    canonical_chat_path: str
    served_model_identity_status: str
    free_mechanism: str
    billing_cap_behavior: str
    account_key_posture: str
    commercial_use_posture: str
    tool_call_status: str
    context_status: str
    status: str
    adapter_fit: str
    auth_env: str
    catalog_path: str
    model_ids: list[str]
    evidence: list[IntakeEvidence]
    blockers: list[str]
    next_action: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProviderIntakeRecord":
        secret_field = _find_secret_like_field(raw)
        if secret_field:
            raise IntakeValidationError(f"secret-like field is forbidden: {secret_field}")
        known_fields = {field.name for field in fields(cls)}
        missing_fields = sorted(known_fields - set(raw))
        if missing_fields:
            raise IntakeValidationError(f"missing required field: {missing_fields[0]}")
        unknown_fields = sorted(set(raw) - known_fields)
        if unknown_fields:
            raise IntakeValidationError(f"unknown field: {unknown_fields[0]}")
        provider_id = raw.get("provider_id")
        schema_version = raw.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise IntakeValidationError("schema_version must be integer 1")
        if not isinstance(provider_id, str) or not PROVIDER_ID_PATTERN.fullmatch(provider_id):
            raise IntakeValidationError("provider_id must be a lowercase slug")
        for field_name in ("name", "next_action"):
            if not _is_safe_single_line_text(raw.get(field_name)):
                raise IntakeValidationError(f"{field_name} must be non-empty single-line text")
        for field_name in ("aliases", "blockers"):
            values = raw.get(field_name)
            if (
                not isinstance(values, list)
                or any(not _is_safe_single_line_text(value) for value in values)
                or len(values) != len(set(values))
            ):
                raise IntakeValidationError(f"{field_name} must contain unique single-line strings")
        auth_env = raw.get("auth_env")
        if not isinstance(auth_env, str) or not AUTH_ENV_PATTERN.fullmatch(auth_env):
            raise IntakeValidationError("auth_env must be an uppercase environment variable name or none")
        for field_name, allowed in ENUM_FIELDS.items():
            if raw.get(field_name) not in allowed:
                raise IntakeValidationError(f"{field_name} has an unsupported value")
        if raw.get("free_mechanism") == "public_shared_key" and raw.get("status") != "do_not_use":
            raise IntakeValidationError("public_shared_key requires status do_not_use")
        for field_name in ("source_urls", "official_urls"):
            urls = raw.get(field_name)
            if (
                not isinstance(urls, list)
                or any(not _is_http_url(url) for url in urls)
                or len(urls) != len(set(urls))
            ):
                raise IntakeValidationError(f"{field_name} must contain unique HTTP(S) URLs")
        claimed_base_url = raw.get("claimed_base_url")
        if not isinstance(claimed_base_url, str) or (claimed_base_url and not _is_http_url(claimed_base_url)):
            raise IntakeValidationError("claimed_base_url must be an empty string or an HTTP(S) URL")
        for field_name in ("catalog_path", "canonical_chat_path"):
            if not _is_safe_endpoint_path(raw.get(field_name)):
                raise IntakeValidationError(f"{field_name} must be an absolute safe endpoint path")
        model_ids = raw.get("model_ids")
        if (
            not isinstance(model_ids, list)
            or any(not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id) for model_id in model_ids)
            or len(model_ids) != len(set(model_ids))
        ):
            raise IntakeValidationError("model_ids must contain unique non-empty model identifiers")
        evidence = raw.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise IntakeValidationError("evidence must contain at least one entry")
        values = dict(raw)
        values["evidence"] = [IntakeEvidence.from_dict(item) for item in evidence]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutomationDecision:
    action: str
    sheet_ready: bool
    scaffold_ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntakeChange:
    action: str
    changed_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrySyncResult:
    action: str
    changed_fields: tuple[str, ...]
    path: Path
    written: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "changed_fields": list(self.changed_fields),
            "path": str(self.path),
            "written": self.written,
        }


def canonical_record_dict(record: ProviderIntakeRecord) -> dict[str, Any]:
    payload = record.to_dict()
    for field_name in CANONICAL_LIST_FIELDS:
        payload[field_name] = sorted(payload[field_name])
    payload["evidence"] = sorted(
        payload["evidence"],
        key=_evidence_key,
    )
    return payload


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        item["observed_at"],
        item["source_class"],
        item["url"],
        item["claim"],
    )


def compare_intake_records(
    existing: ProviderIntakeRecord,
    proposed: ProviderIntakeRecord,
) -> IntakeChange:
    if existing.provider_id != proposed.provider_id:
        raise IntakeValidationError("provider_id mismatch")
    terminal_reopening = (
        existing.status in TERMINAL_STATUSES and proposed.status != existing.status
    )
    discovery_downgrade = (
        existing.status in DISCOVERY_STATUS_RANK
        and proposed.status in DISCOVERY_STATUS_RANK
        and DISCOVERY_STATUS_RANK[proposed.status]
        < DISCOVERY_STATUS_RANK[existing.status]
    )
    if terminal_reopening or discovery_downgrade:
        raise IntakeValidationError(
            f"status transition is forbidden: {existing.status} -> {proposed.status}"
        )
    if proposed.status == "sheet_ready_candidate" and not evaluate_automation_readiness(
        proposed
    ).sheet_ready:
        raise IntakeValidationError("sheet_ready_candidate is not qualified")

    existing_payload = canonical_record_dict(existing)
    proposed_payload = canonical_record_dict(proposed)
    for field_name in APPEND_ONLY_LIST_FIELDS:
        if not set(existing_payload[field_name]).issubset(proposed_payload[field_name]):
            raise IntakeValidationError(f"cannot erase prior {field_name}")
    if not {_evidence_key(item) for item in existing_payload["evidence"]}.issubset(
        {_evidence_key(item) for item in proposed_payload["evidence"]}
    ):
        raise IntakeValidationError("cannot erase prior evidence")
    changed_fields = tuple(
        field_name
        for field_name in existing_payload
        if existing_payload[field_name] != proposed_payload[field_name]
    )
    return IntakeChange(
        action="changed" if changed_fields else "unchanged",
        changed_fields=changed_fields,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif stat.S_ISLNK(os.lstat(path).st_mode):
        raise IntakeValidationError(f"registry record is a symlink: {path}")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, getattr(errno, "EMULTIHOP", errno.ELOOP)}:
            raise IntakeValidationError(f"registry record is a symlink: {path}") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise IntakeValidationError(f"registry record is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _record_json(record: ProviderIntakeRecord) -> str:
    return json.dumps(
        canonical_record_dict(record),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _atomic_replace_text(path: Path, content: str, expected_sha256: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise IntakeValidationError(f"registry record is a symlink: {path}")
        if _sha256(_read_regular_file_nofollow(path)) != expected_sha256:
            raise IntakeValidationError("concurrent registry change detected")
        os.replace(temp_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temp_path.unlink(missing_ok=True)


def sync_intake_record(
    proposed: ProviderIntakeRecord,
    registry_dir: Path,
    *,
    dry_run: bool = False,
    expected_sha256: str | None = None,
) -> RegistrySyncResult:
    registry_dir = Path(registry_dir)
    if proposed.status == "sheet_ready_candidate" and not evaluate_automation_readiness(
        proposed
    ).sheet_ready:
        raise IntakeValidationError("sheet_ready_candidate is not qualified")
    if registry_dir.is_symlink():
        raise IntakeValidationError(f"registry directory is a symlink: {registry_dir}")
    target = registry_dir / f"{proposed.provider_id}.json"
    if target.is_symlink():
        raise IntakeValidationError(f"registry record is a symlink: {target}")
    if not target.exists():
        if expected_sha256 not in {None, "missing"}:
            raise IntakeValidationError("concurrent registry change detected")
        result = RegistrySyncResult(
            action="new",
            changed_fields=tuple(canonical_record_dict(proposed)),
            path=target,
            written=not dry_run,
        )
        if dry_run:
            return result
        registry_dir.mkdir(parents=True, exist_ok=True)
        try:
            _exclusive_write_text(target, _record_json(proposed))
        except FileExistsError:
            raise IntakeValidationError("concurrent registry change detected") from None
        return result
    if not target.is_file():
        raise IntakeValidationError(f"registry record is not a regular file: {target}")

    current_bytes = _read_regular_file_nofollow(target)
    current_sha256 = _sha256(current_bytes)
    if expected_sha256 is not None and expected_sha256 != current_sha256:
        raise IntakeValidationError("concurrent registry change detected")
    try:
        current_payload = json.loads(current_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeValidationError(f"existing registry record is invalid: {exc}") from None
    if not isinstance(current_payload, dict):
        raise IntakeValidationError("existing registry record must be a JSON object")
    existing = ProviderIntakeRecord.from_dict(current_payload)
    change = compare_intake_records(existing, proposed)
    result = RegistrySyncResult(
        action=change.action,
        changed_fields=change.changed_fields,
        path=target,
        written=change.action == "changed" and not dry_run,
    )
    if result.written:
        _atomic_replace_text(target, _record_json(proposed), current_sha256)
    return result


def build_registry_context(registry_dir: Path, *, limit: int = 200) -> dict[str, Any]:
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise IntakeValidationError("context limit must be between 1 and 1000")
    registry_dir = Path(registry_dir)
    if registry_dir.is_symlink():
        raise IntakeValidationError(f"registry directory is a symlink: {registry_dir}")
    if not registry_dir.exists():
        paths: list[Path] = []
    elif not registry_dir.is_dir():
        raise IntakeValidationError(f"registry path is not a directory: {registry_dir}")
    else:
        paths = sorted(registry_dir.glob("*.json"), key=lambda path: path.name)

    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise IntakeValidationError(f"registry record is not a regular file: {path}")
        payload_bytes = _read_regular_file_nofollow(path)
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntakeValidationError(f"registry record is invalid: {path}: {exc}") from None
        if not isinstance(payload, dict):
            raise IntakeValidationError(f"registry record must be a JSON object: {path}")
        record = ProviderIntakeRecord.from_dict(payload)
        if path.stem != record.provider_id:
            raise IntakeValidationError(
                f"registry filename/provider_id mismatch: {path.name} != {record.provider_id}"
            )
        canonical = canonical_record_dict(record)
        rows.append(
            {
                "provider_id": record.provider_id,
                "name": record.name,
                "status": record.status,
                "blockers": canonical["blockers"],
                "newest_evidence_at": max(item.observed_at for item in record.evidence),
                "next_action": record.next_action,
                "sha256": _sha256(payload_bytes),
            }
        )
    selected = rows[:limit]
    return {
        "schema_version": 1,
        "missing_sha256": "missing",
        "total": len(rows),
        "returned": len(selected),
        "truncated": len(selected) < len(rows),
        "providers": selected,
    }


def evaluate_automation_readiness(record: ProviderIntakeRecord) -> AutomationDecision:
    terminal_action = {"do_not_use": "rejected", "closed": "closed"}.get(record.status)
    if terminal_action:
        return AutomationDecision(
            action=terminal_action,
            sheet_ready=False,
            scaffold_ready=False,
            blockers=(),
        )
    if record.status != "sheet_ready_candidate":
        return AutomationDecision(
            action="blocked",
            sheet_ready=False,
            scaffold_ready=False,
            blockers=(f"status:{record.status}",),
        )

    qualification_blockers = list(record.blockers)
    if not record.official_urls:
        qualification_blockers.append("official_urls")
    if not any(item.source_class in OFFICIAL_EVIDENCE_CLASSES for item in record.evidence):
        qualification_blockers.append("evidence")
    if not record.claimed_base_url:
        qualification_blockers.append("claimed_base_url")
    if record.endpoint_shape not in {"openai_v1", "openai_compatible_nonstandard_path"}:
        qualification_blockers.append("endpoint_shape")
    if record.free_mechanism not in {"strict_zero_catalog", "recurring_free_quota", "free_model"}:
        qualification_blockers.append("free_mechanism")
    if record.account_key_posture not in {"not_required", "no_card"}:
        qualification_blockers.append("account_key_posture")
    if (
        (record.account_key_posture == "not_required" and record.auth_env != "none")
        or (record.account_key_posture == "no_card" and record.auth_env == "none")
    ):
        qualification_blockers.append("auth_env")
    if record.billing_cap_behavior != "fails_closed":
        qualification_blockers.append("billing_cap_behavior")
    if record.commercial_use_posture != "allowed":
        qualification_blockers.append("commercial_use_posture")
    if record.served_model_identity_status not in {"exact", "documented_alias"}:
        qualification_blockers.append("served_model_identity_status")
    if record.tool_call_status != "verified":
        qualification_blockers.append("tool_call_status")
    if record.context_status != "verified_128k_plus":
        qualification_blockers.append("context_status")
    if record.adapter_fit == "generic_config_with_allowlist" and not record.model_ids:
        qualification_blockers.append("model_ids")
    if qualification_blockers:
        return AutomationDecision(
            action="blocked",
            sheet_ready=False,
            scaffold_ready=False,
            blockers=tuple(dict.fromkeys(qualification_blockers)),
        )

    runtime_blockers: list[str] = []
    if record.account_key_posture == "not_required" and record.auth_env == "none":
        runtime_blockers.append("runtime:keyless_remote")
    if record.canonical_chat_path != "/chat/completions":
        runtime_blockers.append("runtime:nonstandard_chat_path")
    if record.catalog_path != "/models":
        runtime_blockers.append("runtime:nonstandard_catalog_path")
    if runtime_blockers:
        return AutomationDecision(
            action="manual_code_required",
            sheet_ready=True,
            scaffold_ready=False,
            blockers=tuple(runtime_blockers),
        )

    generic_adapters = {
        "generic_config_only",
        "generic_config_with_allowlist",
        "generic_config_with_catalog_pricing",
    }
    if record.adapter_fit in {"provider_specific_runtime", "generic_plus_normalizer"}:
        return AutomationDecision(
            action="manual_code_required",
            sheet_ready=True,
            scaffold_ready=False,
            blockers=(f"adapter_fit:{record.adapter_fit}",),
        )
    if record.adapter_fit == "not_integrable":
        return AutomationDecision(
            action="rejected",
            sheet_ready=False,
            scaffold_ready=False,
            blockers=("adapter_fit:not_integrable",),
        )
    if record.adapter_fit not in generic_adapters:
        return AutomationDecision(
            action="blocked",
            sheet_ready=False,
            scaffold_ready=False,
            blockers=(f"adapter_fit:{record.adapter_fit}",),
        )
    return AutomationDecision(
        action="scaffold_ready",
        sheet_ready=True,
        scaffold_ready=True,
        blockers=(),
    )


def render_provider_sheet(record: ProviderIntakeRecord) -> str:
    decision = evaluate_automation_readiness(record)
    if not decision.sheet_ready:
        raise IntakeValidationError(f"{record.provider_id} is not ready for a provider sheet")
    provider_class = PROVIDER_CLASS_BY_FREE_MECHANISM[record.free_mechanism]
    openai_compat = {
        "openai_v1": "full",
        "openai_compatible_nonstandard_path": "partial",
    }.get(record.endpoint_shape, "none")
    reviewed = max(evidence.observed_at for evidence in record.evidence)
    reviewed_display = date.fromisoformat(reviewed).strftime("%d/%m/%Y")
    sources = sorted({*record.source_urls, *record.official_urls, *(item.url for item in record.evidence)})
    blocker_text = ", ".join(decision.blockers) if decision.blockers else "none"
    source_lines = "\n".join(f"- {url}" for url in sources)
    evidence_lines = "\n".join(
        f"- {item.observed_at} · `{item.source_class}` · {item.claim} ({item.url})"
        for item in sorted(record.evidence, key=lambda item: (item.observed_at, item.url, item.claim))
    )
    return (
        "---\n"
        f"id: {record.provider_id}\n"
        f"name: {_yaml_scalar(record.name)}\n"
        "status: candidate\n"
        f"provider_class: {provider_class}\n"
        f"strict_zero: {'true' if record.free_mechanism == 'strict_zero_catalog' else 'false'}\n"
        "integration: none\n"
        f"base_url: {record.claimed_base_url}\n"
        f"auth_env: {record.auth_env}\n"
        f"openai_compat: {openai_compat}\n"
        f"last_reviewed: {reviewed_display}\n"
        "---\n\n"
        f"# {record.name} — provider sheet\n\n"
        f"Last reviewed: {reviewed_display} · Status: **candidate**\n\n"
        "## Verdict\n\n"
        f"Machine intake status: `{record.status}`. Next action: {record.next_action}\n\n"
        "## Identity and free access\n\n"
        f"- **Endpoint shape:** `{record.endpoint_shape}`\n"
        f"- **Canonical chat path:** `{record.canonical_chat_path}`\n"
        f"- **Free mechanism:** `{record.free_mechanism}`\n"
        f"- **Billing cap:** `{record.billing_cap_behavior}`\n"
        f"- **Account/key posture:** `{record.account_key_posture}`\n"
        f"- **Commercial use:** `{record.commercial_use_posture}`\n"
        f"- **Model identity:** `{record.served_model_identity_status}`\n"
        f"- **Tools:** `{record.tool_call_status}`\n"
        f"- **Context:** `{record.context_status}`\n\n"
        "## Automation readiness\n\n"
        f"- **Action:** `{decision.action}`\n"
        f"- **Adapter fit:** `{record.adapter_fit}`\n"
        f"- **Blockers:** {blocker_text}\n"
        "- **Generated state:** documentation/scaffold only; not registered, invokable, activated, or live-smoked.\n\n"
        "## Evidence\n\n"
        f"{evidence_lines}\n\n"
        "## Sources\n\n"
        f"{source_lines}\n\n"
        "## See also\n\n"
        "- [`README.md`](README.md) — provider sheet index.\n"
        "- [`../prds/provider-intake-verification-prd.md`](../prds/provider-intake-verification-prd.md) — intake gates.\n"
    )


def build_integration_scaffold(record: ProviderIntakeRecord) -> dict[str, Any]:
    decision = evaluate_automation_readiness(record)
    if not decision.scaffold_ready:
        raise IntakeValidationError(f"{record.provider_id} is not ready for an integration scaffold")
    provider_class = PROVIDER_CLASS_BY_FREE_MECHANISM[record.free_mechanism]
    free_access_proof = {
        "generic_config_with_catalog_pricing": "provider_free_catalog_pricing",
        "generic_config_with_allowlist": "provider_free_model_allowlist",
        "generic_config_only": "provider_free_endpoint",
    }[record.adapter_fit]
    return {
        "schema_version": 1,
        "provider_id": record.provider_id,
        "state": "generated_disabled",
        "apply_automatically": False,
        "requires_live_smoke": True,
        "provider_config": {
            "display_name": record.name,
            "enabled": False,
            "base_url": record.claimed_base_url,
            "catalog_path": record.catalog_path,
            "chat_path": record.canonical_chat_path,
            "activation_policy": "configured_credentials" if record.auth_env != "none" else "always",
            "provider_class": provider_class,
            "free_scope": "model" if record.free_mechanism in {"strict_zero_catalog", "free_model"} else "provider",
            "free_access_proof": free_access_proof,
            "auth_env": record.auth_env,
        },
        "candidate_model_ids": sorted(record.model_ids),
        "required_gates": [
            "targeted_tests",
            "simplify",
            "review_code",
            "credential_presence",
            "catalog_identity_smoke",
            "chat_smoke",
            "tool_smoke",
            "streaming_smoke",
            "context_smoke",
            "billing_cap_readback",
        ],
    }


def _exclusive_write_text(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def generate_artifacts(record: ProviderIntakeRecord, output_dir: Path) -> dict[str, Path]:
    decision = evaluate_automation_readiness(record)
    sheet = render_provider_sheet(record)
    scaffold = (
        json.dumps(build_integration_scaffold(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if decision.scaffold_ready
        else None
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "provider_sheet": output_dir / f"{record.provider_id}.md",
    }
    if scaffold is not None:
        targets["integration_scaffold"] = output_dir / f"{record.provider_id}.integration.json"
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise IntakeValidationError(f"artifact already exists: {existing[0]}")

    created: list[Path] = []
    try:
        _exclusive_write_text(targets["provider_sheet"], sheet)
        created.append(targets["provider_sheet"])
        if scaffold is not None:
            _exclusive_write_text(targets["integration_scaffold"], scaffold)
            created.append(targets["integration_scaffold"])
    except FileExistsError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise IntakeValidationError(f"artifact already exists: {exc.filename}") from None
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return targets


def generated_intake_schema() -> dict[str, Any]:
    safe_text = {
        "type": "string",
        "minLength": 1,
        "pattern": r"^(?=.*\S)[^\x00-\x1f\x7f\x85\u2028\u2029]+$",
    }
    string_list = {"type": "array", "items": safe_text, "uniqueItems": True}
    # Mirrors _is_http_url: http(s) scheme, non-empty host, no credentials in the netloc, and
    # no whitespace or control character anywhere. A schema that accepted
    # `https://user:secret@host` would green-light a credential for every external validator.
    secret_parameter_pattern = (
        r"[?&#][^=&#]*(?:"
        r"[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]|"
        r"[Kk][Ee][Yy]|"
        r"[Aa][Uu][Tt][Hh](?:[Oo][Rr][Ii][Zz][Aa][Tt][Ii][Oo][Nn])?|"
        r"[Tt][Oo][Kk][Ee][Nn]|"
        r"[Ss][Ee][Cc][Rr][Ee][Tt]|"
        r"[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|"
        r"[Cc][Rr][Ee][Dd][Ee][Nn][Tt][Ii][Aa][Ll]"
        r")="
    )
    url = {
        "type": "string",
        "pattern": "^https?://[^\\s/?#@\\x00-\\x1f\\x7f]+(?:[/?#][^\\s\\x00-\\x1f\\x7f]*)?$",
        "not": {"pattern": secret_parameter_pattern},
    }
    url_list = {"type": "array", "items": url, "uniqueItems": True}
    # Mirrors _is_safe_endpoint_path: absolute, no whitespace, no traversal segment.
    endpoint_path = {
        "type": "string",
        "pattern": "^/\\S*$",
        "not": {"pattern": "(?:^|/)\\.\\.(?:/|$)"},
    }
    properties: dict[str, Any] = {
        "schema_version": {"const": 1},
        "provider_id": {"type": "string", "pattern": PROVIDER_ID_PATTERN.pattern},
        "name": safe_text,
        "aliases": string_list,
        "source_urls": url_list,
        "official_urls": url_list,
        "claimed_base_url": {"type": "string", "anyOf": [{"const": ""}, url]},
        "endpoint_shape": {"enum": sorted(ENUM_FIELDS["endpoint_shape"])},
        "canonical_chat_path": endpoint_path,
        "served_model_identity_status": {"enum": sorted(ENUM_FIELDS["served_model_identity_status"])},
        "free_mechanism": {"enum": sorted(ENUM_FIELDS["free_mechanism"])},
        "billing_cap_behavior": {"enum": sorted(ENUM_FIELDS["billing_cap_behavior"])},
        "account_key_posture": {"enum": sorted(ENUM_FIELDS["account_key_posture"])},
        "commercial_use_posture": {"enum": sorted(ENUM_FIELDS["commercial_use_posture"])},
        "tool_call_status": {"enum": sorted(ENUM_FIELDS["tool_call_status"])},
        "context_status": {"enum": sorted(ENUM_FIELDS["context_status"])},
        "status": {"enum": sorted(ENUM_FIELDS["status"])},
        "adapter_fit": {"enum": sorted(ENUM_FIELDS["adapter_fit"])},
        "auth_env": {"type": "string", "pattern": AUTH_ENV_PATTERN.pattern},
        "catalog_path": endpoint_path,
        "model_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": MODEL_ID_PATTERN.pattern},
            "uniqueItems": True,
        },
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_class", "url", "observed_at", "claim"],
                "properties": {
                    "source_class": {"enum": sorted(SOURCE_CLASSES)},
                    "url": url,
                    # `format` is annotation-only for most validators, so the calendar shape
                    # ISO_DATE_PATTERN enforces at runtime has to be asserted as a pattern.
                    "observed_at": {"type": "string", "format": "date", "pattern": ISO_DATE_PATTERN.pattern},
                    "claim": safe_text,
                },
            },
        },
        "blockers": string_list,
        "next_action": safe_text,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ficelle.ai/schemas/provider-intake-v1.json",
        "title": "Ficelle provider intake record",
        "type": "object",
        "additionalProperties": False,
        "required": [field.name for field in fields(ProviderIntakeRecord)],
        "properties": properties,
    }
