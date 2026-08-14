from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ficelle.providers.base import ProviderAccess


@dataclass(frozen=True)
class ProviderAuthPorts:
    provider_access: Callable[[str, dict[str, Any], bool], ProviderAccess]
    credential_source_label: Callable[[str | None], str | None]


def provider_auth_row(source: str, config: dict[str, Any], *, ports: ProviderAuthPorts) -> dict[str, Any]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    provider_cfg = providers.get(source) if isinstance(providers.get(source), dict) else {}
    access = ports.provider_access(source, provider_cfg, False)
    # The record's own verdict, not a second copy of it: this row and `invoke_model` must
    # answer the same question, and every time each composed its own predicate they drifted.
    invokable = access.can_invoke
    base_url_for_row = access.base_url or provider_cfg.get("base_url")
    return auth_row(
        invokable,
        access.key,
        access.reason,
        base_url_for_row,
        credential_source_label=ports.credential_source_label,
        key_reason=access.key_reason,
    )


def auth_status(config: dict[str, Any], *, ports: ProviderAuthPorts) -> dict[str, dict[str, Any]]:
    providers = config.get("providers") if isinstance(config.get("providers"), dict) else {}
    return {str(source): provider_auth_row(str(source), config, ports=ports) for source in providers}


def auth_row(
    invokable: bool,
    key: Any,
    reason: str,
    base_url: Any,
    *,
    credential_source_label: Callable[[str | None], str | None],
    key_reason: str | None = None,
) -> dict[str, Any]:
    """The public shape of one provider's credential state. Never carries key material.

    ``key_source`` answers "does a key resolve, and from where" and ``invokable`` answers
    "would a request be sent" — two questions, because they have two different fixes. A key
    that resolves and still cannot be used keeps ``reason`` (``missing base_url for provider
    <source>``) rather than reading ``configured``, so no caller can mistake it for one that
    works; and it keeps its ``key_source``, so no caller can mistake it for one with no key
    and answer it with `ficelle set-key`.

    ``key_reason`` is the resolution source when ``reason`` had to be spent on that
    diagnostic; both are coarse redacted strings owned by the resolver, never the value.
    """
    return {
        "invokable": invokable,
        "reason": "configured" if key and invokable else reason,
        "key_source": credential_source_label(key_reason or reason) if key else None,
        "base_url": base_url,
    }


def invokable_provider_sources(auth: Mapping[str, Any]) -> list[str]:
    """Providers this install can actually call.

    ``invokable`` is the same gate ``invoke_model`` applies before sending anything —
    a resolved key with a base URL, or the keyless-local exemption — so an empty list
    means every completion will fail with ``credentials unavailable``, whatever the
    catalog says. The reference providers' catalogs are readable without a key, so
    ``ficelle models`` cannot be used to answer this question.
    """
    return [str(source) for source, row in auth.items() if isinstance(row, dict) and row.get("invokable")]


def unusable_key_provider_reasons(auth: Mapping[str, Any]) -> dict[str, str]:
    """Providers whose key resolves and still cannot be called, mapped to why, in configured order.

    These are the ones `ficelle set-key` cannot help: the key is already there and something
    else is missing — today only a `base_url`, which is the one way to hold a key and stay
    non-invokable. The value is the row's own ``reason`` verbatim, so the advice names what
    is actually missing instead of repeating the question.
    """
    return {
        str(source): str(row.get("reason") or "the stored key cannot be used")
        for source, row in auth.items()
        if isinstance(row, dict) and not row.get("invokable") and row.get("key_source")
    }


def unconfigured_provider_sources(auth: Mapping[str, Any]) -> list[str]:
    """The providers a user would have to give a key to, in configured order.

    Narrower than "not invokable", which is what it used to be: a provider already holding
    a key is in ``unusable_key_provider_reasons`` instead, because storing that same key
    again is not what fixes it.
    """
    accounted = set(invokable_provider_sources(auth)) | set(unusable_key_provider_reasons(auth))
    return [str(source) for source in auth if str(source) not in accounted]


def provider_key_setup_commands(
    sources: Sequence[str],
    key_urls: Mapping[str, str],
    *,
    command: str = "ficelle",
) -> list[str]:
    """One ready-to-paste ``ficelle set-key`` line per provider, unindented.

    The key-creation URLs come from the caller's registry (``PROVIDER_KEY_URLS``)
    rather than from anything written here, so the closed pack's providers stay
    unnamed in the open core and no second copy of a URL can drift. A provider the
    registry does not know still gets its command — the command is the actionable
    half, the link is the convenience.

    ``command`` exists for the one surface that can be read before the CLI is runnable:
    setup prints this block having just installed the console script, and a shell that
    cannot resolve ``ficelle`` yet would meet "ready-to-paste" with `command not found`.
    Every other caller is reached *through* the CLI, which is proof enough that the bare
    name works, so they keep it.
    """
    commands = [f"{command} set-key {source}" for source in sources]
    width = max((len(command) for command in commands), default=0)
    lines: list[str] = []
    for source, command in zip(sources, commands):
        url = str(key_urls.get(source) or "").strip()
        lines.append(f"{command.ljust(width)}  # create a key at {url}" if url else command)
    return lines


def unusable_key_provider_lines(reasons: Mapping[str, str]) -> list[str]:
    """One ``<provider>  # <reason>`` line per stored-but-unusable key, unindented.

    Aligned like ``provider_key_setup_commands`` because the two blocks are printed
    together: one lists what to store, the other what to fix. No command, because there
    is none — the reason is the whole actionable content.
    """
    width = max((len(source) for source in reasons), default=0)
    return [f"{source.ljust(width)}  # {reason}" for source, reason in reasons.items()]


def unusable_key_provider_block(reasons: Mapping[str, str]) -> list[str]:
    """The whole block — heading, indented lines, trailing blank — or nothing at all.

    Unlike the keyless half, whose framing is deliberately worded per surface, this
    sentence is the same everywhere it is printed, so it is owned here rather than copied:
    a reword must reach every surface at once. Callers add their own follow-up if they
    have one. Empty for an empty mapping, so no surface prints a heading over no lines.
    """
    if not reasons:
        return []
    return [
        "These providers already hold a key, and something else is missing:",
        "",
        *(f"  {line}" for line in unusable_key_provider_lines(reasons)),
        "",
    ]
