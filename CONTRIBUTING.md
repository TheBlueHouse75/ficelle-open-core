# Contributing to Ficelle

Ficelle Core is released under the Business Source License 1.1. Contributions to the
open Core are welcome; the private Ficelle Pro package is developed separately.

## Development setup

Requirements: Python 3.11+ and `uv`.

```bash
uv sync --extra dev
uv run pytest -q
```

Keep changes focused and add tests for behavior changes. Before opening a pull request:

```bash
uv run pytest -q
uv run python -m build
```

Do not commit API keys, license keys, customer data, generated environments, or local
runtime state.

By submitting a contribution, you agree that it is licensed under the repository's
current and future license terms.

For security issues, do not open a public issue; follow [`SECURITY.md`](SECURITY.md).
