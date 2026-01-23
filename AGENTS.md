# Repository Guidelines

## Project Structure & Module Organization
- `src/tazotron/`: Python package entry point (`tazotron:main`), extend here for core logic and CLI.
- `tests/`: pytest suite stubbed via `tests/test_*.py`; add unit/integration coverage for new code.
- `docs/`: design notes (e.g., `ct_necrosis_pipeline.md`); update when workflows change.
- `scripts/`: helper scripts for ops/experiments; keep CLI-friendly and documented.
- `configs/`, `reports/`, `notebooks/`: configuration snapshots, generated artifacts, and explorations; avoid committing large/secret data.

## Build, Test, and Development Commands
- Setup (Python 3.12): `uv sync --group dev` installs runtime + dev tools into `.venv`.
- Run app: `uv run tazotron` executes the package entry point.
- Lint: `uv run ruff check .` (static analysis); format with `uv run ruff format .`.
- Type hints: `uv run ty src tests` for static typing feedback.
- Tests: `uv run pytest` (strict markers, doctests enabled).
- Optional hooks: `uv run pre-commit run --all-files` before pushing.

## Coding Style & Naming Conventions
- Python, 4-space indents, line length 120, double quotes preferred (Ruff formatter defaults).
- Follow PEP 8 naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep functions small, typed, and documented where behavior is non-obvious; prefer pure functions in `src/tazotron`.
- Run Ruff/formatter before committing to satisfy the enforced rule set (`tool.ruff` config).

## Testing Guidelines
- Pytest with markers `fast` (lightweight) and `slow` (heavy); place tests under `tests/` as `test_*.py`.
- Mirror the `src/` package layout under `tests/`, adding `__init__.py` files as needed; keep shared fixtures in `tests/conftest.py`.
- Doctests run automatically; update inline examples when APIs change.
- For new features, add unit tests plus an integration-style check if the CLI path is affected.
- Use fixtures and parametrization to keep tests fast; avoid relying on network or undisclosed data.

## Commit & Pull Request Guidelines
- Follow Conventional Commits observed in history (`feat:`, `chore:`, `docs:`, etc.); keep messages imperative and scoped.
- PRs should include: summary of changes, linked issue/task, test evidence (`uv run pytest`, `ruff check`, etc.), and notes on data or config impacts.
- Keep diffs focused; prefer smaller PRs with clear rationale. Update `docs/` when altering pipelines or expected outputs.

## Pragmatism
- Avoid over-engineering: this is an application, not a general-purpose library. Implement only the necessary scenarios.

## Approvals & Tooling
- If a command fails due to cache or permission restrictions, ask the user for approval or rerun with requested permissions.
- For uv install/upgrade/sync and for linting/formatting commands, request approval up front when needed.

## Imports
- Do not use relative imports; always use absolute import paths within the project.
