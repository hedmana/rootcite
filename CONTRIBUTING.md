# Contributing

This is a solo project, but it is built as if it were not. The conventions below
are the point, not the overhead.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python is pinned
to 3.12 in `.python-version`.

```sh
uv sync --all-extras --dev     # create .venv and install everything
uv run pre-commit install      # lint and format on every commit
```

Day to day:

```sh
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format .           # format
uv run pre-commit run -a       # everything CI runs, locally
```

Add a runtime dependency with `uv add <pkg>`, a dev one with
`uv add --dev <pkg>`. Commit the resulting `uv.lock`: CI installs with
`--locked` and fails if the lockfile is stale.

## Workflow

Every change follows the same loop:

1. **Open an issue** describing the change and referencing the plan step it comes
   from (`Plan step: PR 7 (Phase 2)`).
2. **Branch** off `main` using `<type>/<short-slug>`, for example
   `feat/openalex-client` or `fix/crawler-backoff`.
3. **Commit atomically.** One logical change per commit, even though the PR is
   squash-merged at the end. The commit trail inside a PR is where the reasoning
   lives.
4. **Open a PR** that closes the issue (`Closes #12`) and answers three questions:
   what changed, why, and how to verify it by hand.
5. **Squash-merge** once CI is green.

Use draft PRs for anything that wants feedback before it is finished.

A release is tagged every three to four merged PRs (`v0.1`, `v0.2`, ...) so the
repository history reads as a sequence of working states rather than a flat list
of commits.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org). A single short
subject line, imperative mood, no trailing period.

```
feat: add resumable snowball crawler
fix: retry OpenAlex 429 with exponential backoff
docs: document field config schema
chore: pin python to 3.12
test: cover cited-works pagination
refactor: extract edge dedup from graph builder
```

## Code

- Names over comments. A comment explains *why*, never *what*.
- Prefer fewer lines. Prefer fewer round-trips and fewer allocations.
- Nothing field-specific outside `fields/*.yaml`. If the word "GNN" appears in
  `src/`, it is a bug: the pipeline is field-agnostic and the GNN literature is
  merely the first field it is pointed at.
- No secrets in code, output, or fixtures.
