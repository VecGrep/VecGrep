# Contributing to VecGrep

Thanks for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/VecGrep/VecGrep
cd VecGrep
uv sync --extra dev
```

### Install the git hooks

Run once after cloning to enable the pre-commit lint check:

```bash
git config core.hooksPath .githooks
```

This runs `ruff` against staged Python files before every commit and blocks the commit if any violations are found.

## Development

Run the server locally:
```bash
uv run vecgrep
```

Test against a codebase:
```bash
uv run python -c "
from vecgrep.server import _do_index, search_code
print(_do_index('/path/to/project'))
print(search_code('your query here', '/path/to/project'))
"
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Open a pull request against `main`

## Guidelines

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality where practical
- Follow existing code style (ruff for formatting/linting)
- All `unsafe` usage (if any) must have a comment explaining why

## Reporting Bugs

Open a GitHub issue with steps to reproduce and your Python/OS version.
