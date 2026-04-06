# Contributing to sliceval

## Setup

```bash
git clone https://github.com/kartikeyamandhar/sliceval.git
cd sliceval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pip install pytest
```

## Running Tests

```bash
# Unit + integration tests
pytest tests/ -v

# Stress tests only (slower)
pytest tests/test_stress.py -v

# Single test file
pytest tests/test_evaluator.py -v
```

## Before Submitting a PR

1. All tests pass: `pytest tests/ -v`
2. No new warnings in stress tests: `pytest tests/test_stress.py -v --tb=short`
3. If you added a feature, add tests for it
4. If you changed the public API, update the README

## What We're Looking For

Check the [issues](https://github.com/kartikeyamandhar/sliceval/issues) page. Good first contributions:

- Adding new metrics
- Improving discovery algorithms
- Adding new integrations (W&B, HTML reports)
- Documentation improvements
- Bug reports with reproducible examples

## Code Style

- No linters enforced yet, but keep it consistent with existing code
- Type hints where practical
- Docstrings on all public functions
- Tests for every new feature

## Releasing

Maintainers only. Bump version in `pyproject.toml`, push, create a GitHub release. PyPI publish is automated.
