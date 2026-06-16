# Contributing

Thank you for your interest in contributing to `strands-aidr`!

## Development setup

```bash
git clone https://github.com/CrowdStrike/strands-aidr
cd strands-aidr
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src

# Unit tests
pytest tests/test_model.py -v

# Integration tests (requires real credentials)
CS_AIDR_TOKEN=pts_... CS_AIDR_BASE_URL=... pytest tests/test_integration.py -v
```

All checks must pass before submitting a pull request.

## Pull requests

- Keep changes focused — one logical change per PR.
- Add or update tests for any new behavior.
- Update the README if you change the public API or add configuration options.
- Use clear, descriptive commit messages.

## Reporting issues

Open an issue on [GitHub](https://github.com/CrowdStrike/strands-aidr/issues) with steps to reproduce, expected behavior, and actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
