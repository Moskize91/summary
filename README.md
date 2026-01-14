# Summary

A summary tool.

## Quick Start

### Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python test.py
```

### Add Your Code

- Add your source code in the `summary/` directory
- Add tests in the `tests/` directory
- Update dependencies in `pyproject.toml` under `dependencies = []`

## Features

- Modern PEP 621 compliant `pyproject.toml`
- Pre-configured development tools:
  - `pytest` for testing
  - `ruff` for linting and formatting
  - `pylint` for additional linting
  - `pyright` for type checking
- GitHub Actions workflows for CI/CD
- VSCode settings for Python development
- Example test suite

## Project Structure

```
.
├── summary/                # Your source code goes here
├── tests/                  # Test files
│   └── test_example.py    # Example tests (replace with your tests)
├── .github/workflows/      # GitHub Actions CI/CD
├── .vscode/               # VSCode settings
├── pyproject.toml         # Project configuration
├── test.py               # Test runner script
└── README.md             # This file
```

## Development Tools

Run linting:
```bash
ruff check .
pylint ./summary/**/*.py ./tests/**/*.py
```

Run type checking:
```bash
pyright summary tests
```

Run formatting:
```bash
ruff format .
```

Run tests:
```bash
pytest tests/ -v
# or
python test.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.