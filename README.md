# Python Project Template

A modern Python project template with best practices and pre-configured tooling.

## Quick Start

After cloning this template, follow these steps to customize it for your project:

### 1. Replace All Placeholders

Search and replace the following placeholders throughout the project:

| Placeholder | Replace With | Files to Update |
|------------|--------------|-----------------|
| `template-python` | Your project name (lowercase, hyphen-separated) | `pyproject.toml`, `README.md` |
| `src` | Your source code directory name (optional, can keep as `src`) | Directory name, `.github/workflows/check-pr.yml`, `README.md` |
| `Your Name` | Your actual name | `pyproject.toml`, `LICENSE` |
| `your.email@example.com` | Your email address | `pyproject.toml` |
| `your-username` | Your GitHub username | `pyproject.toml` |
| `[year]` | Current year (e.g., 2026) | `LICENSE` |
| `[fullname]` | Your full name or organization | `LICENSE` |

### 2. Key Files to Edit

#### `pyproject.toml` (Lines 2-11, 38-39)
```toml
name = "your-project-name"
description = "Your project description"
authors = [{name = "Your Name", email = "your.email@example.com"}]

[project.urls]
Homepage = "https://github.com/your-username/your-project-name"
Repository = "https://github.com/your-username/your-project-name"
```

#### `LICENSE` (Line 3)
```
Copyright (c) 2026 Your Name
```

#### `README.md`
Update the title and description to match your project.

#### Rename `src/` directory (Optional)
If you want to use your project name instead of `src/`:
1. Rename the directory: `mv src/ your-project-name/`
2. Update `.github/workflows/check-pr.yml` (lines 27, 31): replace `src` with your directory name
3. Update this README file to reflect the new directory name

### 3. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python test.py
```

### 4. Add Your Code

- Add your source code in the `src/` directory
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
├── src/                    # Your source code goes here
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
pylint ./src/**/*.py ./tests/**/*.py
```

Run type checking:
```bash
pyright src tests
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