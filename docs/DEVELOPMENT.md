# Development Guide

## Setup

### 1. Install System Dependencies

Install Graphviz (required for visualization):

**macOS:**
```shell
brew install graphviz
```

**Linux (Ubuntu/Debian):**
```shell
sudo apt-get install graphviz
```

### 2. Create Python Environment

Setup Python env
```shell
python -m venv .venv
. ./.venv/bin/activate
```

### 3. Install Python Dependencies

```shell
poetry install
```

## Code Quality

### Linting and Formatting

Run ruff to check and fix code issues, and format code:

```shell
ruff check --fix scripts summary tests && ruff format scripts summary tests
```