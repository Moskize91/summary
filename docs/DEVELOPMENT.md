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