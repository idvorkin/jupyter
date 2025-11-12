# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal collection of Jupyter notebooks for data science, algorithm practice, and data analysis. The repository includes notebooks covering topics like NLP, ML, data visualization, health data analysis, and algorithm practice problems.

## Development Environment Setup

This project uses `uv` for Python package management and virtual environment handling.

### Initial Setup

```bash
# Install dependencies using uv
uv sync

# Install pre-commit hooks
pre-commit install
```

### Dependencies

- Core dependencies are defined in `pyproject.toml`
- Dev dependencies (including JupyterLab) are managed via `[tool.uv.dev-dependencies]`
- Key tools: pandas, matplotlib, scikit-learn, nltk, spacy, jupytext, altair, seaborn

## Common Development Commands

### Running Jupyter

```bash
# Start JupyterLab
jupyter lab

# With CPU limiting (if needed to prevent SSH session drops)
cpulimit -l 90 jupyter lab
```

### Testing

```bash
# Run all tests
just test

# Or directly with uv
uv run pytest
```

### Code Quality

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Format and lint (handled by pre-commit):
# - Ruff for Python (.py, .ipynb)
# - Biome for JS/TS/JSON
# - Prettier for Markdown and HTML
```

### Jupyter/Python Roundtrip with JupyText

```bash
# Sync notebook with Python file and format with black
jupytext --sync --pipe black <notebook>.ipynb
```

### Running Marimo Notebooks

```bash
# Local development (default - RECOMMENDED for local work)
marimo edit weight_analysis_marimo.py --watch

# Remote access via tunnel (use when user wants remote access or mentions "0.0.0.0")
# IMPORTANT: When using --host 0.0.0.0, check the hostname and provide clickable link
marimo edit weight_analysis_marimo.py --host 0.0.0.0 --port 5001 --watch

# Run as an app (code hidden, interactive UI only)
marimo run weight_analysis_marimo.py --watch

# Execute as a Python script
python weight_analysis_marimo.py

# Convert Jupyter notebook to marimo
marimo convert notebook.ipynb -o notebook.py

# Export marimo to other formats (HTML, IPYNB, markdown)
marimo export weight_analysis_marimo.py
```

**Important notes:**

- **ALWAYS use `--watch`** for auto-reload on file changes
- **Only use `--host 0.0.0.0`** when user explicitly needs remote/tunnel access (e.g., "run on 0.0.0.0")
- **Providing access links**: When using `--host 0.0.0.0`, check the hostname with `hostname` command and extract port/token from marimo's output:
  - If hostname matches pattern like `C-XXXX` or similar: provide link as `http://<hostname>:PORT?access_token=<TOKEN>`
  - Example: hostname `C-5001`, marimo shows port `5001` and token `abc123` → `http://c-5001:5001?access_token=abc123` (lowercase hostname)
  - Extract both the port number and access token from marimo's stdout (look for the URL line)
- Default host is `127.0.0.1` (localhost only, most secure)

### Type Checking Notebooks

```bash
# Convert notebooks to Python and run mypy
./mypy-all-notebooks.sh
```

### Notebook Diffing

```bash
# View code-only diff (ignoring metadata/output)
nbdiff --ignore-metadata --ignore-details --ignore-output <notebook1>.ipynb <notebook2>.ipynb
```

## Code Architecture

### Notebook Organization

- **Algorithm Practice**: Array.ipynb, Heap.ipynb, Recursion.ipynb, StackAndQueue.ipynb, Strings.ipynb, SubArray.ipynb, Tree Questions.ipynb, BitTwiddling.ipynb
- **Data Analysis**: Weight Analysis.ipynb (health data), SleepAnalysis.ipynb, WeekAnalysis.ipynb, ProductivityGraphs.ipynb, stock_analysis.ipynb
- **ML/Data Science**: PlayML.ipynb, PlayNLP.py (NLP with personal journals), TF-IFD.ipynb, PlayCustomerData.ipynb
- **Visualization**: PlayAnimation.ipynb, pandasPdfCdf.ipynb, sympy.ipynb
- **General Python**: ElegantPython.ipynb, Math.ipynb

### Python Modules

- `pandas_util.py`: Utility functions for pandas operations
  - Provides monkey-patched `.toPercent()` method for Series
  - `time_it()` helper for performance measurement
  - Uses arrow for time handling

- `PlayNLP.py`: JupyText-synced Python version of NLP notebook (uses `# %% [markdown]` format)

- `weight_analysis_marimo.py`: Marimo notebook for weight analysis
  - Reads from `data/HealthAutoExport-2010-03-03-2025-05-30.json`
  - Uses modern JSON data source (migrated from CSV)

### Data Files

- Health data stored in `data/HealthAutoExport-*.json` (JSON format preferred over legacy CSV)

## JupyterLab Configuration

### Vim Mode

- JupyterLab uses jupyterlab-vim extension
- Key bindings include custom escape mappings (fj, fk)
- Reference the vim configuration in README.md for keybindings

### Extensions Used

- jupyterlab-vim: Vim mode in cells
- jupyterlab-lsp: Language server support
- nbdime: Notebook diffing

## Pre-commit Hooks

The project uses pre-commit with:

1. **Ruff**: Python linting and formatting (replaces black, flake8, isort)
2. **Biome**: JS/TS/JSON linting and formatting
3. **Prettier**: Markdown and HTML formatting only
4. **Dasel**: YAML/JSON validation
5. **Local test hook**: Runs `just fast-test` on commits

## Development Workflow Notes

### Working with Notebooks

- Edit in VS Code/Cursor (now well-supported for .ipynb files)
- Use JupyText for Python/notebook roundtripping when version control is important
- Run notebooks through nbdime for meaningful diffs
- Use marimo for reactive notebook development:
  - marimo notebooks are stored as pure Python files (Git-friendly)
  - Reactive execution: cells auto-update when dependencies change
  - Can be executed as scripts or served as interactive apps
  - Example in repo: `weight_analysis_marimo.py`

### Python File Format

- Some notebooks are maintained as both .ipynb and .py files via JupyText
- The .py files use percent format (`# %%`) for cell delimitation
- JupyText metadata is embedded in the Python file header

### Data Analysis Pattern

- Health data analysis migrated from CSV to JSON format (see weight_analysis_marimo.py)
- Uses HealthAutoExport app for iOS health data extraction
- Typical stack: pandas + matplotlib/altair + seaborn for visualization

## Package Management

- Modern workflow uses `uv` (fast Rust-based package manager)
- Legacy files exist (Pipfile, requirements.txt) but pyproject.toml is authoritative
- Virtual environment managed by uv in `.venv/`
