# List available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest

# Run fast tests
fast-test:
    uv run pytest

# Test marimo notebook execution
test-marimo:
    uv run python weight_analysis_marimo.py

# Build WASM export
build:
    uv run marimo export html-wasm weight_analysis_marimo.py -o public --mode edit --show-code --force

# Deploy to Surge
deploy: build
    npm run deploy
