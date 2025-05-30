# List available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest

# Run fast tests
fast-test:
    uv run pytest
