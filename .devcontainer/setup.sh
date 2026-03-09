#!/usr/bin/env bash
# Post-create setup for the r/science causal graph dev container.
# Runs automatically after the container is created.
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install --quiet -r api/requirements.txt -r pipeline/requirements.txt

# Dev tools
pip install --quiet pytest pytest-asyncio ruff

# spaCy's CLI is broken on Python 3.14 (pydantic v1 incompatibility).
# Install the model wheel directly instead.
echo "=== Installing spaCy en_core_web_sm model ==="
pip install --quiet \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

echo "=== Installing Node.js frontend dependencies ==="
if command -v node &>/dev/null; then
    cd frontend && npm install --silent && cd ..
else
    echo "  Node.js not found — skipping frontend install."
    echo "  Install Node.js 18+ and run: cd frontend && npm install"
fi
