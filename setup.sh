#!/usr/bin/env bash
set -e

STAMP=".install_hooks_done"

pip install poetry
echo "📦 Installing Poetry dependencies..."
poetry install --no-root

if [ -f "$STAMP" ]; then
    echo "✅ Environment already configured — skipping install_hooks."
else
    echo "🚀 Running install_hooks for the first time..."
    poetry run python -m install_hooks
    echo "✅ install_hooks completed successfully."
    touch "$STAMP"
fi
