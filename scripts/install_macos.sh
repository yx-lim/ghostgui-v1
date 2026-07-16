#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found."
    if command -v brew >/dev/null 2>&1; then
        echo "Homebrew is available. Install Python with:"
        echo "  brew install python"
    else
        echo "Install Python 3.10 or newer from https://www.python.org/ or Homebrew."
    fi
    exit 1
fi

if command -v brew >/dev/null 2>&1; then
    echo "Homebrew found. Using existing python3: $(python3 --version)"
fi

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
else
    echo "Reusing existing .venv."
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

echo
echo "GhostGUI installed successfully."
echo "Run it with:"
echo "  source .venv/bin/activate"
echo "  ghostgui"
echo
echo "Or use:"
echo "  bash scripts/run_macos.sh"
