#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Helper Functions for Colored Output ---
info() {
    # Blue
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

success() {
    # Green
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

error() {
    # Red
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
    exit 1
}

# --- Main Script ---
info "Starting MochiRAG development environment setup..."

# 1. Check for Python 3.10+
info "Checking for Python 3.10 or higher..."
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' &> /dev/null; then
    error "Python 3.10 or higher is required. Please install it and make it available as 'python3'."
fi
success "Python version check passed."

# 2. Set up Virtual Environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    info "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        info "Detected 'uv'. Using it to create virtual environment."
        uv venv
    else
        info "'uv' not found. Using standard 'python3 -m venv'."
        python3 -m venv "$VENV_DIR"
    fi
    success "Virtual environment created in './$VENV_DIR/'."
fi

# 3. Install Dependencies
info "Installing project dependencies (including test dependencies)..."
if command -v uv &> /dev/null; then
    info "Using 'uv' to install dependencies."
    uv pip install ".[test]"
else
    info "Using 'pip' to install dependencies."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install ".[test]"
fi
success "All dependencies installed successfully."

# 4. Final Instructions
echo
success "Setup complete!"
info "To activate the virtual environment, run the following command:"
echo
echo "  source $VENV_DIR/bin/activate"
echo
info "After activation, you can run the backend and frontend as described in README.md."