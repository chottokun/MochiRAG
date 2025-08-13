#!/bin/bash
# This script sets up the development environment for the MochiRAG project.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_VERSION="python3.10"
VENV_DIR="venv"

# --- Functions ---
print_info() {
    echo "INFO: $1"
}

print_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Main Script ---
print_info "Starting development environment setup..."

# 1. Check for Python
if ! command -v $PYTHON_VERSION &> /dev/null
then
    if command -v python3 &> /dev/null
    then
        PYTHON_VERSION="python3"
        print_info "python3.10 not found, falling back to python3."
    else
        print_error "Neither python3.10 nor python3 could be found. Please install it."
    fi
fi
print_info "Python version check passed."

# 2. Setup Poetry and Virtual Environment
print_info "Installing poetry..."
pip install --upgrade pip
pip install poetry

print_info "Configuring Poetry to create venv inside the project..."
poetry config virtualenvs.in-project true

# This command will now create and use the ./venv directory automatically
print_info "Generating lock file and installing dependencies..."
poetry lock
poetry install --extras "dev test"

print_info "Setup complete!"
print_info "To activate the virtual environment, run: source venv/bin/activate"
