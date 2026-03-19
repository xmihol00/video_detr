#!/bin/bash

# Script to create the conda environment for VideoDETR/CNNSearch
# Usage: ./create_env.sh

ENV_NAME="cnnSearch"
CONDA_FILE="environment.yml"

echo "=================================================="
echo "Creating Conda Environment: $ENV_NAME"
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "$CONDA_FILE" ]; then
    echo "Error: $CONDA_FILE not found in the current directory."
    exit 1
fi

# Create the environment
echo "Creating environment from $CONDA_FILE..."
conda env create -f "$CONDA_FILE"

# Check if creation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Environment '$ENV_NAME' created successfully!"
    echo "=================================================="
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "Error: Failed to create environment '$ENV_NAME'."
    echo "=================================================="
    exit 1
fi
