#!/bin/bash

# Remove the old environment if it exists
if [ -d new_env ]; then
    echo "Removing old virtual environment..."
    rm -rf new_env
else
    echo "No existing virtual environment found."
fi

# Create a new virtual environment
echo "Creating a new virtual environment..."
python3 -m venv new_env

# Activate the new virtual environment
echo "Activating the virtual environment..."
source new_env/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate the virtual environment."
    exit 1
fi
    echo "Installing dependencies..."
    pip install requests bs4 pandas selenium matplotlib numpy scikit-learn mlxtend > /dev/null

echo "Environment setup complete!"
