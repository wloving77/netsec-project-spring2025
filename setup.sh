#!/bin/bash

# Navigate to project
cd gan_ids_project || exit

echo "Assumed python3 available on \$PATH"

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt