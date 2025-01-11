#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit --no-cache-dir

# Run the application
python -m streamlit run main.py --server.port=8501 --server.address=0.0.0.0
