#!/bin/bash

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Make sure streamlit is installed in the correct environment
pip install streamlit --upgrade

# Run the application
streamlit run main.py --server.port $PORT --server.address 0.0.0.0
