#!/bin/bash
echo "========================================"
echo "  FinRisk MultiAgent - Startup"
echo "========================================"
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo ""
echo "Starting Streamlit app..."
python -m streamlit run app.py
