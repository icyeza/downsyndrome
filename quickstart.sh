#!/bin/bash
# Quick Start Script for Down Syndrome Classification Project
# Linux/Mac version

set -e

echo "========================================================================"
echo "  DOWN SYNDROME CLASSIFICATION SYSTEM - QUICK START"
echo "========================================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\S+')
echo "✓ Python $python_version detected"
echo ""

# Check virtual environment
echo "🔍 Checking for virtual environment..."
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠ Virtual environment not detected!"
    echo "  Run: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "📁 Creating project directories..."
mkdir -p models
mkdir -p data/train
mkdir -p data/test
mkdir -p uploads
mkdir -p logs
echo "✓ Directories created"
echo ""

echo "========================================================================"
echo "  QUICK START OPTIONS"
echo "========================================================================"
echo ""
echo "1. Train the model:"
echo "   jupyter notebook notebook/down_syndrome.ipynb"
echo ""
echo "2. Start the API:"
echo "   cd api && python app.py"
echo ""
echo "3. Open the dashboard:"
echo "   cd ui && python -m http.server 8000"
echo "   Then open http://localhost:8000 in your browser"
echo ""
echo "4. Run load tests:"
echo "   locust -f tests/locustfile.py -H http://localhost:5000"
echo ""
echo "5. Deploy with Docker:"
echo "   docker-compose up -d"
echo ""
echo "========================================================================"
echo "  Setup complete! ✓"
echo "========================================================================"
