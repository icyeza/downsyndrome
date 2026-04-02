#!/usr/bin/env python3
"""
Quick Start Script for Down Syndrome Classification Project
Initializes and runs the complete pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}\n")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.8+ required (found {version.major}.{version.minor})")
        return False

def check_dependencies():
    """Check if key packages are installed"""
    try:
        import tensorflow
        print(f"✓ TensorFlow {tensorflow.__version__} installed")
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    
    try:
        import flask
        print(f"✓ Flask {flask.__version__} installed")
    except ImportError:
        print("✗ Flask not installed")
        return False
    
    try:
        import kagglehub
        print(f"✓ Kagglehub installed")
    except ImportError:
        print("✗ Kagglehub not installed")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = [
        'models',
        'data/train',
        'data/test',
        'uploads',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified directory: {dir_path}")

def main():
    """Main initialization script"""
    
    print_header("DOWN SYNDROME CLASSIFICATION SYSTEM - QUICK START")
    
    print("📋 Checking system requirements...\n")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ System requirements not met. Please upgrade Python.")
        sys.exit(1)
    
    print("\n📦 Checking dependencies...\n")
    
    # Check if virtual environment is active
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠ Virtual environment not detected!")
        print("  Run: python -m venv venv && source venv/bin/activate")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n📥 Installing dependencies...\n")
        if not run_command("pip install -r requirements.txt", "Installing packages"):
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    print("\n📁 Setting up directories...\n")
    create_directories()
    
    print_header("QUICK START OPTIONS")
    print("""
1. Run Jupyter Notebook (train model)
   jupyter notebook notebook/down_syndrome.ipynb

2. Start Flask API
   cd api && python app.py

3. Open Dashboard
   open ui/index.html
   (or use: cd ui && python -m http.server 8000)

4. Run Load Tests
   locust -f tests/locustfile.py -H http://localhost:5000

5. Deploy with Docker
   docker-compose up -d

6. View Documentation
   cat README.md
   cat SETUP_GUIDE.md

    """)
    
    print_header("NEXT STEPS")
    print("""
✓ Environment is ready!

To get started:
1. Run the Jupyter notebook to train the model
2. Start the API server
3. Open the dashboard in your browser
4. Make predictions on test images

For detailed instructions, see SETUP_GUIDE.md
    """)

if __name__ == "__main__":
    main()
