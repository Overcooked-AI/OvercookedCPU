#!/bin/bash
# quickstart.sh
# Quick start script for MAPPO Overcooked training

echo "======================================"
echo "MAPPO Overcooked Quick Start"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment found"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate  # Works for both Unix and Windows
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Test environment
echo ""
echo "Testing environment setup..."
python overcooked_mappo_env.py
if [ $? -eq 0 ]; then
    echo "✓ Environment test passed"
else
    echo "✗ Environment test failed"
    exit 1
fi

# Run quick training demo
echo ""
echo "======================================"
echo "Starting quick training demo"
echo "Layout: cramped_room"
echo "Iterations: 10 (for demo purposes)"
echo "======================================"
echo ""

python train_mappo.py \
    --layout cramped_room \
    --iterations 10 \
    --workers 2 \
    --checkpoint-freq 5

echo ""
echo "======================================"
echo "Quick start completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check results in ./results/"
echo "  2. Run full training:"
echo "     python train_mappo.py --iterations 500"
echo "  3. Evaluate checkpoint:"
echo "     python train_mappo.py --eval-checkpoint <path>"
echo ""
echo "For more options, run:"
echo "  python train_mappo.py --help"
echo ""