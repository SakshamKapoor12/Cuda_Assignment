#!/bin/bash

echo "========================================"
echo "   CUDA Capstone Project Builder        "
echo "========================================"

# 1. Compile
echo "[INFO] Compiling..."
make clean
make all

# 2. Run with different sizes
echo "[INFO] Running N-Body Simulation..."
./bin/nbody_sim 8192

# 3. Visualize
echo "[INFO] Generating Visualization..."
python3 scripts/visualize.py

echo "========================================"
echo "[SUCCESS] Project completed."
echo "Check 'simulation_result.png' for output."
echo "========================================"