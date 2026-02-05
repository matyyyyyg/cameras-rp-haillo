#!/bin/bash
# DeGirum Setup Script for Raspberry Pi 5 + Hailo-8
#
# This script installs DeGirum PySDK and verifies the setup.
#
# Usage:
#   chmod +x setup_degirum.sh
#   ./setup_degirum.sh

set -e

echo "=============================================="
echo "DeGirum PySDK Setup for Raspberry Pi + Hailo"
echo "=============================================="

# Check if running on Raspberry Pi
if [[ ! -f /proc/device-tree/model ]]; then
    echo "Warning: Not running on Raspberry Pi"
else
    MODEL=$(cat /proc/device-tree/model)
    echo "Device: $MODEL"
fi

# Check Hailo device
echo ""
echo "Checking Hailo device..."
if lsmod | grep -q hailo; then
    echo "  [OK] Hailo kernel module loaded"
else
    echo "  [WARNING] Hailo kernel module not loaded"
    echo "  Try: sudo modprobe hailo_pci"
fi

# Check HailoRT installation
echo ""
echo "Checking HailoRT..."
if dpkg -l | grep -q hailort; then
    HAILO_VERSION=$(dpkg -l | grep hailort | head -1 | awk '{print $3}')
    echo "  [OK] HailoRT installed: $HAILO_VERSION"
else
    echo "  [WARNING] HailoRT not found"
    echo "  Install with: sudo apt install hailo-all"
fi

# Install DeGirum packages
echo ""
echo "Installing DeGirum PySDK..."
pip install --upgrade degirum degirum_tools

# Verify installation
echo ""
echo "Verifying DeGirum installation..."
python3 -c "
import degirum as dg
import degirum_tools
print(f'  DeGirum version: {dg.__version__}')
print('  [OK] DeGirum PySDK installed successfully')
" || {
    echo "  [ERROR] Failed to import DeGirum"
    exit 1
}

# Test Hailo connection via DeGirum
echo ""
echo "Testing Hailo connection via DeGirum..."
python3 -c "
import degirum as dg

try:
    zoo = dg.connect('@local', 'degirum/hailo', '')
    models = zoo.list_models()
    print(f'  [OK] Connected to local Hailo')
    print(f'  Available models: {len(models)}')

    # Show some face models
    face_models = [m for m in models if 'face' in m.lower()][:3]
    if face_models:
        print('  Face models found:')
        for m in face_models:
            print(f'    - {m}')
except Exception as e:
    print(f'  [ERROR] Could not connect: {e}')
    print('  Make sure Hailo device is connected and HailoRT is installed')
    exit(1)
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Test scripts available:"
echo "  1. python test_degirum_face.py         # Face detection only"
echo "  2. python test_degirum_gender.py       # Face + Gender"
echo "  3. python test_degirum_pipeline.py     # Full pipeline"
echo ""
echo "Quick test:"
echo "  python test_degirum_face.py --list-models"
echo ""
