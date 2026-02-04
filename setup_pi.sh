#!/bin/bash
# Raspberry Pi Setup Script
# Run this on the Raspberry Pi after transferring the code

set -e

echo "================================================"
echo "🥧 CCTV Analytics - Raspberry Pi Setup"
echo "================================================"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-venv git cmake

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install opencv-python numpy flask flask-cors

# Try to install Hailo
echo "🚀 Installing Hailo support..."
if pip install hailo-platform 2>/dev/null; then
    echo "✅ Hailo Python bindings installed"
else
    echo "⚠️  Hailo not available - will use CPU fallback"
fi

# Install optional dependencies
echo "📦 Installing optional dependencies..."
pip install mtcnn-opencv insightface onnxruntime || echo "⚠️  Some optional packages failed"

# Create logs directory
mkdir -p logs

# Check for Hailo hardware
echo ""
echo "🔍 Checking for Hailo hardware..."
if command -v hailortcli &> /dev/null; then
    hailortcli fw-control identify || echo "⚠️  Hailo firmware not detected"
else
    echo "⚠️  hailortcli not found - Hailo SDK may not be installed"
    echo "   Install with: sudo apt install hailo-all"
fi

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Download Hailo models (if using Hailo):"
echo "   ./download_hailo_models.sh"
echo ""
echo "2. Test detection:"
echo "   source venv/bin/activate"
echo "   python -m src.main --backend hailo --source 0"
echo ""
echo "3. Run dashboard:"
echo "   python run_dashboard.py"
echo ""
echo "4. Access at: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
