#!/bin/bash
# Deployment Package Creator
# Packages CCTV Analytics for Raspberry Pi deployment

set -e

echo "================================================"
echo "Creating Raspberry Pi Deployment Package"
echo "================================================"
echo ""

# Project directory
PROJECT_DIR="/Users/krishanu8219/Documents/Face detection"
OUTPUT_FILE="cctv-analytics-deploy.tar.gz"

cd "$PROJECT_DIR"

echo "📦 Packaging files..."

# Create deployment package
tar -czf "$OUTPUT_FILE" \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='uploads/*' \
  --exclude='venv' \
  --exclude='.git' \
  src/ \
  models/ \
  templates/ \
  logs/.gitkeep \
  requirements.txt \
  run_dashboard.py \
  download_models.py \
  RASPBERRY_PI_HAILO_GUIDE.md \
  README.md

echo "✅ Package created: $OUTPUT_FILE"
echo ""
echo "📊 Package size:"
ls -lh "$OUTPUT_FILE"
echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Transfer to Raspberry Pi:"
echo "   scp $OUTPUT_FILE pi@raspberry-pi.local:~/"
echo ""
echo "2. On Raspberry Pi, extract:"
echo "   tar -xzf ~/cctv-analytics-deploy.tar.gz -C ~/cctv-analytics"
echo ""
echo "3. Follow deployment_plan.md for setup instructions"
echo ""
