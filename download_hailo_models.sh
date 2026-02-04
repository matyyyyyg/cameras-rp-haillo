#!/bin/bash
# Download Hailo models for CCTV Analytics System
# Includes face detection and gender classification models

set -e

MODELS_DIR="${1:-models/hailo}"
mkdir -p "$MODELS_DIR"

echo "=== Hailo Model Downloader ==="
echo "Target directory: $MODELS_DIR"
echo ""

# Auto-detect Hailo chip variant (hailo8 vs hailo8l)
HAILO_ARCH="hailo8"  # Default to Hailo-8 (26 TOPS)
if command -v hailortcli &> /dev/null; then
    CHIP_INFO=$(hailortcli fw-control identify 2>/dev/null || true)
    if echo "$CHIP_INFO" | grep -qi "hailo-8l\|8L\|HAILO8L"; then
        HAILO_ARCH="hailo8l"
        echo "Detected: Hailo-8L (13 TOPS)"
    elif echo "$CHIP_INFO" | grep -qi "hailo-8[^l]\|HAILO8[^L]"; then
        HAILO_ARCH="hailo8"
        echo "Detected: Hailo-8 (26 TOPS)"
    else
        echo "Could not detect chip variant, defaulting to hailo8l (RPi 5 AI HAT)"
    fi
else
    echo "hailortcli not found, defaulting to hailo8l (RPi 5 AI HAT)"
fi
echo "Using architecture: $HAILO_ARCH"
echo ""

# Face Detection Models (from Hailo Model Zoo)
echo "📥 Downloading Face Detection models..."

# RetinaFace MobileNet (recommended for face detection)
if [ ! -f "$MODELS_DIR/retinaface_mobilenet_v1.hef" ]; then
    echo "  Downloading retinaface_mobilenet_v1.hef ($HAILO_ARCH)..."
    wget -q --show-progress -O "$MODELS_DIR/retinaface_mobilenet_v1.hef" \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/${HAILO_ARCH}/retinaface_mobilenet_v1.hef" || \
        echo "  ⚠️ Failed to download retinaface_mobilenet_v1.hef"
else
    echo "  ✅ retinaface_mobilenet_v1.hef already exists (delete to re-download for $HAILO_ARCH)"
fi

# SCRFD 2.5g (faster face detection alternative)
if [ ! -f "$MODELS_DIR/scrfd_2.5g.hef" ]; then
    echo "  Downloading scrfd_2.5g.hef ($HAILO_ARCH)..."
    wget -q --show-progress -O "$MODELS_DIR/scrfd_2.5g.hef" \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/${HAILO_ARCH}/scrfd_2.5g.hef" || \
        echo "  ⚠️ Failed to download scrfd_2.5g.hef"
else
    echo "  ✅ scrfd_2.5g.hef already exists (delete to re-download for $HAILO_ARCH)"
fi

# SCRFD 10g (higher accuracy face detection)
if [ ! -f "$MODELS_DIR/scrfd_10g.hef" ]; then
    echo "  Downloading scrfd_10g.hef ($HAILO_ARCH)..."
    wget -q --show-progress -O "$MODELS_DIR/scrfd_10g.hef" \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/${HAILO_ARCH}/scrfd_10g.hef" || \
        echo "  ⚠️ Failed to download scrfd_10g.hef"
else
    echo "  ✅ scrfd_10g.hef already exists (delete to re-download for $HAILO_ARCH)"
fi

echo ""
echo "📥 Downloading Gender Classification model..."

# Gender Classification from DeGirum (FairFace trained)
# Note: DeGirum models are available via their PySDK, we'll create a placeholder
GENDER_MODEL="$MODELS_DIR/yolov8n_fairface_gender.hef"
if [ ! -f "$GENDER_MODEL" ]; then
    echo "  ⚠️ Gender model needs to be downloaded from DeGirum AI Hub"
    echo ""
    echo "  To download the gender model:"
    echo "  1. Install DeGirum PySDK: pip install degirum"
    echo "  2. Run: python -c \"import degirum as dg; zoo = dg.connect_model_zoo(); zoo.download_model('yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8l_1')\""
    echo ""
    echo "  Or download manually from: https://hub.degirum.com/degirum/hailo"
else
    echo "  ✅ yolov8n_fairface_gender.hef already exists"
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models in $MODELS_DIR:"
ls -la "$MODELS_DIR"/*.hef 2>/dev/null || echo "  No .hef files found yet"
echo ""
echo "Next steps:"
echo "1. Ensure Hailo SDK is installed: pip install hailo-platform"
echo "2. For gender model, use DeGirum PySDK or manual download"
echo "3. Run: python -c 'from src.hailo_detector import check_hailo_setup; print(check_hailo_setup())'"
