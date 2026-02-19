#!/bin/bash
# Download models for the face analysis pipeline.
#
# HSE MobileNet age/gender model (Apache 2.0 license)
# Source: https://github.com/av-savchenko/HSE_FaceRec_tf
#
# The model is distributed as a TensorFlow .pb file and must be converted
# to ONNX before use on the Pi.  Requires: pip install tensorflow tf2onnx
#
# Usage: bash scripts/download_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models"

mkdir -p "$MODEL_DIR"

# --- HSE MobileNet Age/Gender ONNX ---
AGE_GENDER_MODEL="$MODEL_DIR/age_gender.onnx"
PB_FILE="$MODEL_DIR/age_gender_tf2_224_deep-03-0.13-0.97.pb"

if [ -f "$AGE_GENDER_MODEL" ]; then
    echo "Age/gender model already exists: $AGE_GENDER_MODEL"
else
    echo "Downloading HSE MobileNet age/gender .pb model..."
    curl -L -o "$PB_FILE" \
        "https://github.com/av-savchenko/HSE_FaceRec_tf/raw/master/age_gender_identity/age_gender_tf2_224_deep-03-0.13-0.97.pb"

    if [ ! -f "$PB_FILE" ]; then
        echo "ERROR: Download failed."
        echo "See: https://github.com/av-savchenko/HSE_FaceRec_tf"
        exit 1
    fi

    echo "Downloaded .pb: $(du -h "$PB_FILE" | cut -f1)"
    echo "Converting to ONNX (requires tensorflow and tf2onnx)..."

    python -m tf2onnx.convert \
        --graphdef "$PB_FILE" \
        --output "$AGE_GENDER_MODEL" \
        --inputs "input_1:0" \
        --outputs "age_pred/Softmax:0,gender_pred/Sigmoid:0"

    if [ -f "$AGE_GENDER_MODEL" ]; then
        echo "Converted: $AGE_GENDER_MODEL ($(du -h "$AGE_GENDER_MODEL" | cut -f1))"
        rm -f "$PB_FILE"
        echo "Cleaned up .pb file."
    else
        echo "ERROR: ONNX conversion failed."
        echo "Make sure tensorflow and tf2onnx are installed: pip install tensorflow tf2onnx"
        exit 1
    fi
fi

echo ""
echo "Models ready in: $MODEL_DIR"
ls -lh "$MODEL_DIR"
