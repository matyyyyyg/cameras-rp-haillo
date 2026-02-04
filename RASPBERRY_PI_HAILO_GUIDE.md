# Raspberry Pi 5 + Hailo-8 Deployment Guide

This guide covers deploying the CCTV Analytics System on Raspberry Pi 5 with Hailo-8 AI accelerator for **30-40+ FPS** performance.

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **Board** | Raspberry Pi 5 (4GB or 8GB RAM recommended) |
| **AI Accelerator** | Raspberry Pi AI HAT+ with Hailo-8 (26 TOPS / ~28 TFLOP) |
| **Storage** | 32GB+ microSD or SSD |
| **Camera** | Raspberry Pi Camera Module 3 (manual focus) |
| **Power** | 5V 5A USB-C power supply |

## Performance Comparison

| Platform | Backend | FPS | Notes |
|----------|---------|-----|-------|
| MacBook (M1/M2) | CPU (MTCNN) | 3-5 FPS | Development |
| Raspberry Pi 5 | CPU (OpenCV SSD) | 5-10 FPS | Basic |
| Raspberry Pi 5 + Hailo-8L | Hardware (13 TOPS) | 25-30 FPS | Budget option |
| **Raspberry Pi 5 + Hailo-8** | **Hardware (26 TOPS)** | **30-40+ FPS** | **Client spec** |

## Step 1: Raspberry Pi Setup

### 1.1 Install Raspberry Pi OS
```bash
# Use Raspberry Pi Imager to install Raspberry Pi OS (64-bit)
# Recommended: Bookworm Desktop (for camera support)
```

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y
```

## Step 2: Install Hailo-8L Software

### 2.1 Add Hailo Repository
```bash
# Add Hailo's apt repository
curl -fsSL https://hailo.ai/install/hailo-rpi5-apt.sh | sudo bash
```

### 2.2 Install Hailo Runtime
```bash
# Install HailoRT and Python bindings
sudo apt install hailo-all -y

# Verify installation
hailortcli fw-control identify
```

### 2.3 Install Python Bindings
```bash
pip install hailo-platform
```

## Step 3: Clone and Setup Project

### 3.1 Clone Repository
```bash
git clone <your-repo-url> ~/cctv-analytics
cd ~/cctv-analytics
```

### 3.2 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install Dependencies
```bash
# Basic dependencies
pip install opencv-python numpy

# For CPU fallback
pip install mtcnn-opencv

# For InsightFace age/gender (optional, slower)
pip install insightface onnxruntime
```

## Step 4: Download Hailo Models

### 4.1 Face Detection Models
```bash
# Create models directory
mkdir -p models/hailo

# Download RetinaFace (recommended for face detection)
wget -O models/hailo/retinaface_mobilenet_v1.hef \
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.8.0/retinaface_mobilenet_v1.hef"

# Alternative: SCRFD (faster, slightly less accurate)
wget -O models/hailo/scrfd_2.5g.hef \
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.8.0/scrfd_2.5g.hef"
```

### 4.2 Age/Gender Models (Optional)
For age/gender estimation on Hailo, you need to compile custom models:

```bash
# Clone Hailo Model Zoo
git clone https://github.com/hailo-ai/hailo_model_zoo.git

# Follow compilation instructions for age/gender models
# This requires Hailo Dataflow Compiler (DFC)
```

**Note**: For simpler setup, use CPU-based age/gender with InsightFace 
and Hailo only for face detection (still achieves 15-20 FPS).

## Step 5: Run with Hailo Acceleration

### 5.1 Basic Usage
```bash
# Activate environment
source venv/bin/activate

# Run with Hailo + InsightFace (recommended)
python test_unified_hailo_insightface.py --input camera --display

# Or fastest option (Hailo + Caffe)
python test_unified_hailo_caffe.py --input camera --display
```

### 5.2 With Video File
```bash
python test_unified_hailo_insightface.py --video cctv_footage.mp4 --save-detections
```

### 5.3 Headless Mode (Server)
```bash
# Run as background service
nohup python test_unified_hailo_insightface.py \
  --input camera \
  --save-detections \
  > /var/log/cctv-analytics.log 2>&1 &
```

## Step 6: Create Systemd Service

### 6.1 Create Service File
```bash
sudo nano /etc/systemd/system/cctv-analytics.service
```

Add:
```ini
[Unit]
Description=CCTV Analytics Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cctv-analytics
ExecStart=/home/pi/cctv-analytics/venv/bin/python test_unified_hailo_insightface.py \
  --input camera \
  --save-detections
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 6.2 Enable Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cctv-analytics
sudo systemctl start cctv-analytics

# Check status
sudo systemctl status cctv-analytics
```

## Troubleshooting

### Hailo Not Detected
```bash
# Check device
lspci | grep Hailo
hailortcli fw-control identify

# If not found, check connections and reboot
sudo reboot
```

### Low FPS
1. Check if Hailo is being used: Look for "Hailo_8L" in logs
2. Reduce input resolution: `--source "/dev/video0?width=640&height=480"`
3. Use SCRFD instead of RetinaFace (faster)

### Camera Issues
```bash
# List cameras
v4l2-ctl --list-devices

# Test camera
libcamera-hello
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                            │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │   Camera    │───▶│  Hailo-8L    │───▶│  Python App    │ │
│  │ (USB/CSI)   │    │  (13 TOPS)   │    │  (Tracking +   │ │
│  │             │    │              │    │   Logging)     │ │
│  └─────────────┘    └──────────────┘    └────────────────┘ │
│                           │                      │          │
│                           ▼                      ▼          │
│                    Face Detection          CSV/JSONL        │
│                    Age/Gender              Logs             │
│                    ~25-30 FPS                               │
└─────────────────────────────────────────────────────────────┘
```

## Model Performance on Hailo-8L

| Model | Input Size | FPS | Use Case |
|-------|------------|-----|----------|
| RetinaFace MobileNet | 640x640 | ~25 FPS | Best accuracy |
| SCRFD 2.5G | 640x480 | ~40 FPS | Faster, good accuracy |
| YuNet | 320x320 | ~60+ FPS | Fastest, basic accuracy |

## Resources

- [Hailo RPi5 Examples](https://github.com/hailo-ai/hailo-rpi5-examples)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo Developer Documentation](https://hailo.ai/developer-zone/)
- [Hailo Community Forum](https://community.hailo.ai/)
