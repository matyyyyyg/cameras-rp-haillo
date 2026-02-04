# Raspberry Pi Deployment - Quick Start

Follow these steps to deploy on Raspberry Pi 5 + Hailo-8.

## Step 1: Create Package (on Mac)

```bash
cd "/Users/krishanu8219/Documents/Face detection"
./create_deploy_package.sh
```

This creates `cctv-analytics-deploy.tar.gz`

## Step 2: Transfer to Raspberry Pi

```bash
# Replace with your Pi's hostname or IP
scp cctv-analytics-deploy.tar.gz pi@raspberry-pi.local:~/
```

## Step 3: Setup on Raspberry Pi

```bash
# SSH into Pi
ssh pi@raspberry-pi.local

# Extract code
mkdir -p cctv-analytics
tar -xzf cctv-analytics-deploy.tar.gz -C cctv-analytics
cd cctv-analytics

# Run setup
chmod +x setup_pi.sh
./setup_pi.sh

# Download Hailo models
chmod +x download_hailo_models.sh
./download_hailo_models.sh
```

## Step 4: Test

```bash
# Activate environment
source venv/bin/activate

# Test face detection with Hailo + InsightFace (recommended)
python test_unified_hailo_insightface.py --input camera --display

# Or test fastest option (Hailo + Caffe)
python test_unified_hailo_caffe.py --input camera --display
```

## Step 5: Production (Auto-Start)

```bash
# Create service
sudo cp deployment_plan.md /tmp/
# Follow "Phase 4: Production Setup" in deployment_plan.md

sudo systemctl enable cctv-analytics
sudo systemctl start cctv-analytics
```

## Troubleshooting

**Hailo not detected:**
```bash
lspci | grep Hailo
hailortcli fw-control identify
```

**Low FPS:**
- Check logs for "Hailo" confirmation
- Reduce camera resolution to 640x480
- Use SCRFD model instead of RetinaFace

**Can't access dashboard:**
```bash
# Find Pi's IP
hostname -I

# Access at http://<IP>:8080
```

---

For detailed instructions see [deployment_plan.md](file:///Users/krishanu8219/.gemini/antigravity/brain/b462acc7-2b79-4840-9b4a-bb642c78fe2c/deployment_plan.md)
