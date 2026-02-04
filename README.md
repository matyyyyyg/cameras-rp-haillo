# CCTV Face Analytics System

**Real-time face detection, gender classification, and person tracking optimized for CCTV footage.**

* **Optimized for:** Raspberry Pi 5 + Hailo-8 AI Accelerator  
* **Use Case:** CCTV analytics with challenging angles, poor lighting, and motion blur  
* **Performance:** 25-30 FPS on hardware-accelerated setups

---

## � **New to This Project? Start Here:**

| Guide | Time | Purpose |
|-------|------|---------|
| **[QUICK_START_CHECKLIST.md](QUICK_START_CHECKLIST.md)** | 5 min | Get set up fast |
| **[TESTING_ROADMAP.md](TESTING_ROADMAP.md)** | 10 min | Visual testing guide |
| **[TEAM_TESTING_GUIDE.md](TEAM_TESTING_GUIDE.md)** | 2-3 hrs | Complete testing instructions |
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | - | Find any documentation |

**** All Guides:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) has the complete list

---

## �* Quick Start for Team Testing

### Prerequisites
- **Hardware:** Raspberry Pi 5 + Hailo-8 (or any Linux/macOS system for testing)
- **Python:** 3.10+
- **Camera/Video:** Webcam, video file, or RTSP stream

### Quick Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download Hailo face detection models (for Raspberry Pi 5 + Hailo-8)
bash download_hailo_models.sh

# 3. Run a test script (choose based on your needs)
python test_unified_hailo_caffe.py --input camera --display
```

**See [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) for complete setup and testing guide.**

---

## * What This System Does

1. **Detects faces** in video streams (optimized for CCTV footage)
2. **Classifies gender** with 90-95% accuracy
3. **Estimates age** (8 age groups)
4. **Tracks individuals** across frames with stable IDs
5. **Logs detections** to CSV/JSON for analytics

### Key Features

- **Face Detection**: MTCNN (best for CCTV profiles) & OpenCV SSD (speed)
- **Gender Classification**: Two optimized implementations (Option A & B)
- **Age Estimation**: 8-bucket classification with confidence scores
- **Person Tracking**: IoU-based tracking with stable IDs
- **Logging**: CSV and JSONL output with ISO 8601 timestamps
- **Enhancement**: CLAHE preprocessing for low-light CCTV footage

---

## 📁 Project Structure

```
📦 Face detection Raspberry/
├── 📂 src/                          # Core pipeline code
│   ├── main.py                      # Main detection pipeline
│   ├── detectors.py                 # Face detection backends (MTCNN, OpenCV)
│   ├── classification.py            # Age/gender classification
│   ├── tracking.py                  # Multi-person tracking
│   ├── logging_utils.py             # CSV/JSON logging
│   ├── hailo_detector.py            # Hailo hardware detector
│   ├── hailo_gender_classifier.py   # Hailo gender classifier
│   └── web_server.py                # Web dashboard server
│
├── 📂 models/                       # Pre-trained models
│   ├── *.caffemodel                 # OpenCV DNN models
│   ├── *.prototxt                   # Model architectures
│   └── hailo/                       # Hailo-optimized models (if available)
│
├── 📂 logs/                         # Auto-generated detection logs
│   ├── detections.csv               # CSV format logs
│   └── *.jsonl                      # JSON Lines format
│
├── 📂 templates/                    # Web dashboard UI
│   └── index.html
│
├── 📂 uploads/                      # Upload folder for video files
│
├── 🧪 HAILO TEST SCRIPTS (Raspberry Pi 5 + Hailo-8)
│   ├── test_unified_hailo_caffe.py       # Hailo + Caffe (18-22 FPS, 85-90% acc)
│   ├── test_unified_hailo_deepface.py    # Hailo + DeepFace (8-12 FPS, 93-95% acc)
│   ├── test_unified_hailo_insightface.py # Hailo + InsightFace (12-16 FPS, 95-97% acc)
│   ├── test_unified_hailo_degirum.py     # Hailo + DeGirum (12-15 FPS, 90-95% acc)
│   └── test_gender_insightface.py        # CPU-only baseline (15-20 FPS)
│
├── ** DOCUMENTATION
│   ├── README.md                    # This file - project overview
│   ├── TESTING_INSTRUCTIONS.md      # Complete testing guide with troubleshooting
│   ├── VALIDATION_REPORT.md         # Script validation and fixes applied
│   ├── HAILO_FACE_DETECTION.md      # Hailo implementation details
│   ├── QUICK_START_CHECKLIST.md     # 5-min setup checklist
│   ├── TEAM_TESTING_GUIDE.md        # Comprehensive testing methodology
│   └── DOCUMENTATION_INDEX.md       # Master documentation index
│
├── 🔧 SETUP SCRIPTS
│   ├── setup_pi.sh                 # Automated Pi setup
│   ├── download_models.py          # Model downloader
│   ├── download_hailo_models.sh    # Hailo model setup
│   └── create_deploy_package.sh    # Create deployment package
│
└── * REQUIREMENTS
    ├── requirements.txt             # Standard dependencies
    └── requirements-production.txt  # Production deployment
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Hardware:** Raspberry Pi 5 with Hailo-8 AI HAT (26 TOPS)
- **OS:** Raspberry Pi OS (64-bit) or Ubuntu 22.04+
- **Python:** 3.9, 3.10, or 3.11

### Step 1: System Setup (Raspberry Pi 5)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Hailo runtime
sudo apt install hailo-all

# Verify Hailo device
hailortcli fw-control identify
lsmod | grep hailo
```

### Step 2: Install Python Dependencies

```bash
cd "Face detection Raspberry"

# Install base requirements
pip install -r requirements.txt

# Install specific model dependencies (based on which test you want to run)
pip install deepface tf-keras        # For test_unified_hailo_deepface.py
pip install insightface onnxruntime  # For test_unified_hailo_insightface.py
pip install degirum degirum-tools    # For test_unified_hailo_degirum.py
```

### Step 3: Download Hailo Models

```bash
# Download RetinaFace face detection model for Hailo-8
bash download_hailo_models.sh

# Verify models downloaded
ls -lh models/hailo/
# Should see: retinaface_mobilenet_v1.hef (or scrfd_2.5g.hef)
```

### Step 4: Run Your First Test

```bash
# Test with webcam
python test_unified_hailo_caffe.py --input camera --display

# Test with video file
python test_unified_hailo_caffe.py --input path/to/video.mp4 --display --save results.json
```

**Troubleshooting?** See [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) for detailed setup and common issues.

---

## 🧪 Testing the Gender Classification System

**All test scripts use Hailo-8 RetinaFace for face detection (25-30 FPS).**  
They differ only in the gender classification model:

### Test Script 1: Hailo + Caffe (FASTEST)
**Performance:** 18-22 FPS | **Accuracy:** 85-90%

```bash
python test_unified_hailo_caffe.py --input test_video.mp4 --display
```

### Test Script 2: Hailo + DeepFace (MOST ACCURATE)
**Performance:** 8-12 FPS | **Accuracy:** 93-95%

```bash
pip install deepface tf-keras
python test_unified_hailo_deepface.py --input test_video.mp4 --display
```

### Test Script 3: Hailo + InsightFace (BEST BALANCE)
**Performance:** 12-16 FPS | **Accuracy:** 95-97% | **Bonus:** Age estimation

```bash
pip install insightface onnxruntime
python test_unified_hailo_insightface.py --input test_video.mp4 --display
```

### Test Script 4: Hailo + DeGirum Cloud
**Performance:** 12-15 FPS | **Accuracy:** 90-95% | **Requires:** Internet + API key

```bash
pip install degirum degirum-tools
# Add API token to env.ini (get free token from https://cs.degirum.com)
python test_unified_hailo_degirum.py --input test_video.mp4 --display
```

### CPU-Only Baseline (No Hailo)
**Performance:** 15-20 FPS | **Accuracy:** 95-97%

```bash
python test_gender_insightface.py --video test_video.mp4
```

**Full testing guide:** See [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md)

---

## * Testing Checklist for Team

### 1. Environment Setup [x]
- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (check `models/` directory)

### 2. Basic Pipeline Test [x]
- [ ] Test with webcam: `python -m src.main`
- [ ] Test with video file: `python -m src.main --source test_video.mp4`
- [ ] Verify logs created in `logs/` directory
- [ ] Check CSV output format

### 3. Gender Detection Test [x]
- [ ] Choose Option A or B (see above)
- [ ] Install DeGirum SDK (if using Option A/B)
- [ ] Run on CCTV test footage
- [ ] Record FPS and accuracy observations
- [ ] Compare with baseline tests (optional)

### 4. Performance Benchmarks 📊
Record these metrics in your test report:
- **FPS:** Frames per second achieved
- **Accuracy:** Percentage of correct gender classifications
- **Face Detection Rate:** Faces detected / total faces
- **Hardware Used:** Pi 5, laptop, etc.
- **Video Quality:** Resolution, lighting conditions

### 5. Known Issues & Edge Cases 🐛
Test these scenarios:
- [ ] Side profile faces
- [ ] Poor lighting / night footage
- [ ] Multiple people in frame
- [ ] Motion blur from fast movement
- [ ] Small/distant faces

---

## 🖥️ Running the Web Dashboard

```bash
# Start the web server
python run_dashboard.py

# Open browser to http://localhost:5000
# Upload videos for processing
# View real-time detection results
```

---

## 📊 Output Format

### CSV Logs (`logs/detections.csv`)
```csv
timestamp,person_id,age_group,age_confidence,gender,gender_confidence
2026-01-07T10:30:45.123456,1,25-32,0.92,Male,0.87
2026-01-07T10:30:45.234567,2,18-24,0.88,Female,0.91
```

### JSONL Logs (`logs/*.jsonl`)
```json
{"timestamp": "2026-01-07T10:30:45.123456", "person_id": 1, "age_group": "25-32", "age_confidence": 0.92, "gender": "Male", "gender_confidence": 0.87}
```

---

## 🔧 Command Line Options (Main Pipeline)

```bash
python -m src.main [OPTIONS]

Options:
  --source, -s       Video source (default: 0 for webcam)
                     Examples: 0, video.mp4, rtsp://camera/stream
  
  --camera-id, -c    Camera identifier (default: cam_01)
  
  --backend, -b      Detection backend (default: mtcnn)
                     - mtcnn: Best for CCTV (side faces, small faces)
                     - multiscale: Good for small faces with OpenCV
                     - opencv: Fast, frontal faces only
  
  --confidence       Detection confidence threshold (default: 0.3)
  
  --enhance-image    Enable image enhancement for CCTV (default: enabled)
  --no-enhance       Disable image enhancement
  
  --log-path, -l     CSV log file path (default: logs/detections.csv)
  --jsonl            Also log to JSONL format
  
  --no-display       Disable video display (headless mode)
  --log-interval     Log every N frames (default: 1)
  --debug            Enable debug logging
```

### Examples

```bash
# CCTV RTSP stream with MTCNN detector
python -m src.main --source "rtsp://admin:pass@192.168.1.100/stream" --backend mtcnn

# Video file with OpenCV detector (faster)
python -m src.main --source cctv_footage.mp4 --backend opencv --confidence 0.5

# Headless processing with JSONL logging
python -m src.main --source input.mp4 --no-display --jsonl

# Debug mode for troubleshooting
python -m src.main --debug --backend mtcnn
```

---

## 📞 Troubleshooting & Support

### Common Issues

**1. "No module named 'cv2'"**
```bash
pip install opencv-python
```

**2. "ModuleNotFoundError: No module named 'mtcnn'"**
```bash
pip install mtcnn-opencv
```

**3. "Cannot open camera/video"**
- Check video file path is correct
- For webcam, try different indices: 0, 1, 2
- For RTSP, verify network connectivity and credentials

**4. "DeGirum authentication failed"**
- Verify `env.ini` has correct token format
- Sign up at https://cs.degirum.com for free tier
- Check internet connection (cloud mode requires internet)

**5. Low FPS on Raspberry Pi**
- Ensure Hailo-8 drivers installed correctly
- Use Option A (Hybrid) instead of Option B
- Reduce video resolution
- Use `--backend opencv` for faster detection

**6. Models not found**
```bash
python download_models.py
# Or check models/ directory for required files
```

### Performance Tuning

| Scenario | Recommendation |
|----------|---------------|
| **Maximum Speed** | `--backend opencv` + Option A |
| **Maximum Accuracy** | `--backend mtcnn` + Option B |
| **CCTV Side Profiles** | `--backend mtcnn` |
| **Frontal Faces Only** | `--backend opencv` |
| **Small/Distant Faces** | `--backend multiscale` |
| **Best Overall** | Option A with MTCNN |

---

## 📈 Expected Performance (Raspberry Pi 5 + Hailo-8)

| Test Script | Face Detection | Gender Model | FPS | Accuracy |
|------------|---------------|--------------|-----|----------|
| **test_unified_hailo_caffe.py** | Hailo RetinaFace | Caffe Adience | 18-22 | 85-90% |
| **test_unified_hailo_deepface.py** | Hailo RetinaFace | DeepFace VGG | 8-12 | 93-95% |
| **test_unified_hailo_insightface.py** | Hailo RetinaFace | InsightFace Buffalo_l | 12-16 | 95-97% |
| **test_unified_hailo_degirum.py** | Hailo RetinaFace | DeGirum MobileNetV2 | 12-15 | 90-95% |
| **test_gender_insightface.py** | InsightFace (CPU) | InsightFace Buffalo_l | 15-20 | 95-97% |

**Key Insight:** All Hailo scripts use the same face detector (RetinaFace at 25-30 FPS). The overall FPS is limited by the gender classifier running on CPU.

*Performance varies with video resolution, face count, and lighting conditions*

---

## * Deployment to Production

### Raspberry Pi Deployment

1. **Transfer project to Pi:**
```bash
# On your computer
scp -r "Face detection Raspberry" pi@raspberrypi.local:~/

# Or use the deployment package
./create_deploy_package.sh
scp cctv-analytics-deploy.tar.gz pi@raspberrypi.local:~/
```

2. **On Raspberry Pi:**
```bash
ssh pi@raspberrypi.local
cd ~/Face\ detection\ Raspberry
chmod +x setup_pi.sh
./setup_pi.sh
```

3. **See detailed guides:**
- [QUICKSTART_PI.md](QUICKSTART_PI.md) - Quick deployment
- [RASPBERRY_PI_HAILO_GUIDE.md](RASPBERRY_PI_HAILO_GUIDE.md) - Full Hailo setup
- [PRODUCTION_CONFIG.md](PRODUCTION_CONFIG.md) - Production optimization

---

## 🤝 Team Testing Workflow

### Step 1: Create Test Report

Save as `TEST_REPORT_[YourName].md`:

```markdown
# Test Report

**Tester:** [Your Name]  
**Date:** 2026-01-07  
**Hardware:** [Pi 5 + Hailo / Laptop / etc.]

## Configuration Tested
- Implementation: [Option A / B / Caffe / etc.]
- Video Source: [Webcam / File / RTSP]
- Resolution: [1920x1080 / etc.]
- Backend: [mtcnn / opencv / etc.]

## Performance Metrics
- **FPS:** [X]
- **Faces Detected:** [X]
- **Gender Accuracy:** [X%]
- **Processing Time:** [X seconds]

## Issues Found
1. [List any bugs or problems]
2. [Include error messages]

## Observations
- [Quality notes]
- [Speed observations]
- [Accuracy comments]

## Recommendation
[Should we use this option? Why or why not?]
```

### Step 2: Share Results

- Save test reports in project root
- Share screenshots/videos if helpful
- Note any configuration changes needed

---

## ** Additional Documentation

| Document | Purpose |
|----------|---------|
| [GENDER_DETECTION_SETUP.md](GENDER_DETECTION_SETUP.md) | Detailed gender detection comparison |
| [RASPBERRY_PI_HAILO_GUIDE.md](RASPBERRY_PI_HAILO_GUIDE.md) | Full Hailo-8 hardware setup |
| [QUICKSTART_PI.md](QUICKSTART_PI.md) | Quick Pi deployment |
| [PRODUCTION_CONFIG.md](PRODUCTION_CONFIG.md) | Production optimization |

---

## * Quick Reference - Testing Priority

### ⭐ Priority 1: Basic Functionality
```bash
python -m src.main --source test_video.mp4
```
- [x] Does it detect faces?
- [x] Does it show age/gender?
- [x] Does it create log files?

### ⭐⭐ Priority 2: Gender Detection (CRITICAL)
```bash
python test_unified_hailo_insightface.py --video test_video.mp4
```
- [x] Does it classify gender accurately?
- [x] What's the FPS?
- [x] How does it perform on side profiles?

### ⭐⭐⭐ Priority 3: Compare Classifiers
```bash
# Test different classifiers and compare
python test_unified_hailo_caffe.py --video cctv.mp4
python test_unified_hailo_insightface.py --video cctv.mp4
python test_unified_hailo_deepface.py --video cctv.mp4
```
- [x] Which is faster?
- [x] Which is more accurate?
- [x] Which works best for CCTV footage?

---

## 📄 License & Credits

### Pre-trained Models
- **Face Detection:** OpenCV DNN (BSD License)
- **Age/Gender:** Adience Dataset (Academic use)
- **MTCNN:** Joint Face Detection and Alignment
- **DeGirum:** Commercial API (Free tier: 100k inferences/month)

### Hardware Optimization
- **Hailo-8:** AI acceleration for edge devices
- Optimized for Raspberry Pi 5

---

**For questions or issues, check the documentation files or create a test report.**

**Ready to deploy? See [QUICKSTART_PI.md](QUICKSTART_PI.md) for Raspberry Pi setup!**
