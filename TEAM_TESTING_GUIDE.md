# Team Testing Guide

**CCTV Face Analytics System - Unified Hailo Testing**

## Testing Objectives

Your mission is to:
1. Test the unified Hailo-8 RetinaFace detection pipeline
2. Compare 4 different gender classification models
3. Find the best configuration for CCTV footage
4. Report performance metrics and issues

**Estimated Testing Time:** 2-3 hours

---

## Before You Start

### What You Need

- **Hardware:** Raspberry Pi 5 + Hailo-8 (required) OR any computer for CPU baseline only
- **Test Video:** CCTV footage or sample video with multiple people
- **Internet:** Required for DeGirum cloud model only
- **Python:** Version 3.9, 3.10, or 3.11

### Quick System Check

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Verify Hailo device (Pi 5 only)
hailortcli fw-control identify
lsmod | grep hailo

# Check project structure
ls -la
# Should see: src/, models/, test_unified_hailo_*.py scripts
```

---

## Phase 1: Basic Setup (15 minutes)

### Step 1: Install Dependencies

```bash
cd "/Users/krishanu8219/Documents/Face detection Raspberry"

# Install core dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy; print('Core dependencies OK')"
```

### Step 2: Verify Models

```bash
# Check models
ls -la models/
ls -la models/hailo/

# Should see:
# - gender_deploy.prototxt, gender_net.caffemodel (Caffe)
# - age_deploy.prototxt, age_net.caffemodel (Age)
# - hailo/retinaface_mobilenet_v1.hef (Hailo face detection)

# If missing Hailo model:
bash download_hailo_models.sh
```

### Step 3: Quick Smoke Test

```bash
# Test fastest setup (Hailo + Caffe)
python test_unified_hailo_caffe.py --input camera --display

# CPU-only fallback (no Hailo needed)
python test_gender_insightface.py --video test_video.mp4
```

**Checkpoint:** If you see face detection with gender labels, proceed!

---

## Phase 2: Unified Hailo Testing (30 minutes)

**Important:** All 4 Hailo scripts use the same RetinaFace face detector (25-30 FPS). We're comparing gender classifiers only.

### Test 1: Caffe (Fastest)

**Expected:** 18-22 FPS, 85-90% accuracy

```bash
python test_unified_hailo_caffe.py --video your_video.mp4 --display
```

**Record:**
- FPS: _______
- Gender accuracy: Good / Fair / Poor
- Issues: _______

### Test 2: InsightFace (Best Balance - RECOMMENDED)

**Expected:** 12-16 FPS, 95-97% accuracy, includes age estimation

```bash
pip install insightface onnxruntime
python test_unified_hailo_insightface.py --video your_video.mp4 --display
```

**Record:**
- FPS: _______
- Gender accuracy: _______
- Age accuracy: _______
- Issues: _______

### Test 3: DeepFace (Most Accurate)

**Expected:** 8-12 FPS, 93-95% accuracy

```bash
pip install deepface tf-keras
python test_unified_hailo_deepface.py --video your_video.mp4 --display
```

**Record:**
- FPS: _______
- Gender accuracy: _______
- Worth speed trade-off: Yes / No

### Test 4: DeGirum (Cloud Option)

**Expected:** 12-15 FPS, 90-95% accuracy

```bash
# First-time setup
pip install degirum degirum-tools

# Get API key from https://cs.degirum.com (free)
cat > env.ini << EOF
[DEFAULT]
DEGIRUM_CLOUD_TOKEN = your_token_here
EOF

# Run test
python test_unified_hailo_degirum.py --video your_video.mp4 --display
```

**Record:**
- FPS: _______
- Gender accuracy: _______
- Internet dependency OK: Yes / No

### CPU Baseline (No Hailo)

```bash
python test_gender_insightface.py --video your_video.mp4
```

**Expected:** 3-6 FPS (shows Hailo benefit)

---

## Phase 3: Comparison & Decision (20 minutes)

### Performance Matrix

| Classifier | FPS | Accuracy | Age | Setup | Internet |
|-----------|-----|----------|-----|-------|----------|
| Hailo + Caffe |  | 85-90% | No | Easy | No |
| Hailo + InsightFace |  | 95-97% | Yes | Medium | No |
| Hailo + DeepFace |  | 93-95% | No | Medium | No |
| Hailo + DeGirum |  | 90-95% | No | Medium | Yes |
| CPU InsightFace |  | 95-97% | Yes | Easy | No |

### Decision Guide

**Choose Caffe if:**
- Speed is critical (18-22 FPS)
- 85-90% accuracy acceptable
- Simplest setup needed

**Choose InsightFace if:** (RECOMMENDED)
- Want best balance (12-16 FPS)
- Need 95%+ accuracy
- Want age estimation

**Choose DeepFace if:**
- Maximum accuracy needed
- Can accept 8-12 FPS
- Don't need age

**Choose DeGirum if:**
- Internet always available
- Want cloud updates
- 12-15 FPS acceptable

**Your Recommendation:**
```
Based on testing, I recommend: _______________

Reasons:
1. _________________________________
2. _________________________________
3. _________________________________
```

---

## Phase 4: Edge Case Testing (30 minutes)

Test challenging scenarios:

### Scenario 1: Side Profiles
```bash
python test_unified_hailo_insightface.py --video side_profile.mp4
```
**Result:** Detected / Not Detected / Partial

### Scenario 2: Poor Lighting
```bash
python test_unified_hailo_insightface.py --video night_video.mp4
```
**Result:** Good / Fair / Poor

### Scenario 3: Multiple People
```bash
python test_unified_hailo_insightface.py --video crowd.mp4
```
**Result:** All detected / Some missed / Confusion

### Scenario 4: Small/Distant Faces
```bash
python test_unified_hailo_insightface.py --video wide_angle.mp4
```
**Result:** Detected / Missed / Partial

---

## Phase 5: Create Test Report

Save as: `TEST_REPORT_[YourName]_[Date].md`

```markdown
# CCTV Analytics Test Report

## Tester Information
- Name: [Your Name]
- Date: [2026-01-07]
- Hardware: [Pi 5 + Hailo-8 / Other]
- Test Duration: [X hours]

## Test Videos
1. [video1.mp4] - [Description]
2. [video2.mp4] - [Description]

## Results Summary

### Unified Hailo Tests

#### Caffe
- FPS: [X]
- Accuracy: [X%]
- Notes: [...]

#### InsightFace
- FPS: [X]
- Accuracy: [X%]
- Age accuracy: [X%]
- Notes: [...]

#### DeepFace
- FPS: [X]
- Accuracy: [X%]
- Notes: [...]

#### DeGirum
- FPS: [X]
- Accuracy: [X%]
- Notes: [...]

## Edge Cases
- Side profiles: [Result]
- Poor lighting: [Result]
- Multiple people: [Result]
- Small faces: [Result]

## Final Recommendation

**Recommended Configuration:**
- Classifier: [Caffe / InsightFace / DeepFace / DeGirum]
- Expected FPS: [X]
- Expected Accuracy: [X%]
- Hardware: Pi 5 + Hailo-8
- Internet: [Required / Not Required]

**Reasons:**
[Your reasoning]

## Issues Found
1. [Issue description]
   - Severity: Critical / Major / Minor
   - Reproduction: [Steps]
```

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install [module_name]
# Or reinstall all:
pip install -r requirements.txt
```

### "Cannot open video"
```bash
# Check file exists
ls -la your_video.mp4

# Try absolute path
python test_unified_hailo_caffe.py --video /full/path/to/video.mp4
```

### "Hailo device not found"
```bash
# Check Hailo
hailortcli fw-control identify
lsmod | grep hailo

# If not found:
sudo apt install hailo-all
sudo reboot
```

### "DeGirum authentication failed"
```bash
# Check env.ini
cat env.ini
# Token should be: DEGIRUM_CLOUD_TOKEN = dg_xxxxxx
```

### Low FPS
```bash
# Try fastest option
python test_unified_hailo_caffe.py --video your_video.mp4

# Check Hailo is active (should see "Hailo" in logs)
```

---

## Testing Checklist

### Must Do:
- [ ] Install dependencies
- [ ] Test Hailo + Caffe (fastest)
- [ ] Test Hailo + InsightFace (best balance)
- [ ] Compare all 4 classifiers
- [ ] Create test report

### Should Do:
- [ ] Test edge cases
- [ ] Test CPU baseline for comparison
- [ ] Record performance metrics

### Nice to Have:
- [ ] Test with multiple videos
- [ ] Test RTSP stream
- [ ] Create comparison charts

---

## Need Help?

**Documentation:**
- [README.md](README.md) - Main guide
- [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) - Complete testing guide
- [RASPBERRY_PI_HAILO_GUIDE.md](RASPBERRY_PI_HAILO_GUIDE.md) - Hardware setup

**Common Issues:**
- Most issues = missing dependencies (`pip install -r requirements.txt`)
- No detection = check Hailo device (`hailortcli fw-control identify`)
- Low FPS = use Caffe classifier

---

## After Testing

1. Save your test report in project root
2. Share key findings with team
3. Recommend best configuration
4. Document any issues

**Example Summary:**
```
QUICK SUMMARY:
- Best classifier: InsightFace (14 FPS, 96% accurate)
- Production ready: YES
- Issues found: 0
- Recommendation: Deploy Hailo + InsightFace on Pi 5
```

---

**Good luck with testing!**
