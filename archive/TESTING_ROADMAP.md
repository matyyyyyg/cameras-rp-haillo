# * Testing Roadmap - Visual Guide

**Follow this flowchart to test the CCTV analytics system**

---

## 📍 Where Are You?

```
┌─────────────────────────────────────────────────────────┐
│  START: I just received this project                    │
│  ↓                                                       │
│  Read: QUICK_START_CHECKLIST.md (5 min)                │
│  Do: pip install -r requirements.txt                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Verify Everything Works                        │
│  ↓                                                       │
│  Run: python -m src.main --source 0                     │
│  Expected: Webcam opens with face detection             │
│                                                          │
│  [x] Works? → Continue to Step 2                         │
│  [!] Fails? → Check troubleshooting in README.md         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Test with Video File                           │
│  ↓                                                       │
│  Run: python -m src.main --source uploads/video.mp4     │
│  Check: FPS, detection accuracy, log files created      │
│                                                          │
│  Record: FPS = _____, Accuracy = Good/Fair/Poor         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  DECISION POINT: Do you have Raspberry Pi 5 + Hailo?    │
│                                                          │
│  YES → Continue to Step 3A (Hardware Testing)           │
│  NO  → Continue to Step 3B (Software Testing)           │
└─────────────────────────────────────────────────────────┘
         ↓ YES                              ↓ NO
         │                                  │
         ↓                                  ↓
┌──────────────────────┐         ┌──────────────────────┐
│  STEP 3A: Hardware   │         │  STEP 3B: Software   │
│  ↓                   │         │  ↓                   │
│  Install DeGirum SDK │         │  Install DeGirum SDK │
│  Get API token       │         │  Get API token       │
│  Test Option A       │         │  Test Option A       │
│  (Hybrid mode)       │         │  (Cloud mode)        │
│                      │         │                      │
│  Expected:           │         │  Expected:           │
│  25-30 FPS          │         │  15-20 FPS          │
│  90-95% accuracy    │         │  90-95% accuracy    │
└──────────────────────┘         └──────────────────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4: Gender Detection Testing (CRITICAL)             │
│  ↓                                                       │
│  Test all options and compare:                          │
│                                                          │
│  ┌─────────────────────────────────────────────┐       │
│  │ Option A (Hybrid)                            │       │
│  │ Run: python option_a_hailo_degirum.py       │       │
│  │ Record: FPS, accuracy, side profile handling│       │
│  └─────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────┐       │
│  │ Option B (Full DeGirum)                      │       │
│  │ Run: python option_b_degirum_full.py        │       │
│  │ Record: FPS, accuracy, CCTV performance     │       │
│  └─────────────────────────────────────────────┘       │
│  ┌─────────────────────────────────────────────┐       │
│  │ Baseline (Caffe)                             │       │
│  │ Run: python test_gender_caffe.py            │       │
│  │ Record: FPS, accuracy                        │       │
│  └─────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 5: Compare Results                                 │
│  ↓                                                       │
│  Fill comparison table:                                  │
│                                                          │
│  | Option | FPS | Accuracy | Side Profiles | Best For? |│
│  |--------|-----|----------|---------------|-----------|│
│  | A      |     |          |               |           |│
│  | B      |     |          |               |           |│
│  | Caffe  |     |          |               |           |│
│                                                          │
│  Decision: Which performs best for YOUR use case?       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 6: Edge Case Testing                              │
│  ↓                                                       │
│  Test challenging scenarios:                             │
│  ✓ Side profiles (45-90 degree angles)                  │
│  ✓ Poor lighting / night footage                        │
│  ✓ Multiple people in frame                             │
│  ✓ Motion blur                                          │
│  ✓ Small/distant faces                                  │
│                                                          │
│  Record: Which option handles each best?                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 7: Create Test Report                             │
│  ↓                                                       │
│  Use template in TEAM_TESTING_GUIDE.md                  │
│  Include:                                                │
│  • Performance metrics                                   │
│  • Comparison results                                    │
│  • Issues found                                          │
│  • Your recommendation                                   │
│                                                          │
│  Save as: TEST_REPORT_[YourName]_[Date].md             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  FINAL DECISION                                          │
│  ↓                                                       │
│  Based on your testing, which configuration is best?    │
│                                                          │
│  For Speed:    Option A + OpenCV backend                │
│  For Accuracy: Option B + MTCNN backend                 │
│  For Offline:  Caffe baseline                           │
│  For Balance:  Option A + MTCNN backend ⭐ RECOMMENDED  │
│                                                          │
│  Document your choice and reasoning in test report      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  DONE! Share results with team                          │
│                                                          │
│  Next steps:                                             │
│  • Deploy chosen option to production                    │
│  • See QUICKSTART_PI.md for Raspberry Pi deployment     │
│  • See PRODUCTION_CONFIG.md for optimization            │
└─────────────────────────────────────────────────────────┘
```

---

## 🚦 Quick Decision Tree

### "What should I test first?"

```
Do you have < 1 hour?
  ├─ YES → Test basic pipeline + Option A only
  └─ NO  → Follow full TEAM_TESTING_GUIDE.md

Is speed more important than accuracy?
  ├─ YES → Focus on Option A + OpenCV backend
  └─ NO  → Focus on Option B + MTCNN backend

Do you have Raspberry Pi 5 + Hailo?
  ├─ YES → Test hardware-accelerated options
  └─ NO  → Test software options first

Do you have internet connection?
  ├─ YES → Test DeGirum options (A & B)
  └─ NO  → Test offline options (Caffe, InsightFace)
```

---

## 📊 Expected Test Timeline

```
Quick Test (1 hour):
├─ Setup (15 min)
├─ Basic pipeline test (15 min)
├─ Option A test (20 min)
└─ Quick report (10 min)

Standard Test (2-3 hours):
├─ Setup (15 min)
├─ Basic pipeline (30 min)
├─ All gender options (60 min)
├─ Edge cases (30 min)
└─ Detailed report (15 min)

Comprehensive Test (4+ hours):
├─ Everything above
├─ Hardware testing
├─ All baselines
├─ Multiple test videos
└─ Production deployment test
```

---

## * Success Criteria

### Minimum Requirements:
- [x] Basic pipeline runs without errors
- [x] Face detection works on test video
- [x] At least one gender option tested
- [x] FPS and accuracy metrics recorded
- [x] Test report created

### Good Testing:
- [x] All of above +
- [x] Both Option A and B tested
- [x] Comparison table filled out
- [x] Edge cases tested
- [x] Clear recommendation made

### Excellent Testing:
- [x] All of above +
- [x] All baseline models tested
- [x] Multiple test videos used
- [x] Screenshots/videos recorded
- [x] Issues documented with reproduction steps
- [x] Production deployment tested

---

## 🔄 Testing Loop

```
For each gender detection option:
  1. Run on test video
  2. Record metrics:
     - FPS
     - Face detection count
     - Gender accuracy
     - Success rate
  3. Test edge cases
  4. Note strengths/weaknesses
  5. Move to next option
  
After all options tested:
  1. Compare results
  2. Choose best option
  3. Test chosen option thoroughly
  4. Document in report
```

---

## 📝 Quick Commands Reference

```bash
# Basic pipeline
python -m src.main --source video.mp4

# Gender Option A (Hybrid - Fast)
python option_a_hailo_degirum.py --video video.mp4

# Gender Option B (DeGirum - Accurate)
python option_b_degirum_full.py --video video.mp4

# Baseline Caffe (Offline)
python test_gender_caffe.py video.mp4

# With different backend
python -m src.main --source video.mp4 --backend mtcnn
python -m src.main --source video.mp4 --backend opencv
python -m src.main --source video.mp4 --backend multiscale

# Web dashboard
python run_dashboard.py
# Then open: http://localhost:5000
```

---

## 🆘 When Things Go Wrong

```
ERROR encountered?
  ├─ Check error message
  ├─ Search in README.md troubleshooting
  ├─ Check TEAM_TESTING_GUIDE.md common issues
  ├─ Verify dependencies installed
  └─ Document in test report

Low FPS?
  ├─ Try different backend (opencv is fastest)
  ├─ Reduce video resolution
  ├─ Close other applications
  └─ Check hardware specs

No detections?
  ├─ Verify video file plays
  ├─ Try different confidence threshold
  ├─ Try different backend
  └─ Check models directory exists

Gender prediction fails?
  ├─ Verify DeGirum SDK installed
  ├─ Check env.ini has valid token
  ├─ Try baseline (Caffe) instead
  └─ Verify internet connection (for cloud)
```

---

## 🎓 Learning Path

### Beginner:
1. Read QUICK_START_CHECKLIST.md
2. Run basic tests
3. Follow TEAM_TESTING_GUIDE.md

### Intermediate:
1. Understand different backends
2. Compare all gender options
3. Test edge cases
4. Optimize for your use case

### Advanced:
1. Read all documentation
2. Deploy to Raspberry Pi
3. Optimize for production
4. Contribute improvements

---

## 📍 Where to Get Help

```
Question about...
├─ Setup → QUICK_START_CHECKLIST.md
├─ Testing → TEAM_TESTING_GUIDE.md
├─ Gender models → GENDER_DETECTION_SETUP.md
├─ Commands → README.md
├─ Raspberry Pi → RASPBERRY_PI_HAILO_GUIDE.md
├─ Deployment → QUICKSTART_PI.md
└─ Files → PROJECT_ORGANIZATION.md
```

---

**Follow this roadmap and you'll successfully test the entire system! ***

**Start here: [QUICK_START_CHECKLIST.md](QUICK_START_CHECKLIST.md)**
