# Face Detection System for Raspberry Pi

A real-time face detection system with gender/age classification and person tracking. Built for my university project using Raspberry Pi 5 with Hailo-8 AI accelerator.

## What it does

- Detects faces in video/camera feed
- Estimates gender and age
- Tracks people across frames (same person keeps same ID)
- Re-identifies people when they leave and come back (ReID feature)

## Hardware

- Raspberry Pi 5
- Hailo-8 AI HAT (for fast face detection)
- Camera or video file

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download the Hailo face detection model
bash download_hailo_models.sh

# Run with camera
python test_unified_hailo_insightface.py --display

# Run with video file
python test_unified_hailo_insightface.py --input video.mp4 --display
```

## Main Files

| File | What it does |
|------|--------------|
| `test_unified_hailo_insightface.py` | Main script - run this one |
| `src/unified_hailo_face.py` | Hailo face detection |
| `src/classification.py` | Gender/age classification (InsightFace) |
| `src/kalman_tracker.py` | Person tracking + ReID |

## Command Line Options

```bash
python test_unified_hailo_insightface.py [options]

--input         Video file or 'camera' (default: camera)
--display       Show video window
--face-conf     Face detection threshold (default: 0.4)
--sensor-id     Camera name for logs (default: SENSOR_001)
--log           Save detections to JSON file
```

## How it Works

1. **Face Detection** - Hailo-8 runs RetinaFace model (~30 FPS)
2. **Classification** - InsightFace extracts gender, age, and face embedding
3. **Tracking** - Kalman filter tracks faces across frames
4. **ReID** - When someone leaves and returns, they get the same ID back

## ReID (Re-Identification)

The system remembers people for 10 minutes after they leave the frame. When they come back, it matches their face embedding and gives them the same ID.

Settings in `src/kalman_tracker.py`:
```python
REID_SIMILARITY_THRESHOLD = 0.4   # How similar faces need to be (0-1)
GALLERY_MAX_AGE_SECONDS = 600     # Remember for 10 minutes
```

## Output Example

```json
{
  "sensor_id": "SENSOR_001",
  "timestamp": "2026-02-05 14:30:45.123",
  "detections": [
    {
      "id": 1,
      "age": 25.3,
      "gender": "male",
      "confidence": 0.92,
      "bbox": {"xc": 320, "yc": 240, "width": 100, "height": 120}
    }
  ]
}
```

## Troubleshooting

**No faces detected:**
- Lower confidence: `--face-conf 0.3`
- Check lighting
- Make sure face is visible to camera

**Hailo not found:**
```bash
sudo apt install hailo-all
lsmod | grep hailo
```

**Low FPS:**
- Normal on CPU-only mode
- Hailo should give 25-30 FPS

## Project Structure

```
Face detection Raspberry/
├── src/
│   ├── unified_hailo_face.py    # Hailo detector
│   ├── classification.py        # Gender/age
│   └── kalman_tracker.py        # Tracking + ReID
├── models/
│   └── hailo/                   # Hailo model files
├── test_unified_hailo_insightface.py  # Main script
└── requirements.txt
```

## Dependencies

- OpenCV
- NumPy
- InsightFace
- HailoRT (for Raspberry Pi)

---

Made for university project - face analytics on edge devices.
