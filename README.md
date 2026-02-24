# Face Detection System for Raspberry Pi

A real-time face detection system with gender/age classification and person tracking. Built for my university project using Raspberry Pi 5 with Hailo-8 AI accelerator.

## What it does

- Detects faces in video/camera feed using Hailo-8 hardware acceleration
- Estimates gender and age using ONNX MobileNet model
- Tracks people across frames with Kalman filter (same person keeps same ID)
- Quality gate filters out blurry, dark, or tiny faces
- Outputs structured JSON for downstream analytics

## Hardware

- Raspberry Pi 5
- Hailo-8 AI HAT (for fast face detection)
- Pi Camera or USB camera (or video file)

## Quick Setup

```bash
cd face_analysis_NEW

# Install dependencies
pip install -r requirements.txt

# Download age/gender ONNX model
bash scripts/download_models.sh

# Run with Pi Camera
python main.py --display

# Run with video file
python main.py --input video.mp4 --display

# Run with custom confidence and logging
python main.py --display --face-conf 0.3 --log detections.jsonl

# Save annotated video + periodic snapshots
python main.py --display --output-video out.mp4 --snapshot-dir snapshots/
```

## Main Files

| File | What it does |
|------|--------------|
| `face_analysis_NEW/main.py` | Main script - run this one |
| `src/detection/hailo_detector.py` | Hailo-8 face detection (RetinaFace/SCRFD) |
| `src/classification/classifier.py` | Age/gender classification (ONNX MobileNet) |
| `src/kalman_tracking/tracker.py` | Kalman filter tracking + Hungarian assignment |
| `src/utils/face_crop.py` | Face cropping + quality gate |
| `src/utils/types.py` | Shared data types |
| `src/utils/json_logger.py` | JSONL output logger |

## Command Line Options

```bash
python face_analysis_NEW/main.py [options]

--input             Video file or 'camera' (default: camera)
--display           Show video window
--face-conf         Face detection threshold (default: 0.5)
--sensor-id         Camera name for logs (default: SENSOR_001)
--log               Save detections to JSONL file
--output-video      Save annotated video to file
--snapshot-dir      Save periodic snapshot images
--snapshot-interval Seconds between snapshots (default: 60)
--resolution        Camera resolution (default: 1280x960)
--max-age           Frames before dropping lost track (default: 60)
--min-hits          Confirmations before showing track (default: 3)
--iou-threshold     Tracking IoU threshold (default: 0.15)
--no-ids            Hide track IDs in display
```

## How it Works

1. **Face Detection** - Hailo-8 runs RetinaFace/SCRFD model with hardware acceleration
2. **Quality Gate** - Filters out blurry, dark, or tiny face crops
3. **Classification** - ONNX MobileNet estimates age (101-bin softmax) and gender (sigmoid)
4. **Tracking** - Kalman filter + Hungarian algorithm tracks people across frames

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
face_analysis_NEW/
├── main.py                        # Pipeline orchestrator
├── requirements.txt
├── scripts/
│   └── download_models.sh         # Download ONNX model
├── models/
│   └── hailo/                     # Hailo HEF model files
└── src/
    ├── detection/
    │   └── hailo_detector.py      # Hailo face detection
    ├── classification/
    │   └── classifier.py          # ONNX age/gender
    ├── kalman_tracking/
    │   └── tracker.py             # Kalman + Hungarian tracking
    └── utils/
        ├── types.py               # Shared data types
        ├── face_crop.py           # Face crop + quality gate
        └── json_logger.py         # JSONL logger
```

## Dependencies

- OpenCV
- NumPy
- ONNX Runtime
- HailoRT (for Raspberry Pi)
- SciPy (optional, for Hungarian assignment)


