# CCTV Analytics System - Production Configuration

**Optimized for: Raspberry Pi 5 + Hailo-8L (26 TFLOPS)**

## Target KPIs

| Metric | Target |
|--------|--------|
| **Gender Accuracy** | ≥99% |
| **Age Tolerance** | ±5 years |
| **Detection** | No false positives |
| **Performance** | 30-40 FPS |

## Hardware Configuration

| Component | Specification | Performance |
|-----------|---------------|-------------|
| Board | Raspberry Pi 5 (4GB+) | - |
| AI Accelerator | Hailo-8L (26 TFLOPS) | 30-40 FPS |
| Detection Model | RetinaFace MobileNet | Hardware accelerated |
| Classification | InsightFace (CPU) | Real-time |

## Production Settings

### Detection
- **Backend**: Hailo RetinaFace (hardware accelerated)
- **Confidence**: 0.5 (catch all faces, filter by classification)
- **Resolution**: Full (1.0x - no downscaling)
- **Frame skip**: 1 (process every frame)

### Classification (99% Gender Accuracy)
- **Engine**: InsightFace (buffalo_l)
- **Detection size**: 800x800 (increased for better features)
- **Gender confidence threshold**: 0.85 (reject uncertain predictions)
- **Face padding**: 50% (more context)
- **Min face upscale**: 160px (InsightFace) / 120px (crop)
- **Fallback**: Caffe models (always loaded)

### Video Processing
- **Frame skip**: 1 (every frame)
- **Scale factor**: 1.0 (full resolution)
- **Playback speed**: 1.0x (real-time)
- **Cache TTL**: 45 seconds

## Expected Performance

| Metric | Production (Hailo-8L) |
|--------|----------------------|
| **FPS** | 30-40+ |
| **Detection latency** | <30ms |
| **Classification time** | ~50ms/face |
| **Power consumption** | ~15W total |

## Deployment

```bash
# Production system - Recommended (Best balance)
python test_unified_hailo_insightface.py --input camera --save-detections

# Fastest option
python test_unified_hailo_caffe.py --input camera --save-detections

# Most accurate option
python test_unified_hailo_deepface.py --input camera --save-detections
```

See [INSTALL.md](INSTALL.md) for full deployment steps.
