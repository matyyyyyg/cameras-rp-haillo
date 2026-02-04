#!/usr/bin/env python3
"""
DeGirum Full Pipeline Test Script

Tests Face Detection + Gender + Age using DeGirum's compound model pipelining.
This is the recommended approach for production - runs entirely on Hailo hardware.

Usage:
    python test_degirum_pipeline.py                    # Use camera
    python test_degirum_pipeline.py --input video.mp4 # Use video file
    python test_degirum_pipeline.py --image photo.jpg # Single image
"""

import argparse
import time
import sys

try:
    import degirum as dg
    import degirum_tools
except ImportError:
    print("ERROR: DeGirum PySDK not installed")
    print("Install with: pip install degirum degirum_tools")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not installed")
    print("Install with: pip install opencv-python")
    sys.exit(1)


def list_all_models(zoo):
    """List all relevant models in the zoo."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS IN DEGIRUM ZOO")
    print("=" * 60)

    for category in ["face", "gender", "age"]:
        print(f"\n{category.upper()} MODELS:")
        print("-" * 40)
        try:
            models = zoo.list_models()
            matching = [m for m in models if category.lower() in m.lower()]
            for m in sorted(matching)[:10]:
                print(f"  {m}")
            if len(matching) > 10:
                print(f"  ... and {len(matching) - 10} more")
            if not matching:
                print("  (none found)")
        except Exception as e:
            print(f"  Error: {e}")


def create_compound_model(zoo, face_model_name, gender_model_name, age_model_name=None):
    """
    Create a compound model that pipelines face detection with gender/age classification.

    The compound model:
    1. Runs face detection
    2. Crops each detected face
    3. Runs gender classification on each crop
    4. (Optionally) Runs age estimation on each crop
    """
    print("\nCreating compound pipeline model...")

    # Load individual models
    print(f"  Loading face model: {face_model_name}")
    face_model = zoo.load_model(face_model_name)

    print(f"  Loading gender model: {gender_model_name}")
    gender_model = zoo.load_model(gender_model_name)

    age_model = None
    if age_model_name:
        print(f"  Loading age model: {age_model_name}")
        try:
            age_model = zoo.load_model(age_model_name)
        except Exception as e:
            print(f"  Warning: Could not load age model: {e}")
            age_model = None

    # Create compound model using DeGirum tools
    # CroppingAndClassifyingCompoundModel:
    #   - Takes detector output (face boxes)
    #   - Crops each detection with padding
    #   - Runs classifier on each crop
    print("  Building compound model pipeline...")

    try:
        compound_model = degirum_tools.CroppingAndClassifyingCompoundModel(
            face_model,
            gender_model,
            crop_extent=0.3  # 30% padding around face
        )
        print("  Compound model created successfully")
        return compound_model, age_model
    except Exception as e:
        print(f"  ERROR creating compound model: {e}")
        print("  Falling back to manual pipeline...")
        return None, None


def run_manual_pipeline(face_model, gender_model, age_model, frame):
    """
    Manual pipeline when compound model is not available.
    Returns list of detections with gender/age.
    """
    results = []

    # Step 1: Detect faces
    face_result = face_model(frame)

    for det in face_result.results:
        bbox = det.get('bbox', [0, 0, 0, 0])
        face_conf = det.get('score', 0)
        x1, y1, x2, y2 = map(int, bbox)

        # Extract face crop with padding
        h, w = frame.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1
        pad_w, pad_h = int(box_w * 0.3), int(box_h * 0.3)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        face_crop = frame[cy1:cy2, cx1:cx2]

        if face_crop.size == 0:
            continue

        # Step 2: Gender classification
        gender = "Unknown"
        gender_conf = 0.0
        try:
            gender_result = gender_model(face_crop)
            if gender_result.results:
                gender = gender_result.results[0].get('label', 'Unknown')
                gender_conf = gender_result.results[0].get('score', 0)
        except:
            pass

        # Step 3: Age estimation (if available)
        age = 0
        age_conf = 0.0
        if age_model:
            try:
                age_result = age_model(face_crop)
                if age_result.results:
                    # Age models may return label like "25-32" or a number
                    age_label = age_result.results[0].get('label', '0')
                    age_conf = age_result.results[0].get('score', 0)
                    # Try to parse age
                    if '-' in str(age_label):
                        parts = str(age_label).split('-')
                        age = (int(parts[0]) + int(parts[1])) / 2
                    else:
                        age = float(age_label)
            except:
                pass

        results.append({
            'bbox': (x1, y1, x2, y2),
            'face_conf': face_conf,
            'gender': gender,
            'gender_conf': gender_conf,
            'age': age,
            'age_conf': age_conf
        })

    return results


def test_pipeline_video(zoo, args, source):
    """Test the full pipeline on video/camera."""

    # Try to create compound model
    compound_model, age_model = create_compound_model(
        zoo, args.face_model, args.gender_model, args.age_model
    )

    # If compound model failed, load models separately
    if compound_model is None:
        print("\nUsing manual pipeline mode...")
        face_model = zoo.load_model(args.face_model)
        gender_model = zoo.load_model(args.gender_model)
        try:
            age_model = zoo.load_model(args.age_model) if args.age_model else None
        except:
            age_model = None
        use_compound = False
    else:
        use_compound = True
        face_model = gender_model = None

    # Open video source
    if source == "camera":
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"\nOpening video: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    print("Processing... Press 'q' to quit")
    print(f"Mode: {'Compound Model' if use_compound else 'Manual Pipeline'}")

    frame_count = 0
    start_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start = time.time()

        # Run inference
        if use_compound:
            # Compound model returns combined results
            try:
                result = compound_model(frame)
                detections = []
                for det in result.results:
                    detections.append({
                        'bbox': tuple(map(int, det.get('bbox', [0,0,0,0]))),
                        'face_conf': det.get('score', 0),
                        'gender': det.get('label', 'Unknown'),
                        'gender_conf': det.get('score', 0),
                        'age': 0,
                        'age_conf': 0
                    })
            except Exception as e:
                detections = []
        else:
            # Manual pipeline
            detections = run_manual_pipeline(face_model, gender_model, age_model, frame)

        # Draw results
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            gender = det['gender']
            gender_conf = det['gender_conf']
            age = det['age']

            # Color by gender
            if 'male' in gender.lower():
                color = (255, 150, 0)  # Blue
            elif 'female' in gender.lower():
                color = (147, 20, 255)  # Pink
            else:
                color = (128, 128, 128)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            if age > 0:
                label = f"{gender} {int(age)}y ({gender_conf:.2f})"
            else:
                label = f"{gender} ({gender_conf:.2f})"

            cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*9, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculate FPS
        frame_time = time.time() - frame_start
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / frame_time) if frame_time > 0 else fps_display

        # Draw stats
        mode_text = "Compound" if use_compound else "Manual"
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"DeGirum Pipeline ({mode_text})", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("DeGirum Face+Gender+Age Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="DeGirum Full Pipeline Test (Face + Gender + Age)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with camera
    python test_degirum_pipeline.py

    # Test with video file
    python test_degirum_pipeline.py --input video.mp4

    # List available models
    python test_degirum_pipeline.py --list-models

    # Use cloud inference (requires token)
    python test_degirum_pipeline.py --cloud --token YOUR_TOKEN
        """
    )

    parser.add_argument('--input', type=str, default='camera',
                        help='Input source: camera or video file path')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to process')

    # Model selection
    parser.add_argument('--face-model', type=str,
                        default='yolov8n_relu6_face--640x640_quant_hailort_hailo8_1',
                        help='Face detection model')
    parser.add_argument('--gender-model', type=str,
                        default='yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8_1',
                        help='Gender classification model')
    parser.add_argument('--age-model', type=str,
                        default='arcface_gender_age--112x112_quant_hailort_hailo8_1',
                        help='Age estimation model (optional)')
    parser.add_argument('--no-age', action='store_true',
                        help='Disable age estimation')

    # Infrastructure
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--cloud', action='store_true',
                        help='Use cloud inference instead of local Hailo')
    parser.add_argument('--token', type=str, default='',
                        help='DeGirum cloud token')

    args = parser.parse_args()

    if args.no_age:
        args.age_model = None

    # Setup inference target
    if args.cloud:
        if not args.token:
            print("ERROR: --token required for cloud inference")
            return 1
        inference_host = "@cloud"
        token = args.token
        print("Using DeGirum CLOUD inference")
    else:
        inference_host = "@local"
        token = ""
        print("Using LOCAL Hailo inference")

    # Connect to model zoo
    print(f"\nConnecting to DeGirum model zoo...")
    try:
        zoo = dg.connect(inference_host, "degirum/hailo", token)
        print("Connected successfully")
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Check Hailo device: lsmod | grep hailo")
        print("  2. Check HailoRT: dpkg -l | grep hailo")
        print("  3. Update DeGirum: pip install --upgrade degirum degirum_tools")
        return 1

    # List models if requested
    if args.list_models:
        list_all_models(zoo)
        return 0

    # Run pipeline
    test_pipeline_video(zoo, args, args.input)

    return 0


if __name__ == "__main__":
    sys.exit(main())
