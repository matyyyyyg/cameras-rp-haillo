#!/usr/bin/env python3
"""
DeGirum Gender Classification Test Script

Tests YOLOv8-gender model on local Hailo hardware using DeGirum PySDK.
This script manually crops faces and runs gender classification.

Usage:
    python test_degirum_gender.py                    # Use camera
    python test_degirum_gender.py --input video.mp4 # Use video file
    python test_degirum_gender.py --image photo.jpg # Single image
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


def list_available_models(zoo, filter_text="gender"):
    """List available models in the zoo matching filter."""
    print(f"\nAvailable models containing '{filter_text}':")
    print("-" * 60)
    try:
        models = zoo.list_models()
        matching = [m for m in models if filter_text.lower() in m.lower()]
        for m in sorted(matching)[:20]:
            print(f"  {m}")
        return matching
    except Exception as e:
        print(f"  Error listing models: {e}")
        return []


def extract_face_crop(frame, bbox, padding=0.3):
    """Extract face crop with padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Add padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * padding)
    pad_h = int(box_h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    return frame[y1:y2, x1:x2].copy()


def test_gender_video(face_model, gender_model, source):
    """Test face detection + gender classification on video/camera."""
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

    frame_count = 0
    start_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start = time.time()

        # Step 1: Detect faces
        face_result = face_model(frame)

        # Step 2: Classify gender for each face
        for det in face_result.results:
            bbox = det.get('bbox', [0, 0, 0, 0])
            face_conf = det.get('score', 0)
            x1, y1, x2, y2 = map(int, bbox)

            # Extract face crop
            face_crop = extract_face_crop(frame, bbox)
            if face_crop.size == 0:
                continue

            # Run gender classification
            try:
                gender_result = gender_model(face_crop)
                if gender_result.results:
                    gender = gender_result.results[0].get('label', 'Unknown')
                    gender_conf = gender_result.results[0].get('score', 0)
                else:
                    gender = "Unknown"
                    gender_conf = 0
            except Exception as e:
                gender = "Error"
                gender_conf = 0

            # Color based on gender
            if gender.lower() == 'male':
                color = (255, 150, 0)  # Blue
            elif gender.lower() == 'female':
                color = (147, 20, 255)  # Pink
            else:
                color = (128, 128, 128)  # Gray

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{gender} ({gender_conf:.2f})"
            cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*10, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculate FPS
        frame_time = time.time() - frame_start
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / frame_time) if frame_time > 0 else fps_display

        # Draw stats
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(face_result.results)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "DeGirum Face+Gender", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("DeGirum Gender Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")


def test_gender_image(face_model, gender_model, image_path):
    """Test on single image."""
    print(f"\nTesting on image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    # Detect faces
    start = time.time()
    face_result = face_model(image)
    face_time = time.time() - start

    print(f"Face detection: {face_time*1000:.1f}ms, found {len(face_result.results)} faces")

    # Classify each face
    for i, det in enumerate(face_result.results):
        bbox = det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)

        face_crop = extract_face_crop(image, bbox)

        start = time.time()
        gender_result = gender_model(face_crop)
        gender_time = time.time() - start

        if gender_result.results:
            gender = gender_result.results[0].get('label', 'Unknown')
            gender_conf = gender_result.results[0].get('score', 0)
        else:
            gender = "Unknown"
            gender_conf = 0

        print(f"  Face {i+1}: {gender} (conf={gender_conf:.2f}) - {gender_time*1000:.1f}ms")

        # Draw
        color = (255, 150, 0) if gender.lower() == 'male' else (147, 20, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{gender} ({gender_conf:.2f})"
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("DeGirum Gender Classification", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="DeGirum Gender Classification Test")
    parser.add_argument('--input', type=str, default='camera',
                        help='Input source: camera, video file path')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to process')
    parser.add_argument('--face-model', type=str,
                        default='yolov8n_relu6_face--640x640_quant_hailort_hailo8_1',
                        help='Face detection model name')
    parser.add_argument('--gender-model', type=str,
                        default='yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8_1',
                        help='Gender classification model name')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--cloud', action='store_true',
                        help='Use cloud inference instead of local Hailo')
    parser.add_argument('--token', type=str, default='',
                        help='DeGirum cloud token (required for --cloud)')
    args = parser.parse_args()

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
        print(f"ERROR: Failed to connect to DeGirum: {e}")
        return 1

    # List models if requested
    if args.list_models:
        list_available_models(zoo, "face")
        list_available_models(zoo, "gender")
        list_available_models(zoo, "age")
        return 0

    # Load models
    print(f"\nLoading face model: {args.face_model}")
    try:
        face_model = zoo.load_model(args.face_model)
        print("Face model loaded")
    except Exception as e:
        print(f"ERROR: Failed to load face model: {e}")
        list_available_models(zoo, "face")
        return 1

    print(f"\nLoading gender model: {args.gender_model}")
    try:
        gender_model = zoo.load_model(args.gender_model)
        print("Gender model loaded")
    except Exception as e:
        print(f"ERROR: Failed to load gender model: {e}")
        list_available_models(zoo, "gender")
        return 1

    # Run test
    if args.image:
        test_gender_image(face_model, gender_model, args.image)
    else:
        test_gender_video(face_model, gender_model, args.input)

    return 0


if __name__ == "__main__":
    sys.exit(main())
