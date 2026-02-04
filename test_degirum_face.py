#!/usr/bin/env python3
"""
DeGirum Face Detection Test Script

Tests YOLOv8-face model on local Hailo hardware using DeGirum PySDK.
Run this first to verify DeGirum + Hailo setup is working.

Usage:
    python test_degirum_face.py                    # Use camera
    python test_degirum_face.py --input video.mp4 # Use video file
    python test_degirum_face.py --image photo.jpg # Single image
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


def list_available_models(zoo, filter_text="face"):
    """List available models in the zoo matching filter."""
    print(f"\nAvailable models containing '{filter_text}':")
    print("-" * 60)
    try:
        models = zoo.list_models()
        matching = [m for m in models if filter_text.lower() in m.lower()]
        for m in sorted(matching)[:20]:  # Limit to 20
            print(f"  {m}")
        if len(matching) > 20:
            print(f"  ... and {len(matching) - 20} more")
        return matching
    except Exception as e:
        print(f"  Error listing models: {e}")
        return []


def test_face_detection_image(model, image_path: str):
    """Test face detection on a single image."""
    print(f"\nTesting on image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    # Run inference
    start = time.time()
    result = model(image)
    elapsed = time.time() - start

    print(f"Inference time: {elapsed*1000:.1f}ms")
    print(f"Detected faces: {len(result.results)}")

    # Draw results
    for i, det in enumerate(result.results):
        bbox = det.get('bbox', [0, 0, 0, 0])
        conf = det.get('score', 0)
        x1, y1, x2, y2 = map(int, bbox)

        print(f"  Face {i+1}: bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("DeGirum Face Detection", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_face_detection_video(model, source, output_video=None, display=True):
    """Test face detection on video/camera stream."""
    if source == "camera":
        print("\nOpening camera...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"\nOpening video: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    # Setup video writer if output requested
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"Saving output to: {output_video}")

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

        # Run inference
        result = model(frame)

        # Draw results
        for det in result.results:
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('score', 0)
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate FPS
        frame_time = time.time() - frame_start
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / frame_time) if frame_time > 0 else fps_display

        # Draw stats
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(result.results)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "DeGirum + Hailo", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Write frame to output video
        if video_writer:
            video_writer.write(frame)

        # Display if requested
        if display:
            cv2.imshow("DeGirum Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print progress
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}: {len(result.results)} faces, FPS: {fps_display:.1f}")

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\nVideo saved to: {output_video}")
    if display:
        cv2.destroyAllWindows()

    # Summary
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="DeGirum Face Detection Test")
    parser.add_argument('--input', type=str, default='camera',
                        help='Input source: camera, video file path')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to process')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Save annotated output video to this path')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display window (useful for headless/SSH)')
    parser.add_argument('--model', type=str,
                        default='yolov8n_relu6_face--640x640_quant_hailort_hailo8_1',
                        help='Model name from DeGirum zoo (use hailo8 for Hailo-8, hailo8l for Hailo-8L)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available face models and exit')
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
        print("\nTroubleshooting:")
        print("  1. Ensure Hailo device is connected: lsmod | grep hailo")
        print("  2. Ensure HailoRT is installed: dpkg -l | grep hailo")
        print("  3. Try: pip install --upgrade degirum degirum_tools")
        return 1

    # List models if requested
    if args.list_models:
        list_available_models(zoo, "face")
        return 0

    # Load face detection model
    print(f"\nLoading model: {args.model}")
    try:
        model = zoo.load_model(args.model)
        print(f"Model loaded successfully")
        print(f"  Input size: {model.input_shape}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("\nAvailable face models:")
        list_available_models(zoo, "face")
        return 1

    # Run test
    if args.image:
        test_face_detection_image(model, args.image)
    else:
        test_face_detection_video(
            model,
            args.input,
            output_video=args.output_video,
            display=not args.no_display
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
