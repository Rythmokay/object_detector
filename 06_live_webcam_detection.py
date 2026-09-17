"""
06_live_webcam_detection.py
---------------------------
Live laptop webcam real-time object detection application.
Captures live feed from your integrated/external camera, performs real-time
YOLO object detection via OpenCV DNN, displays FPS, and supports live snapshots.
"""

import os
import sys
import time
import argparse
import importlib.util
from datetime import datetime
import cv2

# Dynamically import ObjectDetector from 02_object_detector.py
_detector_path = os.path.join(os.path.dirname(__file__), "02_object_detector.py")
_spec = importlib.util.spec_from_file_location("detector_module", _detector_path)
_detector_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detector_module)
ObjectDetector = _detector_module.ObjectDetector


def start_live_webcam(
    camera_id: int = 0,
    conf_threshold: float = 0.40,
    output_dir: str = "output",
):
    """
    Launch live laptop camera object detection stream.

    :param camera_id: System camera index (default 0 for main laptop camera).
    :param conf_threshold: Initial confidence threshold for filtering detections.
    :param output_dir: Directory to save webcam snapshots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print(" OpenCV Real-Time Live Webcam Object Detection ")
    print("=" * 65)
    print(f"[*] Initializing laptop camera (Device Index: {camera_id})...")

    # Initialize Object Detector
    try:
        detector = ObjectDetector(conf_threshold=conf_threshold)
    except Exception as e:
        print(f"[!] Error loading detection model: {e}")
        print("    Please run 'python 01_download_models.py' first.")
        sys.exit(1)

    # Open video capture from camera
    cap = cv2.VideoCapture(camera_id)

    # Fallback to backend flags if primary capture fails
    if not cap.isOpened():
        print(f"[!] Warning: Camera index {camera_id} failed with default backend. Retrying with AVFoundation/V4L2...")
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"\n[!] ERROR: Could not open laptop camera (Index {camera_id}).")
        print("    Please verify:")
        print("    1. Your laptop camera is connected and unblocked.")
        print("    2. Terminal/IDE has camera access permissions in System Settings.")
        print("    3. If using an external USB webcam, try running with: python 06_live_webcam_detection.py --camera 1")
        sys.exit(1)

    # Request HD resolution if supported by webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[✓] Camera stream initialized successfully ({frame_width}x{frame_height}).")
    print("\n--- Live Controls ---")
    print("  [ Press 'q' or 'ESC' ] : Quit live webcam detection")
    print("  [ Press 's' ]          : Save high-resolution snapshot to output/")
    print("  [ Press 'c' ]          : Cycle confidence threshold (0.25 -> 0.40 -> 0.60)")
    print("=" * 65 + "\n")

    current_conf = conf_threshold
    conf_options = [0.25, 0.40, 0.60]

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Error: Empty frame received from webcam stream.")
                break

            # Mirror frame horizontally for natural selfie view
            frame = cv2.flip(frame, 1)

            # Update confidence threshold if changed
            detector.conf_threshold = current_conf

            # Run real-time detection on webcam frame
            detections = detector.detect(frame)

            # Draw visual bounding boxes & class labels
            annotated_frame = detector.draw_detections(frame, detections)

            # Calculate live FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff) if fps > 0 else (1.0 / time_diff)

            # Draw Top Status Bar Overlay
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 40), (15, 15, 15), -1)
            status_text = (
                f"LIVE WEBCAM | FPS: {fps:.1f} | Conf: {current_conf:.2f} | "
                f"Objects Detected: {len(detections)}"
            )
            cv2.putText(
                annotated_frame,
                status_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw Bottom Controls Bar
            cv2.rectangle(annotated_frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)
            controls_text = "Controls: [q] Quit  [s] Take Snapshot  [c] Cycle Confidence"
            cv2.putText(
                annotated_frame,
                controls_text,
                (12, frame.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Display live window
            cv2.imshow("Live Laptop Object Detection (OpenCV)", annotated_frame)

            # Check key presses (1ms wait)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # 'q' or ESC
                print("[i] User requested live stream exit.")
                break
            elif key == ord("s"):  # Take snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(output_dir, f"webcam_snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, annotated_frame)
                print(f"[✓] Saved snapshot frame to: {snap_path}")
            elif key == ord("c"):  # Cycle confidence threshold
                curr_idx = conf_options.index(current_conf) if current_conf in conf_options else 1
                next_idx = (curr_idx + 1) % len(conf_options)
                current_conf = conf_options[next_idx]
                print(f"[*] Changed detection confidence threshold to: {current_conf:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera stream released cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Live Laptop Camera Object Detection with OpenCV"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 for built-in laptop camera).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Initial confidence threshold (default: 0.40).",
    )
    args = parser.parse_args()

    start_live_webcam(camera_id=args.camera, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
