"""
04_detect_video.py
------------------
Video stream and webcam real-time object detection pipeline.
Processes video frame-by-frame with FPS counter overlay, live object logging,
and optional video output saving.
"""

import os
import sys
import time
import argparse
import importlib.util
from collections import Counter
import cv2

# Import ObjectDetector dynamically from 02_object_detector.py
_detector_path = os.path.join(os.path.dirname(__file__), "02_object_detector.py")
_spec = importlib.util.spec_from_file_location("detector_module", _detector_path)
_detector_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detector_module)
ObjectDetector = _detector_module.ObjectDetector


def process_video_stream(
    source: str,
    detector: ObjectDetector,
    save_output: bool = False,
    output_dir: str = "output",
    show_window: bool = False,
    max_frames: int = None,
):
    """
    Process video stream or webcam feed for real-time object detection.

    :param source: File path to video file or integer string '0' for webcam.
    :param detector: Instance of ObjectDetector.
    :param save_output: Whether to save the annotated output video.
    :param output_dir: Directory to save the output video file.
    :param show_window: Whether to display interactive cv2 window.
    :param max_frames: Optional limit on total frames to process (useful for headless testing).
    """
    # Check if source is webcam index or video path
    if source.isdigit():
        video_source = int(source)
        source_name = f"Webcam_{video_source}"
    else:
        video_source = source
        source_name = os.path.basename(source)
        if not os.path.exists(source):
            print(f"[!] Error: Video file not found: {source}")
            return

    print(f"\n[*] Opening video stream: {source_name}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[!] Error: Unable to open video source: {source_name}")
        return

    # Extract video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    fps_input = fps_input if fps_input > 0 else 30.0

    video_writer = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_filename = f"detected_{os.path.splitext(source_name)[0]}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        # H.264 / mp4v video writer codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            out_path, fourcc, fps_input, (frame_width, frame_height)
        )
        print(f"[*] Saving output video to: {out_path}")

    frame_count = 0
    total_detections_count = 0
    start_time = time.time()

    print("[*] Processing frames... Press 'q' in window to exit early.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            frame_start = time.time()

            # Perform detection on current frame
            detections = detector.detect(frame)
            total_detections_count += len(detections)

            # Draw detections
            annotated_frame = detector.draw_detections(frame, detections)

            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0.0

            # Draw overlay banner on top of frame
            cv2.rectangle(annotated_frame, (0, 0), (frame_width, 35), (20, 20, 20), -1)
            fps_text = f"FPS: {fps:.1f} | Frame: {frame_count} | Objects: {len(detections)}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Save frame if video writer active
            if video_writer:
                video_writer.write(annotated_frame)

            # Show interactive GUI window if enabled
            if show_window:
                try:
                    cv2.imshow("OpenCV Real-Time Object Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[i] User quit video playback.")
                        break
                except cv2.error:
                    show_window = False  # Headless mode fallback

            # Stop if max_frames threshold hit
            if max_frames and frame_count >= max_frames:
                print(f"[*] Reached maximum frame count limit ({max_frames}). Stopping.")
                break

    finally:
        elapsed_time = time.time() - start_time
        cap.release()
        if video_writer:
            video_writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print("\n" + "=" * 60)
    print(" Video Processing Summary ")
    print("=" * 60)
    print(f" Total Frames Processed : {frame_count}")
    print(f" Total Objects Detected  : {total_detections_count}")
    print(f" Total Time Elapsed     : {elapsed_time:.2f} seconds")
    print(f" Average Processing FPS : {avg_fps:.1f} FPS")
    if save_output:
        print(f" Saved Video File       : {out_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Object Detection - Video & Webcam Stream Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file or webcam index (e.g., '0'). Defaults to sample video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold filter (0.0 to 1.0). Default is 0.4.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video stream to output/ directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive OpenCV GUI video window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames processed (useful for testing).",
    )
    args = parser.parse_args()

    detector = ObjectDetector(conf_threshold=args.conf)

    # Resolve video source
    if args.video:
        source = args.video
    else:
        sample_video = os.path.join(
            os.path.dirname(__file__), "samples", "pedestrians_walk.avi"
        )
        if os.path.exists(sample_video):
            print("[*] No video source specified. Using sample video 'pedestrians_walk.avi'.")
            source = sample_video
        else:
            print("[!] Sample video missing. Defaulting to webcam index 0.")
            source = "0"

    process_video_stream(
        source=source,
        detector=detector,
        save_output=args.save,
        output_dir="output",
        show_window=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
