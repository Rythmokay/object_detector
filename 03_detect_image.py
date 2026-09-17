"""
03_detect_image.py
------------------
Image object detection pipeline.
Processes static images using ObjectDetector, displays detection logs,
and saves annotated visual results to disk.
"""

import os
import sys
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


def run_image_detection(
    image_path: str,
    detector: ObjectDetector,
    output_dir: str = "output",
    show_window: bool = False,
) -> str:
    """
    Run object detection on a single image file.

    :param image_path: Path to input image file.
    :param detector: Instance of ObjectDetector.
    :param output_dir: Directory where annotated result will be saved.
    :param show_window: Whether to open an OpenCV window displaying the image.
    :return: Filepath of the saved annotated image.
    """
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return ""

    print(f"\n[*] Processing image: {os.path.basename(image_path)}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Error: Failed to decode image file {image_path}")
        return ""

    # Run detection
    detections = detector.detect(image)

    # Summarize detected classes
    class_counts = Counter([d["label"] for d in detections])
    print(f"[+] Total objects detected: {len(detections)}")
    for label, count in class_counts.items():
        print(f"    - {label.capitalize()}: {count}")

    # Print detailed coordinates
    for i, d in enumerate(detections, 1):
        x, y, w, h = d["box"]
        print(
            f"  [{i}] {d['label'].capitalize()} ({d['confidence']*100:.1f}%) -> Box: [x={x}, y={y}, w={w}, h={h}]"
        )

    # Draw bounding boxes and text
    annotated_image = detector.draw_detections(image, detections)

    # Save output image
    os.makedirs(output_dir, exist_ok=True)
    filename = "detected_" + os.path.basename(image_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, annotated_image)
    print(f"[✓] Saved annotated image to: {save_path}")

    # Optionally show window if desktop environment is interactive
    if show_window:
        try:
            cv2.imshow(f"Detection - {os.path.basename(image_path)}", annotated_image)
            print("[i] Press any key in the image window to proceed...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("[!] Note: Graphical window not available in headless mode.")

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Object Detection - Single & Batch Image Processor"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a specific image file. If omitted, processes all sample images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold filter (0.0 to 1.0). Default is 0.4.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save annotated result images.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive OpenCV GUI image window.",
    )
    args = parser.parse_args()

    # Initialize Object Detector
    detector = ObjectDetector(conf_threshold=args.conf)

    if args.image:
        run_image_detection(args.image, detector, args.output_dir, args.show)
    else:
        print("[*] No specific image provided. Running on all sample images in 'samples/'...")
        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        if not os.path.exists(samples_dir):
            print("[!] Error: 'samples/' folder missing. Please run 01_download_models.py")
            sys.exit(1)

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        sample_files = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.lower().endswith(valid_extensions)
        ]

        if not sample_files:
            print("[!] No sample images found. Run 01_download_models.py first.")
            sys.exit(1)

        for img_path in sorted(sample_files):
            run_image_detection(img_path, detector, args.output_dir, args.show)

    print("\n" + "=" * 60)
    print("[✓] Image detection pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
