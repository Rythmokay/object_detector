"""
05_batch_benchmark.py
---------------------
Performance benchmarking & detection metric evaluator script.
Evaluates OpenCV DNN ObjectDetector performance across varying confidence
thresholds, measures inference latency, and generates a report.
"""

import os
import sys
import time
import importlib.util
from collections import Counter
import cv2

# Import ObjectDetector dynamically from 02_object_detector.py
_detector_path = os.path.join(os.path.dirname(__file__), "02_object_detector.py")
_spec = importlib.util.spec_from_file_location("detector_module", _detector_path)
_detector_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detector_module)
ObjectDetector = _detector_module.ObjectDetector


def benchmark_samples(thresholds: list = [0.25, 0.50, 0.75]):
    """
    Run benchmarks across sample images for multiple confidence thresholds.

    :param thresholds: List of float confidence thresholds to test.
    """
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(samples_dir):
        print("[!] Error: 'samples/' folder missing. Please run 01_download_models.py")
        sys.exit(1)

    image_files = [
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("[!] Error: No sample images found in 'samples/'.")
        sys.exit(1)

    print("=" * 70)
    print(" OpenCV Object Detection - Performance Benchmarks ")
    print("=" * 70)
    print(f" Found {len(image_files)} sample image(s) for benchmarking.\n")

    report_lines = []
    report_lines.append("OpenCV Object Detection Benchmark Report")
    report_lines.append("=" * 55)
    report_lines.append(f"{'Conf Thresh':<12} | {'Total Obj':<10} | {'Avg Time (ms)':<15} | {'Est FPS':<10}")
    report_lines.append("-" * 55)

    print(report_lines[2])
    print(report_lines[3])

    for thresh in thresholds:
        detector = ObjectDetector(conf_threshold=thresh)
        total_detections = 0
        execution_times = []
        class_summary = Counter()

        for img_path in image_files:
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Warm-up run
            detector.detect(image)

            # Timed run
            t0 = time.perf_counter()
            detections = detector.detect(image)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            execution_times.append(latency_ms)
            total_detections += len(detections)

            for d in detections:
                class_summary[d["label"]] += 1

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        est_fps = 1000.0 / avg_time if avg_time > 0 else 0

        line = f"{thresh:<12.2f} | {total_detections:<10d} | {avg_time:<15.2f} | {est_fps:<10.1f}"
        print(line)
        report_lines.append(line)

    report_lines.append("=" * 55)

    # Save benchmark report text file
    report_path = os.path.join(output_dir, "benchmark_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n" + "=" * 70)
    print(f"[✓] Benchmark completed! Detailed report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_samples()
