"""
01_download_models.py
---------------------
Automated model and sample asset downloader for OpenCV Object Detection.
Downloads pre-trained YOLO (Darknet) model architecture, weights, class labels,
and sample media files into local project directories.
"""

import os
import sys
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Model configuration: YOLOv4-tiny (Darknet format) with fallbacks
MODEL_FILES = {
    "yolov4-tiny.cfg": [
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    ],
    "yolov4-tiny.weights": [
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "https://pjreddie.com/media/files/yolov3-tiny.weights",
    ],
    "coco.names": [
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
    ],
}

SAMPLE_MEDIA = {
    "soccer_players.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg",
    "basketball.png": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/basketball1.png",
    "pedestrians_walk.avi": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
}


def ensure_directories():
    """Create project directories if they do not exist."""
    for folder in [MODELS_DIR, SAMPLES_DIR, OUTPUT_DIR]:
        os.makedirs(folder, exist_ok=True)
        print(f"[+] Directory ready: {folder}")


def download_file(urls, destination_path, file_description):
    """
    Download a file from a list of candidate URLs with streaming progress.

    :param urls: List of string URLs or a single URL string.
    :param destination_path: Target filepath on disk.
    :param file_description: Human readable description for logging.
    """
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f"[✓] {file_description} already exists at {destination_path}")
        return True

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        print(f"[*] Downloading {file_description} from:\n    {url}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            bytes_downloaded = 0
            chunk_size = 8192

            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            percent = (bytes_downloaded / total_size) * 100
                            sys.stdout.write(
                                f"\r    Progress: {bytes_downloaded}/{total_size} bytes ({percent:.1f}%)"
                            )
                            sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"[✓] Successfully downloaded {file_description}")
            return True
        except Exception as e:
            print(f"[!] Warning: Download failed from {url}. Error: {e}")
            if os.path.exists(destination_path):
                os.remove(destination_path)

    print(f"[✗] Failed to download {file_description} from all sources.")
    return False


def setup_assets():
    """Download required model weights, class names, and sample images/video."""
    print("=" * 60)
    print(" OpenCV Object Detection - Asset Downloader ")
    print("=" * 60)

    ensure_directories()

    print("\n--- Downloading Pre-trained YOLO Model & Class Names ---")
    for filename, urls in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        success = download_file(urls, dest, filename)
        if not success:
            print(f"[!] CRITICAL: Failed to download required model file: {filename}")
            sys.exit(1)

    print("\n--- Downloading Sample Test Media ---")
    for filename, url in SAMPLE_MEDIA.items():
        dest = os.path.join(SAMPLES_DIR, filename)
        download_file(url, dest, filename)

    print("\n" + "=" * 60)
    print("[✓] All model weights and sample media are ready for object detection!")
    print("=" * 60)


if __name__ == "__main__":
    setup_assets()
