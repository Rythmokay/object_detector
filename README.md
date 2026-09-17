# Modular OpenCV Object Detection System

A clean, production-ready, and modular Object Detection project built with Python and OpenCV's Deep Neural Network (`cv2.dnn`) module. 

This project utilizes pre-trained YOLO (Darknet) models to detect **80 common object classes** (such as people, cars, dogs, sports equipment, chairs, laptops, cell phones, etc.) in real time with high performance and zero external heavy deep learning framework overhead (no PyTorch/TensorFlow required).

---

## 📁 Project Architecture & File Structure

The project is structured logically into numbered scripts (`01_*.py` to `06_*.py`) for modularity and ease of maintenance:

```text
proj2/
├── requirements.txt              # Project dependencies (opencv-python, numpy, requests)
├── 01_download_models.py         # Automated downloader for model weights & test media
├── 02_object_detector.py         # Core reusable ObjectDetector class & visual drawing engine
├── 03_detect_image.py            # Single & batch image object detection pipeline
├── 04_detect_video.py            # Real-time video file detection pipeline
├── 05_batch_benchmark.py         # Performance benchmarking & metrics generator
├── 06_live_webcam_detection.py   # Live laptop webcam real-time object detection application
├── models/                       # (Generated) YOLO weights, config & COCO class names
├── samples/                      # (Generated) Test images and video files
└── output/                       # (Generated) Annotated output images, videos & logs
```

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment (`venv`)

Create and activate an isolated Python virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate

# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install all required Python packages specified in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📖 Usage Instructions

### 📷 Live Laptop Camera Object Detection (`06_live_webcam_detection.py`)

To run real-time object detection directly on your **laptop camera / webcam**:

```bash
python 06_live_webcam_detection.py
```

#### Interactive Camera Controls:
- **`q`** or **`ESC`**: Exit live webcam stream cleanly.
- **`s`**: Take a high-resolution snapshot and save it to `output/webcam_snapshot_<timestamp>.jpg`.
- **`c`**: Cycle detection confidence threshold dynamically (0.25 ➔ 0.40 ➔ 0.60).

---

### Step 1: Download Model Weights & Sample Media (`01_download_models.py`)

Run the downloader script to fetch pre-trained model weights (`yolov4-tiny`), class labels (`coco.names`), and sample test images/videos:

```bash
python 01_download_models.py
```

### Step 2: Core Object Detector Module (`02_object_detector.py`)

This script contains the core `ObjectDetector` class wrapping OpenCV DNN inference, Non-Maximum Suppression (NMS), and color bounding box rendering:

```bash
python 02_object_detector.py
```

### Step 3: Run Object Detection on Images (`03_detect_image.py`)

Process sample images or custom images:

```bash
# Process all sample images
python 03_detect_image.py

# Process a custom image file
python 03_detect_image.py --image path/to/your_image.jpg --conf 0.50
```

### Step 4: Run Video Detection (`04_detect_video.py`)

Detect objects in video files:

```bash
# Process sample video file and save output video to output/
python 04_detect_video.py --save
```

### Step 5: Run Benchmarking (`05_batch_benchmark.py`)

Evaluate detection speed (FPS) and object counts:

```bash
python 05_batch_benchmark.py
```

---

## 🏷️ Supported Object Classes (COCO Dataset - 80 Classes)

`person`, `bicycle`, `car`, `motorbike`, `aeroplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `sofa`, `pottedplant`, `bed`, `diningtable`, `toilet`, `tvmonitor`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`.
