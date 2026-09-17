"""
02_object_detector.py
---------------------
Core object detection module wrapping OpenCV DNN.
Provides the ObjectDetector class for model loading, forward inference,
non-maximum suppression (NMS) filtering, and bounding box drawing.
"""

import os
import cv2
import numpy as np


class ObjectDetector:
    """
    OpenCV DNN Object Detector using pre-trained Darknet/YOLO models.
    """

    def __init__(
        self,
        config_path: str = "models/yolov4-tiny.cfg",
        weights_path: str = "models/yolov4-tiny.weights",
        classes_path: str = "models/coco.names",
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.3,
        input_size: tuple = (416, 416),
    ):
        """
        Initialize the detector with paths to model config, weights, and class names.

        :param config_path: Path to Darknet model .cfg file.
        :param weights_path: Path to model .weights file.
        :param classes_path: Path to .names text file containing class labels.
        :param conf_threshold: Minimum confidence score to filter detections.
        :param nms_threshold: IoU threshold for Non-Maximum Suppression.
        :param input_size: Tuple (width, height) for image blob resizing.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Resolve paths
        self.config_path = os.path.abspath(config_path)
        self.weights_path = os.path.abspath(weights_path)
        self.classes_path = os.path.abspath(classes_path)

        # Validate file existence
        for p, label in [
            (self.config_path, "Config"),
            (self.weights_path, "Weights"),
            (self.classes_path, "Classes"),
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[!] {label} file not found at: {p}\n"
                    "    Please run 'python 01_download_models.py' first!"
                )

        # Load classes and generate deterministic colors
        self.classes = self._load_classes(self.classes_path)
        self.colors = self._generate_colors(len(self.classes))

        # Load neural network using OpenCV DNN
        print(f"[*] Loading OpenCV DNN model from Darknet framework...")
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Determine layer names for output
        layer_names = self.net.getLayerNames()
        out_layer_indices = self.net.getUnconnectedOutLayers()
        if len(out_layer_indices.shape) == 1:
            self.output_layers = [layer_names[i - 1] for i in out_layer_indices]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in out_layer_indices]

        print(f"[✓] Model successfully loaded ({len(self.classes)} classes).")

    @staticmethod
    def _load_classes(path: str) -> list:
        """Load class labels line-by-line from file."""
        with open(path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes

    @staticmethod
    def _generate_colors(num_classes: int) -> list:
        """Generate distinct, vibrant RGB colors for each class."""
        np.random.seed(42)  # Deterministic colors
        colors = np.random.randint(50, 255, size=(num_classes, 3), dtype="uint8")
        return [tuple(int(c) for c in color) for color in colors]

    def detect(self, image: np.ndarray) -> list:
        """
        Perform object detection on an input BGR image frame.

        :param image: Input OpenCV BGR image numpy array.
        :return: List of detection dictionaries containing bounding box, label, and confidence.
        """
        height, width = image.shape[:2]

        # Construct a 4D blob from the image for input to the network
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        # Forward pass inference
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Parse raw network outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > self.conf_threshold:
                    # Bounding box center coordinates and dimensions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Top-left corner position
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else indices
            for i in flat_indices:
                x, y, w, h = boxes[i]
                cid = class_ids[i]
                conf = confidences[i]
                label = self.classes[cid] if cid < len(self.classes) else "unknown"
                color = self.colors[cid] if cid < len(self.colors) else (0, 255, 0)

                detections.append(
                    {
                        "class_id": cid,
                        "label": label,
                        "confidence": conf,
                        "box": (max(0, x), max(0, y), w, h),
                        "color": color,
                    }
                )

        return detections

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw styled bounding boxes and class labels on a copy of the image.

        :param image: Input OpenCV BGR image array.
        :param detections: List of detection dictionaries from detect().
        :return: Annotated BGR image numpy array.
        """
        annotated = image.copy()

        for det in detections:
            x, y, w, h = det["box"]
            label = det["label"]
            conf = det["confidence"]
            color = det["color"]  # BGR tuple

            # Draw outer rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label text formatting
            text = f"{label.capitalize()}: {conf * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Get text bounding box size
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Draw label background box
            label_y1 = max(0, y - text_h - 10)
            label_y2 = y
            cv2.rectangle(
                annotated,
                (x, label_y1),
                (x + text_w + 10, label_y2),
                color,
                cv2.FILLED,
            )

            # Draw text inside background box (white font for readability)
            text_pos = (x + 5, y - 5 if y - 5 > 10 else y + text_h + 5)
            cv2.putText(
                annotated,
                text,
                (x + 5, max(15, y - 5)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated


if __name__ == "__main__":
    print("--- Running ObjectDetector Module Verification ---")
    detector = ObjectDetector()
    print(f"Loaded {len(detector.classes)} object classes.")
    sample_img = np.zeros((416, 416, 3), dtype=np.uint8)
    results = detector.detect(sample_img)
    print(f"Verification test completed smoothly. Found {len(results)} detections in blank frame.")
