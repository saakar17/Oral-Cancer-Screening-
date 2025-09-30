import os
import time
from typing import List, Dict
import torch
import cv2
from ultralytics import YOLO

class YOLOInferenceEngine:
    def __init__(self, model_path: str, class_names: List[str], device: str = "cpu"):
        """
        Initialize YOLO engine.
        """
        self.device = device
        self.model = YOLO(model_path)
        self.class_names = class_names

        if not torch.cuda.is_available() and device == "cuda":
            print("⚠️ CUDA not available, falling back to CPU")

    def _process_result(self, result, image_path: str, inference_time: float) -> Dict:
        """
        Convert YOLO result into app-compatible dictionary.
        """
        detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cname
            })

        # Read original image for display
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr is not None else None

        return {
            "image_path": image_path,
            "original_image": image_rgb,
            "detections": detections,
            "num_detections": len(detections),
            "inference_time": inference_time
        }

    def predict_single_image(self, image_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> Dict:
        """
        Run inference on a single image.
        """
        try:
            start = time.time()
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            elapsed = time.time() - start
            return self._process_result(results[0], image_path, elapsed)
        except Exception as e:
            return {
                "image_path": image_path,
                "error": str(e),
                "detections": [],
                "num_detections": 0,
                "inference_time": 0
            }

    def predict_directory(self, input_dir: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45, save_results: bool = False) -> List[Dict]:
        """
        Run inference on all images in a directory.
        """
        results_all = []
        for fname in os.listdir(input_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                fpath = os.path.join(input_dir, fname)
                res = self.predict_single_image(fpath, conf_threshold, iou_threshold)
                results_all.append(res)
        return results_all

