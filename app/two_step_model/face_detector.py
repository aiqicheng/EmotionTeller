"""
Face Detection Module
Extracted from DetectionBaseline.ipynb

This module provides face detection functionality using YOLO model.
"""

import cv2
import numpy as np
import pandas as pd
import ast
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


class FaceDetector:
    """
    Face detector using YOLO model.
    
    This class provides functionality to:
    - Load pretrained YOLO face detection model
    - Detect faces in images
    - Evaluate detection performance using IoU metrics
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.2):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            confidence_threshold: Minimum confidence for face detection
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the face detection model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"Face detection model loaded successfully from {self.model_path}")
    
    def detect_faces(self, image: np.ndarray) -> List[List[float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of bounding boxes in percentage format [x%, y%, w%, h%]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        (h, w) = image.shape[:2]
        
        # YOLO expects RGB format, convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference
        results = self.model.predict(
            source=image_rgb,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Collect detections
        bboxes = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # Convert to percentage of width/height * 100
                    x_pct = 100 * x1 / w
                    y_pct = 100 * y1 / h
                    w_pct = 100 * (x2 - x1) / w
                    h_pct = 100 * (y2 - y1) / h
                    box_pct = np.array([x_pct, y_pct, w_pct, h_pct], dtype=float).tolist()
                    bboxes.append(box_pct)
        
        return bboxes
    
    def detect_faces_from_path(self, image_path: str) -> Tuple[Optional[np.ndarray], List[List[float]]]:
        """
        Detect faces from image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_array, bounding_boxes)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Warning] Cannot read {image_path}")
            return None, []
        
        bboxes = self.detect_faces(image)
        return image, bboxes
    
    def evaluate_detection(self, df_results: pd.DataFrame, iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate face detection performance using IoU metrics.
        
        Args:
            df_results: DataFrame with detection results containing columns:
                       - 'bboxes_pct_detected': detected bounding boxes
                       - 'bboxes_pct_labelled': ground truth bounding boxes
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            Dictionary containing evaluation metrics
        """
        def iou(boxA, boxB):
            """Calculate Intersection over Union (IoU) between two bounding boxes."""
            xA1, yA1, wA, hA = boxA
            xA2, yA2 = xA1 + wA, yA1 + hA
            xB1, yB1, wB, hB = boxB
            xB2, yB2 = xB1 + wB, yB1 + hB
            
            interX1, interY1 = max(xA1, xB1), max(yA1, yB1)
            interX2, interY2 = min(xA2, xB2), min(yA2, yB2)
            interW, interH = max(0, interX2 - interX1), max(0, interY2 - interY1)
            interArea = interW * interH
            areaA, areaB = wA * hA, wB * hB
            union = areaA + areaB - interArea
            return interArea / union if union > 0 else 0.0
        
        # Basic TP/FP/FN metrics
        TP = FP = FN = 0
        all_dets, all_gts = [], {}
        
        for _, row in df_results.iterrows():
            img_id = row["file_name"]
            preds = row["bboxes_pct_detected"] or []
            gts = row["bboxes_pct_labelled"] or []
            all_gts[img_id] = gts
            matched = set()
            
            # Check each detected bbox
            for pred in preds:
                best_iou = 0
                best_gt = None
                # Loop through all gt of one image
                for gi, gt in enumerate(gts):
                    if gi in matched:
                        continue
                    iou_val = iou(pred, gt)
                    if iou_val > best_iou:
                        best_iou, best_gt = iou_val, gi
                
                # Record highest iou(detection, ground truth)
                if best_iou >= iou_threshold:
                    TP += 1
                    matched.add(best_gt)
                else:
                    FP += 1
            
            # Unmatched ground truth
            FN += (len(gts) - len(matched))
            
            # For mAP later
            for box in preds:
                all_dets.append((img_id, 1.0, box))  # assume conf=1.0 for now
        
        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        # Compute AP@0.5
        all_dets.sort(key=lambda x: x[1], reverse=True)
        tp, fp = [], []
        gt_counter = sum(len(v) for v in all_gts.values())
        matched_gts = {img_id: np.zeros(len(gt)) for img_id, gt in all_gts.items()}
        
        for img_id, conf, det_box in all_dets:
            best_iou, best_gt = 0.0, -1
            gts = all_gts.get(img_id, [])
            for gi, gt in enumerate(gts):
                iou_val = iou(det_box, gt)
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gi
            
            if best_iou >= iou_threshold and best_gt >= 0 and matched_gts[img_id][best_gt] == 0:
                tp.append(1)
                fp.append(0)
                matched_gts[img_id][best_gt] = 1
            else:
                tp.append(0)
                fp.append(1)
        
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall_curve = tp / (gt_counter + 1e-9)
        precision_curve = tp / (tp + fp + 1e-9)
        
        # 11-point interpolation AP (VOC2007)
        recall_points = np.linspace(0, 1, 11)
        ap = 0.0
        for r in recall_points:
            p = np.max(precision_curve[recall_curve >= r]) if np.any(recall_curve >= r) else 0
            ap += p
        ap = float(ap / 11.0)
        
        return {
            "IoU_threshold": iou_threshold,
            "TruePos": TP,
            "FalsePos": FP,
            "FalseNeg": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "mAP@0.5": ap,
        }
    
    def visualize_detection(self, image: np.ndarray, detected_boxes: List[List[float]], 
                           ground_truth_boxes: List[List[float]] = None, 
                           save_path: str = None, show: bool = True) -> np.ndarray:
        """
        Visualize face detection results.
        
        Args:
            image: Input image
            detected_boxes: Detected bounding boxes in percentage format
            ground_truth_boxes: Ground truth bounding boxes in percentage format
            save_path: Optional path to save the visualization
            show: Whether to display the image
            
        Returns:
            Annotated image
        """
        H, W = image.shape[:2]
        
        def pct_box_to_pixel(box_pct):
            x_pct, y_pct, w_pct, h_pct = box_pct
            x1 = int(round((x_pct / 100) * W))
            y1 = int(round((y_pct / 100) * H))
            x2 = int(round(((x_pct + w_pct) / 100) * W))
            y2 = int(round(((y_pct + h_pct) / 100) * H))
            return max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
        
        def draw_boxes(img, boxes, color, label):
            for b in boxes:
                x1, y1, x2, y2 = pct_box_to_pixel(b)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, max(10, y1-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        canvas = image.copy()
        print(f"Number of detections: {len(detected_boxes)}")
        draw_boxes(canvas, detected_boxes, (0, 255, 0), "det")  # green
        
        if ground_truth_boxes:
            print(f"Number of ground truth: {len(ground_truth_boxes)}")
            draw_boxes(canvas, ground_truth_boxes, (0, 0, 255), "gt")  # red
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, canvas)
        
        if show:
            try:
                from google.colab.patches import cv2_imshow
                cv2_imshow(canvas)
            except Exception:
                import matplotlib.pyplot as plt
                plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
        
        return canvas


def process_dataset_detection(metadata_csv: str, image_folder: str, 
                             model_path: str,
                             output_csv: str = None, confidence_threshold: float = 0.2) -> pd.DataFrame:
    """
    Process a dataset for face detection evaluation.
    
    Args:
        metadata_csv: Path to CSV file with image metadata
        image_folder: Folder containing images
        model_path: Path to YOLO face detection model (.pt)
        output_csv: Optional path to save results
        confidence_threshold: Detection confidence threshold
        
    Returns:
        DataFrame with detection results
    """
    detector = FaceDetector(model_path, confidence_threshold)
    
    # Load dataframe
    df = pd.read_csv(metadata_csv)
    results = []
    
    # Process each image
    for i in range(len(df)):
        file_path = os.path.join(image_folder, df["file_name"].iloc[i])
        image, bboxes = detector.detect_faces_from_path(file_path)
        
        if image is None:
            continue
        
        # Get ground truth bounding boxes
        bboxes_labelled = ast.literal_eval(df['objects'].iloc[i])['bbox']
        
        # Record per image
        results.append({
            "file_name": df["file_name"].iloc[i],
            "num_faces_detected": len(bboxes),
            "bboxes_pct_detected": bboxes,
            "num_faces_labelled": len(bboxes_labelled),
            "bboxes_pct_labelled": bboxes_labelled,
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    if output_csv:
        df_results.to_csv(output_csv, index=False)
        print(f"Detection results saved to {output_csv}")
    
    return df_results


if __name__ == "__main__":
    # Example usage
    data_folder = "/content/drive/MyDrive/emo/"
    model_folder = data_folder + "/BaselineModels/"
    
    model_path = model_folder + "yolo11n-face-best.pt"
    
    # Initialize detector
    detector = FaceDetector(model_path)
    
    # Process dataset
    df_path = data_folder + "train_meta.csv"
    image_folder = data_folder + "ImageData/"
    output_csv = model_folder + "detection.csv"
    
    results = process_dataset_detection(df_path, image_folder, model_path, output_csv)
    
    # Evaluate performance
    metrics = detector.evaluate_detection(results)
    print("Detection Metrics:", metrics)
    
    # Save metrics
    with open(model_folder + "detection_eval.json", "w") as f:
        json.dump(metrics, f, indent=4)
