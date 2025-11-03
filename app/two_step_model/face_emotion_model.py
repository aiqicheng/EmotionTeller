"""
Face Emotion Model
Unified model that combines face detection and emotion classification.

This module provides the main FaceEmotionModel class that integrates:
- Face detection using OpenCV DNN
- Emotion classification using PyTorch models
- End-to-end inference pipeline
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import torchvision.transforms as T

from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier


class FaceEmotionModel(nn.Module):
    """
    Unified face emotion recognition model.
    
    This class combines face detection and emotion classification into a single pipeline.
    It can detect faces in images and predict emotions for each detected face.
    """
    
    def __init__(self, face_detector: FaceDetector, emotion_classifier: EmotionClassifier, 
                 transform: Optional[T.Compose] = None):
        """
        Initialize the FaceEmotionModel.
        
        Args:
            face_detector: FaceDetector instance for face detection
            emotion_classifier: EmotionClassifier instance for emotion prediction
            transform: Optional transform for preprocessing images
        """
        super().__init__()
        self.face_detector = face_detector
        self.emotion_classifier = emotion_classifier
        self.transform = transform
        
        # Set up transforms if not provided
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _get_default_transform(self) -> T.Compose:
        """Get default image preprocessing transform."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    
    def _bbox_to_pixels(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert percentage bounding box to pixel coordinates.
        
        Args:
            bbox: Bounding box in percentage format [x%, y%, w%, h%]
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x, y, width, height) in pixels
        """
        x_pct, y_pct, w_pct, h_pct = bbox
        
        x = int(round((x_pct / 100) * img_width))
        y = int(round((y_pct / 100) * img_height))
        w = int(round((w_pct / 100) * img_width))
        h = int(round((h_pct / 100) * img_height))
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        return x, y, w, h
    
    def detect_and_classify(self, image: Union[np.ndarray, str]) -> List[Dict]:
        """
        Detect faces and classify emotions in an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            List of dictionaries containing detection and classification results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from path: {image}")
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_height, img_width = image.shape[:2]
        
        # Step 1: Detect faces
        bboxes = self.face_detector.detect_faces(image)
        
        results = []
        for bbox in bboxes:
            # Convert bbox to pixel coordinates
            x, y, w, h = self._bbox_to_pixels(bbox, img_width, img_height)
            
            # Extract face region
            face = image[y:y+h, x:x+w]
            
            # Convert BGR to RGB for emotion classification
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Step 2: Classify emotion
            try:
                emotion, confidence = self.emotion_classifier.predict(face_pil)
                
                results.append({
                    'bbox': [x, y, w, h],
                    'bbox_pct': bbox,
                    'emotion': emotion,
                    'confidence': confidence,
                    'face_image': face_rgb
                })
            except Exception as e:
                print(f"Error classifying face: {e}")
                results.append({
                    'bbox': [x, y, w, h],
                    'bbox_pct': bbox,
                    'emotion': 'Unknown',
                    'confidence': 0.0,
                    'face_image': face_rgb
                })
        
        return results
    
    def forward(self, image: Union[np.ndarray, str]) -> List[Dict]:
        """
        Forward pass for PyTorch compatibility.
        
        Args:
            image: Input image
            
        Returns:
            List of detection and classification results
        """
        return self.detect_and_classify(image)
    
    def visualize_results(self, image: np.ndarray, results: List[Dict], 
                        save_path: Optional[str] = None, show: bool = True) -> np.ndarray:
        """
        Visualize detection and classification results.
        
        Args:
            image: Original image
            results: Results from detect_and_classify
            save_path: Optional path to save visualization
            show: Whether to display the image
            
        Returns:
            Annotated image
        """
        canvas = image.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{emotion}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(canvas, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(canvas, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"Visualization saved to {save_path}")
        
        if show:
            try:
                from google.colab.patches import cv2_imshow
                cv2_imshow(canvas)
            except Exception:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title('Face Emotion Detection Results')
                plt.show()
        
        return canvas
    
    def process_batch(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to their results
        """
        batch_results = {}
        
        for image_path in image_paths:
            try:
                results = self.detect_and_classify(image_path)
                batch_results[image_path] = results
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                batch_results[image_path] = []
        
        return batch_results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'face_detector': {
                'prototxt_path': self.face_detector.prototxt_path,
                'weights_path': self.face_detector.weights_path,
                'confidence_threshold': self.face_detector.confidence_threshold
            },
            'emotion_classifier': {
                'device': self.emotion_classifier.device,
                'num_classes': len(self.emotion_classifier.label2id) if self.emotion_classifier.label2id else 0,
                'classes': list(self.emotion_classifier.label2id.keys()) if self.emotion_classifier.label2id else []
            }
        }
        
        return info


def create_face_emotion_model(prototxt_path: str, weights_path: str, 
                             emotion_model_path: str, 
                             confidence_threshold: float = 0.2,
                             device: str = None) -> FaceEmotionModel:
    """
    Create a FaceEmotionModel with pre-trained components.
    
    Args:
        prototxt_path: Path to face detection prototxt file
        weights_path: Path to face detection weights file
        emotion_model_path: Path to trained emotion classification model
        confidence_threshold: Face detection confidence threshold
        device: Device for emotion classifier
        
    Returns:
        Initialized FaceEmotionModel
    """
    # Initialize face detector
    face_detector = FaceDetector(prototxt_path, weights_path, confidence_threshold)
    
    # Initialize emotion classifier
    emotion_classifier = EmotionClassifier(device)
    emotion_classifier.load_model(emotion_model_path)
    
    # Create unified model
    model = FaceEmotionModel(face_detector, emotion_classifier)
    
    return model

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR numpy array."""
    # Convert to numpy array
    img_rgb = np.array(pil_image)

    # If the image has an alpha channel, drop it
    if img_rgb.shape[-1] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    # Convert RGB (PIL) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

def demo_inference(image: Image.Image, model: FaceEmotionModel, 
                  save_path: Optional[str] = None) -> List[Dict]:
    """
    Run inference on a single image and visualize results.
    
    Args:
        image: Input image
        model: FaceEmotionModel instance
        save_path: Optional path to save visualization
        
    Returns:
        results: Detection and classification results
        ann: annotated image
    """
    # Load image
    image = pil_to_cv2(image)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Run inference
    results = model.detect_and_classify(image)
    
    # Visualize results
    ann = model.visualize_results(image, results, save_path)
    
    results = pd.DataFrame(results)

    return results, ann


if __name__ == "__main__":
    # Example usage
    data_folder = "/content/drive/MyDrive/emo/"
    model_folder = data_folder + "/BaselineModels/"
    
    # Model paths
    prototxt_path = model_folder + "deploy.prototxt"
    weights_path = model_folder + "res10_300x300_ssd_iter_140000.caffemodel"
    emotion_model_path = model_folder + "best_overall.pt"
    
    # Create model
    model = create_face_emotion_model(prototxt_path, weights_path, emotion_model_path)
    
    # Get model info
    info = model.get_model_info()
    print("Model Information:")
    print(f"Face Detector: {info['face_detector']}")
    print(f"Emotion Classifier: {info['emotion_classifier']}")
    
    # Example inference
    image_path = "path/to/your/image.jpg"  # Replace with actual image path
    try:
        results = demo_inference(image_path, model)
        print(f"Inference completed successfully!")
    except Exception as e:
        print(f"Inference failed: {e}")
        print("Please provide a valid image path for testing.")