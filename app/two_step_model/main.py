"""
Main Script for Face Emotion Recognition
This script provides a command-line interface and example usage for the refactored emotion recognition system.

Usage:
    python main.py --mode train --data_folder /path/to/data
    python main.py --mode inference --image_path /path/to/image.jpg
    python main.py --mode demo --image_folder /path/to/images
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Import our modules
from .face_detector import FaceDetector, process_dataset_detection
from .image_cropper import ImageCropper, process_dataset_cropping
from .emotion_classifier import EmotionClassifier, combine_datasets
from .face_emotion_model import FaceEmotionModel, create_face_emotion_model, demo_inference


def setup_paths(data_folder: str) -> Dict[str, str]:
    """
    Setup file paths based on data folder.
    
    Args:
        data_folder: Root data folder path
        
    Returns:
        Dictionary with all necessary paths
    """
    data_folder = Path(data_folder)
    model_folder = data_folder / "BaselineModels"
    
    paths = {
        'data_folder': str(data_folder),
        'model_folder': str(model_folder),
        'image_folder': str(data_folder / "ImageData"),
        'faces_folder': str(data_folder / "FACES_headshots_6classes"),
        'train_meta': str(data_folder / "train_meta.csv"),
        'test_meta': str(data_folder / "test_meta.csv"),
        'faces_meta': str(data_folder / "FACES_headshots_6classes" / "facesdata.csv"),
        'face_detection_model': str(model_folder / "yolo11n-face-best.pt"),
        'emotion_model': str(model_folder / "best_overall.pt"),
        'cropped_folder': str(model_folder / "cropped_single"),
        'train_crops': str(model_folder / "trainval_with_crops.csv"),
        'test_crops': str(model_folder / "test_with_crops.csv"),
        'combined_data': str(model_folder / "trainval_subset_addfaces.csv"),
        'detection_results': str(model_folder / "detection.csv"),
        'detection_eval': str(model_folder / "detection_eval.json"),
    }
    
    return paths


def train_pipeline(data_folder: str, epochs: int = 20, batch_size: int = 16, 
                   lr: float = 3e-4, arch: str = "resnet18", 
                   freeze_backbone: bool = True, val_size: float = 0.2):
    """
    Run the complete training pipeline.
    
    Args:
        data_folder: Root data folder path
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        arch: Model architecture
        freeze_backbone: Whether to freeze backbone
        val_size: Validation set size
    """
    print("=== Starting Training Pipeline ===")
    paths = setup_paths(data_folder)
    
    # Step 1: Face Detection (if not already done)
    print("\n1. Face Detection...")
    if not os.path.exists(paths['detection_results']):
        print("Running face detection on dataset...")
        process_dataset_detection(
            paths['train_meta'], paths['image_folder'],
            paths['face_detection_model'],
            paths['detection_results']
        )
    else:
        print("Face detection results already exist.")
    
    # Step 2: Image Cropping (if not already done)
    print("\n2. Image Cropping...")
    if not os.path.exists(paths['train_crops']):
        print("Cropping training images...")
        process_dataset_cropping(
            paths['train_meta'], paths['image_folder'],
            paths['cropped_folder'], paths['train_crops']
        )
    else:
        print("Training crops already exist.")
    
    if not os.path.exists(paths['test_crops']):
        print("Cropping test images...")
        process_dataset_cropping(
            paths['test_meta'], paths['image_folder'],
            paths['cropped_folder'], paths['test_crops']
        )
    else:
        print("Test crops already exist.")
    
    # Step 3: Combine Datasets (if not already done)
    print("\n3. Combining Datasets...")
    if not os.path.exists(paths['combined_data']):
        print("Combining HGEL and FACES datasets...")
        combine_datasets(paths['train_crops'], paths['faces_meta'], paths['combined_data'])
    else:
        print("Combined dataset already exists.")
    
    # Step 4: Train Emotion Classifier
    print("\n4. Training Emotion Classifier...")
    if not os.path.exists(paths['emotion_model']):
        print("Training emotion classification model...")
        
        # Load combined dataset
        combined_df = pd.read_csv(paths['combined_data'])
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            combined_df, test_size=val_size, random_state=42,
            stratify=combined_df["emotion"]
        )
        
        # Train model
        classifier = EmotionClassifier()
        result = classifier.train_model(
            train_df, val_df, arch=arch, epochs=epochs,
            batch_size=batch_size, lr=lr, freeze_backbone=freeze_backbone,
            output_dir=str(Path(paths['model_folder']) / "classification_runs")
        )
        
        print(f"Training completed! Best validation accuracy: {result['best_val_acc']:.4f}")
        
        # Save the best model
        import shutil
        best_model_path = result['best_path']
        shutil.copyfile(best_model_path, paths['emotion_model'])
        print(f"Best model saved to {paths['emotion_model']}")
    else:
        print("Emotion classification model already exists.")
    
    print("\n=== Training Pipeline Completed ===")


def inference_pipeline(image: Image.Image, data_folder: str, 
                      confidence_threshold: float = 0.2,
                      save_visualization: Optional[str] = None):
    """
    Run inference on a single image.
    
    Args:
        image: Input image
        data_folder: Root data folder path with the model files
        confidence_threshold: Face detection confidence threshold
        save_visualization: Optional path to save visualization
    Returns:
        Detection and classification results and an annotated image
    """
    print("=== Starting Inference Pipeline ===")
    paths = setup_paths(data_folder)
    
    # Check if model exists
    if not os.path.exists(paths['emotion_model']):
        print(f"Error: Emotion model not found at {paths['emotion_model']}")
        print("Please run training pipeline first.")
        return
    
    # Create model
    print("Loading models...")
    model = create_face_emotion_model(
        paths['face_detection_model'], paths['emotion_model'],
        confidence_threshold
    )
    
    # Run inference
    return demo_inference(image=image, 
                             model = model, 
                             save_path = save_visualization)
    

def batch_inference(image_folder: str, data_folder: str,
                   confidence_threshold: float = 0.2,
                   output_folder: Optional[str] = None):
    """
    Run inference on a batch of images.
    
    Args:
        image_folder: Folder containing images
        data_folder: Root data folder path
        confidence_threshold: Face detection confidence threshold
        output_folder: Optional folder to save visualizations
    """
    print("=== Starting Batch Inference Pipeline ===")
    paths = setup_paths(data_folder)
    
    # Check if model exists
    if not os.path.exists(paths['emotion_model']):
        print(f"Error: Emotion model not found at {paths['emotion_model']}")
        print("Please run training pipeline first.")
        return
    
    # Create model
    print("Loading models...")
    model = create_face_emotion_model(
        paths['face_detection_model'], paths['emotion_model'],
        confidence_threshold
    )
    
    # Get image files
    image_folder = Path(image_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to process.")
    
    # Setup output folder
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    all_results = {}
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            # Run inference
            results = model.detect_and_classify(str(image_file))
            all_results[str(image_file)] = results
            
            # Save visualization if output folder specified
            if output_folder:
                image = cv2.imread(str(image_file))
                if image is not None:
                    save_path = output_folder / f"{image_file.stem}_result.jpg"
                    model.visualize_results(image, results, str(save_path), show=False)
            
            print(f"  Detected {len(results)} faces")
            for j, result in enumerate(results):
                print(f"    Face {j+1}: {result['emotion']} ({result['confidence']:.3f})")
                
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            all_results[str(image_file)] = []
    
    # Save summary
    if output_folder:
        summary_path = output_folder / "inference_summary.csv"
        summary_data = []
        for image_path, results in all_results.items():
            for result in results:
                summary_data.append({
                    'image_path': image_path,
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'bbox_x': result['bbox'][0],
                    'bbox_y': result['bbox'][1],
                    'bbox_w': result['bbox'][2],
                    'bbox_h': result['bbox'][3]
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            print(f"Inference summary saved to {summary_path}")
    
    print(f"\nBatch inference completed! Processed {len(image_files)} images.")
    print("=== Batch Inference Pipeline Completed ===")


def demo_pipeline(data_folder: str, confidence_threshold: float = 0.2):
    """
    Run a demo with sample images.
    
    Args:
        data_folder: Root data folder path
        confidence_threshold: Face detection confidence threshold
    """
    print("=== Starting Demo Pipeline ===")
    paths = setup_paths(data_folder)
    
    # Check if model exists
    if not os.path.exists(paths['emotion_model']):
        print(f"Error: Emotion model not found at {paths['emotion_model']}")
        print("Please run training pipeline first.")
        return
    
    # Create model
    print("Loading models...")
    model = create_face_emotion_model(
        paths['face_detection_model'], paths['emotion_model'],
        confidence_threshold
    )
    
    # Get model info
    info = model.get_model_info()
    print("\nModel Information:")
    print(f"Face Detector: {info['face_detector']['confidence_threshold']} confidence threshold")
    print(f"Emotion Classes: {info['emotion_classifier']['classes']}")
    print(f"Device: {info['emotion_classifier']['device']}")
    
    # Look for sample images
    sample_images = []
    for folder in [paths['image_folder'], paths['faces_folder']]:
        if os.path.exists(folder):
            folder_path = Path(folder)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = [f for f in folder_path.iterdir() 
                     if f.suffix.lower() in image_extensions]
            sample_images.extend(images[:3])  # Take first 3 images from each folder
    
    if not sample_images:
        print("No sample images found for demo.")
        return
    
    print(f"\nRunning demo on {len(sample_images)} sample images...")
    
    # Process sample images
    for i, image_file in enumerate(sample_images):
        print(f"\nDemo {i+1}: {image_file.name}")
        try:
            results = demo_inference(str(image_file), model)
        except Exception as e:
            print(f"Error in demo: {e}")
    
    print("\n=== Demo Pipeline Completed ===")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Face Emotion Recognition System')
    parser.add_argument('--mode', choices=['train', 'inference', 'batch', 'demo'], 
                       required=True, help='Mode to run')
    parser.add_argument('--data_folder', type=str, required=True,
                       help='Root data folder path')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--image_folder', type=str,
                       help='Folder containing images for batch inference')
    parser.add_argument('--output_folder', type=str,
                       help='Output folder for batch inference results')
    parser.add_argument('--save_viz', type=str,
                       help='Path to save visualization')
    parser.add_argument('--confidence', type=float, default=0.2,
                       help='Face detection confidence threshold')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['resnet18', 'vgg16_bn'],
                       help='Model architecture')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone during training')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder not found: {args.data_folder}")
        sys.exit(1)
    
    if args.mode == 'inference' and not args.image_path:
        print("Error: --image_path required for inference mode")
        sys.exit(1)
    
    if args.mode == 'batch' and not args.image_folder:
        print("Error: --image_folder required for batch mode")
        sys.exit(1)
    
    if args.mode == 'inference' and not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    if args.mode == 'batch' and not os.path.exists(args.image_folder):
        print(f"Error: Image folder not found: {args.image_folder}")
        sys.exit(1)
    
    # Run the appropriate pipeline
    try:
        if args.mode == 'train':
            train_pipeline(args.data_folder, args.epochs, args.batch_size,
                          args.lr, args.arch, args.freeze_backbone, args.val_size)
        elif args.mode == 'inference':
            inference_pipeline(args.image_path, args.data_folder,
                              args.confidence, args.save_viz)
        elif args.mode == 'batch':
            batch_inference(args.image_folder, args.data_folder,
                           args.confidence, args.output_folder)
        elif args.mode == 'demo':
            demo_pipeline(args.data_folder, args.confidence)
    
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
