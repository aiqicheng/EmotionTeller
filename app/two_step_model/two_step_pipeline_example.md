# Two-Step Pipeline Examples

This document provides example commands and usage patterns for `two_step_pipeline.py`.

## Command-Line Interface (CLI) Examples

### 1. Single Image Processing

```bash
# Basic usage - process a single image
python two_step_pipeline.py --image path/to/image.jpg

# Process image and save results to CSV
python two_step_pipeline.py --image path/to/image.jpg --out_csv results.csv
```

### 2. Batch Processing (Folder of Images)

```bash
# Process all images in a folder and save to CSV
python two_step_pipeline.py --folder path/to/images --out_csv predictions.csv

# Example with specific paths
python two_step_pipeline.py --folder /content/drive/MyDrive/emo/ImageData --out_csv /content/drive/MyDrive/emo/predictions.csv
```

### 3. Video Processing

```bash
# Process a video file
python two_step_pipeline.py --video path/to/video.mp4 --out_video output_annotated.mp4
```

## Programmatic Usage Examples

### Basic Usage

```python
from two_step_pipeline import Config, load_detector, load_classifier, run_on_image, run_on_folder

# Create config (auto-detects Colab vs local)
cfg = Config()

# Load models
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)

# Process a single image
result = run_on_image("path/to/image.jpg", det_model, cls_model, cfg)
print(f"Found {len(result['faces'])} faces")
for face in result['faces']:
    print(f"  Emotion: {face['pred_label']}, Confidence: {face['det_conf']:.2f}")
```

### Custom Configuration

```python
from two_step_pipeline import Config, load_detector, load_classifier, run_on_image

# Custom config with specific paths (override auto-detection)
cfg = Config(
    detector_path="/custom/path/to/detector.pt",
    classifier_path="/custom/path/to/classifier.pt",
    det_conf_thres=0.25,  # Lower threshold for more detections
    det_iou_thres=0.45,
    crop_expand=1.3,      # Expand crop by 30%
    draw=True,            # Generate visualization
    device="cuda"         # Force GPU
)

# Load and run
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)
result = run_on_image("image.jpg", det_model, cls_model, cfg)
```

### Batch Processing with Custom Config

```python
from two_step_pipeline import Config, run_on_folder

# Configure for batch processing
cfg = Config(
    det_conf_thres=0.3,
    batch_size=64,        # Larger batch for faster processing
    draw=False,            # Skip visualization for batch
    device="cuda"
)

# Process entire folder
output_csv = run_on_folder(
    in_dir="/path/to/images",
    out_csv="/path/to/output.csv",
    cfg=cfg
)
print(f"Results saved to {output_csv}")
```

### Accessing Results Programmatically

```python
from two_step_pipeline import Config, load_detector, load_classifier, run_on_image, faces_to_df
import pandas as pd

cfg = Config()
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)

# Process image
result = run_on_image("image.jpg", det_model, cls_model, cfg)

# Convert to DataFrame
df = faces_to_df(result)
print(df)

# Access individual faces
for face in result['faces']:
    bbox = face['bbox_xyxy']
    emotion = face['pred_label']
    confidence = face['det_conf']
    probs = face['pred_probs']
    print(f"Face at {bbox}: {emotion} (conf={confidence:.2f})")
    print(f"  All probabilities: {dict(zip(cfg.class_names, probs))}")
```

### Processing Multiple Images

```python
from two_step_pipeline import Config, load_detector, load_classifier, run_on_image
from pathlib import Path

cfg = Config()
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)

# Process multiple images
image_dir = Path("/path/to/images")
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

results = []
for img_path in image_files:
    result = run_on_image(str(img_path), det_model, cls_model, cfg)
    results.append(result)
    print(f"Processed {img_path.name}: {len(result['faces'])} faces")

# Aggregate results
total_faces = sum(len(r['faces']) for r in results)
print(f"Total faces detected: {total_faces}")
```

## Google Colab Specific Examples

### Colab Setup and Usage

```python
# Mount Google Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# The script auto-detects Colab environment
# No need to change paths manually!
from two_step_pipeline import Config, load_detector, load_classifier, run_on_folder

cfg = Config()  # Automatically uses Colab paths
print(f"Using detector: {cfg.detector_path}")
print(f"Using classifier: {cfg.classifier_path}")

# Process images
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)

# Run on Colab paths
result = run_on_image(
    "/content/drive/MyDrive/emo/ImageData/test_image.jpg",
    det_model, cls_model, cfg
)
```

### Override Colab Auto-Detection

```python
from two_step_pipeline import Config

# If you want to use different paths in Colab
cfg = Config(
    detector_path="/content/drive/MyDrive/custom/path/detector.pt",
    classifier_path="/content/drive/MyDrive/custom/path/classifier.pt"
)
```

## Advanced Usage

### Custom Class Names

```python
from two_step_pipeline import Config

# Define custom class names (must match training order)
cfg = Config(
    class_names=["Neutral", "Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust"]
)
```

### Performance Optimization

```python
from two_step_pipeline import Config

# Optimize for speed
cfg = Config(
    batch_size=128,       # Larger batches
    half=True,            # FP16 (if GPU supports it)
    det_conf_thres=0.4,   # Higher threshold = fewer detections = faster
    draw=False            # Skip visualization
)

# For accuracy
cfg = Config(
    det_conf_thres=0.2,   # Lower threshold = more detections
    crop_expand=1.5,      # More context around faces
    batch_size=16        # Smaller batches for stability
)
```

### Save Results in Different Formats

```python
from two_step_pipeline import Config, run_on_image, pack_image_prediction_row
import json
import pandas as pd

cfg = Config()
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)

result = run_on_image("image.jpg", det_model, cls_model, cfg)

# Save as JSON
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

# Save in evaluation format (matches GT CSV format)
import cv2
img = cv2.imread("image.jpg")
H, W = img.shape[:2]
row = pack_image_prediction_row("image.jpg", result['faces'], W, H)

# Convert to DataFrame and save
df = pd.DataFrame([row])
df.to_csv("result.csv", index=False)
```

## Integration with Evaluation Pipeline

```python
from two_step_pipeline import Config, run_on_folder
from evaluation_pipeline_map import evaluate

# Generate predictions
cfg = Config()
output_csv = run_on_folder(
    in_dir="/path/to/test/images",
    out_csv="/path/to/predictions.csv",
    cfg=cfg
)

# Evaluate against ground truth
mAP, per_class = evaluate(
    gt_csv="/path/to/ground_truth.csv",
    pred_csv="/path/to/predictions.csv",
    iou_thr=0.5
)

print(f"mAP@0.5: {mAP:.4f}")
for cls, metrics in per_class.items():
    print(f"{cls}: AP={metrics['AP']:.4f}")
```

## Troubleshooting

### Check Model Paths

```python
from two_step_pipeline import Config, _is_colab

print(f"Running in Colab: {_is_colab()}")
cfg = Config()
print(f"Detector path: {cfg.detector_path}")
print(f"Classifier path: {cfg.classifier_path}")

# Check if files exist
from pathlib import Path
print(f"Detector exists: {Path(cfg.detector_path).exists()}")
print(f"Classifier exists: {Path(cfg.classifier_path).exists()}")
```

### Debug Model Loading

```python
from two_step_pipeline import Config, load_detector, load_classifier

cfg = Config()
print("Loading detector...")
try:
    det_model = load_detector(cfg)
    print("✓ Detector loaded successfully")
except Exception as e:
    print(f"✗ Detector failed: {e}")

print("Loading classifier...")
try:
    cls_model = load_classifier(cfg)
    print("✓ Classifier loaded successfully")
    print(f"  Architecture: {cls_model.__class__.__name__}")
    print(f"  Classes: {cfg.class_names}")
except Exception as e:
    print(f"✗ Classifier failed: {e}")
```

## Complete Workflow Example

```python
"""
Complete workflow: Load models once, process multiple images
"""
from two_step_pipeline import Config, load_detector, load_classifier, run_on_image
from pathlib import Path

# Setup
cfg = Config(
    det_conf_thres=0.3,
    draw=True  # Generate visualizations
)

print("Loading models...")
det_model = load_detector(cfg)
cls_model = load_classifier(cfg)
print("Models loaded!")

# Process images
image_dir = Path("/path/to/images")
for img_path in image_dir.glob("*.jpg"):
    print(f"\nProcessing {img_path.name}...")
    
    result = run_on_image(str(img_path), det_model, cls_model, cfg)
    
    print(f"  Found {len(result['faces'])} faces")
    for i, face in enumerate(result['faces']):
        print(f"    Face {i+1}: {face['pred_label']} "
              f"(det_conf={face['det_conf']:.2f})")
    
    if result['visualization_path']:
        print(f"  Visualization saved to {result['visualization_path']}")
```

