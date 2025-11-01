# Face Emotion Recognition System

This project provides a complete face emotion recognition system that combines face detection and emotion classification. The system has been refactored from Jupyter notebooks into modular Python scripts for better maintainability and deployment.

## Features

- **Face Detection**: Uses OpenCV DNN with SSD ResNet10 model for robust face detection
- **Emotion Classification**: Supports 7 emotion classes (Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise)
- **End-to-End Pipeline**: Complete workflow from raw images to emotion predictions
- **Modular Design**: Separate modules for detection, cropping, classification, and unified inference
- **Command-Line Interface**: Easy-to-use CLI for training, inference, and batch processing
- **Visualization**: Built-in visualization of detection and classification results

## Project Structure

```
EmotionTeller/
├── face_detector.py          # Face detection module
├── image_cropper.py          # Image cropping module  
├── emotion_classifier.py     # Emotion classification module
├── face_emotion_model.py     # Unified model combining all components
├── main.py                   # Main script with CLI interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── best_overall.pt           # Trained emotion classification model
├── deploy.prototxt           # Face detection model configuration
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model weights
├── train_meta.csv            # Training dataset metadata
├── test_meta.csv            # Test dataset metadata
└── DetectionBaseline.ipynb   # Original notebooks (for reference)
    CropImage.ipynb
    ClassificationBaseline.ipynb
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the required model files**:
   - `deploy.prototxt` - Face detection model configuration
   - `res10_300x300_ssd_iter_140000.caffemodel` - Face detection model weights
   - `best_overall.pt` - Trained emotion classification model

## Usage

### Command-Line Interface

The main script provides several modes of operation:

#### 1. Training Pipeline
Train the complete emotion recognition system:
```bash
python main.py --mode train --data_folder /path/to/your/data
```

Additional training options:
```bash
python main.py --mode train --data_folder /path/to/data \
    --epochs 50 --batch_size 32 --lr 1e-4 --arch resnet18 --freeze_backbone
```

#### 2. Single Image Inference
Run inference on a single image:
```bash
python main.py --mode inference --data_folder /path/to/data \
    --image_path /path/to/image.jpg --save_viz /path/to/output.jpg
```

#### 3. Batch Inference
Process multiple images in a folder:
```bash
python main.py --mode batch --data_folder /path/to/data \
    --image_folder /path/to/images --output_folder /path/to/results
```

#### 4. Demo Mode
Run a demo with sample images:
```bash
python main.py --mode demo --data_folder /path/to/data
```

### Programmatic Usage

You can also use the modules directly in your Python code:

```python
from face_emotion_model import create_face_emotion_model, demo_inference

# Create the unified model
model = create_face_emotion_model(
    prototxt_path="deploy.prototxt",
    weights_path="res10_300x300_ssd_iter_140000.caffemodel", 
    emotion_model_path="best_overall.pt"
)

# Run inference
results = demo_inference("path/to/image.jpg", model)

# Process results
for result in results:
    print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.3f}")
```

## Data Structure

The system expects the following data structure:

```
data_folder/
├── ImageData/                    # Source images
├── FACES_headshots_6classes/     # FACES dataset
├── BaselineModels/               # Model files and outputs
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── best_overall.pt
│   ├── cropped_single/          # Cropped face images
│   └── classification_runs/     # Training outputs
├── train_meta.csv               # Training metadata
└── test_meta.csv                # Test metadata
```

## Model Architecture

### Face Detection
- **Model**: OpenCV DNN SSD ResNet10 (300x300)
- **Input**: RGB images of any size
- **Output**: Bounding boxes with confidence scores
- **Confidence Threshold**: 0.2 (adjustable)

### Emotion Classification  
- **Model**: ResNet18 (or VGG16) with ImageNet pretrained weights
- **Input**: 224x224 RGB face crops
- **Output**: 7 emotion classes with confidence scores
- **Classes**: Happy, Sad, Angry, Fear, Disgust, Neutral, Surprise

## Training Process

The training pipeline consists of 4 main steps:

1. **Face Detection**: Evaluate face detection performance on the dataset
2. **Image Cropping**: Extract face regions using ground truth bounding boxes
3. **Dataset Combination**: Combine cropped images with FACES dataset
4. **Model Training**: Train emotion classification model with cross-validation

## Performance

The system achieves good performance on the combined dataset:
- **Face Detection**: High precision and recall with IoU@0.5 evaluation
- **Emotion Classification**: Trained with class-weighted loss and data augmentation
- **End-to-End**: Complete pipeline from raw images to emotion predictions

## Customization

### Adding New Emotion Classes
1. Update the `TARGET_CLASSES` list in `emotion_classifier.py`
2. Retrain the model with new labeled data
3. Update the label mappings

### Using Different Models
1. Modify the `build_model()` function in `emotion_classifier.py`
2. Add support for new architectures (e.g., EfficientNet, Vision Transformer)
3. Adjust input preprocessing accordingly

### Adjusting Detection Parameters
1. Modify `confidence_threshold` in `FaceDetector`
2. Adjust IoU threshold for evaluation
3. Fine-tune detection parameters for your specific use case

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The system automatically detects and uses available GPUs. For CPU-only usage, set `device='cpu'` in `EmotionClassifier`.

2. **Memory Issues**: Reduce batch size or image size if you encounter memory errors.

3. **Model Loading Errors**: Ensure all model files are present and paths are correct.

4. **Import Errors**: Make sure all dependencies are installed and Python path includes the project directory.

### Performance Optimization

1. **Batch Processing**: Use batch inference for multiple images
2. **Model Quantization**: Consider quantizing models for faster inference
3. **GPU Acceleration**: Ensure CUDA is properly installed for GPU acceleration

## Contributing

To extend or modify the system:

1. **Add New Features**: Extend the existing classes with new methods
2. **Improve Models**: Experiment with different architectures and training strategies  
3. **Enhance Visualization**: Add more visualization options for results
4. **Optimize Performance**: Implement optimizations for faster inference

## License

This project is based on the original Emotion Teller baseline notebooks. Please refer to the original project for licensing information.

## Acknowledgments

- Original Emotion Teller project and baseline notebooks
- OpenCV DNN face detection models
- PyTorch and torchvision for deep learning framework
- FACES dataset for additional training data
