from .utils import run_inference
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

def load_yolo11m_model():
    WEIGHTS = Path("../YOLO_training/runs/yolo11m_finetuned4/weights/last.pt")      
    assert WEIGHTS.exists(), f"Model weights not found at {WEIGHTS.resolve()}"
    model = YOLO(str(WEIGHTS))
    return model

def run_yolo11m_model(image: Image.Image = None):
    model = load_yolo11m_model()
    return run_inference(model, image)

