from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from ultralytics import YOLO


## Helper functions
def get_available_devices() -> List[str]:
    devices = ["cpu"]
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                devices.insert(0, "cuda")
        except Exception:
            pass
        try:
            if torch.backends.mps.is_available(): 
                devices.insert(0, "mps")
        except Exception:
            pass
    return devices

def results_to_df(results) -> pd.DataFrame:
    """Convert a single Ultralytics Results object to a tidy DataFrame."""
    if results is None or results.boxes is None or results.boxes.xyxy is None or len(results.boxes) == 0:
        return pd.DataFrame(columns=["x1","y1","x2","y2","confidence","class_id","class_name"])

    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],))
    cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
    names = results.names if hasattr(results, "names") else {}

    rows = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        cid = int(cls[i]) if i < len(cls) else -1
        rows.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "confidence": float(conf[i]), "class_id": cid, "class_name": names.get(cid, str(cid))
        })
    return pd.DataFrame(rows)

def annotate_image(results) -> Image.Image:
    """Return annotated PIL image from Ultralytics results."""
    return Image.fromarray(results.plot())

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

DEVICES = get_available_devices()
DEVICE = DEVICES[0] 
OUTPUT_DIR = Path("outputs")

CONF_THRES = 0.25
IOU_THRES  = 0.45
IMG_SIZE   = 640
MAX_DET    = 300
AGNOSTIC_NMS = False
FP16 = False

def load_yolo_model():
    WEIGHTS = Path("yolo_model/last.pt")      
    assert WEIGHTS.exists(), f"Model weights not found at {WEIGHTS.resolve()}"
    model = YOLO(str(WEIGHTS))
    return model


def upload_image(model, image) -> Union[Tuple[Image.Image,Image.Image, pd.DataFrame], None]:
    img = image.convert("RGB")
    results_list = model.predict(
        source=np.array(img),
        conf=CONF_THRES, iou=IOU_THRES, device=DEVICE,
        imgsz=int(IMG_SIZE), max_det=int(MAX_DET),
        agnostic_nms=AGNOSTIC_NMS, half=FP16, verbose=False
    )
    res = results_list[0]
    ann = annotate_image(res)
    df = results_to_df(res)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUTPUT_DIR / "uploaded_annotated.jpg"
    ann.save(out_img)
    if not df.empty:
        save_df(df, OUTPUT_DIR / "uploaded_detections.csv")
    return img, ann, df

def webcam(model, image) -> Union[Tuple[Image.Image,Image.Image, pd.DataFrame], None]:
    img = image.convert("RGB")
    results_list = model.predict(
        source=np.array(img),
        conf=CONF_THRES, iou=IOU_THRES, device=DEVICE,
        imgsz=int(IMG_SIZE), max_det=int(MAX_DET),
        agnostic_nms=AGNOSTIC_NMS, half=FP16, verbose=False
    )
    res = results_list[0]
    ann = annotate_image(res)
    df = results_to_df(res)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img = OUTPUT_DIR / "webcam_snapshot_annotated.jpg"
    ann.save(out_img)
    if not df.empty:
        save_df(df, OUTPUT_DIR / "webcam_snapshot_detections.csv")
    return img, ann, df


class model_output():
    def __init__(self,  
                 webcam: bool = False):
        self.webcam = webcam
    
    def run_model(self, image: Image.Image = None):
        model = load_yolo_model()
        if self.webcam:
            return upload_image(model, image)
        else:
            return webcam(model, image)
        