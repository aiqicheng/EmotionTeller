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

def _expand_and_clip_box(xyxy: np.ndarray, expand: float, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # make it square then expand
    side = max(w, h) * expand
    x1n = max(0, int(round(cx - side/2)))
    y1n = max(0, int(round(cy - side/2)))
    x2n = min(W-1, int(round(cx + side/2)))
    y2n = min(H-1, int(round(cy + side/2)))
    return np.array([x1n, y1n, x2n, y2n], dtype=np.int32)

import cv2
def annotate_image(res) -> Image.Image:
    expand_factor = 1.3
    font_scale = 0.6
    font_thickness = 1
    H, W = res.plot().shape[:2]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = res.boxes.conf.cpu().numpy().astype(np.float32)
    emo_dic = {0:'Neutral',1:'Happy',2:'Surprise',3:'Sad',4:'Angry',5:'Fear',6:'Disgust'}
    emotions = [emo_dic[int(x)] for x in res.boxes.cls.tolist()]
    faces = []
    for i, (xyxy, score, emotion) in enumerate(zip(boxes_xyxy, confs, emotions)):
        xyxy_exp = _expand_and_clip_box(xyxy, expand_factor, W, H)
        faces.append({
            "det_idx": i,
            "xyxy_exp": xyxy_exp.tolist(),
            "det_conf": float(score),
            "emotion": emotion 
        })
    canvas = cv2.cvtColor(res.orig_img.copy(), cv2.COLOR_RGB2BGR)  
    for f in faces:
        x1, y1, x2, y2 = map(int, f["xyxy_exp"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{f["emotion"]} ({f["det_conf"]:.2f})'
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        # --- Determine label placement dynamically ---
        # Default: top-left
        label_x, label_y = x1 + 2, y1 - 4
        box_top = y1 - th - 6
        box_bottom = y1

        # If no space above (text would go off-frame)
        if box_top < 0:
            # Try bottom-left
            if y2 + th + 6 < H:
                label_y = y2 + th + 4
                box_top = y2
                box_bottom = y2 + th + 6
            # If also no space at bottom, try middle-right
            elif x2 + tw + 6 < W:
                label_x = x2 + 6
                label_y = y1 + (y2 - y1)//2
                box_top = label_y - th//2 - 3
                box_bottom = label_y + th//2 + 3
            # Else middle-left (fallback)
            elif x1 - tw - 6 > 0:
                label_x = x1 - tw - 6
                label_y = y1 + (y2 - y1)//2
                box_top = label_y - th//2 - 3
                box_bottom = label_y + th//2 + 3

        # Draw filled background rectangle
        cv2.rectangle(canvas, (label_x - 2, box_top), (label_x + tw + 2, box_bottom), (0, 255, 0), -1)

        # Draw text
        cv2.putText(canvas, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (0, 0, 0), cfg.font_thickness, cv2.LINE_AA)
    custom_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return custom_img

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

DEVICES = get_available_devices()
DEVICE = DEVICES[0] 

CONF_THRES = 0.25
IOU_THRES  = 0.45
IMG_SIZE   = 1024
MAX_DET    = 300
AGNOSTIC_NMS = False
FP16 = False

def load_yolo_model():
    WEIGHTS = Path("../YOLO_training/runs/yolo11m_finetuned2/weights/last.pt")      
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

    return ann, df

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
    return ann, df


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
        