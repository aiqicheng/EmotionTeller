# pipeline.py
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
from PIL import Image
import json, csv, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd

import cv2


# --- Detector (Ultralytics YOLO) ---
from ultralytics import YOLO  # pip install ultralytics

# ---------------------------
# Configuration
# ---------------------------
def _is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _get_default_paths():
    """Get default paths based on environment (Colab vs local)."""
    if _is_colab():
        # Colab paths
        return {
            "detector_path": "/content/drive/MyDrive/emo/BaselineModels/runs/detect_finetune_yolo11n_face_train/weights/best.pt",
            "classifier_path": "/content/drive/MyDrive/emo/BaselineModels/runs/classification_resnet18_V2/best_overall.pt"
        }
    else:
        # Local paths
        return {
            "detector_path": "./BaselineModels/yolo11n-face-best.pt",
            "classifier_path": "./BaselineModels/best_overall.pt"
        }

@dataclass
class Config:
    # Paths will be auto-detected based on environment (Colab vs local)
    # You can override these by passing custom paths when creating Config
    detector_path: str = None
    classifier_path: str = None
    class_names: List[str] = None               # e.g. ["Neutral","Happy","Angry","Surprise","Sad","Fear","Disgust"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    det_conf_thres: float = 0.30
    det_iou_thres: float = 0.50                  # NMS threshold (Ultralytics handles this internally)
    crop_expand: float = 1.2                     # expand bbox to include a little context
    input_size: int = 224                        # classifier input size
    batch_size: int = 32
    half: bool = False                           # True if your GPU supports fp16 safely
    draw: bool = True
    font_scale: float = 0.6
    font_thickness: int = 1

    def __post_init__(self):
        if self.class_names is None:
            # TODO: set to your classifier's label order
            self.class_names = ["Neutral","Happy","Angry","Surprise","Sad","Fear","Disgust"]
        
        # Auto-detect paths if not provided
        default_paths = _get_default_paths()
        if self.detector_path is None:
            self.detector_path = default_paths["detector_path"]
        if self.classifier_path is None:
            self.classifier_path = default_paths["classifier_path"]

# ---------------------------
# Utilities
# ---------------------------
# --- helpers to align with your label format ---


def _xyxy_to_xywh_percent(x1, y1, x2, y2, W, H):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    # convert to percent of width/height
    return [
        (x1 / W) * 100.0,
        (y1 / H) * 100.0,
        (w  / W) * 100.0,
        (h  / H) * 100.0,
    ]

def pack_image_prediction_row(image_path, faces_out, W, H):
    """Return a dict shaped like your GT rows.
    file_name,objects,original_width,original_height,emotions
    where 'objects' is a JSON-like string with bbox (percent) & categories & scores
    """
    file_name = Path(image_path).name
    boxes = []
    cats  = []
    scores = []
    for f in faces_out:
        # IMPORTANT: use the detector bbox (not expanded) for IoU-based eval
        x1, y1, x2, y2 = map(int, f["bbox_xyxy"])  # original detector box
        boxes.append(_xyxy_to_xywh_percent(x1, y1, x2, y2, W, H))
        cats.append(f["pred_label"])
        # score for ranking: combine det_conf * cls_prob_of_pred (optional but common)
        probs = f["pred_probs"]
        cls_prob = probs[f["pred_class_id"]]
        scores.append(float(f["det_conf"] * cls_prob))

    objects = {"bbox": boxes, "categories": cats, "scores": scores}
    emotions = cats  # flat list, same order as boxes

    return {
        "file_name": file_name,
        "objects": json.dumps(objects),           # keep as JSON string
        "original_width": W,
        "original_height": H,
        "emotions": json.dumps(emotions)
    }

def _xywh_to_xyxy(xywh: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    # Ultralytics results are usually xyxy already; this is here if you need it
    # xywh: (cx, cy, w, h)
    cx, cy, w, h = xywh
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

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

def _crop(img: np.ndarray, xyxy_int: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy_int.tolist()
    return img[y1:y2, x1:x2]

def build_cls_transform(img_size: int):
    # Match the preprocessing you used during classifier training
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # TODO: set your mean/std from training
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

def faces_to_df(result):
    rows = [
        {
            "image_path": result["image_path"],
            "x1": f["bbox_xyxy"][0],
            "y1": f["bbox_xyxy"][1],
            "x2": f["bbox_xyxy"][2],
            "y2": f["bbox_xyxy"][3],
            "confidence": f["det_conf"],         # detector confidence
            "emotion": f["pred_label"],          # classifier label
            "prob_of_emotion": f["pred_probs"][f["pred_class_id"]],  # optional
        }
        for f in result.get("faces", [])
    ]
    return pd.DataFrame(rows)


# ---------------------------
# Load models
# ---------------------------
def load_detector(cfg: Config):
    model = YOLO(cfg.detector_path)
    return model

import torch
from collections import OrderedDict

def _strip_module_prefix(sd):
    out = OrderedDict()
    for k, v in sd.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out

def build_classifier_from_arch(arch: str, num_classes: int):
    import torchvision.models as tvm
    arch = (arch or "resnet18").lower()
    if arch in ["resnet18", "resnet18_v2", "resnet18_bn"]:
        m = tvm.resnet18(weights=None)  # torchvision >=0.13: use `weights=None`
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    # Fallback (customize if you trained other nets)
    m = tvm.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def load_classifier(cfg: Config):
    path = cfg.classifier_path
    print(f"Loading classifier checkpoint from {path} ...")
    ckpt = torch.load(path, map_location=cfg.device)

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(
            f"Expected a dict checkpoint with 'model_state'. Got type={type(ckpt)} "
            f"keys={list(ckpt.keys())[:20] if isinstance(ckpt, dict) else 'N/A'}"
        )

    state_dict = _strip_module_prefix(ckpt["model_state"])
    arch = ckpt.get("arch", "resnet18")
    label2id = ckpt.get("label2id", None)

    # Derive class names from the checkpoint (keeps training order)
    if label2id:
        # id2label[i] = label
        id2label = [None] * (max(label2id.values()) + 1)
        for label, idx in label2id.items():
            id2label[idx] = label
        cfg.class_names = id2label
    num_classes = len(cfg.class_names)

    # Build the same architecture you trained
    model = build_classifier_from_arch(arch, num_classes)

    # Load weights (strict=True first; fall back to strict=False with a note)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True), []
    except RuntimeError as e:
        print(f"Strict load failed ({e}). Retrying with strict=False …")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded with strict=False. Missing: {list(missing)[:6]}  Unexpected: {list(unexpected)[:6]}")

    model.to(cfg.device).eval()
    if cfg.half and cfg.device == "cuda":
        model.half()
    print(f"Classifier ready. arch={arch}, num_classes={num_classes}, classes={cfg.class_names}")
    return model


# ---------------------------
# Inference: one image
# ---------------------------
@torch.inference_mode()
def run_on_image(img_path: str, det_model, cls_model, cfg: Config) -> Dict[str, Any]:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # A) Detect faces
    det_preds = det_model.predict(
        source=img_rgb,
        verbose=False,
        conf=cfg.det_conf_thres,
        iou=cfg.det_iou_thres
    )
    # Ultralytics returns a list; take the first (single image)
    det = det_preds[0]
    # Boxes in xyxy
    if det.boxes is None or len(det.boxes) == 0:
        return {
            "image_path": img_path,
            "faces": [],
            "visualization_path": None
        }

    boxes_xyxy = det.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = det.boxes.conf.cpu().numpy().astype(np.float32)

    # B) Crop faces (+expand) and batch-classify
    transform = build_cls_transform(cfg.input_size)
    crops, meta = [], []  # keep mapping back to image
    for i, (xyxy, score) in enumerate(zip(boxes_xyxy, confs)):
        xyxy_exp = _expand_and_clip_box(xyxy, cfg.crop_expand, W, H)
        crop = _crop(img_rgb, xyxy_exp)
        if crop.size == 0:
            continue
        crops.append(transform(crop))  # to tensor
        meta.append({
            "det_idx": i,
            "xyxy": xyxy.tolist(),
            "xyxy_exp": xyxy_exp.tolist(),
            "det_conf": float(score)
        })

    faces_out = []
    if crops:
        batch = torch.stack(crops, dim=0).to(cfg.device)
        if cfg.half and cfg.device == "cuda":
            batch = batch.half()
        logits = cls_model(batch)       # (N, C)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)

        for m, pid, p in zip(meta, pred_ids, probs):
            faces_out.append({
                "bbox_xyxy": m["xyxy"],
                "bbox_xyxy_expanded": m["xyxy_exp"],
                "det_conf": m["det_conf"],
                "pred_class_id": int(pid),
                "pred_label": cfg.class_names[int(pid)],
                "pred_probs": p.tolist()
            })

    # C) Optional drawing
    vis_path = None
    if cfg.draw and faces_out:
        canvas = img_bgr.copy()
        for f in faces_out:
            x1, y1, x2, y2 = map(int, f["bbox_xyxy_expanded"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{f["pred_label"]} ({f["det_conf"]:.2f})'
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.font_thickness)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (0, 0, 0), cfg.font_thickness, cv2.LINE_AA)
        vis_path = str(Path(img_path.replace(".", "_annotated.")))
        cv2.imwrite(vis_path, canvas)

    return {
        "image_path": img_path,
        "faces": faces_out,
        "visualization_path": vis_path
    }

# ---------------------------
# Inference: folder of images
# ---------------------------
def run_on_folder(in_dir: str, out_csv: str, cfg: Config = Config()):
    det_model = load_detector(cfg)
    cls_model = load_classifier(cfg)

    img_paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        img_paths += list(Path(in_dir).rglob(ext))
    img_paths = sorted(img_paths)

    rows = []
    for p in img_paths:
        p = str(p)
        # run detection+classification
        res = run_on_image(p, det_model, cls_model, cfg)

        # get dimensions
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        row = pack_image_prediction_row(
            res["image_path"],
            res["faces"],   # list of dicts from the pipeline
            W, H
        )
        rows.append(row)

    # write the CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name","objects","original_width","original_height","emotions"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return out_csv


# ---------------------------
# (Optional) Video streaming
# ---------------------------
@torch.inference_mode()
def run_on_video(video_path: str, out_path: str, cfg: Config = Config()):
    det_model = load_detector(cfg)
    cls_model = load_classifier(cfg)
    transform = build_cls_transform(cfg.input_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        det = det_model.predict(source=frame_rgb, verbose=False, conf=cfg.det_conf_thres, iou=cfg.det_iou_thres)[0]
        if det.boxes is not None and len(det.boxes) > 0:
            boxes_xyxy = det.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = det.boxes.conf.cpu().numpy().astype(np.float32)

            crops, metas = [], []
            for xyxy, sc in zip(boxes_xyxy, confs):
                xyxy_exp = _expand_and_clip_box(xyxy, cfg.crop_expand, W, H)
                crop = _crop(frame_rgb, xyxy_exp)
                if crop.size == 0: 
                    continue
                crops.append(transform(crop))
                metas.append((xyxy_exp, sc))

            if crops:
                batch = torch.stack(crops, 0).to(cfg.device)
                if cfg.half and cfg.device == "cuda": batch = batch.half()
                logits = cls_model(batch)
                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                ids = probs.argmax(1)

                for (xyxy_e, sc), pid in zip(metas, ids):
                    x1,y1,x2,y2 = map(int, xyxy_e)
                    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                    label = f"{cfg.class_names[int(pid)]} ({sc:.2f})"
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.font_thickness)
                    cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0,255,0), -1)
                    cv2.putText(frame_bgr, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, (0,0,0), cfg.font_thickness, cv2.LINE_AA)

        writer.write(frame_bgr)

    cap.release()
    writer.release()

# ---------------------------
# CLI entry (example)
# ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, help="Path to a single image")
    ap.add_argument("--folder", type=str, help="Folder with images")
    ap.add_argument("--video", type=str, help="Path to a video file")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_video", type=str, default="out.mp4")
    args = ap.parse_args()

    cfg = Config()

    if args.image:
        det = load_detector(cfg)
        cls = load_classifier(cfg)
        r = run_on_image(args.image, det, cls, cfg)
        print(json.dumps(r, indent=2))

    if args.folder:
        run_on_folder(args.folder, out_json=args.out_json, out_csv=args.out_csv, cfg=cfg)

    if args.video:
        run_on_video(args.video, out_path=args.out_video, cfg=cfg)
