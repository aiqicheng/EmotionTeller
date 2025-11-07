# evaluate_pipeline_map.py
import csv, json, math, argparse, ast
from collections import defaultdict
from pathlib import Path
import cv2

# -------------- Box conversions --------------
def _bbox_to_pixels(bbox, W, H):
    """bbox = [x, y, w, h] in one of: pixels, [0–1], or [0–100] percent."""
    x, y, w, h = bbox
    vals = [abs(x), abs(y), abs(w), abs(h)]
    m = max(vals) if vals else 0.0

    # Heuristic on units
    if m <= 1.0:         # relative [0..1]
        px = x * W; py = y * H; pw = w * W; ph = h * H
    elif m <= 100.0:     # percent [0..100]
        px = (x/100.0) * W; py = (y/100.0) * H
        pw = (w/100.0) * W; ph = (h/100.0) * H
    else:                # absolute pixels
        px, py, pw, ph = x, y, w, h

    return [px, py, pw, ph]

def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

# -------------- Data loading --------------
def _parse_objects_cell(s):
    """
    Handles either JSON (our preds) or Python-literal-like strings (your GT).
    Returns dict with keys 'bbox', 'categories', optionally 'scores'.
    """
    s = s.strip()
    try:
        # First try JSON
        return json.loads(s)
    except Exception:
        # Fallback to Python literal (e.g., "{'bbox': ..., 'categories': ...}")
        return ast.literal_eval(s)

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def expand_gt_rows(gt_rows):
    """Return dict: img -> list of GT items: {bbox_xyxy, label, used=False}"""
    per_img = {}
    for r in gt_rows:
        W = int(float(r["original_width"]))
        H = int(float(r["original_height"]))
        objs = _parse_objects_cell(r["objects"])
        bboxes = objs.get("bbox", [])
        cats   = objs.get("categories", [])
        file_name = r["file_name"]
        items = []
        for b, c in zip(bboxes, cats):
            px = _bbox_to_pixels(b, W, H)
            xyxy = _xywh_to_xyxy(px)
            items.append({"bbox": xyxy, "label": c, "used": False})
        per_img[file_name] = items
    return per_img

def expand_pred_rows(pred_rows):
    """Return list of preds with fields:
    (img, label, score, bbox_xyxy)
    """
    out = []
    for r in pred_rows:
        W = int(float(r["original_width"]))
        H = int(float(r["original_height"]))
        file_name = r["file_name"]
        objs = _parse_objects_cell(r["objects"])
        bboxes = objs.get("bbox", [])
        cats   = objs.get("categories", [])
        scores = objs.get("scores", [1.0]*len(cats))
        for b, c, s in zip(bboxes, cats, scores):
            px = _bbox_to_pixels(b, W, H)
            xyxy = _xywh_to_xyxy(px)
            out.append({"img": file_name, "label": c, "score": float(s), "bbox": xyxy})
    return out

# -------------- AP computation --------------
def voc_ap(rec, prec):
    """11-point interpolation or continuous? We'll do continuous (trapezoidal)."""
    # Add boundary points
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    # Make precision non-increasing
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    # Integrate
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i-1]) * mpre[i]
    return ap

def compute_ap_for_class(preds, gts_per_img, cls_name, iou_thr=0.5):
    """preds: list of {img,label,score,bbox} filtered to label==cls_name"""
    preds = sorted([p for p in preds if p["label"] == cls_name], key=lambda x: -x["score"])
    # Count GTs of this class
    npos = sum(1 for items in gts_per_img.values() for it in items if it["label"] == cls_name)
    if npos == 0:
        return None  # undefined AP if no GT of this class

    tp = [0] * len(preds)
    fp = [0] * len(preds)

    # Reset "used" flags
    for items in gts_per_img.values():
        for it in items:
            it["used"] = False

    for i, p in enumerate(preds):
        gts = gts_per_img.get(p["img"], [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if gt["label"] != cls_name:
                continue
            iou = iou_xyxy(p["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0 and (not gts[best_j]["used"]):
            tp[i] = 1
            gts[best_j]["used"] = True
        else:
            fp[i] = 1

    # Cumulate
    ctp = []
    cfp = []
    s1 = s2 = 0
    for i in range(len(preds)):
        s1 += tp[i]; s2 += fp[i]
        ctp.append(s1); cfp.append(s2)

    # Precision/Recall
    prec = []
    rec = []
    for i in range(len(preds)):
        prec.append(ctp[i] / max(1, (ctp[i] + cfp[i])))
        rec.append(ctp[i] / npos)

    ap = voc_ap(rec, prec)
    return {
        "AP": ap,
        "precision": prec[-1] if prec else 0.0,
        "recall": rec[-1] if rec else 0.0,
        "npos": npos,
        "n_pred": len(preds)
    }

def evaluate(gt_csv, pred_csv, iou_thr=0.5):
    gt_rows = load_csv(gt_csv)
    pred_rows = load_csv(pred_csv)

    gts_per_img = expand_gt_rows(gt_rows)
    preds_all   = expand_pred_rows(pred_rows)

    # class set from GT (safer than from preds)
    cls_set = set()
    for items in gts_per_img.values():
        for it in items:
            cls_set.add(it["label"])
    cls_list = sorted(cls_set)

    per_class = {}
    ap_list = []
    for c in cls_list:
        res = compute_ap_for_class(preds_all, gts_per_img, c, iou_thr=iou_thr)
        if res is None:
            per_class[c] = {"AP": None, "precision": None, "recall": None, "npos": 0, "n_pred": 0}
        else:
            per_class[c] = res
            ap_list.append(res["AP"])

    mAP = sum(ap_list) / len(ap_list) if ap_list else 0.0
    return mAP, per_class

# -------------- Visualization --------------
def _calculate_adaptive_scale(W: int, H: int, base_font_scale: float = 1.0, 
                               base_thickness: int = 2, reference_size: int = 1000):
    """
    Calculate adaptive font scale and thickness based on image resolution.
    
    Args:
        W: Image width
        H: Image height
        base_font_scale: Base font scale (default 1.0)
        base_thickness: Base line thickness (default 2)
        reference_size: Reference image size for scaling (default 1000 pixels)
    
    Returns:
        Tuple of (adaptive_font_scale, adaptive_thickness)
    """
    # Use the smaller dimension to ensure consistent scaling
    min_dim = min(W, H)
    # Scale factor based on reference size
    scale_factor = min_dim / reference_size
    # Adaptive font scale (clamp to reasonable bounds)
    adaptive_font_scale = max(0.5, min(3.0, base_font_scale * scale_factor))
    # Adaptive thickness (ensure it's at least 1)
    adaptive_thickness = max(1, int(base_thickness * scale_factor))
    return adaptive_font_scale, adaptive_thickness

def get_class_color(class_name: str):
    """Return a distinct BGR color for each emotion class.
    Uses the same color mapping as two_step_pipeline.py for consistency."""
    normalized = class_name.strip().lower()
    color_map = {
        "neutral": (128, 128, 128),    # Gray
        "happy": (0, 255, 0),          # Green
        "angry": (0, 0, 255),          # Red
        "surprise": (255, 165, 0),     # Orange
        "sad": (0, 255, 255),          # Cyan
        "fear": (255, 0, 255),         # Magenta
        "disgust": (0, 128, 255),      # Orange-Red
    }
    # Return the color for the class, or default to white if not found
    return color_map.get(normalized, (255, 255, 255))

def draw_ground_truth(gt_csv: str, image_dir: str, image_filename: str, 
                     output_path: str = None, font_scale: float = 1.0, 
                     font_thickness: int = 2, bbox_thickness: int = 4):
    """
    Draw ground truth bounding boxes and labels on an image.
    
    Args:
        gt_csv: Path to ground truth CSV file
        image_dir: Directory containing the images
        image_filename: Name of the image file to visualize
        output_path: Path to save annotated image (if None, saves as image_filename_gt_annotated.jpg)
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        bbox_thickness: Thickness of bounding box lines
    
    Returns:
        Path to saved annotated image
    """
    # Load ground truth data
    gt_rows = load_csv(gt_csv)
    gts_per_img = expand_gt_rows(gt_rows)
    
    # Find the image in ground truth
    if image_filename not in gts_per_img:
        raise ValueError(f"Image '{image_filename}' not found in ground truth CSV")
    
    # Load the image
    image_path = Path(image_dir) / image_filename
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get image dimensions
    H, W = img.shape[:2]
    
    # Calculate adaptive font scale and thickness based on image resolution
    adaptive_font_scale, adaptive_thickness = _calculate_adaptive_scale(
        W, H, font_scale, font_thickness
    )
    adaptive_bbox_thickness = max(2, int(bbox_thickness * (min(W, H) / 1000.0)))
    
    # Get ground truth annotations for this image
    gt_items = gts_per_img[image_filename]
    
    # Draw bounding boxes and labels
    for item in gt_items:
        x1, y1, x2, y2 = map(int, item["bbox"])
        label = item["label"]
        color = get_class_color(label)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, adaptive_bbox_thickness)
        
        # Draw label background and text
        label_text = f"{label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                     adaptive_font_scale, adaptive_thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, (0, 0, 0), 
                   adaptive_thickness, cv2.LINE_AA)
    
    # Save annotated image
    if output_path is None:
        stem = Path(image_filename).stem
        suffix = Path(image_filename).suffix
        output_path = str(Path(image_dir) / f"{stem}_gt_annotated{suffix}")
    else:
        output_path = str(output_path)
    
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to: {output_path}")
    return output_path

def draw_all_ground_truth(gt_csv: str, image_dir: str, output_dir: str = None,
                         font_scale: float = 1.0, font_thickness: int = 2, 
                         bbox_thickness: int = 4):
    """
    Draw ground truth annotations for all images in the CSV.
    
    Args:
        gt_csv: Path to ground truth CSV file
        image_dir: Directory containing the images
        output_dir: Directory to save annotated images (if None, saves in image_dir)
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        bbox_thickness: Thickness of bounding box lines
    
    Returns:
        List of paths to saved annotated images
    """
    gt_rows = load_csv(gt_csv)
    gts_per_img = expand_gt_rows(gt_rows)
    
    if output_dir is None:
        output_dir = image_dir
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_paths = []
    for image_filename in gts_per_img.keys():
        try:
            stem = Path(image_filename).stem
            suffix = Path(image_filename).suffix
            output_path = str(Path(output_dir) / f"{stem}_gt_annotated{suffix}")
            
            output_path = draw_ground_truth(
                gt_csv, image_dir, image_filename, 
                output_path=output_path,
                font_scale=font_scale,
                font_thickness=font_thickness,
                bbox_thickness=bbox_thickness
            )
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
    
    print(f"Processed {len(output_paths)} images")
    return output_paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", required=False, help="Ground-truth CSV (your labeled metadata)")
    ap.add_argument("--pred_csv", required=False, help="Predictions CSV from pipeline")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default 0.5)")
    
    # Visualization options
    ap.add_argument("--draw_gt", action="store_true", help="Draw ground truth annotations")
    ap.add_argument("--image_dir", type=str, help="Directory containing images")
    ap.add_argument("--image_filename", type=str, help="Specific image to visualize (if not provided, visualizes all)")
    ap.add_argument("--output_dir", type=str, help="Output directory for annotated images")
    ap.add_argument("--font_scale", type=float, default=1.0, help="Font scale for labels (default 1.0)")
    ap.add_argument("--font_thickness", type=int, default=2, help="Font thickness for labels (default 2)")
    ap.add_argument("--bbox_thickness", type=int, default=4, help="Bounding box line thickness (default 4)")
    
    args = ap.parse_args()
    
    # Visualization mode
    if args.draw_gt:
        if not args.gt_csv or not args.image_dir:
            ap.error("--draw_gt requires --gt_csv and --image_dir")
        
        if args.image_filename:
            # Draw single image
            draw_ground_truth(
                args.gt_csv, 
                args.image_dir, 
                args.image_filename, 
                output_path=args.output_dir,
                font_scale=args.font_scale,
                font_thickness=args.font_thickness,
                bbox_thickness=args.bbox_thickness
            )
        else:
            # Draw all images
            draw_all_ground_truth(
                args.gt_csv, 
                args.image_dir, 
                output_dir=args.output_dir,
                font_scale=args.font_scale,
                font_thickness=args.font_thickness,
                bbox_thickness=args.bbox_thickness
            )
    elif args.gt_csv and args.pred_csv:
        # Evaluation mode
        mAP, per_class = evaluate(args.gt_csv, args.pred_csv, iou_thr=args.iou)

        print(f"mAP@{args.iou:.2f}: {mAP:.4f}")
        for c, r in per_class.items():
            print(f"{c:10s} AP={r['AP'] if r['AP'] is not None else 'NA'} "
                  f"(npos={r['npos']}, n_pred={r['n_pred']})")
    else:
        ap.error("Must provide either (--draw_gt) or (--gt_csv and --pred_csv)")
