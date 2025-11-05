# evaluate_pipeline_map.py
import csv, json, math, argparse, ast
from collections import defaultdict
from pathlib import Path

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", required=True, help="Ground-truth CSV (your labeled metadata)")
    ap.add_argument("--pred_csv", required=True, help="Predictions CSV from pipeline")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default 0.5)")
    args = ap.parse_args()

    mAP, per_class = evaluate(args.gt_csv, args.pred_csv, iou_thr=args.iou)

    print(f"mAP@{args.iou:.2f}: {mAP:.4f}")
    for c, r in per_class.items():
        print(f"{c:10s} AP={r['AP'] if r['AP'] is not None else 'NA'} "
              f"(npos={r['npos']}, n_pred={r['n_pred']})")
