import pandas as pd
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
import torch
from PIL import Image
from IPython.display import display, clear_output
from ultralytics.engine.results import Boxes

emo_dic = {'Neutral':0,'Happy':1,'Surprise':2,'Sad':3,'Angry':4,'Fear':5,'Disgust':6}

def extract_bbox_data(ground_truth_row, prediction_result, cat_class_dict):
    """
    Extracts ground truth and predicted bounding box information and categories/classes for a single image.

    Args:
        ground_truth_row: A pandas Series representing a row from the ground truth DataFrame.
        prediction_result: An object containing the prediction results for the corresponding image.
        cat_class_dict: A dictionary mapping class names to their corresponding class indices.

    Returns:
        A dictionary containing the extracted information.
    """
    file_name = ground_truth_row['file_name']

    class_cat_dict = {v: k for k, v in cat_class_dict.items()}

    # Extract width and height

    width = ground_truth_row['original_width']
    height = ground_truth_row['original_height']

    # Extract ground truth data
    gt_objects = ground_truth_row['objects']
    gt_bboxes = gt_objects['bbox']
    gt_categories = gt_objects['categories']

    # Extract predicted data
    pred_boxes = prediction_result.boxes
    pred_bboxes = pred_boxes.xyxy.tolist()
    pred_confs = pred_boxes.conf.tolist()
    pred_classes = pred_boxes.cls.tolist()
    pred_categories = [class_cat_dict[c] for c in pred_classes if c in class_cat_dict]

    if ground_truth_row['path_to_img'] is not None:
        return {
            'file_name': file_name,
            'gt_bboxes': gt_bboxes,
            'gt_categories': gt_categories,
            'pred_bboxes': pred_bboxes,
            'pred_confs': pred_confs,
            'pred_classes': pred_classes,
            'pred_categories': pred_categories,
            'original_width': width,
            'original_height': height,
            'path_to_img': ground_truth_row['path_to_img']
        }
    return {
        'file_name': file_name,
        'gt_bboxes': gt_bboxes,
        'gt_categories': gt_categories,
        'pred_bboxes': pred_bboxes,
        'pred_confs': pred_confs,
        'pred_classes': pred_classes,
        'pred_categories': pred_categories,
        'original_width': width,
        'original_height': height
    }


def combine_gt_pred(ground_truth_df, pred_results, cat_class_dict):
    """
    Combines the ground truth dataframe with the list of predictions to produce a single dataframe.

    Args:
        ground_truth_df: A pandas Dataframe containing the ground truth for every image.
        prediction_result: A list of objects containing the prediction results for the corresponding image.
        cat_class_dict: A dictionary mapping class names to their corresponding class indices.

    Returns:
        A dataframe that combines the information from ground_truth_df and the prediciont results.
    """
    combined_data = []

    for i in range(len(ground_truth_df)):
        ground_truth_row = ground_truth_df.iloc[i]
        prediction_result = pred_results[i]
        extracted_data = extract_bbox_data(ground_truth_row, prediction_result,cat_class_dict)
        combined_data.append(extracted_data)
    return pd.DataFrame(combined_data)

def xywh2xcycwh(x,y,w,h):
    """
    Transforms from top-left coordinates of bbox to center coordinates.

    Args:
        x: top-left x coordinate
        y: top-left y coordinate
        w: width of box
        h: height of box

    Returns:
        [xc,yc,wc,hc] : the center coordinates for the bounding box.
    """
    return [x + w/2, y + h/2, w/2, h/2]

def label(item):
    """
    Generates the label we will save in the .txt file for the YOLO format

    Args:
        item: A row from the dataframe read from the meta_train.csv or meta_test.csv

    Returns:
        [xc,yc,wc,hc] : the center coordinates for the bounding box.
    """
    objs = item['objects']
    text_label = ''
    for ind, emotion in enumerate(objs['categories']):
        x,y,w,h = np.array(objs['bbox'][ind])/100 # Unpack the list into four variables
        xc,yc,wc,hc = xywh2xcycwh(x,y,w,h) # Pass the four variables to the function
        text_label += f'{emo_dic[emotion]} {xc} {yc} {wc} {hc}\n'
    return text_label

def calculate_iou(box1, box2, box1_format='xywh', box2_format='xywh'):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): Bounding box coordinates. Format specified by box1_format.
        box2 (list): Bounding box coordinates. Format specified by box2_format.
        box1_format (str): Format of box1 ('xywh' or 'xyxy').
        box2_format (str): Format of box2 ('xywh' or 'xyxy').

    Returns:
        float: The IoU score.
    """

    # Convert box1 to xyxy format
    if box1_format == 'xywh':
        box1_x_min, box1_y_min, box1_width, box1_height = box1
        box1_x_max = box1_x_min + box1_width
        box1_y_max = box1_y_min + box1_height
    elif box1_format == 'xyxy':
        box1_x_min, box1_y_min, box1_x_max, box1_y_max = box1
    else:
        raise ValueError("box1_format must be 'xywh' or 'xyxy'")

    # Convert box2 to xyxy format
    if box2_format == 'xywh':
        box2_x_min, box2_y_min, box2_width, box2_height = box2
        box2_x_max = box2_x_min + box2_width
        box2_y_max = box2_y_min + box2_height
    elif box2_format == 'xyxy':
        box2_x_min, box2_y_min, box2_x_max, box2_y_max = box2
    else:
        raise ValueError("box2_format must be 'xywh' or 'xyxy'")


    # Determine the coordinates of the intersection rectangle
    x_min_inter = max(box1_x_min, box2_x_min)
    y_min_inter = max(box1_y_min, box2_y_min)
    x_max_inter = min(box1_x_max, box2_x_max)
    y_max_inter = min(box1_y_max, box2_y_max)

    # Compute the area of intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    area_inter = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    area_box1 = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
    area_box2 = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

    # Compute the Intersection over Union by dividing the intersection area by the union area
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union if area_union > 0 else 0

    return iou

def compute_ap(recall, precision):
    """
    Computes the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        float: The average precision.
    """
    # Append sentinel values to make sure the curve starts and ends at 0 and 1
    mrecall = np.concatenate(([0.], recall, [1.]))
    mprecision = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mprecision.size - 1, 0, -1):
        mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

    # To calculate area under PR curve, look for points where recall changes
    i = np.where(mrecall[1:] != mrecall[:-1])[0]

    # And sum (\Delta recall) * prec
    ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])
    return ap

def evaluate_detections(df_combined, iou_threshold=0.5):
    """
    Evaluates object detection predictions against ground truth.

    Args:
        df_combined (pd.DataFrame): DataFrame with 'gt_bboxes', 'gt_categories',
                                   'pred_bboxes', 'pred_categories', 'pred_confs',
                                   'original_width', 'original_height' columns.
        iou_threshold (float): The IoU threshold for considering a detection as a True Positive.

    Returns:
        tuple: A tuple containing:
            - precision (float): The overall precision.
            - recall (float): The overall recall.
            - average_precisions (dict): A dictionary of AP for each class.
    """
    true_positives = defaultdict(list)
    false_positives = defaultdict(list)
    false_negatives = defaultdict(list)
    total_ground_truths = defaultdict(int)

    for index, row in df_combined.iterrows():
        gt_bboxes = row['gt_bboxes']
        gt_categories = row['gt_categories']
        pred_bboxes = row['pred_bboxes']
        pred_categories = row['pred_categories']
        pred_confs = row['pred_confs']
        original_width = row['original_width']
        original_height = row['original_height']


        # Track which ground truth boxes have been matched to a prediction
        gt_matched = [False] * len(gt_bboxes)

        for i in range(len(pred_bboxes)):
            pred_box = pred_bboxes[i]
            pred_category = pred_categories[i]
            pred_conf = pred_confs[i]

            best_iou = 0
            best_match_idx = -1

            # Find the best matching ground truth box for the current prediction
            for j in range(len(gt_bboxes)):
                gt_box_percent = gt_bboxes[j]
                gt_category = gt_categories[j]

                # Scale the ground truth bounding box to absolute pixel values
                gt_box_scaled = [
                    gt_box_percent[0] * original_width / 100,
                    gt_box_percent[1] * original_height / 100,
                    gt_box_percent[2] * original_width / 100,
                    gt_box_percent[3] * original_height / 100,
                ]

                # Check if the categories match and the ground truth box hasn't been matched yet
                if pred_category == gt_category and not gt_matched[j]:
                    iou = calculate_iou(pred_box, gt_box_scaled, box1_format='xyxy', box2_format='xywh')
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = j

            # Determine if the prediction is a True Positive or False Positive
            if best_iou >= iou_threshold:
                true_positives[pred_category].append({'conf': pred_conf, 'is_tp': True})
                gt_matched[best_match_idx] = True # Mark the ground truth box as matched
            else:
                false_positives[pred_category].append({'conf': pred_conf, 'is_tp': False})

        # Count False Negatives (unmatched ground truth boxes)
        for j in range(len(gt_bboxes)):
            if not gt_matched[j]:
                false_negatives[gt_categories[j]].append(True)

        # Count total ground truths per category
        for category in gt_categories:
            total_ground_truths[category] += 1


    # Calculate Precision, Recall, and AP for each class
    average_precisions = {}
    for category in set(true_positives.keys()) | set(false_positives.keys()):
        detections = sorted(true_positives[category] + false_positives[category], key=lambda x: x['conf'], reverse=True)

        if not detections:
            average_precisions[category] = 0.0
            continue

        tp_list = [d['is_tp'] for d in detections]
        fp_list = [not d['is_tp'] for d in detections]

        tp_cumulative = np.cumsum(tp_list).astype(float)
        fp_cumulative = np.cumsum(fp_list).astype(float)

        precision = tp_cumulative / (tp_cumulative + fp_cumulative)
        recall = tp_cumulative / total_ground_truths.get(category, 0) if total_ground_truths.get(category, 0) > 0 else 0

        average_precisions[category] = compute_ap(recall, precision)

    # Calculate overall Precision and Recall (considering all classes together)
    all_detections = []
    for category in true_positives.keys():
        all_detections.extend(true_positives[category])
    for category in false_positives.keys():
        all_detections.extend(false_positives[category])

    all_detections = sorted(all_detections, key=lambda x: x['conf'], reverse=True)

    all_tp_list = [d['is_tp'] for d in all_detections]
    all_fp_list = [not d['is_tp'] for d in all_detections]

    all_tp_cumulative = np.cumsum(all_tp_list).astype(float)
    all_fp_cumulative = np.cumsum(all_fp_list).astype(float)

    overall_precision = all_tp_cumulative[-1] / (all_tp_cumulative[-1] + all_fp_cumulative[-1]) if len(all_tp_cumulative) > 0 else 0.0
    overall_recall = all_tp_cumulative[-1] / sum(total_ground_truths.values()) if sum(total_ground_truths.values()) > 0 else 0.0


    return overall_precision, overall_recall, average_precisions

def compute_map(average_precisions):
    """
    Computes the mean average precision (mAP).

    Args:
        average_precisions (dict): A dictionary of AP for each class.

    Returns:
        float: The mean average precision.
    """
    if not average_precisions:
        return 0.0
    return np.mean(list(average_precisions.values()))

def plot_bboxes(row, img_path, emo_dic):
    """
    Plots ground truth and predicted bounding boxes on an image.

    Args:
        row (pd.Series): A row from the combined dataframe containing
                         'gt_bboxes', 'gt_categories', 'pred_bboxes',
                         'pred_categories', 'original_width', 'original_height'.
        img_path (Path): The path to the image file.
        emo_dic (dict): Dictionary mapping emotion labels to integers.
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib

    original_width = row['original_width']
    original_height = row['original_height']

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Plot Ground Truth Bboxes (xywh in percentages)
    if row['gt_bboxes']:
        for gt_bbox_percent, gt_category_list in zip(row['gt_bboxes'], row['gt_categories']):
            # Assuming gt_categories is a list, take the first emotion
            gt_category = gt_category_list[0] if gt_category_list else "Unknown"

            # Scale to absolute pixel values (xywh format)
            gt_x_min = gt_bbox_percent[0] * original_width / 100
            gt_y_min = gt_bbox_percent[1] * original_height / 100
            gt_width = gt_bbox_percent[2] * original_width / 100
            gt_height = gt_bbox_percent[3] * original_height / 100

            rect = patches.Rectangle((gt_x_min, gt_y_min), gt_width, gt_height,
                                     linewidth=2, edgecolor='g', facecolor='none', label=f'GT: {gt_category}')
            ax.add_patch(rect)
            plt.text(gt_x_min, gt_y_min - 5, f'GT: {gt_category}', color='green', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))


    # Plot Predicted Bboxes (xyxy format)
    if row['pred_bboxes']:
        for pred_bbox, pred_category_list in zip(row['pred_bboxes'], row['pred_categories']):
            # Assuming pred_categories is a list, take the first emotion
            pred_category = pred_category_list[0] if pred_category_list else "Unknown"

            # pred_bboxes are already in xyxy format
            pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_bbox
            pred_width = pred_x_max - pred_x_min
            pred_height = pred_y_max - pred_y_min

            rect = patches.Rectangle((pred_x_min, pred_y_min), pred_width, pred_height,
                                     linewidth=2, edgecolor='r', facecolor='none', label=f'Pred: {pred_category}')
            ax.add_patch(rect)
            plt.text(pred_x_min, pred_y_min - 5, f'Pred: {pred_category}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    plt.title(f"Ground Truth and Predicted Bounding Boxes for {row['file_name']}")
    plt.show()

def detect_metrics(train_combined, test_combined):
    # @title
    # Compute metrics for train_combined
    precision_train, recall_train, ap_train = evaluate_detections(train_combined, iou_threshold=0.05)
    map50_train = compute_map(ap_train)

    # Compute metrics for test_combined
    precision_test, recall_test, ap_test = evaluate_detections(test_combined, iou_threshold=0.05)
    map50_test = compute_map(ap_test)

    # Compute mAP50-95
    # This requires evaluating at different IoU thresholds (0.5 to 0.95 with step 0.05)
    map_50_95_train_list = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        _, _, ap_iou = evaluate_detections(train_combined, iou_threshold=iou_thresh)
        map_50_95_train_list.append(compute_map(ap_iou))
    map50_95_train = np.mean(map_50_95_train_list)

    map_50_95_test_list = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        _, _, ap_iou = evaluate_detections(test_combined, iou_threshold=iou_thresh)
        map_50_95_test_list.append(compute_map(ap_iou))
    map50_95_test = np.mean(map_50_95_test_list)


    print("Train Metrics:")
    print(f"Precision: {precision_train:.4f}")
    print(f"Recall: {recall_train:.4f}")
    print(f"mAP5: {map50_train:.4f}")

    print("\nTest Metrics:")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"mAP5: {map50_test:.4f}")
    return

def drop_display(train_combined,test_combined):
    # Get the available indices from the dataframes
    train_indices = train_combined.index.tolist()
    test_indices = test_combined.index.tolist()

    # Create dropdown widgets
    train_dropdown = widgets.Dropdown(
        options=train_indices,
        description='Select Train Index:',
        disabled=False,
    )

    test_dropdown = widgets.Dropdown(
    options=test_indices,
    description='Select Test Index:',
    disabled=False,
    )

    # Output area for plots
    output_train = widgets.Output()
    output_test = widgets.Output()

    # Function to update the plot based on the selected index
    def on_train_dropdown_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            with output_train:
                clear_output(wait=True)
                indx = change['new']
                if not train_combined.empty:
                    plot_bboxes(train_combined.iloc[indx], train_combined.iloc[indx]['path_to_img'], emo_dic)
                else:
                    print("train_combined is empty. Cannot plot.")

    def on_test_dropdown_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            with output_test:
                clear_output(wait=True)
                indx = change['new']
                if not test_combined.empty:
                    plot_bboxes(test_combined.iloc[indx], test_combined.iloc[indx]['path_to_img'], emo_dic)
                else:
                    print("test_combined is empty. Cannot plot.")

    # Link the dropdowns to the update function
    train_dropdown.observe(on_train_dropdown_change)
    test_dropdown.observe(on_test_dropdown_change)

    # Display the dropdowns and output areas
    print("Train Set:")
    display(train_dropdown, output_train)

    print("\nTest Set:")
    display(test_dropdown, output_test)

    # Trigger initial plots
    on_train_dropdown_change({'type': 'change', 'name': 'value', 'new': train_dropdown.value})
    on_test_dropdown_change({'type': 'change', 'name': 'value', 'new': test_dropdown.value})
    return

class CustomResult:
    def __init__(self, boxes, orig_shape, names=None):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.names = names # Include names dictionary for class labels

    def __str__(self):
        return f"CustomResult object with Boxes:\n{self.boxes}\nOriginal Shape: {self.orig_shape}"

def two_step_to_yolo(gt,preds):
    results = []
    for ind, r in enumerate(preds):
        boxes = r['faces']
        width = gt['original_width'].iloc[ind]
        height = gt['original_height'].iloc[ind]
        if len(boxes) == 0:
            box_data = torch.empty(0,6, device='cuda:0')
        else:
            box_data = torch.tensor([b['bbox_xyxy']+[b['det_conf']]+[float(emo_dic[b['pred_label']])] for b in boxes], device='cuda:0') 
        original_image_shape = (height,width)
        custom_boxes = Boxes(box_data, orig_shape=original_image_shape)
        results.append(CustomResult(boxes=custom_boxes,
                                            orig_shape=original_image_shape,
                                            names=emo_dic))
    return results