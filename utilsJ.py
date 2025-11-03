import pandas as pd

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

    return {
        'file_name': file_name,
        'gt_bboxes': gt_bboxes,
        'gt_categories': gt_categories,
        'pred_bboxes': pred_bboxes,
        'pred_confs': pred_confs,
        'pred_classes': pred_classes,
        'pred_categories': pred_categories
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
        item: 

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