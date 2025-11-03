"""
Image Cropping Module
Extracted from CropImage.ipynb

This module provides functionality to crop images according to bounding boxes
and prepare datasets for emotion classification training.
"""

import ast
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union


class ImageCropper:
    """
    Image cropper for extracting face regions from images based on bounding boxes.
    
    This class provides functionality to:
    - Parse metadata with bounding box information
    - Crop images according to bounding boxes
    - Save cropped images with proper naming
    - Handle different coordinate formats (pixels, percentages, 0-1 normalized)
    """
    
    def __init__(self, jpeg_quality: int = 95):
        """
        Initialize the image cropper.
        
        Args:
            jpeg_quality: JPEG quality for saving cropped images (1-100)
        """
        self.jpeg_quality = jpeg_quality
    
    def expand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand dataframe by converting objects column and exploding bbox/category pairs.
        
        Args:
            df: DataFrame with 'objects' column containing bbox and category information
            
        Returns:
            Expanded DataFrame with separate rows for each bbox-category pair
        """
        df = df.copy()
        
        # Ensure objects are dicts (not stringified dicts)
        def to_dict(x):
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return {}
            return {}
        
        obj = df['objects'].apply(to_dict)
        
        df_expanded = (
            df.assign(
                bbox=obj.apply(lambda d: d.get('bbox', [])),
                category=obj.apply(lambda d: d.get('categories', [])),
            )
            .explode(['bbox', 'category'], ignore_index=True)
        )
        
        # Drop rows where explode produced NaNs because lists were unequal
        df_expanded = df_expanded.dropna(subset=['bbox', 'category'])
        df_expanded = df_expanded.drop(columns=['emotions', 'source_folder'])
        
        # Add original_width and original_height
        df_expanded['original_width'] = df['original_width']
        df_expanded['original_height'] = df['original_height']
        
        return df_expanded
    
    def _bbox_to_pixels(self, bbox: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to pixel coordinates.
        
        Args:
            bbox: Bounding box [x, y, w, h] in pixels, [0-1], or [0-100] percent
            W: Image width
            H: Image height
            
        Returns:
            Tuple of (left, top, right, bottom) pixel coordinates
        """
        x, y, w, h = bbox
        
        vals = [abs(x), abs(y), abs(w), abs(h)]
        m = max(vals)
        
        if m <= 1.0:        # relative
            px, py, pw, ph = x*W, y*H, w*W, h*H
        elif m <= 100.0:    # percent
            px, py, pw, ph = x*W/100.0, y*H/100.0, w*W/100.0, h*H/100.0
        else:               # pixels
            px, py, pw, ph = x, y, w, h
        
        left = max(0, int(round(px)))
        top = max(0, int(round(py)))
        right = min(W, int(round(px + pw)))
        bottom = min(H, int(round(py + ph)))
        
        if right <= left:  
            right = min(W, left + 1)
        if bottom <= top:  
            bottom = min(H, top + 1)
        
        return left, top, right, bottom
    
    def save_crops_and_annotate(self, df_expanded: pd.DataFrame, images_root: str, 
                               out_root: str, filename_col: str = 'file_name',
                               keep_cols: Tuple[str, ...] = ('file_name', 'category'),
                               drop_failed: bool = True) -> pd.DataFrame:
        """
        Save bbox crops and return a new DataFrame with cropped image information.
        
        Args:
            df_expanded: DataFrame with bbox and category information
            images_root: Root directory containing source images
            out_root: Output directory for cropped images
            filename_col: Column name containing image filenames
            keep_cols: Columns to keep from original DataFrame
            drop_failed: Whether to drop failed crops from results
            
        Returns:
            DataFrame with cropped image paths and metadata
        """
        images_root = Path(images_root)
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        
        counters = {}
        results = []
        
        for _, row in df_expanded.iterrows():
            # Resolve source path
            src_name = str(row.get(filename_col, ""))
            src_path = Path(src_name)
            if not src_path.is_file():
                candidate = images_root / src_path.name
                if candidate.is_file():
                    src_path = candidate
            
            record_base = {k: row.get(k) for k in keep_cols if k in row.index}
            
            # Default failed record
            failed_record = {
                **record_base,
                'cropped_file_path': "",
                'new_width': None,
                'new_height': None,
            }
            
            if not src_path.is_file():
                if not drop_failed:
                    results.append(failed_record)
                continue
            
            try:
                with Image.open(src_path) as im:
                    W, H = im.size
                    W0 = int(row.get('original_width', W) or W)
                    H0 = int(row.get('original_height', H) or H)
                    if abs(W0 - W) > 5 or abs(H0 - H) > 5:
                        W0, H0 = W, H
                    
                    bbox = row['bbox']
                    left, top, right, bottom = self._bbox_to_pixels(bbox, W0, H0)
                    
                    # Clamp to actual file dims
                    left = max(0, min(left, W-1))
                    right = max(left+1, min(right, W))
                    top = max(0, min(top, H-1))
                    bottom = max(top+1, min(bottom, H))
                    
                    crop = im.crop((left, top, right, bottom))
                    new_w = right - left
                    new_h = bottom - top
                    
                    stem = src_path.stem
                    counters[stem] = counters.get(stem, 0) + 1
                    new_name = f"{stem}__crop_{counters[stem]:04d}.jpg"
                    out_path = out_root / new_name
                    
                    crop.save(out_path, format="JPEG", quality=self.jpeg_quality, subsampling=1)
                    
                    results.append({
                        **record_base,
                        'cropped_file_path': str(out_path),
                        'new_width': new_w,
                        'new_height': new_h,
                    })
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                if not drop_failed:
                    results.append(failed_record)
        
        # Build the return df with the exact columns requested
        cols = list(keep_cols) + ['cropped_file_path', 'new_width', 'new_height']
        out_df = pd.DataFrame(results)
        out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
        
        return out_df
    
    def crop_dataset(self, metadata_csv: str, image_folder: str, output_folder: str,
                     train_csv: str = None, test_csv: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Crop entire dataset from metadata CSV.
        
        Args:
            metadata_csv: Path to CSV file with image metadata
            image_folder: Folder containing source images
            output_folder: Folder to save cropped images
            train_csv: Optional path to save training crops CSV
            test_csv: Optional path to save test crops CSV
            
        Returns:
            Tuple of (train_crops_df, test_crops_df)
        """
        # Load and expand metadata
        df = pd.read_csv(metadata_csv)
        df_expanded = self.expand_dataframe(df)
        
        # Crop images
        df_with_crops = self.save_crops_and_annotate(
            df_expanded, image_folder, output_folder, filename_col='file_name'
        )
        
        # Save results
        if train_csv:
            df_with_crops.to_csv(train_csv, index=False)
            print(f"Training crops saved to {train_csv}")
        
        return df_with_crops, None  # Only returning train for now, can be extended for test split


def process_dataset_cropping(metadata_csv: str, image_folder: str, output_folder: str,
                             train_csv: str = None, test_csv: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process dataset for image cropping.
    
    Args:
        metadata_csv: Path to CSV file with image metadata
        image_folder: Folder containing source images
        output_folder: Folder to save cropped images
        train_csv: Optional path to save training crops CSV
        test_csv: Optional path to save test crops CSV
        
    Returns:
        Tuple of (train_crops_df, test_crops_df)
    """
    cropper = ImageCropper()
    return cropper.crop_dataset(metadata_csv, image_folder, output_folder, train_csv, test_csv)


if __name__ == "__main__":
    # Example usage
    data_folder = "/content/drive/MyDrive/emo/"
    model_folder = data_folder + "/BaselineModels/"
    
    # Process training data
    df_path = data_folder + "train_meta.csv"
    image_folder = data_folder + "ImageData/"
    crops_dir = model_folder + "cropped_single/"
    train_csv = model_folder + 'trainval_with_crops.csv'
    
    train_crops, _ = process_dataset_cropping(df_path, image_folder, crops_dir, train_csv)
    print(f"Processed {len(train_crops)} training crops")
    
    # Process test data
    df_path_test = data_folder + "test_meta.csv"
    test_csv = model_folder + 'test_with_crops.csv'
    
    test_crops, _ = process_dataset_cropping(df_path_test, image_folder, crops_dir, test_csv)
    print(f"Processed {len(test_crops)} test crops")
