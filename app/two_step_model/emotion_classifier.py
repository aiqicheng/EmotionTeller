"""
Emotion Classification Module
Extracted from ClassificationBaseline.ipynb

This module provides functionality for training and using emotion classification models
using PyTorch and torchvision models (ResNet18, VGG16).
"""

import os
import json
import random
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from statistics import mean, pstdev
import torchvision.models as models
import torchvision.transforms as T


class ImageDataset(Dataset):
    """
    Generic image classification dataset with built-in label map handling.
    
    Args:
        df: DataFrame with at least [file_path, <label_col>].
        label_col: Column name for the target (default: 'emotion').
        label2id: Optional mapping str -> int. If None, it will be built from df.
        transform: Optional torchvision transform.
        dropna_labels: Drop rows where label_col is NaN (default: True).
        strict_labels: If True and label2id is provided, rows with unknown labels are dropped.
                       If False, unknown labels raise a ValueError.
    """
    
    def __init__(self, df: pd.DataFrame, label_col: str = "emotion", 
                 label2id: Dict[str, int] = None, transform=None,
                 dropna_labels: bool = True, strict_labels: bool = True):
        if dropna_labels:
            df = df.dropna(subset=[label_col]).reset_index(drop=True)
        
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        
        # Build or validate label maps
        if label2id is None:
            classes = sorted(self.df[self.label_col].unique())
            self.label2id = {c: i for i, c in enumerate(classes)}
        else:
            self.label2id = dict(label2id)  # copy
            # Ensure labels in df all exist in provided mapping
            unknown = set(self.df[self.label_col].unique()) - set(self.label2id.keys())
            if unknown:
                if strict_labels:
                    # Drop unknown labels silently to keep splits consistent
                    self.df = self.df[self.df[self.label_col].isin(self.label2id.keys())].reset_index(drop=True)
                else:
                    raise ValueError(f"Found labels not in provided label2id: {unknown}")
        
        self.id2label = {i: c for c, i in self.label2id.items()}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Open each image with PIL, convert to RGB and apply transformation
        
        Returns:
            img: PIL image tensor shape (3, H, W)
            y: class index tensor
        """
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = self.label2id[row["emotion"]]
        return img, torch.tensor(y, dtype=torch.long)
    
    def save_label_maps(self, out_json_path: Union[str, Path]):
        """Save label mappings to JSON file."""
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label},
                     f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_label_maps(in_json_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load label mappings from JSON file."""
        with open(in_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["label2id"], obj["id2label"]
    
    def class_distribution(self) -> Dict[str, int]:
        """Return dict: label -> count on the CURRENT df."""
        return self.df[self.label_col].value_counts().to_dict()
    
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute per-class weights from CURRENT df.
        
        Returns:
            torch.FloatTensor of shape (num_classes,), index = class id
        """
        counts = self.df["emotion"].value_counts().reindex(self.label2id.keys()).fillna(0).values.astype(float)
        # Inverse frequency; guard against zero
        counts[counts == 0] = 1.0
        weights = (1.0 / counts)
        weights = weights / weights.sum() * len(weights)
        # Ordered by class index
        ordered = np.zeros(len(self.label2id))
        for lab, idx in self.label2id.items():
            ordered[idx] = weights[list(self.label2id.keys()).index(lab)]
        return torch.tensor(ordered, dtype=torch.float32)


class EmotionClassifier:
    """
    Emotion classification model trainer and predictor.
    
    This class provides functionality to:
    - Build different model architectures (ResNet18, VGG16)
    - Train emotion classification models
    - Evaluate model performance
    - Save and load trained models
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the emotion classifier.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label2id = None
        self.id2label = None
    
    def build_transforms(self, img_size: int = 224) -> Tuple[T.Compose, T.Compose]:
        """
        Create image preprocessing for train vs. validation.
        
        Args:
            img_size: Target image size
            
        Returns:
            Tuple of (train_transform, val_transform)
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Train transform: resize, augment, totensor, normalization
        train_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        
        # Validation transform: resize, totensor, normalization
        val_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        
        return train_tf, val_tf
    
    def build_model(self, arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
        """
        Initialize model backbone with pretrained weights.
        
        Args:
            arch: Model architecture ('resnet18' or 'vgg16_bn')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            PyTorch model
        """
        arch = arch.lower()
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = m.fc.in_features
            m.fc = nn.Linear(in_feats, num_classes)
        elif arch == "vgg16_bn":
            m = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError("arch must be one of: resnet18, vgg16_bn")
        
        return m
    
    def train_one_epoch(self, model: nn.Module, loader: DataLoader, 
                       criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        One pass over training set.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            
            # Raw score (batch size, num classes)
            logits = model(x)
            pred = logits.argmax(1)
            # Weighted cross entropy
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += x.size(0)
        
        return running_loss / total, correct / total
    
    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        One pass over validation set, without gradient tracking.
        
        Returns:
            Tuple of (average_loss, average_accuracy, y_true, y_pred)
        """
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_y, all_pred = [], []
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
        
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_pred)
        return running_loss / total, correct / total, y_true, y_pred
    
    def set_seed(self, seed: int = 42):
        """Set seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Use deterministic kernels and no auto-tuner
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   arch: str = "resnet18", epochs: int = 20, batch_size: int = 16,
                   lr: float = 3e-4, img_size: int = 224, freeze_backbone: bool = False,
                   output_dir: str = "runs/classification", seed: int = 42) -> Dict:
        """
        Train emotion classification model.
        
        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            arch: Model architecture
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            img_size: Image size
            freeze_backbone: Whether to freeze backbone parameters
            output_dir: Output directory for saving models
            seed: Random seed
            
        Returns:
            Dictionary with training results
        """
        self.set_seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build label maps from TRAIN ONLY
        tmp_train = train_df.dropna(subset=["emotion"])
        classes = sorted(tmp_train["emotion"].unique())
        label2id = {c: i for i, c in enumerate(classes)}
        self.label2id = label2id
        self.id2label = {i: c for c, i in label2id.items()}
        
        # Datasets
        train_tf, val_tf = self.build_transforms(img_size)
        train_ds = ImageDataset(train_df, label_col="emotion", label2id=label2id,
                              transform=train_tf, dropna_labels=True)
        val_ds = ImageDataset(val_df, label_col="emotion", label2id=label2id,
                             transform=val_tf, dropna_labels=True)
        
        print("Train class distribution:", train_ds.class_distribution())
        print("Val   class distribution:", val_ds.class_distribution())
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)
        
        # Model
        model = self.build_model(arch, num_classes=len(classes), pretrained=True).to(self.device)
        
        if freeze_backbone:
            for name, p in model.named_parameters():
                p.requires_grad = False
            # Unfreeze classifier head only
            if arch == "resnet18":
                for p in model.fc.parameters():
                    p.requires_grad = True
            else:
                for p in model.classifier[-1].parameters():
                    p.requires_grad = True
        
        # Loss with class weights
        class_weights = train_ds.compute_class_weights().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer & scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)
        
        # Training loop
        best_val_acc, best_path = 0.0, output_dir / "best.pt"
        epochs_no_improve = 0
        early_stop_patience = 10
        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, _, _ = self.evaluate(model, val_loader, criterion)
            scheduler.step()
            
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save({"model_state": model.state_dict(),
                           "arch": arch,
                           "label2id": label2id}, best_path)
                print(f"  ↳ Saved new best to {best_path}")
            else:
                epochs_no_improve += 1
            
            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping after {epochs_no_improve} epochs without improvement.")
                break
        
        # Final evaluation using best weights
        ckpt = torch.load(best_path, map_location=self.device)
        model = self.build_model(ckpt["arch"], num_classes=len(ckpt["label2id"])).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        _, _, y_true, y_pred = self.evaluate(model, val_loader, criterion)
        
        # Save history plots
        self._save_history_plots(history, output_dir)
        
        # Save model
        self.model = model
        
        return {
            "best_val_acc": best_val_acc,
            "classes": sorted(ckpt["label2id"], key=lambda k: ckpt["label2id"][k]),
            "y_true": y_true,
            "y_pred": y_pred,
            "best_path": str(best_path),
            "history": history,
        }
    
    def _save_history_plots(self, history: Dict, out_dir: Path, prefix: str = ""):
        """Save per-epoch metrics to CSV and loss/acc plots."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV
        hist_df = pd.DataFrame(history)
        hist_csv = out_dir / f"{prefix}training_history.csv"
        hist_df.to_csv(hist_csv, index=False)
        print(f"Saved history to {hist_csv}")
        
        # Loss plot
        plt.figure(figsize=(7, 5))
        plt.plot(history["epoch"], history["train_loss"], label="train loss")
        plt.plot(history["epoch"], history["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Val Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_png = out_dir / f"{prefix}loss_curve.png"
        plt.savefig(loss_png, dpi=150)
        print(f"Saved loss curve to {loss_png}")
        
        # Acc plot
        plt.figure(figsize=(7, 5))
        plt.plot(history["epoch"], history["train_acc"], label="train acc")
        plt.plot(history["epoch"], history["val_acc"], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Val Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_png = out_dir / f"{prefix}accuracy_curve.png"
        plt.savefig(acc_png, dpi=150)
        print(f"Saved accuracy curve to {acc_png}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        ckpt = torch.load(model_path, map_location=self.device)
        self.label2id = ckpt["label2id"]
        self.id2label = {i: c for c, i in self.label2id.items()}
        
        self.model = self.build_model(ckpt["arch"], num_classes=len(self.label2id))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Tuple[str, float]:
        """
        Predict emotion from a single image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Tuple of (predicted_emotion, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        _, val_tf = self.build_transforms()
        image_tensor = val_tf(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_emotion = self.id2label[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_emotion, confidence_score


def combine_datasets(image_df_path: str, faces_df_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    Combine HGEL cropped images with FACES dataset.
    
    Args:
        image_df_path: Path to CSV with cropped image data
        faces_df_path: Path to CSV with FACES dataset data
        output_csv: Optional path to save combined dataset
        
    Returns:
        Combined DataFrame
    """
    # Constants / mappings
    TARGET_CLASSES = ['Neutral', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear', 'Disgust']
    
    # Map FACES labels -> target labels (case-insensitive)
    faces2target = {
        'neutrality': 'Neutral',
        'happiness': 'Happy',
        'anger': 'Angry',
        'surprise': 'Surprise',
        'sadness': 'Sad',
        'fear': 'Fear',
        'disgust': 'Disgust',
    }
    
    # All FACES images share fixed size
    FACES_W, FACES_H = 2835, 3543
    
    # Helper: build absolute path
    def make_abs_path(path_like: str, csv_dir: Path) -> str:
        p = Path(str(path_like))
        return str(p if p.is_absolute() else (csv_dir / p))
    
    # Load data
    image_df = pd.read_csv(image_df_path)
    faces_df = pd.read_csv(faces_df_path)
    
    # HGEL -> unify to ['file_path','emotion','width','height']
    image_file_path = image_df['cropped_file_path']
    image_emotion = image_df['category'].astype(str).str.strip().str.title()
    
    image_out = pd.DataFrame({
        'file_path': image_file_path,
        'emotion': image_emotion,
        'width': image_df['new_width'].astype(int),
        'height': image_df['new_height'].astype(int),
    })
    image_out = image_out[image_out['emotion'].isin(TARGET_CLASSES)].reset_index(drop=True)
    
    # FACES -> unify to ['file_path','emotion','width','height']
    faces_dir = Path(faces_df_path).parent
    faces_file_path = faces_df['img_name'].apply(lambda p: make_abs_path(p, faces_dir))
    faces_emotion = (faces_df['emotion'].astype(str).str.strip().str.lower().map(faces2target))
    
    faces_out = pd.DataFrame({
        'file_path': faces_file_path,
        'emotion': faces_emotion,
        'width': FACES_W,
        'height': FACES_H,
    })
    faces_out = faces_out[faces_out['emotion'].isin(TARGET_CLASSES)].reset_index(drop=True)
    
    # Combine and enforce categorical dtype
    combined_df = pd.concat([image_out, faces_out], ignore_index=True)
    combined_df['emotion'] = pd.Categorical(combined_df['emotion'], categories=TARGET_CLASSES, ordered=False)
    
    # Quick sanity checks
    print('image rows:', len(image_out), '| FACES rows:', len(faces_out), '| Combined:', len(combined_df))
    print('Class distribution:\n', combined_df['emotion'].value_counts(dropna=False))
    
    if output_csv:
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined dataset saved to {output_csv}")
    
    return combined_df


if __name__ == "__main__":
    # Example usage
    data_folder = "/content/drive/MyDrive/emo/"
    model_folder = data_folder + "/BaselineModels/"
    
    # Combine datasets
    image_df_path = model_folder + "trainval_with_crops.csv"
    faces_df_path = data_folder + "FACES_headshots_6classes/facesdata.csv"
    combined_csv = model_folder + "trainval_subset_addfaces.csv"
    
    combined_df = combine_datasets(image_df_path, faces_df_path, combined_csv)
    
    # Train model
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42, 
                                     stratify=combined_df["emotion"])
    
    classifier = EmotionClassifier()
    result = classifier.train_model(train_df, val_df, arch="resnet18", epochs=20)
    
    print("Training completed!")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
