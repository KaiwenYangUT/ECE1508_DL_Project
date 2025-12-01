"""
Comprehensive Metrics Computation Module

This module provides utilities for computing various metrics for:
- Classification (accuracy, precision, recall, F1-score, confusion matrix)
- Part Segmentation (IoU, per-class IoU)
- Semantic Segmentation (mIoU, pixel accuracy, class accuracy)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import json


class ClassificationMetrics:
    """Metrics for point cloud classification tasks"""
    
    def __init__(self, num_classes: int):
        """
        Initialize classification metrics
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.correct_per_class = np.zeros(self.num_classes)
        self.total_per_class = np.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions
        
        Args:
            predictions: Model predictions (B, num_classes) or (B,)
            targets: Ground truth labels (B,)
        """
        if predictions.dim() > 1:
            pred_choice = predictions.data.max(1)[1]
        else:
            pred_choice = predictions
            
        pred_np = pred_choice.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        
        # Per-class accuracy
        for cls in range(self.num_classes):
            mask = target_np == cls
            if mask.sum() > 0:
                self.correct_per_class[cls] += (pred_np[mask] == target_np[mask]).sum()
                self.total_per_class[cls] += mask.sum()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall accuracy
        overall_acc = (predictions == targets).mean()
        
        # Per-class accuracy
        class_acc = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            if self.total_per_class[cls] > 0:
                class_acc[cls] = self.correct_per_class[cls] / self.total_per_class[cls]
        
        # Average class accuracy
        valid_classes = self.total_per_class > 0
        avg_class_acc = class_acc[valid_classes].mean() if valid_classes.sum() > 0 else 0.0
        
        return {
            'overall_accuracy': float(overall_acc),
            'average_class_accuracy': float(avg_class_acc),
            'per_class_accuracy': class_acc.tolist()
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        return confusion_matrix(self.targets, self.predictions, 
                               labels=list(range(self.num_classes)))
    
    def get_classification_report(self, class_names: Optional[List[str]] = None) -> str:
        """
        Get detailed classification report
        
        Args:
            class_names: Optional list of class names
        
        Returns:
            Classification report string
        """
        target_names = class_names if class_names else [f"Class_{i}" for i in range(self.num_classes)]
        return classification_report(self.targets, self.predictions, 
                                    target_names=target_names, zero_division=0)


class SegmentationMetrics:
    """Metrics for segmentation tasks (part and semantic)"""
    
    def __init__(self, num_classes: int, num_parts: Optional[int] = None):
        """
        Initialize segmentation metrics
        
        Args:
            num_classes: Number of object classes
            num_parts: Number of parts (for part segmentation)
        """
        self.num_classes = num_classes
        self.num_parts = num_parts if num_parts else num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_correct = 0
        self.total_seen = 0
        self.total_correct_class = np.zeros(self.num_parts)
        self.total_seen_class = np.zeros(self.num_parts)
        self.total_iou_deno_class = np.zeros(self.num_parts)
        
        # For shape IoU (part segmentation)
        self.shape_ious = {}
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               cur_label: Optional[torch.Tensor] = None):
        """
        Update segmentation metrics
        
        Args:
            predictions: Model predictions (B, N, num_parts) or (B, N)
            targets: Ground truth labels (B, N)
            cur_label: Current object labels for part segmentation (B,)
        """
        if predictions.dim() > 2:
            pred_val = predictions.data.max(2)[1]
        else:
            pred_val = predictions
        
        pred_np = pred_val.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        correct = (pred_np == target_np).sum()
        self.total_correct += correct
        self.total_seen += target_np.size
        
        # Per-class metrics
        for l in range(self.num_parts):
            self.total_seen_class[l] += (target_np == l).sum()
            self.total_correct_class[l] += ((pred_np == l) & (target_np == l)).sum()
            self.total_iou_deno_class[l] += ((pred_np == l) | (target_np == l)).sum()
    
    def update_shape_iou(self, predictions: torch.Tensor, targets: torch.Tensor, 
                         shape_label: int, num_parts: int):
        """
        Update shape IoU for part segmentation
        
        Args:
            predictions: Model predictions (N,)
            targets: Ground truth labels (N,)
            shape_label: Object category label
            num_parts: Number of parts for this shape
        """
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        iou_list = []
        for part in range(num_parts):
            intersection = ((pred_np == part) & (target_np == part)).sum()
            union = ((pred_np == part) | (target_np == part)).sum()
            iou = intersection / (union + 1e-10)
            iou_list.append(iou)
        
        if shape_label not in self.shape_ious:
            self.shape_ious[shape_label] = []
        self.shape_ious[shape_label].append(np.mean(iou_list))
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all computed metrics
        """
        # Overall accuracy
        accuracy = self.total_correct / (self.total_seen + 1e-10)
        
        # Class accuracy
        class_accuracy = self.total_correct_class / (self.total_seen_class + 1e-10)
        avg_class_accuracy = np.mean(class_accuracy)
        
        # IoU
        iou = self.total_correct_class / (self.total_iou_deno_class + 1e-10)
        mean_iou = np.mean(iou)
        
        metrics = {
            'overall_accuracy': float(accuracy),
            'average_class_accuracy': float(avg_class_accuracy),
            'mean_iou': float(mean_iou),
            'per_class_accuracy': class_accuracy.tolist(),
            'per_class_iou': iou.tolist()
        }
        
        # Shape IoU for part segmentation
        if self.shape_ious:
            shape_iou_dict = {}
            for label, ious in self.shape_ious.items():
                shape_iou_dict[f"shape_{label}"] = float(np.mean(ious))
            metrics['shape_ious'] = shape_iou_dict
            metrics['instance_avg_iou'] = float(np.mean([np.mean(ious) for ious in self.shape_ious.values()]))
        
        return metrics


class MetricsTracker:
    """Track metrics over training epochs"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.best_metrics = {}
    
    def add_train_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Add training metrics for an epoch"""
        metrics['epoch'] = epoch
        self.train_metrics.append(metrics)
    
    def add_val_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Add validation metrics for an epoch"""
        metrics['epoch'] = epoch
        self.val_metrics.append(metrics)
        
        # Update best metrics
        for key, value in metrics.items():
            if key != 'epoch':
                if key not in self.best_metrics or value > self.best_metrics[key]['value']:
                    self.best_metrics[key] = {'value': value, 'epoch': epoch}
    
    def add_test_metrics(self, metrics: Dict[str, float]):
        """Add test metrics"""
        self.test_metrics.append(metrics)
    
    def get_best_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get best metrics across all epochs"""
        return self.best_metrics
    
    def save(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            'train': self.train_metrics,
            'validation': self.val_metrics,
            'test': self.test_metrics,
            'best': self.best_metrics
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.train_metrics = data.get('train', [])
        self.val_metrics = data.get('validation', [])
        self.test_metrics = data.get('test', [])
        self.best_metrics = data.get('best', {})


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IoU for each class
    
    Args:
        pred: Predictions (N,)
        target: Ground truth (N,)
        num_classes: Number of classes
    
    Returns:
        Tuple of (iou_per_class, valid_classes_mask)
    """
    ious = np.zeros(num_classes)
    valid = np.zeros(num_classes, dtype=bool)
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        if target_mask.sum() == 0:
            continue
            
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        ious[cls] = intersection / (union + 1e-10)
        valid[cls] = True
    
    return ious, valid
