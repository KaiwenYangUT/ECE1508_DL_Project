"""
Visualization Utilities for PointNet/PointNet++ Results

This module provides comprehensive visualization tools for:
- Training curves (loss, accuracy over epochs)
- Confusion matrices
- Per-class performance
- Point cloud visualizations
- Model comparison plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import json


class TrainingVisualizer:
    """Visualize training progress and results"""
    
    def __init__(self, save_dir: str):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_curves(self, train_metrics: List[Dict], 
                            val_metrics: List[Dict],
                            metric_keys: List[str] = ['overall_accuracy', 'loss'],
                            save_name: str = 'training_curves.png'):
        """
        Plot training and validation curves
        
        Args:
            train_metrics: List of training metrics per epoch
            val_metrics: List of validation metrics per epoch
            metric_keys: Keys of metrics to plot
            save_name: Filename to save the plot
        """
        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric_key in enumerate(metric_keys):
            ax = axes[idx]
            
            # Extract data
            train_epochs = [m['epoch'] for m in train_metrics if metric_key in m]
            train_values = [m[metric_key] for m in train_metrics if metric_key in m]
            val_epochs = [m['epoch'] for m in val_metrics if metric_key in m]
            val_values = [m[metric_key] for m in val_metrics if metric_key in m]
            
            # Plot
            if train_epochs:
                ax.plot(train_epochs, train_values, 'o-', label='Train', linewidth=2, markersize=4)
            if val_epochs:
                ax.plot(val_epochs, val_values, 's-', label='Validation', linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric_key.replace("_", " ").title()} over Epochs', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_name: str = 'confusion_matrix.png',
                            normalize: bool = True):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix (num_classes, num_classes)
            class_names: List of class names
            save_name: Filename to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        n_classes = cm.shape[0]
        if class_names is None:
            class_names = [f'C{i}' for i in range(n_classes)]
        
        # Adjust figure size based on number of classes
        fig_size = max(10, n_classes * 0.5)
        plt.figure(figsize=(fig_size, fig_size))
        
        sns.heatmap(cm, annot=n_classes <= 20, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_accuracy(self, class_accuracies: np.ndarray,
                                class_names: Optional[List[str]] = None,
                                save_name: str = 'per_class_accuracy.png'):
        """
        Plot per-class accuracy as bar chart
        
        Args:
            class_accuracies: Array of per-class accuracies
            class_names: List of class names
            save_name: Filename to save the plot
        """
        n_classes = len(class_accuracies)
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.5), 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        bars = ax.bar(range(n_classes), class_accuracies, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{class_accuracies[i]:.2%}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=np.mean(class_accuracies), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(class_accuracies):.2%}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy plot saved to {save_path}")
    
    def plot_iou_comparison(self, iou_dict: Dict[str, np.ndarray],
                           class_names: Optional[List[str]] = None,
                           save_name: str = 'iou_comparison.png'):
        """
        Plot IoU comparison across different models or classes
        
        Args:
            iou_dict: Dictionary mapping model names to IoU arrays
            class_names: List of class names
            save_name: Filename to save the plot
        """
        n_classes = len(list(iou_dict.values())[0])
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.6), 7))
        
        x = np.arange(n_classes)
        width = 0.8 / len(iou_dict)
        
        for idx, (model_name, ious) in enumerate(iou_dict.items()):
            offset = (idx - len(iou_dict)/2 + 0.5) * width
            ax.bar(x + offset, ious, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('IoU', fontsize=12)
        ax.set_title('IoU Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"IoU comparison plot saved to {save_path}")
    
    def plot_learning_rate_schedule(self, lr_history: List[float],
                                    save_name: str = 'learning_rate_schedule.png'):
        """
        Plot learning rate schedule over epochs
        
        Args:
            lr_history: List of learning rates per epoch
            save_name: Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(lr_history)), lr_history, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning rate schedule saved to {save_path}")


class ModelComparison:
    """Compare multiple models' performance"""
    
    def __init__(self, save_dir: str):
        """
        Initialize model comparison
        
        Args:
            save_dir: Directory to save comparison plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]],
                      save_name: str = 'model_comparison.png'):
        """
        Create a comprehensive model comparison plot
        
        Args:
            model_results: Dict mapping model names to their metrics
            save_name: Filename to save the plot
        """
        metrics = list(list(model_results.values())[0].keys())
        models = list(model_results.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [model_results[model][metric] for model in models]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            bars = ax.bar(range(len(models)), values, color=colors, edgecolor='black')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{values[i]:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison saved to {save_path}")
    
    def create_performance_table(self, model_results: Dict[str, Dict[str, float]],
                                save_name: str = 'performance_table.txt'):
        """
        Create a formatted performance comparison table
        
        Args:
            model_results: Dict mapping model names to their metrics
            save_name: Filename to save the table
        """
        save_path = self.save_dir / save_name
        
        with open(save_path, 'w') as f:
            # Header
            metrics = list(list(model_results.values())[0].keys())
            f.write(f"{'Model':<30} | " + " | ".join([f"{m:<15}" for m in metrics]) + "\n")
            f.write("-" * (30 + len(metrics) * 18) + "\n")
            
            # Data rows
            for model, results in model_results.items():
                f.write(f"{model:<30} | ")
                f.write(" | ".join([f"{results[m]:<15.4f}" for m in metrics]))
                f.write("\n")
        
        print(f"Performance table saved to {save_path}")


def plot_point_cloud_samples(points: np.ndarray, labels: Optional[np.ndarray] = None,
                            save_path: str = 'point_cloud_samples.png',
                            n_samples: int = 4):
    """
    Plot point cloud samples in 3D
    
    Args:
        points: Point cloud data (B, N, 3) or (N, 3)
        labels: Optional labels for coloring (B, N) or (N,)
        save_path: Path to save the plot
        n_samples: Number of samples to plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if points.ndim == 2:
        points = points[np.newaxis, ...]
        if labels is not None:
            labels = labels[np.newaxis, ...]
    
    n_samples = min(n_samples, points.shape[0])
    fig = plt.figure(figsize=(5*n_samples, 5))
    
    for i in range(n_samples):
        ax = fig.add_subplot(1, n_samples, i+1, projection='3d')
        
        pc = points[i]
        if labels is not None:
            scatter = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                               c=labels[i], cmap='tab20', s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Point cloud samples saved to {save_path}")
