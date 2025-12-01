"""
Experiment Logger for PointNet/PointNet++ Training

This module provides comprehensive logging utilities including:
- Console and file logging
- TensorBoard integration (optional)
- Weights & Biases integration (optional)
- Metrics tracking and checkpointing
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import datetime


class ExperimentLogger:
    """Comprehensive experiment logger"""
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
        """
        Initialize experiment logger
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Metrics log file
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        self.metrics_history = []
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.info("=" * 50)
        self.info("HYPERPARAMETERS")
        self.info("=" * 50)
        for key, value in hyperparams.items():
            self.info(f"  {key}: {value}")
        self.info("=" * 50)
        
        # Save to JSON
        hyperparam_file = self.log_dir / f"{self.experiment_name}_hyperparameters.json"
        with open(hyperparam_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            phase: Phase (train/val/test)
        """
        metric_str = f"Epoch {epoch} [{phase}] - "
        metric_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(metric_str)
        
        # Save to history
        metrics_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def log_model_summary(self, model, input_size: tuple):
        """
        Log model summary
        
        Args:
            model: PyTorch model
            input_size: Input tensor size
        """
        self.info("=" * 50)
        self.info("MODEL ARCHITECTURE")
        self.info("=" * 50)
        self.info(str(model))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("=" * 50)
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        self.info("=" * 50)
    
    def log_checkpoint_saved(self, checkpoint_path: str, metrics: Dict[str, float]):
        """
        Log checkpoint save event
        
        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Metrics at checkpoint
        """
        self.info(f"Checkpoint saved: {checkpoint_path}")
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"  Metrics: {metric_str}")


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether TensorBoard logging is enabled
        """
        self.enabled = enabled
        self.writer = None
        
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard logging enabled. Log dir: {log_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values"""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values"""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image"""
        if self.enabled and self.writer:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close the TensorBoard writer"""
        if self.enabled and self.writer:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logging wrapper"""
    
    def __init__(self, project: str, experiment_name: str, 
                 config: Dict[str, Any], enabled: bool = False):
        """
        Initialize W&B logger
        
        Args:
            project: W&B project name
            experiment_name: Name of the experiment
            config: Configuration dictionary
            enabled: Whether W&B logging is enabled
        """
        self.enabled = enabled
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                self.run = wandb.init(
                    project=project,
                    name=experiment_name,
                    config=config
                )
                print(f"W&B logging enabled. Project: {project}, Run: {experiment_name}")
            except ImportError:
                print("Weights & Biases not available. Install with: pip install wandb")
                self.enabled = False
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str):
        """Log model artifact"""
        if self.enabled:
            artifact = self.wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run"""
        if self.enabled:
            self.wandb.finish()


def create_experiment_logger(log_dir: str, experiment_name: str,
                            use_tensorboard: bool = False,
                            use_wandb: bool = False,
                            wandb_project: Optional[str] = None,
                            config: Optional[Dict[str, Any]] = None) -> tuple:
    """
    Create all loggers for an experiment
    
    Args:
        log_dir: Directory for logs
        experiment_name: Name of experiment
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        config: Configuration dictionary
    
    Returns:
        Tuple of (experiment_logger, tensorboard_logger, wandb_logger)
    """
    # Main experiment logger
    exp_logger = ExperimentLogger(log_dir, experiment_name)
    
    # TensorBoard logger
    tb_logger = None
    if use_tensorboard:
        tb_log_dir = os.path.join(log_dir, 'tensorboard')
        tb_logger = TensorBoardLogger(tb_log_dir, enabled=True)
    
    # W&B logger
    wb_logger = None
    if use_wandb and wandb_project:
        wb_logger = WandbLogger(wandb_project, experiment_name, 
                               config or {}, enabled=True)
    
    return exp_logger, tb_logger, wb_logger
