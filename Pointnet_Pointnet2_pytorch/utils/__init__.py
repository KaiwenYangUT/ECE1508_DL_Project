"""
Utils Package for PointNet/PointNet++ Project

This package provides comprehensive utilities for experiment management,
metrics tracking, visualization, logging, and reporting.
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    create_experiment_config,
    save_experiment_config
)

from .metrics import (
    ClassificationMetrics,
    SegmentationMetrics,
    MetricsTracker,
    compute_iou
)

from .visualization import (
    TrainingVisualizer,
    ModelComparison,
    plot_point_cloud_samples
)

from .logger import (
    ExperimentLogger,
    TensorBoardLogger,
    WandbLogger,
    create_experiment_logger
)

from .report import (
    ResultsAnalyzer,
    ReportGenerator,
    create_latex_table
)

__all__ = [
    # Config
    'ExperimentConfig',
    'ModelConfig',
    'TrainingConfig',
    'create_experiment_config',
    'save_experiment_config',
    
    # Metrics
    'ClassificationMetrics',
    'SegmentationMetrics',
    'MetricsTracker',
    'compute_iou',
    
    # Visualization
    'TrainingVisualizer',
    'ModelComparison',
    'plot_point_cloud_samples',
    
    # Logging
    'ExperimentLogger',
    'TensorBoardLogger',
    'WandbLogger',
    'create_experiment_logger',
    
    # Reporting
    'ResultsAnalyzer',
    'ReportGenerator',
    'create_latex_table',
]

__version__ = '1.0.0'
