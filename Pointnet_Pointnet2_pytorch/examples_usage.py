"""
Complete Example Script: Model Training and Evaluation with Full Utilities

This script demonstrates how to:
1. Set up experiment configuration
2. Train a model with enhanced logging
3. Evaluate and visualize results
4. Generate comprehensive reports
5. Compare multiple experiments

Author: Enhanced by utility modules
Date: December 2025
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import (
    ExperimentConfig,
    ClassificationMetrics,
    MetricsTracker,
    TrainingVisualizer,
    ModelComparison,
    create_experiment_logger,
    ResultsAnalyzer,
    ReportGenerator,
    create_latex_table
)


def example_1_basic_metrics():
    """Example 1: Basic metrics computation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Metrics Computation")
    print("="*60)
    
    import torch
    
    # Simulate predictions and targets
    num_classes = 10
    batch_size = 32
    
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create metrics tracker
    metrics = ClassificationMetrics(num_classes)
    metrics.update(predictions, targets)
    
    # Compute results
    results = metrics.compute()
    print(f"\nMetrics Results:")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"  Average Class Accuracy: {results['average_class_accuracy']:.4f}")
    
    # Get confusion matrix
    cm = metrics.get_confusion_matrix()
    print(f"\nConfusion Matrix Shape: {cm.shape}")


def example_2_configuration_management():
    """Example 2: Configuration management"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Configuration Management")
    print("="*60)
    
    # Create configuration
    config = ExperimentConfig()
    config.set('model', 'pointnet2_cls_ssg')
    config.set('batch_size', 24)
    config.set('learning_rate', 0.001)
    config.set('num_epochs', 200)
    
    print("\nConfiguration created:")
    print(config)
    
    # Save configuration
    config_dir = Path('examples/configs')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config.save_yaml(str(config_dir / 'example_config.yaml'))
    config.save_json(str(config_dir / 'example_config.json'))
    print(f"\nConfiguration saved to {config_dir}/")
    
    # Load configuration
    loaded_config = ExperimentConfig.load_json(str(config_dir / 'example_config.json'))
    print(f"\nLoaded configuration:")
    print(f"  Model: {loaded_config.get('model')}")
    print(f"  Batch Size: {loaded_config.get('batch_size')}")


def example_3_logging():
    """Example 3: Enhanced logging"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Enhanced Logging")
    print("="*60)
    
    # Create experiment logger
    log_dir = 'examples/logs'
    exp_logger, tb_logger, wb_logger = create_experiment_logger(
        log_dir=log_dir,
        experiment_name='example_experiment',
        use_tensorboard=False,
        use_wandb=False
    )
    
    # Log hyperparameters
    hyperparams = {
        'model': 'pointnet2_cls_ssg',
        'batch_size': 24,
        'learning_rate': 0.001
    }
    exp_logger.log_hyperparameters(hyperparams)
    
    # Log metrics
    for epoch in range(1, 6):
        train_metrics = {
            'loss': 2.5 - epoch * 0.3,
            'accuracy': 0.5 + epoch * 0.08
        }
        exp_logger.log_metrics(epoch, train_metrics, phase='train')
    
    print(f"\nLogs saved to {log_dir}/")


def example_4_visualization():
    """Example 4: Visualization"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Visualization")
    print("="*60)
    
    import numpy as np
    
    # Create sample training history
    train_metrics = [
        {'epoch': i, 'overall_accuracy': 0.5 + i*0.04, 'loss': 2.0 - i*0.15}
        for i in range(1, 21)
    ]
    val_metrics = [
        {'epoch': i, 'overall_accuracy': 0.48 + i*0.042, 'loss': 2.1 - i*0.14}
        for i in range(1, 21)
    ]
    
    # Create visualizer
    vis_dir = 'examples/visualizations'
    visualizer = TrainingVisualizer(save_dir=vis_dir)
    
    # Plot training curves
    visualizer.plot_training_curves(
        train_metrics, val_metrics,
        metric_keys=['overall_accuracy', 'loss'],
        save_name='example_training_curves.png'
    )
    
    # Plot confusion matrix
    cm = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(cm, np.random.randint(80, 100, 10))
    
    visualizer.plot_confusion_matrix(
        cm,
        class_names=[f'Class_{i}' for i in range(10)],
        save_name='example_confusion_matrix.png'
    )
    
    # Plot per-class accuracy
    class_accs = np.random.uniform(0.7, 0.95, 10)
    visualizer.plot_per_class_accuracy(
        class_accs,
        class_names=[f'Class_{i}' for i in range(10)],
        save_name='example_per_class_accuracy.png'
    )
    
    print(f"\nVisualizations saved to {vis_dir}/")


def example_5_model_comparison():
    """Example 5: Model comparison"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Comparison")
    print("="*60)
    
    # Create sample model results
    model_results = {
        'PointNet': {
            'overall_accuracy': 0.906,
            'average_class_accuracy': 0.891,
            'mean_iou': 0.820
        },
        'PointNet++ SSG': {
            'overall_accuracy': 0.924,
            'average_class_accuracy': 0.915,
            'mean_iou': 0.845
        },
        'PointNet++ MSG': {
            'overall_accuracy': 0.928,
            'average_class_accuracy': 0.921,
            'mean_iou': 0.852
        }
    }
    
    # Create comparison
    comp_dir = 'examples/comparisons'
    comparison = ModelComparison(save_dir=comp_dir)
    
    comparison.compare_models(
        model_results,
        save_name='example_model_comparison.png'
    )
    
    comparison.create_performance_table(
        model_results,
        save_name='example_performance_table.txt'
    )
    
    print(f"\nComparison saved to {comp_dir}/")
    print("\nPerformance Summary:")
    for model, results in model_results.items():
        print(f"  {model}:")
        for metric, value in results.items():
            print(f"    {metric}: {value:.4f}")


def example_6_report_generation():
    """Example 6: Report generation"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Report Generation")
    print("="*60)
    
    # Create sample experiment data
    experiment_data = {
        'name': 'example_experiment',
        'path': 'examples/experiment',
        'config': {
            'model': 'pointnet2_cls_ssg',
            'batch_size': 24,
            'learning_rate': 0.001,
            'num_epochs': 200
        },
        'metrics': [
            {'epoch': i, 'phase': 'train', 'overall_accuracy': 0.5 + i*0.04}
            for i in range(1, 11)
        ] + [
            {'epoch': i, 'phase': 'val', 'overall_accuracy': 0.48 + i*0.042}
            for i in range(1, 11)
        ],
        'best_metrics': {
            'overall_accuracy': 0.924,
            'average_class_accuracy': 0.915,
            'epoch': 150
        }
    }
    
    # Generate report
    report_dir = 'examples/reports'
    report_gen = ReportGenerator(output_dir=report_dir)
    
    report_gen.generate_report(
        experiment_data,
        report_name='example_experiment_report.md'
    )
    
    print(f"\nReport generated in {report_dir}/")


def example_7_full_workflow():
    """Example 7: Complete workflow demonstration"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete Workflow")
    print("="*60)
    
    workflow_dir = Path('examples/full_workflow')
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Configuration
    print("\n1. Creating configuration...")
    config = ExperimentConfig()
    config.update({
        'model': 'pointnet2_cls_ssg',
        'batch_size': 24,
        'learning_rate': 0.001,
        'num_epochs': 200
    })
    config.save_yaml(str(workflow_dir / 'config.yaml'))
    
    # 2. Logging
    print("2. Setting up logging...")
    exp_logger, _, _ = create_experiment_logger(
        log_dir=str(workflow_dir / 'logs'),
        experiment_name='workflow_experiment',
        use_tensorboard=False
    )
    exp_logger.log_hyperparameters(config.config)
    
    # 3. Metrics tracking
    print("3. Tracking metrics...")
    tracker = MetricsTracker()
    for epoch in range(1, 11):
        train_metrics = {
            'overall_accuracy': 0.5 + epoch * 0.04,
            'loss': 2.0 - epoch * 0.15
        }
        val_metrics = {
            'overall_accuracy': 0.48 + epoch * 0.042,
            'loss': 2.1 - epoch * 0.14
        }
        tracker.add_train_metrics(epoch, train_metrics)
        tracker.add_val_metrics(epoch, val_metrics)
    
    tracker.save(str(workflow_dir / 'metrics.json'))
    
    # 4. Visualization
    print("4. Creating visualizations...")
    visualizer = TrainingVisualizer(save_dir=str(workflow_dir / 'visualizations'))
    visualizer.plot_training_curves(
        tracker.train_metrics,
        tracker.val_metrics,
        metric_keys=['overall_accuracy', 'loss']
    )
    
    # 5. Report generation
    print("5. Generating report...")
    report_gen = ReportGenerator(output_dir=str(workflow_dir / 'reports'))
    experiment_data = {
        'name': 'workflow_experiment',
        'path': str(workflow_dir),
        'config': config.config,
        'metrics': tracker.train_metrics + tracker.val_metrics,
        'best_metrics': tracker.get_best_metrics()
    }
    report_gen.generate_report(experiment_data)
    
    print(f"\n✓ Complete workflow executed!")
    print(f"✓ All outputs saved to {workflow_dir}/")
    print(f"\nGenerated files:")
    for file in workflow_dir.rglob('*'):
        if file.is_file():
            print(f"  - {file.relative_to(workflow_dir)}")


def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description='PointNet/PointNet++ Utilities Examples')
    parser.add_argument('--example', type=int, default=0, 
                       help='Run specific example (1-7), or 0 for all')
    args = parser.parse_args()
    
    examples = [
        example_1_basic_metrics,
        example_2_configuration_management,
        example_3_logging,
        example_4_visualization,
        example_5_model_comparison,
        example_6_report_generation,
        example_7_full_workflow
    ]
    
    print("\n" + "="*60)
    print("PointNet/PointNet++ Utilities Examples")
    print("="*60)
    
    if args.example == 0:
        # Run all examples
        for i, example_func in enumerate(examples, 1):
            try:
                example_func()
            except Exception as e:
                print(f"\n❌ Example {i} failed: {str(e)}")
    elif 1 <= args.example <= len(examples):
        # Run specific example
        try:
            examples[args.example - 1]()
        except Exception as e:
            print(f"\n❌ Example {args.example} failed: {str(e)}")
    else:
        print(f"Invalid example number. Choose 1-{len(examples)} or 0 for all.")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
