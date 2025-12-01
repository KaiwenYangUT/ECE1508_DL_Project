"""
Results Analysis and Reporting Tools

This module provides utilities for:
- Analyzing experiment results
- Generating comprehensive reports
- Comparing multiple experiments
- Exporting results in various formats
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import pandas as pd


class ResultsAnalyzer:
    """Analyze and aggregate experimental results"""
    
    def __init__(self, results_dir: str):
        """
        Initialize results analyzer
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        
    def load_experiment(self, exp_name: str, exp_dir: str):
        """
        Load a single experiment's results
        
        Args:
            exp_name: Name of the experiment
            exp_dir: Directory containing experiment results
        """
        exp_path = Path(exp_dir)
        
        experiment_data = {
            'name': exp_name,
            'path': str(exp_path),
            'config': {},
            'metrics': {},
            'best_metrics': {}
        }
        
        # Load configuration
        config_files = list(exp_path.glob('config*.json')) + list(exp_path.glob('config*.yaml'))
        if config_files:
            with open(config_files[0], 'r') as f:
                if config_files[0].suffix == '.json':
                    experiment_data['config'] = json.load(f)
                # YAML loading would require PyYAML
        
        # Load metrics
        metrics_files = list(exp_path.glob('*metrics*.json'))
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                experiment_data['metrics'] = json.load(f)
        
        # Find best metrics
        if experiment_data['metrics']:
            if isinstance(experiment_data['metrics'], list):
                # Find best epoch
                best_val_metrics = max([m for m in experiment_data['metrics'] 
                                       if m.get('phase') == 'val'],
                                      key=lambda x: x.get('overall_accuracy', 0),
                                      default={})
                experiment_data['best_metrics'] = best_val_metrics
            elif isinstance(experiment_data['metrics'], dict):
                experiment_data['best_metrics'] = experiment_data['metrics'].get('best', {})
        
        self.experiments[exp_name] = experiment_data
    
    def load_all_experiments(self):
        """Load all experiments from the results directory"""
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} does not exist")
            return
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                self.load_experiment(exp_dir.name, str(exp_dir))
    
    def compare_experiments(self, metric_key: str = 'overall_accuracy') -> Dict[str, float]:
        """
        Compare experiments based on a specific metric
        
        Args:
            metric_key: Metric to compare
        
        Returns:
            Dictionary mapping experiment names to metric values
        """
        comparison = {}
        for exp_name, exp_data in self.experiments.items():
            best_metrics = exp_data.get('best_metrics', {})
            if metric_key in best_metrics:
                comparison[exp_name] = best_metrics[metric_key]
        
        return dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all experiments
        
        Returns:
            Pandas DataFrame with experiment summaries
        """
        summary_data = []
        
        for exp_name, exp_data in self.experiments.items():
            row = {'Experiment': exp_name}
            
            # Add config info
            config = exp_data.get('config', {})
            row['Model'] = config.get('model', 'Unknown')
            row['Batch Size'] = config.get('batch_size', 'N/A')
            row['Learning Rate'] = config.get('learning_rate', 'N/A')
            row['Epochs'] = config.get('epoch', 'N/A')
            
            # Add best metrics
            best_metrics = exp_data.get('best_metrics', {})
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)) and key != 'epoch':
                    row[key.replace('_', ' ').title()] = f"{value:.4f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_summary(self, output_path: str, format: str = 'csv'):
        """
        Export summary table to file
        
        Args:
            output_path: Path to save the summary
            format: Output format ('csv', 'excel', 'json', 'markdown')
        """
        df = self.get_summary_table()
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=4)
        elif format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(df.to_markdown(index=False))
        
        print(f"Summary exported to {output_path}")


class ReportGenerator:
    """Generate comprehensive experiment reports"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, experiment_data: Dict[str, Any],
                       report_name: str = 'experiment_report.md'):
        """
        Generate a markdown report for an experiment
        
        Args:
            experiment_data: Dictionary containing experiment data
            report_name: Name of the report file
        """
        report_path = self.output_dir / report_name
        
        with open(report_path, 'w') as f:
            # Title
            f.write(f"# Experiment Report: {experiment_data.get('name', 'Unnamed')}\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            config = experiment_data.get('config', {})
            if config:
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")
                for key, value in config.items():
                    f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            # Best Metrics
            f.write("## Best Performance\n\n")
            best_metrics = experiment_data.get('best_metrics', {})
            if best_metrics:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in best_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"| {key.replace('_', ' ').title()} | {value:.4f} |\n")
                    else:
                        f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
            f.write("\n")
            
            # Training History
            f.write("## Training History\n\n")
            metrics = experiment_data.get('metrics', [])
            if isinstance(metrics, list) and metrics:
                f.write("Training progress over epochs:\n\n")
                # Group by phase
                train_metrics = [m for m in metrics if m.get('phase') == 'train']
                val_metrics = [m for m in metrics if m.get('phase') == 'val']
                
                if train_metrics:
                    f.write("### Training Metrics\n\n")
                    f.write(f"- Total epochs: {len(train_metrics)}\n")
                    if 'overall_accuracy' in train_metrics[-1]:
                        f.write(f"- Final training accuracy: {train_metrics[-1]['overall_accuracy']:.4f}\n")
                    f.write("\n")
                
                if val_metrics:
                    f.write("### Validation Metrics\n\n")
                    if 'overall_accuracy' in val_metrics[-1]:
                        f.write(f"- Final validation accuracy: {val_metrics[-1]['overall_accuracy']:.4f}\n")
                    best_val_acc = max([m.get('overall_accuracy', 0) for m in val_metrics])
                    f.write(f"- Best validation accuracy: {best_val_acc:.4f}\n")
                    f.write("\n")
            
            # Files
            f.write("## Output Files\n\n")
            exp_path = Path(experiment_data.get('path', ''))
            if exp_path.exists():
                f.write("Generated files:\n\n")
                for file in exp_path.glob('*'):
                    if file.is_file():
                        f.write(f"- `{file.name}` ({file.stat().st_size / 1024:.2f} KB)\n")
            f.write("\n")
            
            # Conclusion
            f.write("## Summary\n\n")
            f.write(f"This experiment was conducted to evaluate the performance of ")
            f.write(f"{config.get('model', 'the model')} on ")
            f.write(f"{config.get('task', 'the task')}.\n\n")
            
            if best_metrics.get('overall_accuracy'):
                f.write(f"The model achieved an overall accuracy of ")
                f.write(f"{best_metrics['overall_accuracy']:.2%}.\n\n")
        
        print(f"Report generated: {report_path}")
    
    def generate_comparison_report(self, experiments: Dict[str, Dict[str, Any]],
                                   report_name: str = 'comparison_report.md'):
        """
        Generate a comparison report for multiple experiments
        
        Args:
            experiments: Dictionary of experiment data
            report_name: Name of the report file
        """
        report_path = self.output_dir / report_name
        
        with open(report_path, 'w') as f:
            # Title
            f.write("# Model Comparison Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"*Comparing {len(experiments)} experiments*\n\n")
            
            # Summary Table
            f.write("## Performance Summary\n\n")
            f.write("| Experiment | Model | Overall Accuracy | Mean IoU | Epochs |\n")
            f.write("|------------|-------|------------------|----------|--------|\n")
            
            for exp_name, exp_data in experiments.items():
                config = exp_data.get('config', {})
                best_metrics = exp_data.get('best_metrics', {})
                
                model = config.get('model', 'N/A')
                acc = best_metrics.get('overall_accuracy', 0)
                iou = best_metrics.get('mean_iou', 0)
                epochs = config.get('epoch', 'N/A')
                
                f.write(f"| {exp_name} | {model} | {acc:.4f} | {iou:.4f} | {epochs} |\n")
            
            f.write("\n")
            
            # Best Performing Model
            f.write("## Best Performing Models\n\n")
            
            # By accuracy
            best_acc_exp = max(experiments.items(), 
                             key=lambda x: x[1].get('best_metrics', {}).get('overall_accuracy', 0))
            f.write(f"### Highest Accuracy\n\n")
            f.write(f"- **{best_acc_exp[0]}**: ")
            f.write(f"{best_acc_exp[1].get('best_metrics', {}).get('overall_accuracy', 0):.4f}\n\n")
            
            # Individual Experiment Details
            f.write("## Detailed Results\n\n")
            for exp_name, exp_data in experiments.items():
                f.write(f"### {exp_name}\n\n")
                
                config = exp_data.get('config', {})
                f.write("**Configuration:**\n")
                for key in ['model', 'batch_size', 'learning_rate', 'optimizer']:
                    if key in config:
                        f.write(f"- {key}: {config[key]}\n")
                f.write("\n")
                
                best_metrics = exp_data.get('best_metrics', {})
                f.write("**Best Metrics:**\n")
                for key, value in best_metrics.items():
                    if isinstance(value, (int, float)) and key != 'epoch':
                        f.write(f"- {key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")
        
        print(f"Comparison report generated: {report_path}")


def create_latex_table(experiments: Dict[str, Dict[str, Any]], 
                      output_path: str,
                      metrics: List[str] = ['overall_accuracy', 'mean_iou']):
    """
    Create a LaTeX table for paper/report
    
    Args:
        experiments: Dictionary of experiment data
        output_path: Path to save LaTeX table
        metrics: List of metrics to include
    """
    with open(output_path, 'w') as f:
        # Table header
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance Comparison}\n")
        f.write("\\label{tab:results}\n")
        
        # Column specification
        n_cols = len(metrics) + 2  # +2 for model name and architecture
        f.write(f"\\begin{{tabular}}{{{'l' * n_cols}}}\n")
        f.write("\\hline\n")
        
        # Header row
        header = "Model & Architecture"
        for metric in metrics:
            header += f" & {metric.replace('_', ' ').title()}"
        f.write(header + " \\\\\n")
        f.write("\\hline\n")
        
        # Data rows
        for exp_name, exp_data in experiments.items():
            config = exp_data.get('config', {})
            best_metrics = exp_data.get('best_metrics', {})
            
            row = f"{exp_name} & {config.get('model', 'N/A')}"
            for metric in metrics:
                value = best_metrics.get(metric, 0)
                row += f" & {value:.3f}"
            f.write(row + " \\\\\n")
        
        # Table footer
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {output_path}")
