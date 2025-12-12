# PointNet2 MSG Model Performance Visualization Results

This directory contains comprehensive visualization and analysis results for PointNet2 Multi-Scale Grouping (MSG) model experiments conducted on the ModelNet40 dataset.

## Overview

This analysis compares the performance of 5 different PointNet2 MSG model configurations to evaluate the impact of various architectural modifications:
- **Vanilla MSG**: Baseline model (depth=0)
- **Modified Models**: Variations with different width multipliers, depth, and residual connections

## Experiment Configurations

| Model | Width | Depth | Residual | Test Accuracy | Class Accuracy |
|-------|-------|-------|----------|---------------|----------------|
| Modified (w=1, d=1, r=False) | 1.0 | 1 | No | **90.37%** | 86.22% |
| Modified (w=0.8, d=1, r=False) | 0.8 | 1 | No | 90.33% | **86.44%** |
| Vanilla MSG | 1.0 | 0 | No | 89.91% | 86.09% |
| Modified (w=1, d=1, r=True) | 1.0 | 1 | Yes | 89.67% | 85.98% |
| Modified (w=1.5, d=1, r=True) | 1.5 | 1 | Yes | 87.32% | 82.30% |

## Key Findings

### 1. Depth Effect
Adding depth (d=1) to the baseline model improved performance:
- Vanilla MSG (d=0): 89.91%
- Modified (w=1, d=1, r=False): 90.37%
- **Improvement: +0.47%**

### 2. Width Effect
Best performance achieved with moderate width (w=0.8 to w=1.0):
- w=0.8: 90.33%
- w=1.0: 90.37%
- w=1.5: 87.32%
- **Conclusion**: Excessive width (w=1.5) degraded performance

### 3. Residual Connections
Residual connections showed mixed results:
- Without residual: 90.37%
- With residual: 89.67%
- **Difference: -0.70%**
- Residual connections did not improve performance in this configuration

## Generated Visualizations

### Performance Comparison Charts
- `msg_performance_comparison.png` - Overall model performance comparison
- `msg_overall_model_comparison.png` - Comprehensive comparison across all metrics
- `msg_best_test_accuracy.png` - Top model by test accuracy
- `msg_best_class_accuracy.png` - Top model by class accuracy
- `msg_performance_table.png` - Tabular summary of all model results

### Training Analysis
- `msg_training_test_accuracy.png` - Training vs. test accuracy curves
- `msg_learning_curves_20epochs.png` - Learning progression over 20 epochs
- `msg_train_vs_test_final.png` - Final training vs. test accuracy comparison
- `msg_overfitting_analysis.png` - Analysis of overfitting behavior

### Accuracy Metrics
- `msg_class_accuracy.png` - Class-wise accuracy comparison
- `msg_configuration_impact.png` - Impact of different configuration parameters

### Summary Report
- `msg_summary_report.txt` - Detailed text report with rankings and analysis

## Dataset Information

- **Dataset**: ModelNet40
- **Training Epochs**: 21 (across all models)
- **Evaluation Metrics**:
  - Test Instance Accuracy
  - Test Class Accuracy
  - Training Instance Accuracy

## Top Performing Model

**Winner: Modified (w=1, d=1, r=False)**
- Configuration: width=1.0, depth=1, residual=False
- Best Test Accuracy: **90.37%**
- Best Class Accuracy: 86.22%
- Improvement over baseline: +0.47%

## How to Regenerate

To regenerate these visualizations, run:
```bash
python visualize_msg_results.py
```

This script reads from `pointnet2_msg_2.csv` and generates all charts and the summary report.

## Analysis Insights

1. **Optimal Architecture**: The best configuration uses moderate width (w=1.0) with additional depth (d=1) but without residual connections.

2. **Diminishing Returns**: Increasing model width beyond 1.0 led to performance degradation, suggesting potential overfitting or optimization difficulties.

3. **Residual Connections**: While residual connections often help in deep networks, they did not provide benefits in this specific MSG architecture, possibly due to the relatively shallow nature of the modifications.

4. **Consistency**: All models showed good convergence within 21 epochs, with the top 3 models achieving very similar performance (89.91% - 90.37%).

## Related Files

- Training script: `train_classification.py`
- Visualization script: `visualize_msg_results.py`
- Raw results: `pointnet2_msg_2.csv`
- Model implementation: `models/pointnet2_cls_msg.py`

---

*Generated on December 12, 2025 as part of ECE1508 Deep Learning Project*
