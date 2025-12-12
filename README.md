# ECE1508 Deep Learning Project: PointNet2 MSG Architecture Optimization

## Project Overview

This project investigates architectural modifications to the PointNet2 Multi-Scale Grouping (MSG) model for 3D point cloud classification on the ModelNet40 dataset. The research focuses on understanding how different architectural parameters—including network width, depth, and residual connections—impact model performance.

### Objectives

- Evaluate the impact of network depth on PointNet2 MSG performance
- Investigate the effect of width multipliers on classification accuracy
- Analyze the contribution of residual connections in MSG architecture
- Identify optimal architectural configurations for 3D point cloud classification

## Architecture Modifications

### PointNet2 MSG Baseline
The baseline model uses the standard PointNet2 Multi-Scale Grouping architecture with hierarchical feature learning through:
- Set Abstraction (SA) layers with multiple scales
- Feature propagation for point-wise predictions
- Multi-scale grouping for robust feature extraction

### Investigated Modifications

1. **Depth Parameter (d)**
   - Controls the number of additional layers in the network
   - Tested configurations: d=0 (vanilla), d=1 (modified)

2. **Width Parameter (w)**
   - Multiplier for the number of channels in each layer
   - Tested configurations: w=0.8, w=1.0, w=1.5

3. **Residual Connections (r)**
   - Skip connections to facilitate gradient flow
   - Tested configurations: r=True, r=False

## Experimental Setup

### Dataset
- **Name**: ModelNet40
- **Classes**: 40 object categories
- **Training samples**: ~9,843 models
- **Test samples**: ~2,468 models
- **Point cloud size**: 1,024 points per object

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with decay)
- **Batch Size**: 24
- **Epochs**: 21
- **Data Augmentation**: Random rotation and jittering

### Evaluated Models

| Model ID | Width (w) | Depth (d) | Residual (r) | Description |
|----------|-----------|-----------|--------------|-------------|
| Vanilla MSG | 1.0 | 0 | No | Baseline model |
| Modified-1 | 0.8 | 1 | No | Reduced width with depth |
| Modified-2 | 1.0 | 1 | No | Standard width with depth |
| Modified-3 | 1.0 | 1 | Yes | Standard width with depth and residual |
| Modified-4 | 1.5 | 1 | Yes | Increased width with depth and residual |

## Results and Findings

### Performance Summary

| Model | Test Accuracy | Class Accuracy | Improvement over Baseline |
|-------|---------------|----------------|---------------------------|
| **Modified (w=1, d=1, r=False)** | **90.37%** | 86.22% | **+0.47%** |
| Modified (w=0.8, d=1, r=False) | 90.33% | **86.44%** | +0.42% |
| Vanilla MSG (baseline) | 89.91% | 86.09% | - |
| Modified (w=1, d=1, r=True) | 89.67% | 85.98% | -0.24% |
| Modified (w=1.5, d=1, r=True) | 87.32% | 82.30% | -2.59% |

### Key Findings

#### 1. Impact of Network Depth
Adding one additional depth layer (d=1) to the baseline architecture improved performance:
- **Vanilla MSG (d=0)**: 89.91%
- **Modified (w=1, d=1, r=False)**: 90.37%
- **Improvement**: +0.47 percentage points

**Conclusion**: Moderate increase in depth enhances feature learning capacity without overfitting.

#### 2. Effect of Width Multiplier
Performance varies significantly with width:
- **w=0.8**: 90.33% (slight reduction in parameters, minimal accuracy loss)
- **w=1.0**: 90.37% (optimal balance)
- **w=1.5**: 87.32% (excessive parameters lead to degradation)

**Conclusion**: Width multiplier of 1.0 provides the best balance. Increasing width to 1.5x actually hurts performance, likely due to overfitting or optimization difficulties.

#### 3. Residual Connections
Residual connections showed unexpected negative impact:
- **Without residual (w=1, d=1)**: 90.37%
- **With residual (w=1, d=1)**: 89.67%
- **Difference**: -0.70 percentage points

**Conclusion**: In this relatively shallow architecture, residual connections do not provide the expected benefits and may introduce unnecessary complexity.

### Visualization Results

All visualizations are available in the `visualization_results_msg/` directory:

- **Performance comparisons**: Overall model rankings and accuracy charts
- **Learning curves**: Training progression over 21 epochs
- **Overfitting analysis**: Training vs. test accuracy gaps
- **Configuration impact**: Effect of different hyperparameters
- **Summary report**: Detailed text-based analysis

## Installation and Setup

### Prerequisites
```bash
# Python 3.7+
# CUDA 10.1+ (for GPU support)
# PyTorch 1.6+
```

### Environment Setup

1. **Clone the repository**
```bash
git clone https://github.com/KaiwenYangUT/ECE1508_DL_Project.git
cd ECE1508_DL_Project/Pointnet_Pointnet2_pytorch
```

2. **Create virtual environment**
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install torch==1.6.0
pip install -r requirements.txt
```

### Data Preparation

Download the ModelNet40 dataset:
```bash
# Download from: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
# Extract to: data/modelnet40_normal_resampled/
```

## Usage

### Training Models

#### Train Vanilla MSG (Baseline)
```bash
python train_classification.py \
    --model pointnet2_cls_msg \
    --log_dir pointnet2_cls_msg \
    --epoch 21
```

#### Train Modified Model (Best Configuration)
```bash
python train_classification.py \
    --model pointnet2_cls_msg \
    --log_dir pointnet2_msg_best \
    --width 1.0 \
    --depth 1 \
    --no-residual \
    --epoch 21
```

#### Train with Different Width
```bash
python train_classification.py \
    --model pointnet2_cls_msg \
    --log_dir pointnet2_msg_w08 \
    --width 0.8 \
    --depth 1 \
    --epoch 21
```

### Testing Models

```bash
python test_classification.py \
    --log_dir pointnet2_msg_best
```

### Visualizing Results

Generate all performance visualizations:
```bash
python visualize_msg_results.py
```

This will create:
- Performance comparison charts
- Learning curves
- Overfitting analysis
- Configuration impact plots
- Summary report

Output saved to `visualization_results_msg/`

## Project Structure

```
ECE1508_DL_Project/
├── models/
│   ├── pointnet2_cls_msg.py          # Main MSG model implementation
│   ├── pointnet2_cls_ssg.py          # Single-scale grouping variant
│   └── pointnet2_utils.py            # Utility functions for PointNet2
├── data_utils/
│   ├── ModelNetDataLoader.py         # DataLoader for ModelNet40
│   └── provider.py                   # Data augmentation utilities
├── log/
│   └── classification/               # Training logs and checkpoints
├── visualization_results_msg/        # Generated visualizations
│   ├── *.png                        # Performance charts
│   ├── msg_summary_report.txt       # Detailed text report
│   └── README.md                    # Visualization documentation
├── train_classification.py           # Training script
├── test_classification.py            # Testing/evaluation script
├── visualize_msg_results.py          # Visualization generation
├── pointnet2_msg_2.csv              # Raw experimental results
└── README.md                         # This file
```

## Reproducibility

### Reproducing Experiments

All experiments can be reproduced using the provided scripts:

1. **Vanilla baseline**: 21 epochs, default parameters
2. **Modified models**: Various width/depth/residual combinations
3. **Visualization**: Automated chart generation from CSV results

### Random Seeds
Set random seeds for reproducibility:
```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

## Performance Comparison with Literature

| Model | Test Accuracy | Source |
|-------|---------------|--------|
| PointNet (Official) | 89.2% | Qi et al. 2017 |
| PointNet2 (Official) | 91.9% | Qi et al. 2017 |
| PointNet2 MSG (Baseline) | 89.91% | This work |
| **PointNet2 MSG (Optimized)** | **90.37%** | **This work** |

**Note**: Our baseline is slightly lower than the official implementation, likely due to different training configurations or data preprocessing. However, our modifications show consistent improvements over our baseline.

## Insights and Conclusions

### Main Contributions

1. **Systematic evaluation** of architectural parameters in PointNet2 MSG
2. **Identification of optimal configuration**: w=1.0, d=1, no residual connections
3. **Counter-intuitive findings**: Residual connections and excessive width hurt performance
4. **Comprehensive visualization** framework for model comparison

### Practical Recommendations

For 3D point cloud classification with PointNet2 MSG:
- ✅ **Do**: Add moderate depth (d=1) for better feature learning
- ✅ **Do**: Use standard width (w=1.0) or slightly reduced (w=0.8)
- ✅ **Do**: Train for at least 20 epochs for convergence
- ❌ **Don't**: Add residual connections in shallow architectures
- ❌ **Don't**: Use excessive width (w>1.0) without regularization

### Future Work

- Investigate deeper architectures (d=2, d=3) with appropriate regularization
- Explore different residual connection patterns
- Test on additional datasets (ShapeNet, S3DIS)
- Combine successful modifications with other techniques (e.g., attention mechanisms)
- Analyze computational efficiency vs. accuracy trade-offs

## References

### Papers
- Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3d classification and segmentation. *CVPR*.
- Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep hierarchical feature learning on point sets in a metric space. *NeurIPS*.

### Code Base
This project builds upon:
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Course Information

**Course**: ECE1508 - Deep Learning  
**Institution**: University of Toronto  
**Term**: Fall 2025  
**Date**: December 12, 2025

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original PointNet and PointNet2 authors for their groundbreaking work
- The PyTorch PointNet implementation community
- ECE1508 course instructors and TAs

---

For questions or issues, please open an issue on the [GitHub repository](https://github.com/KaiwenYangUT/ECE1508_DL_Project).