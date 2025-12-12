# ECE1508 Deep Learning Project: PointNet2 Architecture Exploration

This repository contains a comprehensive study of PointNet2 SSG (Single-Scale Grouping) architecture variations, exploring the effects of network depth, width, and residual connections on 3D point cloud classification performance.

## Project Overview

This project investigates how architectural modifications to the PointNet2 SSG model affect classification accuracy on the ModelNet40 dataset. We explore three key architectural dimensions:

- **Depth (Deepen)**: Adding extra Set Abstraction layers (0, 1, or 2 additional layers)
- **Width (Widen)**: Scaling the number of channels in MLP layers (0.8, 0.85, 0.9, 1.0, 1.5)
- **Residual Connections**: Adding skip connections between layers (True/False)

### Key Findings

Our experiments on ModelNet40 classification revealed:
- **Best Model**: Deepen=1 achieved **85.54% test accuracy** and **79.42% class accuracy**
- Deepening the network by one layer significantly improved performance over the baseline (79.17%)
- Residual connections showed mixed results depending on network configuration
- Excessive widening (1.5×) degraded performance, likely due to overfitting
- Moderate width reduction (0.9×) maintained competitive performance (81.98% test accuracy)

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.6.0+
- CUDA 10.1+ (for GPU support)

### Setup
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
pip install matplotlib numpy pandas seaborn
```

## Experimental Setup

### Architecture Modifications

The PointNet2 SSG model was modified to support three architectural parameters:

1. **`--deepen <int>`**: Number of additional Set Abstraction layers (default: 0)
   - 0: Original architecture (3 SA layers)
   - 1: Add one extra SA layer (4 SA layers)
   - 2: Add two extra SA layers (5 SA layers)

2. **`--widen <float>`**: Channel width multiplier for MLP layers (default: 1.0)
   - 0.8, 0.85, 0.9: Narrower networks
   - 1.0: Original width
   - 1.5: Wider network

3. **`--residual`**: Enable residual/skip connections between layers (default: False)

### Training Configuration

All experiments were conducted with the following settings:
- **Dataset**: ModelNet40 (40 object classes)
- **Epochs**: 10 per model
- **Batch Size**: 24
- **Learning Rate**: 0.001 (with decay)
- **Optimizer**: Adam
- **Input**: 1024 points per point cloud
- **Use Normals**: No (only XYZ coordinates)

## Classification (ModelNet40)

### Data Preparation
Download alignment **ModelNet40** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Training Commands

#### Baseline Model
```shell
python train_classification.py --model pointnet2_cls_ssg --log_dir ssg_baseline_e10
python test_classification.py --log_dir ssg_baseline_e10
```

#### Architectural Variations

**Depth Experiments:**
```shell
# Deepen=1
python train_classification.py --model pointnet2_cls_ssg --deepen 1 --log_dir ssg_deepen1_e10

# Deepen=2 with Residual
python train_classification.py --model pointnet2_cls_ssg --deepen 2 --residual --log_dir ssg_deepen2_residual_e10
```

**Width Experiments:**
```shell
# Widen=0.9
python train_classification.py --model pointnet2_cls_ssg --widen 0.9 --log_dir ssg_widen0.9_e10

# Widen=1.5
python train_classification.py --model pointnet2_cls_ssg --widen 1.5 --log_dir ssg_widen1.5_e10
```

**Combined Experiments:**
```shell
# Residual only
python train_classification.py --model pointnet2_cls_ssg --residual --log_dir ssg_residual_e10

# Deepen=1 + Residual
python train_classification.py --model pointnet2_cls_ssg --deepen 1 --residual --log_dir ssg_deepen1_residual_e10

# Widen=1.5 + Deepen=1 + Residual
python train_classification.py --model pointnet2_cls_ssg --widen 1.5 --deepen 1 --residual --log_dir ssg_widen1.5_deepen1_residual_e10
```

### Experimental Results

| Configuration | Test Accuracy | Class Accuracy | Training Time |
|--------------|---------------|----------------|---------------|
| **Baseline** (d=0, w=1.0, r=False) | 79.17% | 76.89% | 130.23 min |
| **Deepen=1** ⭐ | **85.54%** | **79.42%** | 135.13 min |
| **Residual=True** | 82.93% | 76.23% | 136.19 min |
| **Widen=0.9** | 81.98% | 75.98% | ~135 min |
| **Widen=1.5** | 74.65% | 70.93% | 138.88 min |
| **Deepen=1 + Residual** | 80.75% | 74.54% | 138.39 min |
| **Deepen=2 + Residual** | 72.19% | 70.46% | 141.34 min |
| **Widen=1.5 + Deepen=1 + Residual** | 72.56% | 64.69% | 148.59 min |

⭐ Best performing model

### Visualization and Analysis

The project includes comprehensive visualization and analysis tools:

```shell
# Generate performance visualizations and summary report
python visualize_results.py

# Parse training logs and analyze configuration effects
python parse_logs_and_update_config_effects.py
```

**Generated Visualizations** (saved in `visualization_results/`):
1. **Training curves**: Accuracy vs. epochs for all models
2. **Comparison plots**: Best test accuracy, class accuracy, and training time
3. **Configuration heatmaps**: Effect of deepen/widen/residual parameters
4. **Summary report**: Detailed performance metrics for each model

### Additional Experiments (Width Scaling Study)

Fine-grained width scaling experiments:
```shell
# Width variations: 0.80, 0.85, 0.90, 1.0
python train_classification.py --model pointnet2_cls_ssg --widen 0.80 --log_dir ssg_widen0.80_e10
python train_classification.py --model pointnet2_cls_ssg --widen 0.85 --log_dir ssg_widen0.85_e10

# With Deepen=1
python train_classification.py --model pointnet2_cls_ssg --widen 0.80 --deepen 1 --log_dir ssg_widen0.80_deepen1_e10
```

## Project Structure

```
.
├── models/
│   ├── pointnet2_cls_ssg.py    # Modified PointNet2 SSG with deepen/widen/residual
│   ├── pointnet2_cls_msg.py    # Multi-scale grouping variant
│   └── pointnet_utils.py       # Utility functions
├── log/classification/          # Training logs and checkpoints
├── Train_Log/                   # Organized training logs by configuration
├── visualization_results/       # Generated plots and summary reports
├── train_classification.py     # Training script
├── test_classification.py      # Testing script
├── visualize_results.py        # Visualization and analysis script
└── parse_logs_and_update_config_effects.py  # Log parsing utility
```

## Key Insights

1. **Depth Matters**: Adding one extra Set Abstraction layer (deepen=1) improved test accuracy by 6.37 percentage points over the baseline.

2. **Width Trade-offs**: 
   - Moderate width reduction (0.9×) maintained performance with fewer parameters
   - Excessive widening (1.5×) led to overfitting and degraded performance

3. **Residual Connections**: 
   - Beneficial for shallow networks (baseline + residual: 82.93%)
   - Mixed results when combined with deepening
   - May require different learning rates or training strategies

4. **Computational Cost**: 
   - Deepening increased training time by ~4% (135 vs 130 minutes)
   - Widening increased training time more significantly (~7-14%)

5. **Overfitting Risk**: Deep + wide + residual combinations showed signs of overfitting, suggesting the need for regularization or longer training.


## References and Acknowledgments

This project builds upon the excellent PointNet/PointNet++ PyTorch implementation by Xu Yan.

### Base Implementation
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

### Original Papers
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) (CVPR 2017)
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) (NeurIPS 2017)

### Additional References
- [halimacc/pointnet3](https://github.com/halimacc/pointnet3)
- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
- [charlesq34/PointNet](https://github.com/charlesq34/pointnet)
- [charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


## Citation
```
@article{Pytorch_Pointnet_Pointnet2,
  Author = {Xu Yan},
  Title = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```
```
@InProceedings{yan20222dpass,
      title={2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds}, 
      author={Xu Yan and Jiantao Gao and Chaoda Zheng and Chao Zheng and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      journal={ECCV}
}
```
## Selected Projects using This Codebase
* [PointConv: Deep Convolutional Networks on 3D Point Clouds, CVPR'19](https://github.com/Young98CN/pointconv_pytorch)
* [On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks, CVPR'20](https://github.com/skywalker6174/3d-isometry-robust)
* [Label-Efficient Learning on Point Clouds using Approximate Convex Decompositions, ECCV'20](https://github.com/matheusgadelha/PointCloudLearningACD)
* [PCT: Point Cloud Transformer](https://github.com/MenghaoGuo/PCT)
* [PSNet: Fast Data Structuring for Hierarchical Deep Learning on Point Cloud](https://github.com/lly007/PointStructuringNet)
* [Stratified Transformer for 3D Point Cloud Segmentation, CVPR'22](https://github.com/dvlab-research/stratified-transformer)
