# PointNet++ for pointcloud classification in MSG model

This repo is based on [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

The goal is to understand how simple architectural changes within the PointNet++ framework influence accuracy on 3D shape classification.

## Install
The latest codes are tested on torch==2.5.1+cu118 and Python 3.11.3


## Classification (ModelNet40)
### Data Preparation
Download alignment **ModelNet** [here](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled) and save in `data/modelnet40_normal_resampled/`.

### Architecture Modifications

The PointNet2 MSG model was modified to support three architectural parameters:

1. **`--deepen <int>`**: Number of additional Set Abstraction layers (default: 0)
   - 0: Original architecture (3 SA layers)
   - 1: Add one extra SA layer (4 SA layers)

2. **`--widen <float>`**: Channel width multiplier for MLP layers (default: 1.0)
   - 0.8: Narrower networks
   - 1.0: Original width
   - 1.5: Wider network

3. **`--residual`**: Enable residual/skip connections between layers (default: False)

To modify the model architecture, please update the parameters in the following files:

- `models/pointnet2_cls_msg.py` (class `get_model`)
- `models/pointnet2_utils.py` (class `PointNetSetAbstractionMsg`)

### Training Configuration

All experiments were conducted with the following settings:
- **Dataset**: ModelNet40 (40 object classes)
- **Epochs**: 20 per model
- **Batch Size**: 4
- **Learning Rate**: 0.001 (with decay)(Default)
- **Optimizer**: Adam
- **Input**: 1024 points per point cloud
- **Use Normals**: Yes

## Run
### pointnet2_msg with normal features
```shell
python train_classification.py --model pointnet2_cls_msg --use_normals --log_dir pointnet2_cls_msg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_msg_normal
```

### Performance of classification in MSG model
| Model | test_instance_acc | class_acc |
|--|--|--|
| vanilla msg model(width=1,deepth=0,res=False)| 88.5 | 86.1 |
| modified msg model(width=1.5,deepth=1,res=True) | 87.1 | 82.3 |
| modified msg model(width=1,deepth=1,res=False) | 90.3 | 86.2 |
| modified msg model(width=1,deepth=1,res=True) |  89.7 | 86.0 |
| modified msg model(width=0.8,deepth=1,res=False) | 90.3 | 86.4 |

## Experimental Results and Key Findings

Our experiments show that the best Multi-Scale Grouping (MSG) variants achieve over **90% test accuracy**, slightly outperforming the vanilla MSG baseline (**88.6%**).

The strongest configurations are:
- **Moderate depth only** (`deepen = 1`), achieving **90.37%** test accuracy.
- **Moderate depth with width shrinkage** (`deepen = 1`, `width shrink = 0.8`), achieving **90.33%** test accuracy.

In contrast, adding residual connections consistently resulted in a slight performance degradation, and excessively wide networks performed worse. Overall, these results suggest a clear trend: **moderate depth provides the best performance trade-off**, while overly complex architectures do not yield further gains.

## MSG vs. SSG Comparison

Compared to Single-Scale Grouping (SSG), Multi-Scale Grouping (MSG) converges faster and more smoothly during training.  
MSG models achieve higher training accuracy, while test accuracy stabilizes earlier, indicating improved training stability.

These results confirm that multi-scale grouping improves robustness to variations in point density. In addition, MSG exhibits significantly reduced overfitting compared to SSG, further supporting its effectiveness for point-cloud classification tasks.

## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)


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
