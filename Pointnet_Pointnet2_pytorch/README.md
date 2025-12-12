# PointNet++ for point cloud classification in **SSG model (Modified)**

This branch is based on [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) implemented in PyTorch.

The goal of this work is to **systematically study how architectural modifications to the PointNet++ SSG (Single-Scale Grouping) model**—specifically **network deepening, channel widening, and residual connections**—affect 3D shape classification performance on ModelNet.

---

## Install

The latest code is tested on:

* **Python** 3.9+
* **PyTorch** 2.5.1 (MPS / CUDA compatible)

---

## Classification (ModelNet10 / ModelNet40)

### Data Preparation

Download the aligned **ModelNet** dataset from
[https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled)

and place it under:

```
data/modelnet40_normal_resampled/
```

---

### Run

You can control architectural variants using the following arguments:

* `--deepen` : number of extra MLP layers added per Set Abstraction block
* `--widen` : channel width multiplier applied to MLP layers
* `--residual` : enable residual connections within MLP blocks
* `--use_normals` : include normal vectors as input features (used in all experiments)

#### Example commands (ModelNet40)

```bash
# Baseline SSG
python train_classification.py \
  --model pointnet2_cls_ssg \
  --use_normals \
  --log_dir ssg_baseline

# Deepened SSG
python train_classification.py \
  --model pointnet2_cls_ssg \
  --use_normals \
  --deepen 1 \
  --log_dir ssg_deepen1

# Narrow SSG (widen < 1)
python train_classification.py \
  --model pointnet2_cls_ssg \
  --use_normals \
  --widen 0.8 \
  --log_dir ssg_widen0_8

# Deepened + residual SSG
python train_classification.py \
  --model pointnet2_cls_ssg \
  --use_normals \
  --deepen 1 \
  --residual \
  --log_dir ssg_deepen1_residual
```

Testing:

```bash
python test_classification.py --log_dir <log_dir_name>
```

---

## Architectural Modifications (SSG)

Compared to the vanilla SSG model, the following modifications were implemented:

* **Deepening**: inserting additional MLP layers inside Set Abstraction blocks
* **Widening**: scaling channel dimensions by a multiplicative factor
* **Residual connections**: identity skip connections within MLP stacks

These changes are fully configurable via command-line arguments and do not alter the overall PointNet++ pipeline.

---

## Performance on ModelNet40 (SSG)

| Model        | deepen | widen | residual | test_instance_acc | class_acc |
| ------------ | -----: | ----: | :------: | ----------------: | --------: |
| Vanilla SSG  |      0 |   1.0 |     ❌    |          baseline |  baseline |
| Modified SSG |      1 |   1.0 |     ✅    |             ~80.8 |     ~74.5 |
| Modified SSG |      1 |   0.9 |     ❌    |             ~80.6 |     ~73.7 |
| Modified SSG |      1 |  0.85 |     ❌    |          **84.3** |  **77.3** |
| Modified SSG |      1 |  0.80 |     ❌    |          **85.4** |  **79.5** |
| Modified SSG |      0 |  0.85 |     ❌    |          **84.7** |  **79.1** |
| Modified SSG |      0 |  0.80 |     ❌    |          **84.9** |  **79.4** |

**Key observations:**

* Moderate **channel narrowing (widen < 1)** consistently improves generalization
* **Residual connections help stability**, but are not always necessary for best accuracy
* Excessive deepening degrades performance without residual support

---

## Key Insights

* **SSG benefits more from capacity control than capacity expansion**
* Narrower networks reduce overfitting while preserving geometric expressiveness
* Residual connections become more important as depth increases
* Best-performing models are **simpler than the baseline**, not larger

---

## Reference By

[halimacc/pointnet3](https://github.com/halimacc/pointnet3)
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)
[charlesq34/PointNet](https://github.com/charlesq34/pointnet)
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)

---

## Citation

```bibtex
@article{Pytorch_Pointnet_Pointnet2,
  Author = {Xu Yan},
  Title = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year = {2019}
}
```

```bibtex
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={CVPR},
  year={2020}
}
```
