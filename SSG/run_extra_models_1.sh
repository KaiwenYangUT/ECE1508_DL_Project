#!/bin/bash

echo "===== Model 1: deepen=1, widen=1.0, residual=True ====="
caffeinate -i python train_classification.py \
  --model pointnet2_cls_ssg \
  --epoch 10 \
  --use_normals \
  --deepen 1 \
  --widen 1.0 \
  --residual \
  --log_dir ssg_deepen1_residual_e10

echo "===== Model 2: deepen=2, widen=1.0, residual=True ====="
caffeinate -i python train_classification.py \
  --model pointnet2_cls_ssg \
  --epoch 10 \
  --use_normals \
  --deepen 2 \
  --widen 1.0 \
  --residual \
  --log_dir ssg_deepen2_residual_e10

echo "===== Model 3: deepen=2, widen=1.0, residual=False ====="
caffeinate -i python train_classification.py \
  --model pointnet2_cls_ssg \
  --epoch 10 \
  --use_normals \
  --deepen 2 \
  --widen 1.0 \
  --log_dir ssg_deepen2_noresidual_e10

echo "===== Model 4: deepen=0, widen=0.9, residual=False ====="
caffeinate -i python train_classification.py \
  --model pointnet2_cls_ssg \
  --epoch 10 \
  --use_normals \
  --widen 0.9 \
  --log_dir ssg_widen0_9_e10

echo "===== All extra models completed 🎉 ====="

