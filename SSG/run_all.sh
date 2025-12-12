#!/bin/bash

echo "Running widen=1.5 model..."
python train_classification.py --model pointnet2_cls_ssg --epoch 10 --use_normals --widen 1.5 --log_dir ssg_widen1_5_e10

echo "Running deepen=1 model..."
python train_classification.py --model pointnet2_cls_ssg --epoch 10 --use_normals --deepen 1 --log_dir ssg_deepen1_e10

echo "Running residual=True model..."
python train_classification.py --model pointnet2_cls_ssg --epoch 10 --use_normals --residual --log_dir ssg_residual_e10

echo "Running combined (widen=1.5, deepen=1, residual=True) model..."
python train_classification.py --model pointnet2_cls_ssg --epoch 10 --use_normals --widen 1.5 --deepen 1 --residual --log_dir ssg_widen_1_5_deepen1_residual_e10

echo "All 4 runs finished 🎉"

