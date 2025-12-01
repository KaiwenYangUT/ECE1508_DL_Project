# Quick Start Guide

This guide will get you up and running with the enhanced PointNet/PointNet++ implementation in 5 minutes.

## Step 1: Installation (2 minutes)

```bash
# Clone the repository (if not already done)
cd Pointnet_Pointnet2_pytorch

# Create and activate conda environment
conda create -n pointnet python=3.7
conda activate pointnet

# Install dependencies
pip install torch==1.6.0 torchvision==0.7.0
pip install numpy tqdm scikit-learn matplotlib seaborn pandas pyyaml
pip install tensorboard  # Optional but recommended
```

## Step 2: Download Data (varies by dataset size)

### For Classification (ModelNet40)
```bash
# Download and prepare
mkdir -p data
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
cd ..
```

## Step 3: Run Your First Experiment (2 minutes)

### Option A: Basic Training (Original)
```bash
python train_classification.py \
    --model pointnet2_cls_ssg \
    --log_dir my_first_experiment
```

### Option B: Enhanced Training (Recommended)
```bash
python train_classification_enhanced.py \
    --model pointnet2_cls_ssg \
    --log_dir my_first_experiment \
    --use_tensorboard \
    --generate_report \
    --epoch 10  # Use 10 epochs for quick test
```

## Step 4: Monitor Training (Optional)

Open a new terminal:
```bash
conda activate pointnet
tensorboard --logdir=log/classification/my_first_experiment/tensorboard
```

Then open browser to `http://localhost:6006`

## Step 5: View Results (1 minute)

After training completes, check:

```bash
# View experiment directory
ls -la log/classification/my_first_experiment/

# Expected structure:
# ├── checkpoints/           # Saved models
# ├── config.json            # Experiment configuration
# ├── config.yaml
# ├── logs/                  # Training logs
# ├── metrics.json           # Metrics history
# ├── reports/               # Auto-generated reports
# ├── visualizations/        # Training curves, plots
# └── tensorboard/           # TensorBoard logs

# View the report
cat log/classification/my_first_experiment/reports/experiment_report.md

# View metrics
cat log/classification/my_first_experiment/metrics.json
```

## Step 6: Analyze Results (1 minute)

Use Python to analyze:

```python
from utils import ResultsAnalyzer

# Load experiment
analyzer = ResultsAnalyzer('log/classification/')
analyzer.load_all_experiments()

# Get summary
summary = analyzer.get_summary_table()
print(summary)

# Export results
analyzer.export_summary('my_results.csv', format='csv')
```

## Quick Examples

### Run the Example Script
```bash
# Run all examples
python examples_usage.py

# Run specific example (1-7)
python examples_usage.py --example 4  # Visualization example
```

### Test the Utilities
```python
# In Python interpreter
from utils import ClassificationMetrics, TrainingVisualizer
import torch

# Test metrics
metrics = ClassificationMetrics(num_classes=10)
predictions = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))
metrics.update(predictions, targets)
results = metrics.compute()
print(results)

# Test visualization
visualizer = TrainingVisualizer('test_visualizations/')
train_metrics = [{'epoch': i, 'overall_accuracy': 0.5 + i*0.05} for i in range(10)]
val_metrics = [{'epoch': i, 'overall_accuracy': 0.48 + i*0.048} for i in range(10)]
visualizer.plot_training_curves(train_metrics, val_metrics)
```

## Common Commands

### Training Commands
```bash
# PointNet (basic)
python train_classification.py --model pointnet_cls --log_dir pointnet_exp

# PointNet++ SSG (recommended)
python train_classification.py --model pointnet2_cls_ssg --log_dir pn2_ssg_exp

# PointNet++ MSG (best accuracy)
python train_classification.py --model pointnet2_cls_msg --log_dir pn2_msg_exp

# With normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pn2_ssg_normals

# Quick test (10 epochs)
python train_classification.py --model pointnet2_cls_ssg --epoch 10 --log_dir quick_test
```

### Testing Commands
```bash
# Test classification
python test_classification.py --log_dir my_first_experiment

# Test with normals
python test_classification.py --use_normals --log_dir pn2_ssg_normals
```

### Analysis Commands
```bash
# Compare all experiments
python -c "
from utils import ResultsAnalyzer
analyzer = ResultsAnalyzer('log/classification/')
analyzer.load_all_experiments()
comparison = analyzer.compare_experiments('overall_accuracy')
for name, acc in comparison.items():
    print(f'{name}: {acc:.4f}')
"

# Generate comparison report
python -c "
from utils import ResultsAnalyzer, ReportGenerator
analyzer = ResultsAnalyzer('log/classification/')
analyzer.load_all_experiments()
report_gen = ReportGenerator('comparison_reports/')
report_gen.generate_comparison_report(analyzer.experiments)
print('Report generated!')
"
```

## Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python train_classification.py --batch_size 16 --log_dir exp

# Solution 2: Reduce number of points
python train_classification.py --num_point 512 --log_dir exp

# Solution 3: Use CPU (slow but works)
python train_classification.py --use_cpu --log_dir exp
```

### Issue: Import Error
```bash
# Make sure you're in the project root
cd Pointnet_Pointnet2_pytorch

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "from utils import ClassificationMetrics; print('Utils OK!')"
```

### Issue: No Data
```bash
# Check data directory
ls -la data/modelnet40_normal_resampled/

# Re-download if needed
cd data
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
```

## Next Steps

1. **Read the full documentation**: See `DOCUMENTATION.md` for comprehensive guides
2. **Try different models**: Experiment with PointNet, PointNet++ SSG, and MSG
3. **Explore utilities**: Run `examples_usage.py` to see all features
4. **Customize configuration**: Edit files in `configs/` directory
5. **Compare experiments**: Use the analysis tools to compare different runs

## Tips for Success

1. **Start small**: Use fewer epochs (10-20) for initial experiments
2. **Use TensorBoard**: Monitor training in real-time with `--use_tensorboard`
3. **Save configurations**: Always specify `--log_dir` with descriptive names
4. **Generate reports**: Enable `--generate_report` to document experiments
5. **Compare models**: Train multiple variants and use comparison tools

## Getting Help

- **Documentation**: Read `DOCUMENTATION.md` for detailed information
- **Examples**: Run `examples_usage.py` for working examples
- **Issues**: Check the troubleshooting section above
- **Code**: Review the enhanced training script for implementation details

## Performance Expectations

On ModelNet40 classification:
- **PointNet**: ~90.6% accuracy (fast, baseline)
- **PointNet++ SSG**: ~92.4% accuracy (balanced)
- **PointNet++ MSG**: ~92.8% accuracy (best but slower)

Training time (on single GPU):
- PointNet: ~2-3 hours for 200 epochs
- PointNet++ SSG: ~4-5 hours for 200 epochs
- PointNet++ MSG: ~6-8 hours for 200 epochs

## Success Checklist

- [ ] Environment installed and activated
- [ ] Data downloaded and extracted
- [ ] First training run completed
- [ ] TensorBoard monitoring works
- [ ] Results directory created with all files
- [ ] Visualizations generated
- [ ] Report created
- [ ] Metrics analyzed

**Congratulations!** You're now ready to conduct serious experiments with PointNet/PointNet++!

For advanced usage, refer to `DOCUMENTATION.md`.
