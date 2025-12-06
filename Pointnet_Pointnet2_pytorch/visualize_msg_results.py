import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('visualization_results_msg')
output_dir.mkdir(exist_ok=True)

# Read the CSV file - skip first header row, use second row as column names
df = pd.read_csv('pointnet2_msg_2.csv', skiprows=[0])

# Parse the data - the CSV has a complex structure with multiple model configurations
# Column structure: vanilla msg model (3 cols), modified msg model w=1.5,d=1,r=True (3 cols), 
#                  modified best w=1,d=1,r=False (3 cols), modified best w=1,d=1,r=True (3 cols)

# Clean up the data
df_clean = df.copy()

# Extract epoch numbers (remove '/50' suffix)
df_clean['epoch'] = df_clean['epoch'].astype(str).str.extract(r'(\d+)')[0].astype(float)

# Define model configurations
models = {
    'Vanilla MSG': {
        'train_acc': 'train_instance_acc',
        'test_acc': 'test_instance_acc',
        'class_acc': 'class_acc',
        'color': '#1f77b4',
        'config': 'Baseline'
    },
    'Modified (w=1.5, d=1, r=True)': {
        'train_acc': 'train_instance_acc.1',
        'test_acc': 'test_instance_acc.1',
        'class_acc': 'class_acc.1',
        'color': '#ff7f0e',
        'config': 'width=1.5, depth=1, residual=True'
    },
    'Modified (w=1, d=1, r=False)': {
        'train_acc': 'train_instance_acc.2',
        'test_acc': 'test_instance_acc.2',
        'class_acc': 'class_acc.2',
        'color': '#2ca02c',
        'config': 'width=1, depth=1, residual=False'
    },
    'Modified (w=1, d=1, r=True)': {
        'train_acc': 'train_instance_acc.3',
        'test_acc': 'test_instance_acc.3',
        'class_acc': 'class_acc.3',
        'color': '#d62728',
        'config': 'width=1, depth=1, residual=True'
    },
    'Modified (w=0.8, d=1, r=False)': {
        'train_acc': 'train_instance_acc.4',
        'test_acc': 'test_instance_acc.4',
        'class_acc': 'class_acc.4',
        'color': '#9467bd',
        'config': 'width=0.8, depth=1, residual=False'
    }
}

# ====================================================================================
# 1. Training and Test Accuracy Curves
# ====================================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['test_acc']]].dropna()
    ax1.plot(data['epoch'], data[cols['test_acc']], 
             marker='o', markersize=4, linewidth=2, 
             label=model_name, color=cols['color'], alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Instance Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Test Instance Accuracy Over Training', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 20)

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['train_acc']]].dropna()
    ax2.plot(data['epoch'], data[cols['train_acc']], 
             marker='s', markersize=4, linewidth=2, 
             label=model_name, color=cols['color'], alpha=0.8)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Training Instance Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Training Instance Accuracy Over Training', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 20)

plt.tight_layout()
plt.savefig(output_dir / 'msg_training_test_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_training_test_accuracy.png")
plt.close()

# ====================================================================================
# 2. Class Accuracy Curves
# ====================================================================================
fig, ax = plt.subplots(figsize=(14, 7))

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['class_acc']]].dropna()
    ax.plot(data['epoch'], data[cols['class_acc']], 
            marker='D', markersize=5, linewidth=2.5, 
            label=model_name, color=cols['color'], alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Class Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Test Class Accuracy Comparison Across MSG Configurations', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)

plt.tight_layout()
plt.savefig(output_dir / 'msg_class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_class_accuracy.png")
plt.close()

# ====================================================================================
# 3. Best Performance Comparison
# ====================================================================================
best_results = {}
for model_name, cols in models.items():
    test_data = df_clean[cols['test_acc']].dropna()
    class_data = df_clean[cols['class_acc']].dropna()
    train_data = df_clean[cols['train_acc']].dropna()
    
    best_results[model_name] = {
        'test_acc': test_data.max(),
        'class_acc': class_data.max(),
        'final_test_acc': test_data.iloc[-1] if len(test_data) > 0 else 0,
        'final_class_acc': class_data.iloc[-1] if len(class_data) > 0 else 0,
        'final_train_acc': train_data.iloc[-1] if len(train_data) > 0 else 0,
        'epochs_trained': len(test_data)
    }

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Best test accuracy
model_names = list(best_results.keys())
test_accs = [best_results[m]['test_acc'] * 100 for m in model_names]
colors = [models[m]['color'] for m in model_names]

bars = axes[0, 0].bar(range(len(model_names)), test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Best Test Instance Accuracy', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(range(len(model_names)))
axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, test_accs)):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Best class accuracy
class_accs = [best_results[m]['class_acc'] * 100 for m in model_names]
bars = axes[0, 1].bar(range(len(model_names)), class_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Best Test Class Accuracy', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(range(len(model_names)))
axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, class_accs)):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Final epoch accuracy comparison
final_test = [best_results[m]['final_test_acc'] * 100 for m in model_names]
final_train = [best_results[m]['final_train_acc'] * 100 for m in model_names]

x = np.arange(len(model_names))
width = 0.35
bars1 = axes[1, 0].bar(x - width/2, final_train, width, label='Train', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = axes[1, 0].bar(x + width/2, final_test, width, label='Test', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Final Epoch: Train vs Test Accuracy', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Overfitting analysis (train - test gap)
overfit_gap = [best_results[m]['final_train_acc'] * 100 - best_results[m]['final_test_acc'] * 100 
               for m in model_names]
bars = axes[1, 1].bar(range(len(model_names)), overfit_gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Accuracy Gap (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Overfitting Analysis (Train - Test Gap)', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(range(len(model_names)))
axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
for i, (bar, val) in enumerate(zip(bars, overfit_gap)):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top', 
                    fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'msg_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_performance_comparison.png")
plt.close()

# ====================================================================================
# 4. Learning Curves Comparison (First 20 Epochs)
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Focus on first 20 epochs for better visualization
max_epoch = 20

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['test_acc']]].dropna()
    data_20 = data[data['epoch'] <= max_epoch]
    axes[0, 0].plot(data_20['epoch'], data_20[cols['test_acc']], 
                    marker='o', markersize=6, linewidth=2.5, 
                    label=model_name, color=cols['color'], alpha=0.8)

axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Test Instance Accuracy', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Test Instance Accuracy (First 20 Epochs)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=9, loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['train_acc']]].dropna()
    data_20 = data[data['epoch'] <= max_epoch]
    axes[0, 1].plot(data_20['epoch'], data_20[cols['train_acc']], 
                    marker='s', markersize=6, linewidth=2.5, 
                    label=model_name, color=cols['color'], alpha=0.8)

axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Training Instance Accuracy', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Training Instance Accuracy (First 20 Epochs)', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=9, loc='lower right')
axes[0, 1].grid(True, alpha=0.3)

for model_name, cols in models.items():
    data = df_clean[['epoch', cols['class_acc']]].dropna()
    data_20 = data[data['epoch'] <= max_epoch]
    axes[1, 0].plot(data_20['epoch'], data_20[cols['class_acc']], 
                    marker='D', markersize=6, linewidth=2.5, 
                    label=model_name, color=cols['color'], alpha=0.8)

axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Test Class Accuracy', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Test Class Accuracy (First 20 Epochs)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=9, loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# Convergence speed (epochs to reach 80% test accuracy)
convergence_data = []
for model_name, cols in models.items():
    data = df_clean[['epoch', cols['test_acc']]].dropna()
    epochs_to_80 = data[data[cols['test_acc']] >= 0.80]['epoch'].min()
    if not np.isnan(epochs_to_80):
        convergence_data.append((model_name, epochs_to_80))

if convergence_data:
    conv_models, conv_epochs = zip(*convergence_data)
    conv_colors = [models[m]['color'] for m in conv_models]
    bars = axes[1, 1].bar(range(len(conv_models)), conv_epochs, color=conv_colors, 
                          alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Epochs', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Convergence Speed (Epochs to Reach 80% Test Accuracy)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(range(len(conv_models)))
    axes[1, 1].set_xticklabels(conv_models, rotation=15, ha='right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, conv_epochs)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                        f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
else:
    axes[1, 1].text(0.5, 0.5, 'No models reached 80% accuracy', 
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'msg_learning_curves_20epochs.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_learning_curves_20epochs.png")
plt.close()

# ====================================================================================
# 5. Configuration Impact Analysis
# ====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Group by configuration parameters
config_groups = {
    'Width Effect': [
        ('w=0.8', 'Modified (w=0.8, d=1, r=False)', best_results['Modified (w=0.8, d=1, r=False)']),
        ('w=1.0 (Vanilla)', 'Vanilla MSG', best_results['Vanilla MSG']),
        ('w=1.0 (Modified)', 'Modified (w=1, d=1, r=False)', best_results['Modified (w=1, d=1, r=False)']),
        ('w=1.5', 'Modified (w=1.5, d=1, r=True)', best_results['Modified (w=1.5, d=1, r=True)'])
    ],
    'Depth Effect': [
        ('Baseline', 'Vanilla MSG', best_results['Vanilla MSG']),
        ('Depth=1', 'Modified (w=1, d=1, r=False)', best_results['Modified (w=1, d=1, r=False)'])
    ],
    'Residual Effect': [
        ('Without Residual', 'Modified (w=1, d=1, r=False)', best_results['Modified (w=1, d=1, r=False)']),
        ('With Residual', 'Modified (w=1, d=1, r=True)', best_results['Modified (w=1, d=1, r=True)'])
    ]
}

# Width effect
labels = [item[0] for item in config_groups['Width Effect']]
test_vals = [item[2]['test_acc'] * 100 for item in config_groups['Width Effect']]
class_vals = [item[2]['class_acc'] * 100 for item in config_groups['Width Effect']]
x = np.arange(len(labels))
width = 0.35
bars1 = axes[0, 0].bar(x - width/2, test_vals, width, label='Test Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = axes[0, 0].bar(x + width/2, class_vals, width, label='Class Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Width Parameter Effect', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Depth effect
labels = [item[0] for item in config_groups['Depth Effect']]
test_vals = [item[2]['test_acc'] * 100 for item in config_groups['Depth Effect']]
class_vals = [item[2]['class_acc'] * 100 for item in config_groups['Depth Effect']]
x = np.arange(len(labels))
bars1 = axes[0, 1].bar(x - width/2, test_vals, width, label='Test Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = axes[0, 1].bar(x + width/2, class_vals, width, label='Class Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Depth Parameter Effect', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(labels, fontsize=10)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Residual effect
labels = [item[0] for item in config_groups['Residual Effect']]
test_vals = [item[2]['test_acc'] * 100 for item in config_groups['Residual Effect']]
class_vals = [item[2]['class_acc'] * 100 for item in config_groups['Residual Effect']]
x = np.arange(len(labels))
bars1 = axes[1, 0].bar(x - width/2, test_vals, width, label='Test Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = axes[1, 0].bar(x + width/2, class_vals, width, label='Class Acc', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Residual Connection Effect', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(labels, fontsize=10)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Overall comparison
all_models = list(best_results.keys())
test_accs = [best_results[m]['test_acc'] * 100 for m in all_models]
model_colors = [models[m]['color'] for m in all_models]
bars = axes[1, 1].bar(range(len(all_models)), test_accs, color=model_colors, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 1].set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Overall Model Comparison', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(range(len(all_models)))
axes[1, 1].set_xticklabels(all_models, rotation=15, ha='right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, test_accs)):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'msg_configuration_impact.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_configuration_impact.png")
plt.close()

# ====================================================================================
# 6. Performance Summary Table
# ====================================================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Model Configuration', 'Best Test\nAcc (%)', 'Best Class\nAcc (%)', 
           'Final Test\nAcc (%)', 'Final Train\nAcc (%)', 'Overfit\nGap (%)', 'Epochs\nTrained']

for model_name in model_names:
    res = best_results[model_name]
    table_data.append([
        model_name,
        f"{res['test_acc']*100:.2f}",
        f"{res['class_acc']*100:.2f}",
        f"{res['final_test_acc']*100:.2f}",
        f"{res['final_train_acc']*100:.2f}",
        f"{(res['final_train_acc'] - res['final_test_acc'])*100:.2f}",
        f"{res['epochs_trained']}"
    ])

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style the header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style the data rows with alternating colors
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        
        # Highlight best values
        if j in [1, 2]:  # Best test and class accuracy columns
            val = float(table_data[i-1][j])
            col_vals = [float(row[j]) for row in table_data]
            if val == max(col_vals):
                table[(i, j)].set_facecolor('#90EE90')
                table[(i, j)].set_text_props(weight='bold')

plt.title('PointNet2 MSG Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'msg_performance_table.png', dpi=300, bbox_inches='tight')
print(f"Saved: msg_performance_table.png")
plt.close()

# ====================================================================================
# 7. Generate Text Summary Report
# ====================================================================================
report = []
report.append("=" * 80)
report.append("POINTNET2 MSG MODEL PERFORMANCE SUMMARY REPORT")
report.append("=" * 80)
report.append("")
report.append(f"Total Models Evaluated: {len(best_results)}")
report.append(f"Dataset: ModelNet40")
report.append("")

# Find best models
best_test_model = max(best_results.items(), key=lambda x: x[1]['test_acc'])
best_class_model = max(best_results.items(), key=lambda x: x[1]['class_acc'])

report.append("-" * 80)
report.append("TOP PERFORMING MODEL BY TEST INSTANCE ACCURACY:")
report.append("-" * 80)
report.append(f"Model: {best_test_model[0]}")
report.append(f"Configuration: {models[best_test_model[0]]['config']}")
report.append(f"Best Test Accuracy: {best_test_model[1]['test_acc']*100:.2f}%")
report.append(f"Best Class Accuracy: {best_test_model[1]['class_acc']*100:.2f}%")
report.append(f"Final Test Accuracy: {best_test_model[1]['final_test_acc']*100:.2f}%")
report.append(f"Epochs Trained: {best_test_model[1]['epochs_trained']}")
report.append("")

report.append("-" * 80)
report.append("TOP PERFORMING MODEL BY TEST CLASS ACCURACY:")
report.append("-" * 80)
report.append(f"Model: {best_class_model[0]}")
report.append(f"Configuration: {models[best_class_model[0]]['config']}")
report.append(f"Best Test Accuracy: {best_class_model[1]['test_acc']*100:.2f}%")
report.append(f"Best Class Accuracy: {best_class_model[1]['class_acc']*100:.2f}%")
report.append(f"Final Test Accuracy: {best_class_model[1]['final_test_acc']*100:.2f}%")
report.append(f"Epochs Trained: {best_class_model[1]['epochs_trained']}")
report.append("")

report.append("-" * 80)
report.append("ALL MODELS RANKED BY BEST TEST ACCURACY:")
report.append("-" * 80)
sorted_models = sorted(best_results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
for rank, (model_name, results) in enumerate(sorted_models, 1):
    report.append(f"{rank}. {model_name}")
    report.append(f"   Test Accuracy: {results['test_acc']*100:.2f}%")
    report.append(f"   Class Accuracy: {results['class_acc']*100:.2f}%")
    report.append(f"   Epochs: {results['epochs_trained']}")
    report.append("")

report.append("=" * 80)
report.append("KEY FINDINGS:")
report.append("=" * 80)

# Analyze configuration effects
vanilla_test = best_results['Vanilla MSG']['test_acc']
depth_test = best_results['Modified (w=1, d=1, r=False)']['test_acc']
width_15_test = best_results['Modified (w=1.5, d=1, r=True)']['test_acc']
width_08_test = best_results['Modified (w=0.8, d=1, r=False)']['test_acc']
residual_test = best_results['Modified (w=1, d=1, r=True)']['test_acc']

report.append(f"1. Depth Effect: Adding depth (d=1) to baseline:")
report.append(f"   Vanilla (d=0): {vanilla_test*100:.2f}%")
report.append(f"   Modified (w=1, d=1, r=False): {depth_test*100:.2f}%")
report.append(f"   Improvement: {(depth_test - vanilla_test)*100:.2f}%")
report.append("")
report.append(f"2. Width Effect (all with d=1):")
report.append(f"   w=0.8: {width_08_test*100:.2f}%")
report.append(f"   w=1.0: {depth_test*100:.2f}%")
report.append(f"   w=1.5: {width_15_test*100:.2f}%")
report.append(f"   Best width: 1.0 or 0.8")
report.append("")
report.append(f"3. Residual Effect (w=1, d=1):")
report.append(f"   Without residual: {depth_test*100:.2f}%")
report.append(f"   With residual: {residual_test*100:.2f}%")
report.append(f"   Difference: {(residual_test - depth_test)*100:.2f}%")
report.append("")

report.append("=" * 80)
report.append(f"All visualizations saved to '{output_dir}/' directory")
report.append("=" * 80)

# Save report
report_text = '\n'.join(report)
print("\n" + report_text)

with open(output_dir / 'msg_summary_report.txt', 'w') as f:
    f.write(report_text)

print(f"\nSummary report saved to '{output_dir}/msg_summary_report.txt'")
