import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Define unique colors for each model (8 distinct colors with high contrast)
MODEL_COLORS = {
    'Baseline\n(d=0, w=1.0, r=False)': '#e74c3c',  # Bright Red
    'Deepen=1': '#ff8c00',  # Dark Orange
    'Residual=True': '#2ecc71',  # Bright Green
    'Widen=1.5': '#000000',  # Black
    'Deepen=1\n+ Residual': '#3498db',  # Sky Blue
    'Deepen=2\n+ Residual': '#9b59b6',  # Purple
    'Widen=1.5\n+ Deepen=1\n+ Residual': '#e91e63',  # Hot Pink
    'Widen=0.9': '#f4d03f',  # Golden Yellow (very distinct from orange)
}

# Define the training data from logs
training_data = {
    'Baseline\n(d=0, w=1.0, r=False)': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.507622, 0.675508, 0.725610, 0.752033, 0.786280, 0.801220, 0.812398, 0.823679, 0.826626, 0.832012],
        'test_acc': [0.597977, 0.571197, 0.732282, 0.714644, 0.719013, 0.750081, 0.656553, 0.718285, 0.791748, 0.679207],
        'test_class_acc': [0.480279, 0.478563, 0.631557, 0.627745, 0.658936, 0.696960, 0.579154, 0.659641, 0.768923, 0.603394],
        'best_test_acc': 0.791748,
        'best_class_acc': 0.768923,
        'total_time': 130.23,
        'deepen': 0, 'widen': 1.0, 'residual': False
    },
    'Deepen=1': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.508130, 0.645224, 0.703354, 0.727033, 0.759350, 0.774187, 0.788415, 0.801423, 0.804167, 0.811890],
        'test_acc': [0.541828, 0.578883, 0.731877, 0.655178, 0.615210, 0.786246, 0.680016, 0.812136, 0.732605, 0.855421],
        'test_class_acc': [0.437523, 0.507170, 0.645217, 0.578232, 0.540733, 0.716803, 0.607147, 0.761349, 0.654409, 0.794231],
        'best_test_acc': 0.855421,
        'best_class_acc': 0.794231,
        'total_time': 135.13,
        'deepen': 1, 'widen': 1.0, 'residual': False
    },
    'Residual=True': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.526016, 0.688720, 0.741667, 0.773476, 0.782419, 0.802947, 0.807012, 0.813923, 0.831504, 0.833333],
        'test_acc': [0.424757, 0.725081, 0.683333, 0.743042, 0.829288, 0.747654, 0.810194, 0.798382, 0.735761, 0.794903],
        'test_class_acc': [0.302341, 0.623230, 0.576277, 0.664915, 0.754539, 0.664807, 0.736435, 0.762348, 0.651674, 0.732218],
        'best_test_acc': 0.829288,
        'best_class_acc': 0.762348,
        'total_time': 136.19,
        'deepen': 0, 'widen': 1.0, 'residual': True
    },
    'Widen=1.5': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.536179, 0.675102, 0.727541, 0.755488, 0.772154, 0.791565, 0.793293, 0.807114, 0.811077, 0.825000],
        'test_acc': [0.450647, 0.559304, 0.717395, 0.508091, 0.689563, 0.715615, 0.627751, 0.746521, 0.696117, 0.693204],
        'test_class_acc': [0.376322, 0.480973, 0.616505, 0.449360, 0.612148, 0.651825, 0.552357, 0.709344, 0.621273, 0.638158],
        'best_test_acc': 0.746521,
        'best_class_acc': 0.709344,
        'total_time': 138.88,
        'deepen': 0, 'widen': 1.5, 'residual': False
    },
    'Deepen=1\n+ Residual': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.524898, 0.684146, 0.725305, 0.757012, 0.770427, 0.792378, 0.803354, 0.802642, 0.817785, 0.825915],
        'test_acc': [0.708333, 0.553317, 0.705178, 0.758091, 0.662379, 0.761246, 0.807524, 0.801294, 0.762460, 0.771359],
        'test_class_acc': [0.569440, 0.471743, 0.584987, 0.690816, 0.586699, 0.703124, 0.721609, 0.745448, 0.684080, 0.692729],
        'best_test_acc': 0.807524,
        'best_class_acc': 0.745448,
        'total_time': 138.39,
        'deepen': 1, 'widen': 1.0, 'residual': True
    },
    'Deepen=2\n+ Residual': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.522561, 0.667276, 0.721443, 0.751728, 0.767378, 0.782724, 0.799797, 0.811585, 0.822764, 0.824492],
        'test_acc': [0.641019, 0.597411, 0.650243, 0.512217, 0.529126, 0.697006, 0.721926, 0.611812, 0.677104, 0.634061],
        'test_class_acc': [0.527814, 0.515174, 0.554814, 0.478261, 0.460104, 0.637375, 0.704590, 0.552744, 0.632979, 0.597620],
        'best_test_acc': 0.721926,
        'best_class_acc': 0.704590,
        'total_time': 141.34,
        'deepen': 2, 'widen': 1.0, 'residual': True
    },
    'Widen=1.5\n+ Deepen=1\n+ Residual': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.536585, 0.672154, 0.723577, 0.757419, 0.775508, 0.787398, 0.800203, 0.796850, 0.818191, 0.816565],
        'test_acc': [0.498786, 0.624191, 0.589320, 0.555906, 0.725566, 0.616424, 0.461165, 0.333495, 0.552751, 0.592961],
        'test_class_acc': [0.364996, 0.533144, 0.521361, 0.472432, 0.646914, 0.513685, 0.411327, 0.280148, 0.536560, 0.497422],
        'best_test_acc': 0.725566,
        'best_class_acc': 0.646914,
        'total_time': 148.59,
        'deepen': 1, 'widen': 1.5, 'residual': True
    },
    'Widen=0.9': {
        'epochs': list(range(1, 11)),
        'train_acc': [0.515955, 0.670020, 0.723984, 0.750407, 0.773882, 0.794411, 0.813211, 0.822866, 0.824898, 0.826321],
        'test_acc': [0.586246, 0.664320, 0.695874, 0.770307, 0.767718, 0.777346, 0.819822, 0.793689, 0.750728, 0.782120],
        'test_class_acc': [0.460881, 0.600139, 0.608989, 0.649981, 0.696377, 0.708629, 0.759975, 0.750769, 0.694142, 0.709193],
        'best_test_acc': 0.819822,
        'best_class_acc': 0.759975,
        'total_time': 131.32,
        'deepen': 0, 'widen': 0.9, 'residual': False
    },
}

# Create output directory
output_dir = Path('visualization_results')
output_dir.mkdir(exist_ok=True)

# 1. Training and Test Accuracy Curves for All Models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for model_name, data in training_data.items():
    color = MODEL_COLORS[model_name]
    ax1.plot(data['epochs'], data['train_acc'], marker='o', label=model_name, linewidth=2, markersize=4, color=color)
    ax2.plot(data['epochs'], data['test_acc'], marker='s', label=model_name, linewidth=2, markersize=4, color=color)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Training Accuracy vs Epoch', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.4, 0.9])

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Test Instance Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Test Instance Accuracy vs Epoch', fontsize=14, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.2, 0.9])

plt.tight_layout()
plt.savefig(output_dir / 'training_test_accuracy_curves.png', dpi=300, bbox_inches='tight')
print("Saved: training_test_accuracy_curves.png")
plt.close()

# 2. Test Class Accuracy Curves
fig, ax = plt.subplots(figsize=(12, 6))

for model_name, data in training_data.items():
    color = MODEL_COLORS[model_name]
    ax.plot(data['epochs'], data['test_class_acc'], marker='D', label=model_name, linewidth=2, markersize=4, color=color)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Class Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Test Class Accuracy vs Epoch', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.2, 0.85])

plt.tight_layout()
plt.savefig(output_dir / 'test_class_accuracy_curves.png', dpi=300, bbox_inches='tight')
print("Saved: test_class_accuracy_curves.png")
plt.close()

# 3. Best Performance Comparison
model_names = list(training_data.keys())
best_test_accs = [data['best_test_acc'] for data in training_data.values()]
best_class_accs = [data['best_class_acc'] for data in training_data.values()]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width/2, [acc*100 for acc in best_test_accs], width, label='Best Test Instance Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, [acc*100 for acc in best_class_accs], width, label='Best Test Class Accuracy', alpha=0.8)

ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Best Performance Comparison Across Model Configurations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'best_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: best_performance_comparison.png")
plt.close()

# 4. Training Time Comparison
training_times = [data['total_time'] for data in training_data.values()]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(model_names, training_times, alpha=0.8, color=sns.color_palette("viridis", len(model_names)))

ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model Configuration', fontsize=12, fontweight='bold')
ax.set_title('Total Training Time Comparison (10 Epochs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, time) in enumerate(zip(bars, training_times)):
    ax.text(time + 1, i, f'{time:.1f} min', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: training_time_comparison.png")
plt.close()

# 5. Accuracy vs Training Time Scatter Plot
fig, ax = plt.subplots(figsize=(10, 7))

for model_name, data in training_data.items():
    color = MODEL_COLORS[model_name]
    ax.scatter(data['total_time'], data['best_test_acc']*100, 
               s=200, alpha=0.7, color=color, edgecolors='black', linewidth=1.5,
               label=model_name)

ax.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Best Test Instance Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Best Test Accuracy vs Training Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([65, 90])

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_vs_time.png', dpi=300, bbox_inches='tight')
print("Saved: accuracy_vs_time.png")
plt.close()

# 6. Performance Metrics Summary Table
summary_data = []
for model_name, data in training_data.items():
    summary_data.append({
        'Model': model_name.replace('\n', ' '),
        'Deepen': data['deepen'],
        'Widen': data['widen'],
        'Residual': data['residual'],
        'Best Test Acc (%)': f"{data['best_test_acc']*100:.2f}",
        'Best Class Acc (%)': f"{data['best_class_acc']*100:.2f}",
        'Training Time (min)': f"{data['total_time']:.2f}"
    })

df = pd.DataFrame(summary_data)

# Create a table figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                colWidths=[0.25, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('PointNet2 SSG Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
print("Saved: performance_summary_table.png")
plt.close()

# 7. Overfitting Analysis - Train vs Test Accuracy Gap
fig, ax = plt.subplots(figsize=(12, 6))

for model_name, data in training_data.items():
    color = MODEL_COLORS[model_name]
    train_test_gap = np.array(data['train_acc']) - np.array(data['test_acc'])
    ax.plot(data['epochs'], train_test_gap, marker='o', label=model_name, linewidth=2, markersize=4, color=color)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Training - Test Accuracy Gap', fontsize=12, fontweight='bold')
ax.set_title('Overfitting Analysis: Train-Test Accuracy Gap vs Epoch', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: overfitting_analysis.png")
plt.close()

# 8. Configuration Effects Heatmap
configs = []
for model_name, data in training_data.items():
    configs.append([
        data['deepen'],
        data['widen'],
        1 if data['residual'] else 0,
        data['best_test_acc'] * 100,
        data['best_class_acc'] * 100,
        data['total_time']
    ])

config_df = pd.DataFrame(configs, 
                         columns=['Deepen', 'Widen', 'Residual', 'Test Acc (%)', 'Class Acc (%)', 'Time (min)'],
                         index=[name.replace('\n', ' ') for name in model_names])

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Heatmap for Test Accuracy
config_pivot1 = config_df.pivot_table(values='Test Acc (%)', index='Deepen', columns='Widen', aggfunc='mean')
sns.heatmap(config_pivot1, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Test Acc (%)'})
axes[0].set_title('Avg Test Accuracy by Deepen & Widen', fontweight='bold')

# Heatmap for Class Accuracy
config_pivot2 = config_df.pivot_table(values='Class Acc (%)', index='Deepen', columns='Widen', aggfunc='mean')
sns.heatmap(config_pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Class Acc (%)'})
axes[1].set_title('Avg Class Accuracy by Deepen & Widen', fontweight='bold')

# Bar plot for Residual effect
residual_effect = config_df.groupby('Residual')[['Test Acc (%)', 'Class Acc (%)']].mean()
residual_effect.plot(kind='bar', ax=axes[2], alpha=0.8)
axes[2].set_title('Effect of Residual Connections', fontweight='bold')
axes[2].set_xticklabels(['No Residual', 'With Residual'], rotation=0)
axes[2].set_ylabel('Accuracy (%)')
axes[2].legend(['Test Acc', 'Class Acc'])
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'configuration_effects.png', dpi=300, bbox_inches='tight')
print("Saved: configuration_effects.png")
plt.close()

# 9. Final epoch performance comparison
final_train_accs = [data['train_acc'][-1] for data in training_data.values()]
final_test_accs = [data['test_acc'][-1] for data in training_data.values()]

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, [acc*100 for acc in final_train_accs], width, label='Final Train Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, [acc*100 for acc in final_test_accs], width, label='Final Test Accuracy', alpha=0.8)

ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Final Epoch (10) Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'final_epoch_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: final_epoch_comparison.png")
plt.close()

# 10. Create comprehensive summary report
print("\n" + "="*80)
print("POINTNET2 SSG MODEL PERFORMANCE SUMMARY REPORT")
print("="*80)
print(f"\nTotal Models Evaluated: {len(training_data)}")
print(f"Training Duration: 10 epochs each")
print(f"Dataset: ModelNet40")
print(f"Batch Size: 24")
print(f"Learning Rate: 0.001")
print(f"Optimizer: Adam")

print("\n" + "-"*80)
print("TOP 3 MODELS BY TEST INSTANCE ACCURACY:")
print("-"*80)
sorted_by_test = sorted(training_data.items(), key=lambda x: x[1]['best_test_acc'], reverse=True)
for i, (name, data) in enumerate(sorted_by_test[:3], 1):
    print(f"{i}. {name.replace(chr(10), ' ')}")
    print(f"   Test Accuracy: {data['best_test_acc']*100:.2f}%")
    print(f"   Class Accuracy: {data['best_class_acc']*100:.2f}%")
    print(f"   Training Time: {data['total_time']:.2f} minutes")
    print(f"   Config: deepen={data['deepen']}, widen={data['widen']}, residual={data['residual']}")
    print()

print("-"*80)
print("TOP 3 MODELS BY TEST CLASS ACCURACY:")
print("-"*80)
sorted_by_class = sorted(training_data.items(), key=lambda x: x[1]['best_class_acc'], reverse=True)
for i, (name, data) in enumerate(sorted_by_class[:3], 1):
    print(f"{i}. {name.replace(chr(10), ' ')}")
    print(f"   Class Accuracy: {data['best_class_acc']*100:.2f}%")
    print(f"   Test Accuracy: {data['best_test_acc']*100:.2f}%")
    print(f"   Training Time: {data['total_time']:.2f} minutes")
    print(f"   Config: deepen={data['deepen']}, widen={data['widen']}, residual={data['residual']}")
    print()

print("-"*80)
print("FASTEST TRAINING TIMES:")
print("-"*80)
sorted_by_time = sorted(training_data.items(), key=lambda x: x[1]['total_time'])
for i, (name, data) in enumerate(sorted_by_time[:3], 1):
    print(f"{i}. {name.replace(chr(10), ' ')}: {data['total_time']:.2f} minutes")

print("\n" + "="*80)
print("All visualizations saved to 'visualization_results/' directory")
print("="*80)

# Save summary to text file
with open(output_dir / 'summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("POINTNET2 SSG MODEL PERFORMANCE SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    for model_name, data in training_data.items():
        f.write(f"Model: {model_name.replace(chr(10), ' ')}\n")
        f.write(f"Configuration: deepen={data['deepen']}, widen={data['widen']}, residual={data['residual']}\n")
        f.write(f"Best Test Instance Accuracy: {data['best_test_acc']*100:.2f}%\n")
        f.write(f"Best Test Class Accuracy: {data['best_class_acc']*100:.2f}%\n")
        f.write(f"Total Training Time: {data['total_time']:.2f} minutes\n")
        f.write(f"Final Train Accuracy: {data['train_acc'][-1]*100:.2f}%\n")
        f.write(f"Final Test Accuracy: {data['test_acc'][-1]*100:.2f}%\n")
        f.write("-"*80 + "\n\n")

print("\nSummary report saved to 'visualization_results/summary_report.txt'")
