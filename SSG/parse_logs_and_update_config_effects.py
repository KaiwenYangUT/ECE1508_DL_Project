import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def parse_log_file(file_path):
    """Parse a log file and extract training metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract configuration from Namespace line
    namespace_match = re.search(r'Namespace\((.*?)\)', content, re.DOTALL)
    if namespace_match:
        namespace_str = namespace_match.group(1)
        deepen_match = re.search(r'deepen=(\d+)', namespace_str)
        widen_match = re.search(r'widen=([\d.]+)', namespace_str)
        residual_match = re.search(r'residual=(True|False)', namespace_str)
        
        deepen = int(deepen_match.group(1)) if deepen_match else 0
        widen = float(widen_match.group(1)) if widen_match else 1.0
        residual = residual_match.group(1) == 'True' if residual_match else False
    else:
        return None
    
    # Extract best accuracies
    best_test_acc = 0.0
    best_class_acc = 0.0
    
    # Find all test accuracy entries
    test_acc_matches = re.findall(r'Test Instance Accuracy: ([\d.]+), Class Accuracy: ([\d.]+)', content)
    if test_acc_matches:
        for test_acc, class_acc in test_acc_matches:
            test_acc_val = float(test_acc)
            class_acc_val = float(class_acc)
            if test_acc_val > best_test_acc:
                best_test_acc = test_acc_val
            if class_acc_val > best_class_acc:
                best_class_acc = class_acc_val
    
    return {
        'deepen': deepen,
        'widen': widen,
        'residual': residual,
        'best_test_acc': best_test_acc * 100,
        'best_class_acc': best_class_acc * 100
    }

def collect_all_training_data(train_log_dir):
    """Collect all training data from Train_Log directory."""
    data_list = []
    
    train_log_path = Path(train_log_dir)
    
    # Iterate through all subdirectories
    for subdir in train_log_path.iterdir():
        if subdir.is_dir():
            log_file = subdir / 'pointnet2_cls_ssg.txt'
            if log_file.exists():
                print(f"Parsing: {subdir.name}")
                result = parse_log_file(log_file)
                if result:
                    data_list.append(result)
                    print(f"  Deepen={result['deepen']}, Widen={result['widen']}, Residual={result['residual']}")
                    print(f"  Best Test Acc: {result['best_test_acc']:.2f}%, Best Class Acc: {result['best_class_acc']:.2f}%")
    
    return data_list

def create_configuration_effects_plot(data_list, output_path):
    """Create the configuration effects heatmap plot."""
    
    # Convert to DataFrame
    config_df = pd.DataFrame(data_list)
    
    print("\nDataFrame summary:")
    print(config_df.head(20))
    print(f"\nTotal configurations: {len(config_df)}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Heatmap for Test Accuracy
    config_pivot1 = config_df.pivot_table(values='best_test_acc', index='deepen', columns='widen', aggfunc='mean')
    sns.heatmap(config_pivot1, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Test Acc (%)'})
    axes[0].set_title('Avg Test Accuracy by Deepen & Widen', fontweight='bold')
    axes[0].set_xlabel('Widen')
    axes[0].set_ylabel('Deepen')
    
    # Heatmap for Class Accuracy
    config_pivot2 = config_df.pivot_table(values='best_class_acc', index='deepen', columns='widen', aggfunc='mean')
    sns.heatmap(config_pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Class Acc (%)'})
    axes[1].set_title('Avg Class Accuracy by Deepen & Widen', fontweight='bold')
    axes[1].set_xlabel('Widen')
    axes[1].set_ylabel('Deepen')
    
    # Bar plot for Residual effect
    config_df['Residual'] = config_df['residual'].astype(int)
    residual_effect = config_df.groupby('Residual')[['best_test_acc', 'best_class_acc']].mean()
    residual_effect.plot(kind='bar', ax=axes[2], alpha=0.8)
    axes[2].set_title('Effect of Residual Connections', fontweight='bold')
    axes[2].set_xticklabels(['No Residual', 'With Residual'], rotation=0)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].legend(['Test Acc', 'Class Acc'])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

if __name__ == '__main__':
    # Define paths
    train_log_dir = 'Train_Log'
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Collect all training data
    print("Collecting training data from all log files...")
    data_list = collect_all_training_data(train_log_dir)
    
    if not data_list:
        print("No data found!")
    else:
        # Create configuration effects plot
        output_path = output_dir / 'configuration_effects.png'
        create_configuration_effects_plot(data_list, output_path)
        
        print("\n" + "="*80)
        print("Configuration effects visualization updated successfully!")
        print("="*80)
