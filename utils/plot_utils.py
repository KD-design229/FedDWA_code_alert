import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np

def load_latest_log(log_dir):
    """
    Load the latest JSON log file from a directory
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        dict: Loaded JSON data, or None if no files found
    """
    log_files = glob.glob(os.path.join(log_dir, '*.json'))
    if not log_files:
        return None
    
    latest_file = max(log_files, key=os.path.getctime)
    print(f"Loading log from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_training_results(log_dir='./logs_feddwa', save_path='./feddwa_analysis_result.png'):
    """
    Plot training results from a single experiment
    
    Args:
        log_dir: Directory containing log files
        save_path: Path to save the output plot
    """
    data = load_latest_log(log_dir)
    if not data:
        print(f"No log files found in {log_dir}")
        return
    
    # Extract data
    test_acc = data.get('test_acc', [])
    train_loss = data.get('train_loss', [])
    weighted_mean_acc = data.get('test_weighted-mean_acc', [])
    global_acc = data.get('global_acc', [])
    
    rounds = range(1, len(weighted_mean_acc) + 1)
    
    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # 1. Global & Weighted Accuracy
    ax = axes[0]
    ax.plot(rounds, weighted_mean_acc, label='Weighted Mean Acc (Local)', 
            color='#2c3e50', linewidth=2, linestyle='--')
    if global_acc:
        ax.plot(rounds, global_acc, label='Global Model Acc (Full Test Set)', 
                color='#e74c3c', linewidth=3)
    
    ax.set_title('Global Model Performance', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Local Client Accuracy
    ax = axes[1]
    if test_acc:
        client_accs = list(zip(*test_acc))
        for i, accs in enumerate(client_accs):
            ax.plot(rounds, accs, label=f'Client {i}', alpha=0.6, linewidth=1.5)
            
    ax.set_title('Local Client Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    if test_acc and len(test_acc[0]) > 5:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    else:
        ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 3. Local Client Training Loss
    ax = axes[2]
    if train_loss:
        client_losses = list(zip(*train_loss))
        for i, losses in enumerate(client_losses):
            ax.plot(rounds, losses, label=f'Client {i}', alpha=0.6, linewidth=1.5)

    ax.set_title('Local Client Training Loss', fontsize=16, fontweight='bold')
    ax.set_xlabel('Communication Round', fontsize=14)
    ax.set_ylabel('Loss', fontsize=12)
    if train_loss and len(train_loss[0]) <= 5:
        ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {os.path.abspath(save_path)}")
    plt.close()


def plot_comparison(log_dirs, labels, save_path='./comparison_result.png', 
                   colors=None, title='Experiment Comparison'):
    """
    Plot comparison of multiple experiments (for ablation studies)
    
    Args:
        log_dirs: List of log directories to compare
        labels: List of labels for each experiment
        save_path: Path to save the comparison plot
        colors: Optional list of colors for each experiment
        title: Title for the comparison plot
    """
    if colors is None:
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Load all data
    all_data = []
    for log_dir in log_dirs:
        data = load_latest_log(log_dir)
        if data:
            all_data.append(data)
        else:
            print(f"Warning: No data found in {log_dir}")
            all_data.append(None)
    
    if not any(all_data):
        print("No valid data found for comparison")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Global Accuracy Curves
    ax = axes[0]
    for i, (data, label, color) in enumerate(zip(all_data, labels, colors)):
        if data and 'global_acc' in data:
            rounds = range(1, len(data['global_acc']) + 1)
            ax.plot(rounds, data['global_acc'], 
                   label=label, linewidth=2.5, color=color, 
                   marker='o', markersize=4, markevery=max(1, len(rounds)//10))
    
    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Global Accuracy', fontsize=12)
    ax.set_title('Global Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Final Accuracy Bar Chart
    ax = axes[1]
    final_accs = []
    valid_labels = []
    valid_colors = []
    
    for data, label, color in zip(all_data, labels, colors):
        if data and 'global_acc' in data and data['global_acc']:
            final_accs.append(data['global_acc'][-1])
            valid_labels.append(label)
            valid_colors.append(color)
    
    if final_accs:
        bars = ax.bar(valid_labels, final_accs, color=valid_colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax.set_ylabel('Final Global Accuracy', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate x-axis labels if too long
        if any(len(label) > 15 for label in valid_labels):
            ax.set_xticklabels(valid_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {os.path.abspath(save_path)}")
    plt.close()
    
    # Print numerical comparison
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for label, acc in zip(valid_labels, final_accs):
        print(f"{label:30s}: {acc:.4f} ({acc*100:.2f}%)")
    
    if len(final_accs) >= 2:
        baseline_acc = final_accs[0]
        for i, (label, acc) in enumerate(zip(valid_labels[1:], final_accs[1:]), 1):
            improvement = acc - baseline_acc
            relative_improvement = (acc / baseline_acc - 1) * 100
            print(f"\n{label} vs {valid_labels[0]}:")
            print(f"  Absolute improvement: +{improvement:.4f}")
            print(f"  Relative improvement: +{relative_improvement:.2f}%")
    print(f"{'='*60}\n")


def plot_ablation_study(baseline_dir, fedvls_dir, feddecorr_dir, full_dir,
                       save_path='./ablation_study.png'):
    """
    Plot ablation study comparing baseline, FedVLS only, FedDecorr only, and full GPR-FedSense
    
    Args:
        baseline_dir: Log directory for baseline experiment
        fedvls_dir: Log directory for FedVLS-only experiment
        feddecorr_dir: Log directory for FedDecorr-only experiment
        full_dir: Log directory for full GPR-FedSense experiment
        save_path: Path to save the ablation study plot
    """
    log_dirs = [baseline_dir, fedvls_dir, feddecorr_dir, full_dir]
    labels = ['Baseline (FedDWA)', 'FedDWA + FedVLS', 'FedDWA + FedDecorr', 'GPR-FedSense (Full)']
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
    
    plot_comparison(log_dirs, labels, save_path, colors, 
                   title='Ablation Study: GPR-FedSense Components')


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            # Compare multiple experiments
            # Usage: python plot_utils.py compare log_dir1 log_dir2 ... label1 label2 ...
            n_experiments = (len(sys.argv) - 2) // 2
            log_dirs = sys.argv[2:2+n_experiments]
            labels = sys.argv[2+n_experiments:]
            plot_comparison(log_dirs, labels)
        else:
            # Single experiment
            plot_training_results(sys.argv[1])
    else:
        # Default: plot from ./logs_feddwa
        plot_training_results()
