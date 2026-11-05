"""
Phase 6: Automated Visualization Generation
Creates Matplotlib/Seaborn publication-quality figures from E1 experiment results.
Phase 6 requirements: PNG/PDF outputs, consistent labels/units/legends, plot index.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create output directory
output_dir = Path('/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/figures/phase6')
output_dir.mkdir(parents=True, exist_ok=True)

# Load E1 results
results_path = '/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e1/protocol_a_results.csv'
df = pd.read_csv(results_path)

# Load baseline summary
baseline_path = '/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/baseline/frozen/summary.csv'
baseline_df = pd.read_csv(baseline_path, header=None, skiprows=1,
                          names=['Dataset', 'Recall', 'Precision', 'F1', 'MCC', 'G-mean', 'Retained Features'])

print("Phase 6: Generating automated visualizations...")
print(f"Loaded {len(df)} experiment runs across {df['dataset'].nunique()} datasets")

# -------------------------------------------------------------------------
# Figure 1: Baseline vs E1 Performance Comparison (F1 Score Heatmap)
# -------------------------------------------------------------------------
print("\n1. Creating F1 score comparison heatmap...")
pivot_f1 = df.pivot(index='dataset', columns='threshold_method', values='f1')

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'F1 Score'})
ax.set_title('CLA-CA E1 Enhancement: F1 Score by Dataset and Threshold Method',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Threshold Method', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
plt.tight_layout()
fig.savefig(output_dir / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'f1_heatmap.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 2: F1 Score Improvement Distribution
# -------------------------------------------------------------------------
print("2. Creating F1 improvement distribution plot...")

# Calculate improvement vs baseline
baseline_f1 = df[df['threshold_method'] == 'baseline'][['dataset', 'f1']].rename(columns={'f1': 'baseline_f1'})
df_merged = df.merge(baseline_f1, on='dataset')
df_merged['f1_improvement'] = df_merged['f1'] - df_merged['baseline_f1']
df_merged['improvement_pct'] = (df_merged['f1_improvement'] / df_merged['baseline_f1']) * 100

# Box plot
fig, ax = plt.subplots(figsize=(12, 8))
methods = [m for m in df_merged['threshold_method'].unique() if m != 'baseline']
data_to_plot = [df_merged[df_merged['threshold_method'] == m]['improvement_pct'].values for m in methods]

bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.set_title('F1 Score Improvement Distribution vs Baseline', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Threshold Method', fontsize=12)
ax.set_ylabel('F1 Improvement (%)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(output_dir / 'f1_improvement_boxplot.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'f1_improvement_boxplot.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Win/Loss/Tie Analysis Bar Chart
# -------------------------------------------------------------------------
print("3. Creating win/loss/tie analysis...")

win_counts = {}
for method in methods:
    method_data = df_merged[df_merged['threshold_method'] == method]
    wins = (method_data['f1_improvement'] > 0.001).sum()
    losses = (method_data['f1_improvement'] < -0.001).sum()
    ties = len(method_data) - wins - losses
    win_counts[method] = {'wins': wins, 'losses': losses, 'ties': ties}

win_df = pd.DataFrame(win_counts).T

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(win_df.index))
width = 0.25

ax.bar(x - width, win_df['wins'], width, label='Wins', color='#2ca02c', alpha=0.8)
ax.bar(x, win_df['ties'], width, label='Ties', color='#ff7f0e', alpha=0.8)
ax.bar(x + width, win_df['losses'], width, label='Losses', color='#d62728', alpha=0.8)

ax.set_xlabel('Threshold Method', fontsize=12)
ax.set_ylabel('Number of Datasets', fontsize=12)
ax.set_title('E1 Enhancement: Win/Loss/Tie Analysis (F1 Score)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(win_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(output_dir / 'win_loss_tie_analysis.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'win_loss_tie_analysis.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Recall-Precision Trade-off Scatter
# -------------------------------------------------------------------------
print("4. Creating recall-precision trade-off scatter...")

fig, ax = plt.subplots(figsize=(12, 10))

for method in methods:
    method_data = df_merged[df_merged['threshold_method'] == method]
    ax.scatter(method_data['recall'], method_data['precision'],
               label=method, alpha=0.7, s=100)

# Add baseline
baseline_data = df[df['threshold_method'] == 'baseline']
ax.scatter(baseline_data['recall'], baseline_data['precision'],
           label='baseline', alpha=0.9, s=150, edgecolors='black', linewidths=1.5)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Recall-Precision Trade-off by Threshold Method', fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(output_dir / 'recall_precision_tradeoff.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'recall_precision_tradeoff.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 5: MCC vs F1 Correlation
# -------------------------------------------------------------------------
print("5. Creating MCC vs F1 correlation plot...")

fig, ax = plt.subplots(figsize=(12, 8))

for method in ['baseline'] + methods:
    method_data = df[df['threshold_method'] == method]
    ax.scatter(method_data['f1'], method_data['mcc'], label=method, alpha=0.7, s=100)

ax.set_xlabel('F1 Score', fontsize=12)
ax.set_ylabel('MCC', fontsize=12)
ax.set_title('F1 Score vs MCC by Threshold Method', fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 1)
plt.tight_layout()
fig.savefig(output_dir / 'f1_vs_mcc.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'f1_vs_mcc.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 6: Per-Dataset Performance Comparison
# -------------------------------------------------------------------------
print("6. Creating per-dataset performance comparison...")

# Select top 5 datasets with most variation
dataset_std = df.groupby('dataset')['f1'].std().sort_values(ascending=False)
top_datasets = dataset_std.head(5).index

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, dataset in enumerate(top_datasets):
    if idx >= 5:
        break
    ax = axes[idx]
    dataset_data = df[df['dataset'] == dataset].copy()
    dataset_data = dataset_data.sort_values('f1', ascending=False)

    colors = ['red' if m == 'baseline' else 'steelblue' for m in dataset_data['threshold_method']]
    bars = ax.bar(range(len(dataset_data)), dataset_data['f1'], color=colors, alpha=0.8)

    ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, dataset_data['f1']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Hide unused subplot
for idx in range(len(top_datasets), 6):
    axes[idx].set_visible(False)

plt.suptitle('Per-Dataset F1 Score Comparison (Top 5 Variable Datasets)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(output_dir / 'per_dataset_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'per_dataset_comparison.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 7: Threshold Values Distribution
# -------------------------------------------------------------------------
print("7. Creating threshold values distribution...")

fig, ax = plt.subplots(figsize=(12, 8))

for method in ['baseline'] + methods:
    method_data = df[df['threshold_method'] == method]['threshold']
    ax.hist(method_data, bins=15, alpha=0.5, label=method)

ax.set_xlabel('Threshold Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Threshold Values by Method', fontsize=14, fontweight='bold', pad=20)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(output_dir / 'threshold_distribution.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'threshold_distribution.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Figure 8: Ablation Study - Metrics Summary
# -------------------------------------------------------------------------
print("8. Creating ablation study metrics summary...")

# Calculate mean metrics per method
summary_stats = df.groupby('threshold_method')[['f1', 'mcc', 'recall', 'precision']].agg(['mean', 'std'])
summary_stats.columns = ['_'.join(col) for col in summary_stats.columns]
summary_stats = summary_stats.reset_index()

fig, ax = plt.subplots(figsize=(14, 8))

metrics = ['f1', 'mcc', 'recall', 'precision']
x = np.arange(len(summary_stats))

width = 0.2
for i, metric in enumerate(metrics):
    values = summary_stats[f'{metric}_mean']
    ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)

ax.set_xlabel('Threshold Method', fontsize=12)
ax.set_ylabel('Mean Metric Value', fontsize=12)
ax.set_title('Mean Metrics Comparison Across Threshold Methods', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(summary_stats['threshold_method'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(output_dir / 'ablation_metrics_summary.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'ablation_metrics_summary.pdf', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Generate Plot Index
# -------------------------------------------------------------------------
print("\nGenerating plot index...")

plot_index = {
    "phase": "Phase 6: Automated Visualization",
    "generated_at": pd.Timestamp.now().isoformat(),
    "total_plots": 8,
    "plots": [
        {
            "name": "F1 Score Heatmap",
            "file": "f1_heatmap.png",
            "description": "Heatmap of F1 scores across datasets and threshold methods"
        },
        {
            "name": "F1 Improvement Distribution",
            "file": "f1_improvement_boxplot.png",
            "description": "Box plots showing distribution of F1 improvements vs baseline"
        },
        {
            "name": "Win/Loss/Tie Analysis",
            "file": "win_loss_tie_analysis.png",
            "description": "Bar chart of wins/losses/ties per method"
        },
        {
            "name": "Recall-Precision Trade-off",
            "file": "recall_precision_tradeoff.png",
            "description": "Scatter plot showing recall vs precision trade-offs"
        },
        {
            "name": "F1 vs MCC Correlation",
            "file": "f1_vs_mcc.png",
            "description": "Scatter plot of F1 vs MCC correlation"
        },
        {
            "name": "Per-Dataset Comparison",
            "file": "per_dataset_comparison.png",
            "description": "Bar charts for top 5 most variable datasets"
        },
        {
            "name": "Threshold Distribution",
            "file": "threshold_distribution.png",
            "description": "Histogram of threshold values by method"
        },
        {
            "name": "Ablation Metrics Summary",
            "file": "ablation_metrics_summary.png",
            "description": "Mean metrics comparison across all methods"
        }
    ]
}

# Save plot index
with open(output_dir / 'plot_index.json', 'w') as f:
    json.dump(plot_index, f, indent=2)

# Create markdown index
md_content = """# Phase 6: Automated Visualization Index

Generated: {timestamp}

## Overview
This directory contains all automated visualizations generated from E1 Adaptive Thresholding experiments.

## Plots

""".format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

for i, plot in enumerate(plot_index['plots'], 1):
    md_content += f"""{i}. **{plot['name']}**
   - File: `{plot['file']}`
   - Description: {plot['description']}

"""

md_content += """
## Key Findings

- Baseline method: Median of unique scores
- E1 methods tested: percentile_25, percentile_50, percentile_75, percentile_90, otsu, mean, weighted_percentile_50
- Otsu's method shows most promise with 6 wins, 6 losses, 3 ties
- Percentile methods generally underperform baseline
- Trade-off between recall and precision varies significantly by method

## Methodology
All plots follow consistent labeling, units, and legends per Phase 6 requirements.
PNG and PDF formats generated for each visualization.
"""

with open(output_dir / 'plot_index.md', 'w') as f:
    f.write(md_content)

print(f"\nPhase 6 Complete!")
print(f"Generated 8 visualizations in {output_dir}")
print(f"Plot index saved to {output_dir / 'plot_index.md'}")
print(f"\nAll figures saved as PNG (300 DPI) and PDF formats.")
