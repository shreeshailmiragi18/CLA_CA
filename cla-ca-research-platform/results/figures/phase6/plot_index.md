# Phase 6: Automated Visualization Index

Generated: 2026-10-24 13:22:25

## Overview
This directory contains all automated visualizations generated from E1 Adaptive Thresholding experiments.

## Plots

1. **F1 Score Heatmap**
   - File: `f1_heatmap.png`
   - Description: Heatmap of F1 scores across datasets and threshold methods

2. **F1 Improvement Distribution**
   - File: `f1_improvement_boxplot.png`
   - Description: Box plots showing distribution of F1 improvements vs baseline

3. **Win/Loss/Tie Analysis**
   - File: `win_loss_tie_analysis.png`
   - Description: Bar chart of wins/losses/ties per method

4. **Recall-Precision Trade-off**
   - File: `recall_precision_tradeoff.png`
   - Description: Scatter plot showing recall vs precision trade-offs

5. **F1 vs MCC Correlation**
   - File: `f1_vs_mcc.png`
   - Description: Scatter plot of F1 vs MCC correlation

6. **Per-Dataset Comparison**
   - File: `per_dataset_comparison.png`
   - Description: Bar charts for top 5 most variable datasets

7. **Threshold Distribution**
   - File: `threshold_distribution.png`
   - Description: Histogram of threshold values by method

8. **Ablation Metrics Summary**
   - File: `ablation_metrics_summary.png`
   - Description: Mean metrics comparison across all methods


## Key Findings

- Baseline method: Median of unique scores
- E1 methods tested: percentile_25, percentile_50, percentile_75, percentile_90, otsu, mean, weighted_percentile_50
- Otsu's method shows most promise with 6 wins, 6 losses, 3 ties
- Percentile methods generally underperform baseline
- Trade-off between recall and precision varies significantly by method

## Methodology
All plots follow consistent labeling, units, and legends per Phase 6 requirements.
PNG and PDF formats generated for each visualization.
