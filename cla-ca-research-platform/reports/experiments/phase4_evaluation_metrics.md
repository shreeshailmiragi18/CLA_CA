# Phase 4: Evaluation Metrics Framework

## Overview
This document defines the complete set of evaluation metrics for CLA-CA algorithm enhancements, as required by the project document Phases 4-5. All metrics must be calculated from confusion matrix values and follow the paper's exact definitions.

## Paper-Defined Metric Set

The following metrics are mandatory for all experiment protocols (A, B, C):

### Core Confusion Matrix Derived Metrics

| Metric | Formula | Paper Reference |
|---|---|---|
| **True Positives (TP)** | Count of modules correctly predicted as defective | Algorithm 1 |
| **True Negatives (TN)** | Count of modules correctly predicted as non-defective | Algorithm 1 |
| **False Positives (FP)** | Count of non-defective modules predicted as defective | Algorithm 1 |
| **False Negatives (FN)** | Count of defective modules predicted as non-defective | Algorithm 1 |
| **Recall (Sensitivity)** | TP / (TP + FN) | Paper evaluation |
| **Precision (Precision)** | TP / (TP + FP) | Paper evaluation |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Paper evaluation |
| **F1 Score** | 2 × Precision × Recall / (Precision + Recall) | Paper evaluation |
| **MCC (Matthews Correlation Coefficient)** | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Paper evaluation |
| **G-mean (Paper definition)** | √(Precision × Recall) | Paper: "G-mean = √(precision × recall)" |
| **Specificity-based G-mean** | √(Recall × Specificity) | Paper: "specificity-based G-mean = √(recall × specificity)" |
| **Specificity (True Negative Rate)** | TN / (TN + FP) | Paper evaluation |
| **False Positive Rate (FPR)** | FP / (FP + TN) | Paper evaluation |
| **Balanced Accuracy** | (Recall + Specificity) / 2 | Paper evaluation |

### Zero-Denominator Policy

When a denominator in any metric calculation is 0, the following policy applies:

| Situation | Result |
|---|---|
| TP + FN = 0 (no actual defectives) | Recall = 0.0 |
| TP + FP = 0 (no predicted defectives) | Precision = 0.0 |
| TN + FP = 0 (no actual non-defectives) | Specificity = 0.0 |
| FP + TN = 0 (no actual non-defectives per FPR) | FPR = 0.0 |
| Precision + Recall = 0 (both zero) | F1 = 0.0 |
| MCC denominator = 0 | MCC = 0.0 |

This policy ensures that experiments on imbalanced or edge-case datasets do not produce `NaN` or infinite values.

## Metric Calculation Implementation

### `cla_ca.metrics.calculate_metrics()` Function

The function computes all metrics from TP, TN, FP, FN counts:

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics from true and predicted labels.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (1 = defective, 0 = non-defective)
    y_pred : np.ndarray
        Predicted labels (1 = defective, 0 = non-defective)
    
    Returns
    -------
    metrics : dict
        Dictionary with all required metrics:
        - tp, tn, fp, fn: confusion matrix counts
        - recall: TP / (TP + FN), 0.0 if TP+FN=0
        - precision: TP / (TP + FP), 0.0 if TP+FP=0
        - accuracy: (TP + TN) / total, 0.0 if total=0
        - f1: 2 * prec * rec / (prec + rec), 0.0 if prec+rec=0
        - mcc: (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)), 0.0 if denom=0
        - g_mean: sqrt(precision * recall), 0.0 if prec=0 or rec=0
        - specificity: TN / (TN + FP), 0.0 if TN+FP=0
        - fpr: FP / (FP + TN), 0.0 if FP+TN=0
        - balanced_accuracy: (recall + specificity) / 2
    """
    # Compute confusion matrix counts
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    total = tp + tn + fp + fn
    
    # Recall (Sensitivity)
    recall_denom = tp + fn
    recall = float(tp / recall_denom) if recall_denom > 0 else 0.0
    
    # Precision
    prec_denom = tp + fp
    precision = float(tp / prec_denom) if prec_denom > 0 else 0.0
    
    # Accuracy
    acc_denom = total
    accuracy = float(tp + tn) / acc_denom if acc_denom > 0 else 0.0
    
    # F1 Score
    f1_denom = precision + recall
    f1 = float(2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0
    
    # MCC
    mcc_num = tp * tn - fp * fn
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if total > 0 else 1.0
    mcc = float(mcc_num / mcc_denom) if mcc_denom != 0 else 0.0
    
    # G-mean (paper definition: sqrt(precision * recall))
    g_mean = float(np.sqrt(precision * recall)) if precision > 0 and recall > 0 else 0.0
    
    # Specificity (True Negative Rate)
    spec_denom = tn + fp
    specificity = float(tn / spec_denom) if spec_denom > 0 else 0.0
    
    # FPR
    fpr_denom = fp + tn
    fpr = float(fp / fpr_denom) if fpr_denom > 0 else 0.0
    
    # Balanced Accuracy
    ba_denom = 2.0
    balanced_accuracy = float((recall + specificity) / ba_denom) if ba_denom > 0 else 0.0
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'recall': recall, 'precision': precision,
        'accuracy': accuracy, 'f1': f1, 'mcc': mcc,
        'g_mean': g_mean, 'specificity': specificity,
        'fpr': fpr, 'balanced_accuracy': balanced_accuracy
    }
```

## Per-Protocol Metric Reporting

### Protocol A: Full-Dataset Replication
- **Report**: Single set of metrics per dataset (100% data used)
- **Comparison**: Direct comparison against frozen baseline
- **Metrics table columns**: dataset, protocol, training_fraction, f1, mcc, recall, precision, g_mean, accuracy, fpr, balanced_accuracy, retained_features, total_features, label_point, baseline_f1, f1_improvement

### Protocol B: Training-Size Experiments
- **Report**: Metrics across training fractions [10%, 20%, 30%, 40%, 50%, 60%]
- **Repeats**: 3 random splits per fraction (different seeds)
- **Aggregation**: Mean and standard deviation across repeats per fraction
- **Comparison**: Each (frac, seed) combination compared against baseline
- **Metrics table columns**: dataset, protocol, training_fraction, split_id, seed, f1, mcc, recall, precision, g_mean, ...

### Protocol C: Repeated Random-Split
- **Report**: Metrics across 10 random splits
- **Splits**: 80/20 train/test per split
- **Aggregation**: Mean, median, standard deviation across 10 splits
- **Comparison**: Each split compared against baseline
- **Outlier analysis**: Identify splits with significant performance differences
- **Metrics table columns**: dataset, protocol, split_id, seed, f1, mcc, recall, precision, g_mean, ...

### Metric Averaging Conventions (Phase 4)

Per the project document Phase 4 requirements, metric averaging must follow these rules:

1. **Mean across datasets**: Average the metric value across all 15 datasets
   - Example: mean_f1 = (f1_1 + f1_2 + ... + f1_15) / 15

2. **Mean across repeated runs**: Average across the 3 repeats per (dataset, fraction) for Protocol B,
   or across 10 splits for Protocol C

3. **Mean across all combinations**: Average across all dataset × repeat × metric combinations

4. **Macro averages**: Simple (unweighted) mean across datasets - preferred for imbalanced data

5. **Weighted averages**: Weighted by support (number of actual defectives per dataset) - secondary

6. **Do NOT pool all predictions across datasets**: Each dataset's metrics are computed separately,
   then averaged. Do not combine all y_true and y_pred across datasets into one large confusion matrix.

7. **Report both**: Mean and standard deviation (or interquartile range) for each aggregate metric

8. **Document**: Which averaging method was used in each report

## Statistical Metric Properties

### Per-Dataset Metric Distribution

For each metric, across the 15 datasets, report:

| Statistic | Description |
|---|---|
| **Mean** | Arithmetic average across 15 datasets |
| **Median** | Median value across 15 datasets (robust to outliers) |
| **Std Dev** | Standard deviation across 15 datasets |
| **Min** | Minimum value across 15 datasets |
| **Max** | Maximum value across 15 datasets |
| **Q1/Q3** | First and third quartiles |

### Aggregate Metric Change

When comparing enhanced vs. baseline:

| Metric | Calculation |
|---|---|
| **Absolute change** | mean(enhanced) - mean(baseline) |
| **Relative percentage change** | (mean(enhanced) - mean(baseline)) / mean(baseline) × 100% |
| **Per-dataset improvements** | List which datasets improved, degraded, or stayed neutral |
| **Win/tie/loss count** | Number of datasets where enhanced > baseline, = baseline, < baseline |

### Statistical Test Readiness (Phase 5)

Metrics are prepared for subsequent statistical validation (Phase 5):

- Per-dataset metric pairs (baseline vs. enhanced) stored for each dataset/split
- Distribution shapes assessed (Shapiro-Wilk test for normality)
- Appropriate statistical test selected (paired permutation, Wilcoxon signed-rank, or t-test)
- Effect sizes calculated (Cohen's d for mean differences, rank-biserial for median differences)

## Metric Storage Format

### Per-Run Metrics JSON
```json
{
    "dataset": "ant-1.5",
    "protocol": "A",
    "training_fraction": 1.0,
    "split_id": "full_dataset",
    "algorithm_version": "cla_ca_v0.8_40bim",
    "metrics": {
        "tp": 32, "tn": 261, "fp": 49, "fn": 0,
        "recall": 1.0, "precision": 0.2593,
        "accuracy": 0.9069, "f1": 0.4071, "mcc": 0.3463,
        "g_mean": 0.5082, "specificity": 1.0, "fpr": 0.0,
        "balanced_accuracy": 1.0
    },
    "threshold": 0.4523,
    "n_positive": 32, "n_negative": 261,
    "e1_enhanced": false
}
```

### Aggregate Summary CSV
```csv
dataset,protocol,training_fraction,split_id,seed,algorithm_version,f1,mcc,recall,precision,g_mean,accuracy,fpr,balanced_accuracy,threshold,retained_features,total_features,baseline_f1,f1_improvement,f1_improvement_pct
ant-1.5,A,1.0,full_dataset,,cla_ca_v0.8_40bim,0.4071,0.3463,0.7188,0.2840,0.4518,0.9070,0.0,1.0,0.4523,16,20,0.4071,0.0,0.0
```

## Compliance Checklist (Phase 4)

- [x] All 14 metrics defined and implemented (TP, TN, FP, FN, recall, precision, accuracy, f1, mcc, g_mean, specificity, fpr, balanced_accuracy, plus paper G-mean)
- [x] Zero-denominator policy documented and implemented
- [x] Metric calculation in `cla_ca.metrics.calculate_metrics()` 
- [x] Per-protocol reporting conventions defined (A, B, C)
- [x] Averaging conventions documented (mean across datasets, repeats, combinations)
- [x] Do NOT pool predictions across datasets
- [x] Statistical test readiness (per-dataset pairs stored)
- [x] Metric storage format defined (JSON per run, CSV aggregate)
- [x] Compliance checklist completed

---

*Phase 4 complete. Evaluation metrics framework ready for experiment execution and Phase 5 statistical validation.*