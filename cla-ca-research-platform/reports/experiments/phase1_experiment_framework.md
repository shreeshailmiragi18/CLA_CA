# Phase 1: Experiment Framework Design

## Overview
This document designs the fair experiment framework for CLA-CA algorithm enhancements (E0-E5) as specified in the project document. The framework enables controlled comparison between the frozen baseline and enhanced implementations.

## Experiment Interface Design

### Common Interface for All Algorithms
```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class CLAcaExperiment(ABC):
    """Abstract base class for CLA-CA experiment protocols."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'CLAcaExperiment':
        """Fit the algorithm on training data.
        
        For unsupervised CLA-CA: fitting should NOT receive labels 
        unless deliberately changed (supervised variant).
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray, optional
            Target labels. For unsupervised methods, pass None by default.
            
        Returns
        -------
        self : CLAcaExperiment
            Fitted experiment instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Sample matrix of shape (n_samples, n_features)
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (0 = non-defective, 1 = defective)
        """
        pass
    
    @abstractmethod
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly scores for samples in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Sample matrix of shape (n_samples, n_features)
            
        Returns
        -------
        scores : np.ndarray
            Anomaly scores (higher = more likely defective)
        """
        pass
    
    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """Name/identifier for this experiment configuration."""
        pass
    
    @property
    @abstractmethod
    def algorithm_version(self) -> str:
        """Version string identifying this algorithm/configuration."""
        pass
```

## Protocol Definitions

### Protocol A: Full-Dataset Reproduction
**Purpose**: Reproduce baseline results on entire dataset (no train/test split)

**Protocol Details**:
- **Training fraction**: 100% (all data used for both fitting and evaluation)
- **Testing**: Entire dataset used as test set
- **Random seed**: Not applicable (deterministic algorithm)
- **Split identifier**: "full_dataset"
- **Data usage**: 
  - Fit on entire dataset
  - Predict on entire dataset
  - Evaluate against true labels from entire dataset
- **Goal**: Verify that enhanced implementation matches or improves upon frozen baseline

**Implementation**:
```python
def run_protocol_A(dataset_name, algorithm_func, **algorithm_kwargs):
    """Run Protocol A: Full-dataset reproduction."""
    from cla_ca.pipeline import run_claca_pipeline
    
    # Run the algorithm on full dataset
    result = run_claca_pipeline(dataset_name, **algorithm_kwargs)
    
    # result contains: metrics, predictions, label_point, etc.
    # Compare against frozen baseline
    
    return {
        'dataset': dataset_name,
        'protocol': 'A',
        'training_fraction': 1.0,
        'split_id': 'full_dataset',
        'algorithm_version': algorithm_func.__name__,
        **result
    }
```

### Protocol B: Training-Size Experiments
**Purpose**: Evaluate performance across varying training fractions (10%-60%)

**Protocol Details**:
- **Training fractions**: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
- **For each fraction**: 
  - Randomly select `frac` of data for training (no shuffling of order, just subset)
  - Fit CLA-CA on training portion
  - Predict on remaining (1-frac) portion (test set)
  - Evaluate predictions against true test labels
- **Number of repeats**: 3 random splits per fraction (different seeds)
- **Random seeds**: Fixed seeds per fraction (e.g., 42, 123, 456)
- **Split identifier**: f"train_{frac}_seed_{seed}"
- **Data usage**:
  - Train on fraction `frac` of data
  - Test on remaining (1-frac) data
- **Goal**: Observe how performance varies with training set size

**Implementation**:
```python
def run_protocol_B(dataset_name, algorithm_func, **algorithm_kwargs):
    """Run Protocol B: Training-size experiments."""
    from cla_ca.pipeline import run_claca_pipeline
    import numpy as np
    
    fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    seeds = [42, 123, 456]  # 3 repeats per fraction
    results = []
    
    X_df, y_true, is_aeeem = load_dataset(dataset_name)
    n_samples = len(X_df)
    
    for frac in fractions:
        n_train = int(n_samples * frac)
        
        for seed in seeds:
            # Set random seed and select training indices
            np.random.seed(seed)
            train_indices = np.random.choice(n_samples, n_train, replace=False)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            
            X_train = X_df.iloc[train_indices]
            y_train = y_true[train_indices] if y_true is not None else None
            X_test = X_df.iloc[test_indices]
            y_test = y_true[test_indices] if y_true is not None else None
            
            # Fit and predict
            try:
                # Algorithm may or may not use y_train (unsupervised by default)
                algorithm_func.fit(X_train, y_train)
                scores = algorithm_func.predict_scores(X_test)
                y_pred = algorithm_func.predict(X_test)
                
                from cla_ca.metrics import calculate_metrics
                metrics = calculate_metrics(y_test, y_pred)
                
                results.append({
                    'dataset': dataset_name,
                    'protocol': 'B',
                    'training_fraction': frac,
                    'split_id': f"train_{frac}_seed_{seed}",
                    'algorithm_version': algorithm_func.__name__,
                    'seed': seed,
                    'train_size': n_train,
                    'test_size': len(test_indices),
                    **metrics
                })
            except Exception as e:
                results.append({
                    'dataset': dataset_name,
                    'protocol': 'B',
                    'training_fraction': frac,
                    'split_id': f"train_{frac}_seed_{seed}",
                    'algorithm_version': algorithm_func.__name__,
                    'seed': seed,
                    'error': str(e)
                })
    
    return results
```

### Protocol C: Repeated Random-Split
**Purpose**: Evaluate performance across multiple independent random splits

**Protocol Details**:
- **Number of splits**: 10 random splits
- **Train/test ratio**: 80/20 fixed (or configurable)
- **Each split**:
  - Randomly shuffle and split data 80/20
  - Fit on training portion
  - Predict on test portion
  - Evaluate
- **Random seeds**: 10 different seeds (0-9 or fixed set)
- **Split identifier**: f"split_{i}_seed_{seed}" where i=0..9
- **Data usage**:
  - Independent random split each iteration
  - No overlap between splits' training data
- **Goal**: Statistical analysis across multiple random initializations

**Implementation**:
```python
def run_protocol_C(dataset_name, algorithm_func, **algorithm_kwargs):
    """Run Protocol C: Repeated random-split."""
    from cla_ca.pipeline import run_claca_pipeline
    import numpy as np
    
    n_splits = 10
    test_fraction = 0.20
    seeds = list(range(n_splits))  # seeds 0-9
    results = []
    
    X_df, y_true, is_aeeem = load_dataset(dataset_name)
    n_samples = len(X_df)
    n_test = int(n_samples * test_fraction)
    
    for split_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        # Shuffle and split
        shuffled_indices = np.random.permutation(n_samples)
        train_indices = shuffled_indices[:-n_test]  # first 80%
        test_indices = shuffled_indices[-n_test:]  # last 20%
        
        X_train = X_df.iloc[train_indices]
        y_train = y_true[train_indices] if y_true is not None else None
        X_test = X_df.iloc[test_indices]
        y_test = y_true[test_indices] if y_true is not None else None
        
        # Fit and predict
        try:
            algorithm_func.fit(X_train, y_train)
            scores = algorithm_func.predict_scores(X_test)
            y_pred = algorithm_func.predict(X_test)
            
            from cla_ca.metrics import calculate_metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            results.append({
                'dataset': dataset_name,
                'protocol': 'C',
                'split_id': f"split_{split_idx}_seed_{seed}",
                'algorithm_version': algorithm_func.__name__,
                'seed': seed,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                **metrics
            })
        except Exception as e:
            results.append({
                'dataset': dataset_name,
                'protocol': 'C',
                'split_id': f"split_{split_idx}_seed_{seed}",
                'algorithm_version': algorithm_func.__name__,
                'seed': seed,
                'error': str(e)
            })
    
    return results
```

### Protocol D: CPDP (Concept Drift Detection Protocol)
**Purpose**: Evaluate stability under concept drift (if appropriate for CLA-CA)

**Protocol Details**:
- **Applicability**: CLA-CA is unsupervised, so concept drift may not directly apply
- **If enabled**: Simulate drift by incrementally introducing labeled data
- **Number of increments**: 5 drift steps
- **Each step**:
  - Corrupt a portion of features or labels
  - Re-evaluate algorithm performance
  - Measure performance degradation/recovery
- **Split identifier**: f"drift_step_{i}"
- **Goal**: Determine if enhanced algorithm is more robust to concept drift

**Decision**: Protocol D will be conditionally enabled only if the enhancement team determines CLA-CA benefits from concept drift testing. Given CLA-CA's unsupervised nature, this may be deprioritized unless specifically requested.

## Experiment Configuration Structure

### Configuration Files
Create `reports/experiments/` directory with configuration files:

```
reports/experiments/
├── protocol_a_config.json
├── protocol_b_config.json
├── protocol_c_config.json
└── experiment_interface.py
```

### Protocol A Configuration (`protocol_a_config.json`):
```json
{
  "protocol": "A",
  "description": "Full-dataset reproduction",
  "training_fraction": 1.0,
  "split_id": "full_dataset",
  "random_seed": null,
  "datasets": ["ant-1.5", "arc", "berek", "camel-1.0", "ivy-2.0", "jedit-3.2", 
               "synapse-1.1", "tomcat", "xalan-2.4", "xerces-1.2", "EQ", "JDT", "LC", "ML", "PDE"],
  "algorithm_defaults": {
    "bim": {"PROMISE": 40, "AEEEM": 155},
    "alpha": 0.8,
    "normalization": "min-max"
  },
  "compare_against": "frozen_baseline",
  "output_dir": "results/experiments/protocol_a"
}
```

### Protocol B Configuration (`protocol_b_config.json`):
```json
{
  "protocol": "B",
  "description": "Training-size experiments",
  "training_fractions": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
  "repeats_per_fraction": 3,
  "random_seeds": [42, 123, 456],
  "datasets": ["ant-1.5", "arc", "berek", "camel-1.0", "ivy-2.0", "jedit-3.2", 
               "synapse-1.1", "tomcat", "xalan-2.4", "xerces-1.2", "EQ", "JDT", "LC", "ML", "PDE"],
  "algorithm_defaults": {
    "bim": {"PROMISE": 40, "AEEEM": 155},
    "alpha": 0.8,
    "normalization": "min-max"
  },
  "compare_against": "frozen_baseline",
  "output_dir": "results/experiments/protocol_b"
}
```

### Protocol C Configuration (`protocol_c_config.json`):
```json
{
  "protocol": "C",
  "description": "Repeated random-split",
  "n_splits": 10,
  "test_fraction": 0.20,
  "random_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "datasets": ["ant-1.5", "arc", "berek", "camel-1.0", "ivy-2.0", "jedit-3.2", 
               "synapse-1.1", "tomcat", "xalan-2.4", "xerces-1.2", "EQ", "JDT", "LC", "ML", "PDE"],
  "algorithm_defaults": {
    "bim": {"PROMISE": 40, "AEEEM": 155},
    "alpha": 0.8,
    "normalization": "min-max"
  },
  "compare_against": "frozen_baseline",
  "output_dir": "results/experiments/protocol_c"
}
```

## Experiment Runner (Minimal Working Example)

### `run_experiment.py` — Protocol A Runner
```python
#!/usr/bin/env python3
"""
Minimal experiment runner for Protocol A (full-dataset reproduction).
Executes CLA-CA on all 15 datasets and produces results comparable to frozen baseline.
"""

import json
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cla_ca.pipeline import run_claca_pipeline
from cla_ca.metrics import calculate_metrics

# Protocol A configuration
PROTOCOL_A_CONFIG = {
    "protocol": "A",
    "description": "Full-dataset reproduction",
    "training_fraction": 1.0,
    "split_id": "full_dataset",
    "datasets": ["ant-1.5", "arc", "berek", "camel-1.0", "ivy-2.0", "jedit-3.2", 
                 "synapse-1.1", "tomcat", "xalan-2.4", "xerces-1.2", "EQ", "JDT", "LC", "ML", "PDE"],
    "algorithm_defaults": {
        "bim": {"PROMISE": 40, "AEEEM": 155},
        "alpha": 0.8,
        "normalization": "min-max"
    },
    "compare_against": "frozen_baseline",
    "output_dir": "results/experiments/protocol_a"
}

def run_protocol_A():
    """Run Protocol A on all datasets."""
    output_dir = PROTOCOL_A_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Load frozen baseline for comparison
    baseline_df = pd.read_csv('results/baseline/frozen/summary.csv', 
                              header=None, 
                              skiprows=1,
                              names=['Dataset', 'Recall', 'Precision', 'F1', 'MCC', 'G-mean', 'Retained Features'])
    
    results = []
    
    for ds in PROTOCOL_A_CONFIG["datasets"]:
        print(f"Running Protocol A on {ds}...")
        
        try:
            # Run CLA-CA pipeline (this is the "enhanced" or "baseline" algorithm)
            result = run_claca_pipeline(ds)
            
            # Compute metrics if not already present
            # (pipeline already returns metrics, but we standardize the output)
            experiment_result = {
                'dataset': ds,
                'protocol': PROTOCOL_A_CONFIG["protocol"],
                'training_fraction': PROTOCOL_A_CONFIG["training_fraction"],
                'split_id': PROTOCOL_A_CONFIG["split_id"],
                'algorithm_version': f"cla_ca_v{result.get('alpha', 0.8)}_{result.get('bim', 40)}bim",
                'f1': result['metrics']['f1'],
                'mcc': result['metrics']['mcc'],
                'recall': result['metrics']['recall'],
                'precision': result['metrics']['precision'],
                'g_mean': result['metrics']['g_mean'],
                'accuracy': result['metrics']['accuracy'],
                'fpr': result['metrics']['fpr'],
                'balanced_accuracy': result['metrics']['balanced_accuracy'],
                'retained_features': result['retained_features_count'],
                'total_features': result['total_features'],
                'label_point': result['label_point'],
                'bim': result['bim'],
                'alpha': result['alpha']
            }
            
            results.append(experiment_result)
            
            # Compare against frozen baseline
            baseline_row = baseline_df[baseline_df['Dataset'] == ds]
            if not baseline_row.empty:
                baseline_f1 = baseline_row['F1'].values[0]
                improvement = experiment_result['f1'] - baseline_f1
                experiment_result['baseline_f1'] = baseline_f1
                experiment_result['f1_improvement'] = improvement
                experiment_result['f1_improvement_pct'] = (improvement / baseline_f1) * 100 if baseline_f1 != 0 else 0
            
            print(f"  F1={experiment_result['f1']:.4f}", end="")
            if 'f1_improvement' in experiment_result:
                print(f" (baseline: {experiment_result['baseline_f1']:.4f}, improv: {experiment_result['f1_improvement_pct']:.1f}%)", end="")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'dataset': ds,
                'protocol': PROTOCOL_A_CONFIG["protocol"],
                'training_fraction': PROTOCOL_A_CONFIG["training_fraction"],
                'split_id': PROTOCOL_A_CONFIG["split_id"],
                'algorithm_version': 'ERROR',
                'error': str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    
    # Also save comparison summary
    comparison = results_df.merge(
        baseline_df, 
        left_on='dataset', 
        right_on='Dataset', 
        how='outer'
    )
    comparison.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)
    
    print(f"\nProtocol A complete. {len(results)} datasets processed.")
    print(f"Results saved to {output_dir}/")
    
    return results

if __name__ == "__main__":
    run_protocol_A()
```

### Experiment Metrics Standardization
All experiments must calculate and report these metrics (matching paper definitions):

| Metric | Formula | Paper Source |
|---|---|---|
| Recall | TP / (TP + FN) | Algorithm 1 |
| Precision | TP / (TP + FP) | Algorithm 1 |
| F1 | 2 * P * R / (P + R) | Paper evaluation |
| MCC | (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Paper evaluation |
| G-mean | sqrt(precision * recall) | Paper: "G-mean = sqrt(precision * recall)" |
| Specificity G-mean | sqrt(recall * specificity) | Paper: "specificity-based G-mean = sqrt(recall * specificity)" |
| FPR | FP / (FP + TN) | Paper evaluation |
| Balanced Accuracy | (recall + specificity) / 2 | Paper evaluation |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Paper evaluation |

**Zero-denominator policy**: If denominator is 0, return 0.0 for that metric (documented in metrics.py).

## Data Storage Structure

### Per-Run Predictions (stored per experiment)
```
results/experiments/protocol_a/
├── ant-1.5_predictions.csv
├── arc_predictions.csv
├── berek_predictions.csv
├── ...
└── summary.csv

results/experiments/protocol_b/
├── ant-1.5_f0.10_s42.csv
├── ant-1.5_f0.10_s123.csv
├── ant-1.5_f0.20_s42.csv
├── ...
└── summary.csv

results/experiments/protocol_c/
├── ant-1.5_split_0_seed_0.csv
├── ant-1.5_split_0_seed_1.csv
├── ant-1.5_split_1_seed_0.csv
├── ...
└── summary.csv
```

### Per-Run Prediction CSV Format:
```csv
dataset,split_id,algorithm_version,row_index,score,actual,predicted,status
ant-1.5,full_dataset,cla_ca_v0.8_40bim,0,0.4523,1,1,Defective
ant-1.5,full_dataset,cla_ca_v0.8_40bim,1,0.1234,0,0,Non-Defective
...
```

### Experiment Log/Manifest
Each protocol run should produce a manifest:
```json
{
  "run_id": "2026-10-24_Phase1_A_001",
  "protocol": "A",
  "timestamp": "2026-10-24T14:30:00",
  "datasets_run": 15,
  "datasets_succeeded": 15,
  "datasets_failed": 0,
  "baseline_comparison": {
    "ant-1.5": {"baseline_f1": 0.4071, "enhanced_f1": 0.4213, "improvement": 0.0142},
    "berek": {"baseline_f1": 0.8750, "enhanced_f1": 0.8910, "improvement": 0.0160},
    ...
  }
}
```

## Next Steps (Phase 1 Completion)

1. ✅ Design experiment interface (CLAcaExperiment abstract class)
2. ✅ Define protocols A-D with full details
3. ✅ Create configuration files (protocol_a/b/c_config.json)
4. ✅ Create minimal experiment runner (run_experiment.py Protocol A)
5. ✅ Standardize metric definitions and storage formats
6. Create Phase 1 summary report
7. Proceed to Phase 2: Implement enhancement E0 (original CLA-CA unchanged baseline)
8. Begin Phase 3: Implement E1 (adaptive thresholding)

---

*Phase 1 design complete. All protocol definitions, configurations, and runner ready for implementation.*