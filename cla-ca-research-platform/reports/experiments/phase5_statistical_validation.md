# Phase 5: Statistical Validation Framework

## Overview
This document defines the statistical validation methods for comparing baseline CLA-CA against enhanced implementations (E1-E5), as required by the project document Phases 5-10. Proper statistical testing is essential for claiming that any observed improvement is meaningful and not due to chance.

## Experimental Design Principles

### Unit of Analysis
- **Each dataset represents one observational unit**
- **Random seeds/repeats within a dataset are not fully independent**
- **Statistical tests must account for the paired design** (baseline vs. enhanced on the same dataset/split)

### Comparison Protocol
- Baseline and enhanced implementations are evaluated on **identical dataset/split combinations**
- For Protocol A: Same full dataset, different thresholding → paired by dataset
- For Protocol B: Same training fraction/split, different thresholding → paired by (dataset, fraction, seed)
- For Protocol C: Same random split, different thresholding → paired by (dataset, split)

### Hypothesis Testing Framework

| Hypothesis | Description |
|---|---|
| **H0 (null)** | No difference between baseline and enhanced metric values |
| **H1 (alternative)** | Enhanced implementation differs from baseline (two-sided) or improves (one-sided) |

### Statistical Test Selection

| Design | Recommended Test | When to Use |
|---|---|---|
| **Protocol A** (1 dataset, multiple methods) | Wilcoxon signed-rank test | Non-parametric, paired by dataset |
| **Protocol B** (multiple fractions × repeats) | Friedman test + Nemenyi post-hoc | Non-parametric, multiple related samples |
| **Protocol C** (multiple random splits) | Wilcoxon signed-rank test | Paired per split, non-parametric |
| **Large n (≈30+)** | paired t-test | If normality confirmed via Shapiro-Wilk |
| **Small n (<30)** | Wilcoxon signed-rank test | Non-parametric, recommended default |
| **Very small n (<10)** | Exact permutation test | Enumeration of all possible permutations |

### Effect Size Measures

| Metric | Formula | Interpretation |
|---|---|---|
| **Cohen's d** | (mean_diff) / std_diff | Small=0.2, Medium=0.5, Large=0.8 |
| **Rank-biserial r** | (W - n(n+1)/4) / (n(n+1)/4) | -1 to 1, absolute >0.5 large |
| **Pearson r** | t / √(t² + df) | From t-test statistic |
| **Percentage improvement** | (enhanced - baseline) / baseline × 100% | Relative change |

### Multiple Comparisons Correction

When testing multiple enhancements (E1 through E5) or multiple metrics:

| Correction Method | When to Use | Adjusted p-value |
|---|---|---|
| **Bonferroni** | Conservative, few comparisons (≤10) | α_adj = α / m |
| **Holm-Bonferroni** | uniformly more powerful than Bonferroni | Step-down procedure |
| **Benjamini-Hochberg (FDR)** | Many comparisons, want to control false discoveries | Rank-based threshold |
| **Benferroni-Holm** | Recommended default for ≤20 comparisons | Step-down FDR |

**Recommendation**: Use Benjamini-Hochberg FDR control when reporting enhancements E1-E5 across multiple metrics (F1, MCC, Recall). Do not apply overly conservative corrections that mask real but modest improvements.

### Non-Parametric Tests Details

#### Wilcoxon Signed-Rank Test
- **Assumptions**: Paired observations, symmetric distribution of differences (about median)
- **Test statistic W**: Sum of ranks of positive differences
- **p-value**: Based on exact distribution (for n ≤ 20) or normal approximation (n > 20)
- **Effect size r**: r = Z / √n, where Z is the normal approximation of W

#### Friedman Test (Protocol B)
- **Assumptions**: Repeated measurements on same subjects, ordinal or continuous
- **Test statistic χ²_F**: Based on ranks within each block (dataset × fraction)
- **Post-hoc**: Nemenyi test or pairwise Wilcoxon signed-rank with correction
- **Multiple comparisons**: Apply Holm or BH correction

#### Exact Permutation Test
- **Procedure**: 
  1. Compute observed test statistic (mean difference) for paired data
  2. Randomly flip signs of differences (baseline ↑ or enhanced ↑)
  3. Repeat B = 1000+ times to build null distribution
  4. p-value = proportion of permuted statistics ≥ observed statistic
- **Advantages**: No distributional assumptions, exact for any n
- **Recommendation**: Default for Phase 5 when n_datasets ≤ 30

## Phase 5 Analysis Workflow

### Step 1: Data Collection
Store per-dataset metric pairs for each protocol:

| dataset | protocol | training_fraction | split_id | seed | metric | baseline_value | enhanced_value | difference |
|---|---|---|---|---|---|---|---|---|
| ant-1.5 | A | 1.0 | full_dataset | - | f1 | 0.4071 | 0.4213 | +0.0142 |
| ant-1.5 | B | 0.10 | train_0.10_seed_42 | 42 | f1 | 0.1523 | 0.1587 | +0.0064 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Step 2: Distribution Assessment
- Shapiro-Wilk test on differences (baseline - enhanced)
- Visual: Q-Q plot, histogram of differences
- If p > 0.05: differences appear normal → consider parametric test
- If p ≤ 0.05: differences non-normal → use non-parametric test

### Step 3: Test Selection
Based on distribution and sample size:

| Condition | Test |
|---|---|
| n ≥ 30, differences normal | paired t-test |
| n < 30, differences normal | paired t-test (with caution) |
| n < 30, differences non-normal | Wilcoxon signed-rank test |
| Multiple comparisons (E1-E5) | Wilcoxon + BH correction |
| Protocol B (multiple fractions) | Friedman + Nemenyi |

### Step 4: Compute Test Statistic and p-value

#### Wilcoxon Signed-Rank Test (detailed)
```
Differences: d_i = enhanced_i - baseline_i for i = 1, ..., n
Absolute differences: |d_i|
Rank absolute differences (ignoring zeros): R_1, ..., R_k where k = number of non-zero differences
W⁺ = sum of ranks where d_i > 0
W⁻ = sum of ranks where d_i < 0

W = min(W⁺, W⁻)  (for two-sided test)

For n ≤ 20: use exact Wilcoxon distribution
For n > 20: normal approximation
  Z = (W - n(n+1)/4) / √(n(n+1)(2n+1)/24)
  with continuity correction: Z = (W - 0.5 - n(n+1)/4) / √(n(n+1)(2n+1)/24)
  
p-value = 2 × P(Z > |z|)  (two-sided)
```

#### Effect Size (Rank-Biserial r)
```
r = (W⁺ - W⁻) / (n(n+1)/2)  ? Actually:
r = Z / √n  (from the normal approximation Z above)
```

### Step 5: Confidence Intervals

#### Difference Mean CI (95%)
```
mean_diff ± t_{0.025, n-1} × std_diff / √n
```

#### Wilcoxon CI (non-parametric)
- Use bootstrapping: resample paired differences B=1000 times
- Compute percentile interval (2.5th, 97.5th)

#### Relative Percentage Change CI
```
%_change = (mean_diff / mean_baseline) × 100
CI: mean_diff CI transformed to percentage
```

### Step 6: Multiple Comparisons Correction (if testing E1-E5)

| Method | Adjusted p-value Formula |
|---|---|
| **Bonferroni** | p_adj = min(p × m, 1) |
| **Holm-Bonferroni** | Step-down: order p_(1) ≤ p_(2) ≤ ... ≤ p_(m); p_adj(i) = min(p_(i) × (m - i + 1), 1, p_adj(i+1)) |
| **Benjamini-Hochberg (FDR)** | 1. Order p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m) 2. Find largest k such that p_(k) ≤ (k/m) × q 3. p_adj(i) = min_(j≥i) (m × p_(j) / j), capped at 1 |

Where m = number of comparisons, q = FDR level (typically 0.05).

### Step 7: Report Results

For each enhancement E1-E5, report in this order:

1. **Per-dataset results table**: dataset, baseline, enhanced, absolute change, relative change, direction (improve/neutral/degrade)
2. **Aggregate statistics**: mean difference, std dev, median difference, win/tie/loss count
3. **Statistical test**: test name, test statistic, p-value (raw and corrected)
4. **Effect size**: Cohen's d or rank-biserial r, with interpretation
5. **95% CI**: of mean difference (or bootstrap CI for non-parametric)
6. **Limitations**: small n, non-normal distributions, multiple comparisons
7. **Conclusion**: reject/fail to reject H0, practical significance

## Phase 5 Checklist

- [ ] Per-dataset metric pairs collected for all protocols A/B/C
- [ ] Distribution normality assessed (Shapiro-Wilk)
- [ ] Appropriate statistical test selected
- [ ] Test statistics and p-values computed
- [ ] Effect sizes calculated
- [ ] 95% confidence intervals computed
- [ ] Multiple comparisons corrected (BH FDR recommended)
- [ ] Win/tie/loss counts documented
- [ ] Limitations noted (small n=15, non-normal distributions)
- [ ] Report formatted per Phase 5 requirements

## Phase 5 Python Implementation Skeleton

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def statistical_analysis(baseline_df: pd.DataFrame, 
                         enhanced_df: pd.DataFrame,
                         metrics: list = None,
                         alpha: float = 0.05) -> dict:
    """
    Perform statistical comparison of baseline vs enhanced.
    
    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline results with columns: dataset, metric, value
    enhanced_df : pd.DataFrame
        Enhanced results with columns: dataset, metric, value
    metrics : list, optional
        List of metrics to test. If None, test all.
    alpha : float
        Significance level (default 0.05)
    
    Returns
    -------
    results : dict
        Statistical test results
    """
    if metrics is None:
        metrics = ['f1', 'mcc', 'recall', 'precision', 'g_mean']
    
    results = {}
    
    for metric in metrics:
        # Merge baseline and enhanced by dataset
        merged = pd.merge(
            baseline_df[['dataset', 'value']],
            enhanced_df[['dataset', 'value']],
            on='dataset',
            suffixes=('_base', '_enhanced')
        )
        
        d = merged['value_enhanced'] - merged['value_base']
        
        # Descriptive stats
        mean_diff = float(d.mean())
        std_diff = float(d.std())
        median_diff = float(d.median())
        
        # Win/tie/loss
        wins = (d > 0).sum()
        losses = (d < 0).sum()
        ties = (d == 0).sum()
        n = len(d)
        
        # Normality test
        _, shapiro_p = stats.shapiro(d)
        normal = shapiro_p > 0.05
        
        # Test selection and execution
        if normal and n >= 30:
            # paired t-test
            t_stat, p_val = stats.ttest_rel(merged['value_enhanced'], 
                                           merged['value_base'])
            # Cohen's d
            d_cohen = mean_diff / std_diff if std_diff > 0 else 0.0
            test_name = 'paired t-test'
        else:
            # Wilcoxon signed-rank test
            w_stat, p_val = stats.wilcoxon(merged['value_enhanced'], 
                                          merged['value_base'])
            # Rank-biserial r
            z = stats.norm.ppf(1 - p_val / 2) if p_val > 0 else 0
            r_bb = z / np.sqrt(n) if n > 0 else 0.0
            test_name = 'Wilcoxon signed-rank test'
        
        # Confidence interval for mean difference
        if normal and n >= 3:
            se = std_diff / np.sqrt(n)
            h = stats.t.ppf(0.975, df=n-1)  # 95% CI
            ci_lower = mean_diff - h * se
            ci_upper = mean_diff + h * se
        else:
            # Bootstrap CI
            n_boot = 1000
            boot_diffs = []
            rng = np.random.RandomState(42)
            for _ in range(n_boot):
                sample = rng.choice(d, size=len(d), replace=True)
                boot_diffs.append(sample.mean())
            ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
        
        # Multiple comparisons correction (BH FDR across metrics)
        # Will be applied after all metrics processed
        
        results[metric] = {
            'test': test_name,
            'p_raw': float(p_val),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'median_diff': median_diff,
            'wins': int(wins),
            'losses': int(losses),
            'ties': int(ties),
            'n_datasets': n,
            'normal_distribution': normal,
            'cohens_d': float(d_cohen) if 'd_cohen' in dir() else float('nan'),
            'rank_biserial_r': float(r_bb) if 'r_bb' in dir() else float('nan'),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
        }
    
    # Apply Benjamini-Hochberg FDR correction across metrics
    p_values = [results[m]['p_raw'] for m in results]
    _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    for i, metric in enumerate(results):
        results[metric]['p_fdr'] = float(p_adj[i])
    
    return results
```