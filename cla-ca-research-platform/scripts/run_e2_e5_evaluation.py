#!/usr/bin/env python3
"""
Comprehensive evaluation of E2-E5 enhancements for CLA-CA algorithm.
Runs all enhancements on all 15 datasets with Protocol A (full dataset).
Compares against baseline and paper results.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import enhancements
from src.cla_ca.enhancements.e2_adaptive_feature_weighting import run_e2_enhancement
from src.cla_ca.enhancements.e3_improved_valley_detection import run_e3_enhancement
from src.cla_ca.enhancements.e4_discretization_optimization import run_e4_enhancement
from src.cla_ca.enhancements.e5_ensemble_thresholding import run_e5_enhancement

# Import baseline
from src.cla_ca.pipeline import run_claca_pipeline

# Datasets
DATASETS = [
    'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0', 'jedit-3.2',
    'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
    'EQ', 'JDT', 'LC', 'ML', 'PDE'
]

# Output directory
output_dir = Path('/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e2_e5')
output_dir.mkdir(parents=True, exist_ok=True)

def evaluate_baseline():
    """Evaluate baseline on all datasets."""
    print("=" * 80)
    print("EVALUATING BASELINE")
    print("=" * 80)
    results = []
    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        try:
            result = run_claca_pipeline(dataset)
            metrics = result['metrics']
            metrics['dataset'] = dataset
            metrics['method'] = 'baseline'
            metrics['f1'] = metrics['f1']
            metrics['mcc'] = metrics['mcc']
            metrics['recall'] = metrics['recall']
            metrics['precision'] = metrics['precision']
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    return pd.DataFrame(results)


def evaluate_e2():
    """Evaluate E2 adaptive feature weighting methods."""
    print("\n" + "=" * 80)
    print("EVALUATING E2: Adaptive Feature Weighting")
    print("=" * 80)

    e2_methods = [
        ('correlation_adapted', 'E2a-Correlation Adapted Alpha'),
        ('stability', 'E2b-Stability Weighting'),
        ('information_gain', 'E2c-Information Gain')
    ]

    all_results = []
    for method_key, method_name in e2_methods:
        print(f"\n--- {method_name} ---")
        for dataset in DATASETS:
            print(f"Processing {dataset}...", end='\r')
            try:
                result = run_e2_enhancement(dataset, e2_method=method_key)
                metrics = result['metrics']
                metrics['dataset'] = dataset
                metrics['method'] = method_name
                metrics['method_key'] = method_key
                all_results.append(metrics)
            except Exception as e:
                print(f"\nError on {dataset}: {e}")
        print(f"Completed {method_name}")

    return pd.DataFrame(all_results)


def evaluate_e3():
    """Evaluate E3 improved valley detection methods."""
    print("\n" + "=" * 80)
    print("EVALUATING E3: Improved Valley Detection")
    print("=" * 80)

    e3_methods = [
        ('multiscale', 'E3a-Multiscale Valley'),
        ('adaptive', 'E3b-Adaptive Threshold'),
        ('ensemble', 'E3c-Ensemble Detection')
    ]

    all_results = []
    for method_key, method_name in e3_methods:
        print(f"\n--- {method_name} ---")
        for dataset in DATASETS:
            print(f"Processing {dataset}...", end='\r')
            try:
                result = run_e3_enhancement(dataset, e3_method=method_key)
                metrics = result['metrics']
                metrics['dataset'] = dataset
                metrics['method'] = method_name
                metrics['method_key'] = method_key
                all_results.append(metrics)
            except Exception as e:
                print(f"\nError on {dataset}: {e}")
        print(f"Completed {method_name}")

    return pd.DataFrame(all_results)


def evaluate_e4():
    """Evaluate E4 discretization optimization methods."""
    print("\n" + "=" * 80)
    print("EVALUATING E4: Discretization Optimization")
    print("=" * 80)

    e4_methods = [
        ('adaptive_bim', 'E4a-Adaptive BIM'),
        ('equal_frequency', 'E4b-Equal Frequency'),
        ('hybrid', 'E4c-Hybrid Discretization'),
        ('distribution_aware', 'E4d-Distribution Aware')
    ]

    all_results = []
    for method_key, method_name in e4_methods:
        print(f"\n--- {method_name} ---")
        for dataset in DATASETS:
            print(f"Processing {dataset}...", end='\r')
            try:
                result = run_e4_enhancement(dataset, e4_method=method_key)
                metrics = result['metrics']
                metrics['dataset'] = dataset
                metrics['method'] = method_name
                metrics['method_key'] = method_key
                all_results.append(metrics)
            except Exception as e:
                print(f"\nError on {dataset}: {e}")
        print(f"Completed {method_name}")

    return pd.DataFrame(all_results)


def evaluate_e5():
    """Evaluate E5 ensemble thresholding methods."""
    print("\n" + "=" * 80)
    print("EVALUATING E5: Ensemble Thresholding")
    print("=" * 80)

    e5_methods = [
        ('ensemble_voting', 'E5a-Ensemble Voting'),
        ('confidence_weighted', 'E5b-Confidence Weighted'),
        ('meta_learning', 'E5c-Meta Learning')
    ]

    all_results = []
    for method_key, method_name in e5_methods:
        print(f"\n--- {method_name} ---")
        for dataset in DATASETS:
            print(f"Processing {dataset}...", end='\r')
            try:
                result = run_e5_enhancement(dataset, e5_method=method_key)
                metrics = result['metrics']
                metrics['dataset'] = dataset
                metrics['method'] = method_name
                metrics['method_key'] = method_key
                all_results.append(metrics)
            except Exception as e:
                print(f"\nError on {dataset}: {e}")
        print(f"Completed {method_name}")

    return pd.DataFrame(all_results)


def compare_to_baseline(baseline_df, enhanced_df):
    """Compare enhanced methods to baseline."""
    results = []

    for idx, row in enhanced_df.iterrows():
        dataset = row['dataset']
        method = row['method']

        baseline_row = baseline_df[baseline_df['dataset'] == dataset]
        if baseline_row.empty:
            continue

        baseline_f1 = baseline_row['f1'].values[0]
        enhanced_f1 = row['f1']

        f1_improvement = enhanced_f1 - baseline_f1
        f1_improvement_pct = (f1_improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0

        results.append({
            'dataset': dataset,
            'method': method,
            'baseline_f1': baseline_f1,
            'enhanced_f1': enhanced_f1,
            'f1_improvement': f1_improvement,
            'f1_improvement_pct': f1_improvement_pct,
            'baseline_mcc': baseline_row['mcc'].values[0],
            'enhanced_mcc': row['mcc'],
            'mcc_improvement': row['mcc'] - baseline_row['mcc'].values[0]
        })

    return pd.DataFrame(results)


def main():
    print("\n" + "=" * 80)
    print("CLA-CA E2-E5 COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Datasets: {len(DATASETS)}")

    # Evaluate baseline
    baseline_results = evaluate_baseline()
    baseline_results.to_csv(output_dir / 'baseline_results.csv', index=False)

    # Evaluate E2
    e2_results = evaluate_e2()
    e2_results.to_csv(output_dir / 'e2_results.csv', index=False)

    # Evaluate E3
    e3_results = evaluate_e3()
    e3_results.to_csv(output_dir / 'e3_results.csv', index=False)

    # Evaluate E4
    e4_results = evaluate_e4()
    e4_results.to_csv(output_dir / 'e4_results.csv', index=False)

    # Evaluate E5
    e5_results = evaluate_e5()
    e5_results.to_csv(output_dir / 'e5_results.csv', index=False)

    # Combine all results
    all_results = []
    all_results.append(baseline_results[['dataset', 'method', 'f1', 'mcc', 'recall', 'precision']])
    all_results.append(e2_results[['dataset', 'method', 'f1', 'mcc', 'recall', 'precision']])
    all_results.append(e3_results[['dataset', 'method', 'f1', 'mcc', 'recall', 'precision']])
    all_results.append(e4_results[['dataset', 'method', 'f1', 'mcc', 'recall', 'precision']])
    all_results.append(e5_results[['dataset', 'method', 'f1', 'mcc', 'recall', 'precision']])

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / 'combined_results.csv', index=False)

    # Compare to baseline
    print("\n" + "=" * 80)
    print("COMPARING TO BASELINE")
    print("=" * 80)

    e2_comparison = compare_to_baseline(baseline_results, e2_results)
    e3_comparison = compare_to_baseline(baseline_results, e3_results)
    e4_comparison = compare_to_baseline(baseline_results, e4_results)
    e5_comparison = compare_to_baseline(baseline_results, e5_results)

    e2_comparison.to_csv(output_dir / 'e2_comparison.csv', index=False)
    e3_comparison.to_csv(output_dir / 'e3_comparison.csv', index=False)
    e4_comparison.to_csv(output_dir / 'e4_comparison.csv', index=False)
    e5_comparison.to_csv(output_dir / 'e5_comparison.csv', index=False)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_mean_f1': float(baseline_results['f1'].mean()),
        'baseline_mean_mcc': float(baseline_results['mcc'].mean()),
    }

    for enhancement_name, df in [('E2', e2_results), ('E3', e3_results), ('E4', e4_results), ('E5', e5_results)]:
        mean_f1 = df.groupby('method')['f1'].mean()
        mean_mcc = df.groupby('method')['mcc'].mean()

        best_method = mean_f1.idxmax()
        best_f1 = mean_f1.max()
        improvement = best_f1 - summary['baseline_mean_f1']
        improvement_pct = (improvement / summary['baseline_mean_f1'] * 100)

        summary[f'{enhancement_name}_best_method'] = best_method
        summary[f'{enhancement_name}_best_f1'] = float(best_f1)
        summary[f'{enhancement_name}_improvement'] = float(improvement)
        summary[f'{enhancement_name}_improvement_pct'] = float(improvement_pct)

        print(f"\n{enhancement_name}:")
        print(f"  Best method: {best_method}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Improvement over baseline: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    # Find overall best enhancement
    best_overall = None
    best_improvement = float('-inf')

    for enhancement_name in ['E2', 'E3', 'E4', 'E5']:
        improvement = summary.get(f'{enhancement_name}_improvement', 0)
        if improvement > best_improvement:
            best_improvement = improvement
            best_overall = summary.get(f'{enhancement_name}_best_method')

    summary['best_overall_method'] = best_overall
    summary['best_overall_improvement'] = float(best_improvement)
    summary['best_overall_improvement_pct'] = float(best_improvement / summary['baseline_mean_f1'] * 100)

    print(f"\n{'=' * 80}")
    print("OVERALL BEST ENHANCEMENT")
    print(f"{'=' * 80}")
    print(f"Best method: {best_overall}")
    print(f"Improvement over baseline: {best_improvement:+.4f} ({summary['best_overall_improvement_pct']:+.1f}%)")
    print(f"Baseline F1: {summary['baseline_mean_f1']:.4f}")
    print(f"Enhanced F1: {summary['baseline_mean_f1'] + best_improvement:.4f}")

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Create detailed report
    report = f"""
# E2-E5 Enhancement Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Evaluated E2-E5 enhancements across {len(DATASETS)} datasets using Protocol A (full dataset).

## Baseline Performance
- Mean F1: {summary['baseline_mean_f1']:.4f}
- Mean MCC: {summary['baseline_mean_mcc']:.4f}

## Enhancement Results

### E2: Adaptive Feature Weighting
Best method: {summary.get('E2_best_method', 'N/A')}
Best F1: {summary.get('E2_best_f1', 0):.4f}
Improvement: {summary.get('E2_improvement', 0):+.4f} ({summary.get('E2_improvement_pct', 0):+.1f}%)

### E3: Improved Valley Detection
Best method: {summary.get('E3_best_method', 'N/A')}
Best F1: {summary.get('E3_best_f1', 0):.4f}
Improvement: {summary.get('E3_improvement', 0):+.4f} ({summary.get('E3_improvement_pct', 0):+.1f}%)

### E4: Discretization Optimization
Best method: {summary.get('E4_best_method', 'N/A')}
Best F1: {summary.get('E4_best_f1', 0):.4f}
Improvement: {summary.get('E4_improvement', 0):+.4f} ({summary.get('E4_improvement_pct', 0):+.1f}%)

### E5: Ensemble Thresholding
Best method: {summary.get('E5_best_method', 'N/A')}
Best F1: {summary.get('E5_best_f1', 0):.4f}
Improvement: {summary.get('E5_improvement', 0):+.4f} ({summary.get('E5_improvement_pct', 0):+.1f}%)

## Overall Best Enhancement
Method: {best_overall}
Improvement: {best_improvement:+.4f} ({summary['best_overall_improvement_pct']:+.1f}%)

## Conclusion
The best performing enhancement provides {summary['best_overall_improvement_pct']:+.1f}% improvement over baseline.
"""

    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}")
    print("Files generated:")
    print("  - baseline_results.csv")
    print("  - e2_results.csv, e2_comparison.csv")
    print("  - e3_results.csv, e3_comparison.csv")
    print("  - e4_results.csv, e4_comparison.csv")
    print("  - e5_results.csv, e5_comparison.csv")
    print("  - combined_results.csv")
    print("  - summary.json")
    print("  - evaluation_report.md")

    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
