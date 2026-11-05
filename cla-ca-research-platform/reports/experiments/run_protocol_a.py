#!/usr/bin/env python3
"""
Protocol A Experiment Runner: Full-dataset reproduction.

Executes CLA-CA on all 15 datasets using full-dataset protocol (100% training fraction)
and produces results comparable to the frozen baseline.

This is the first protocol in the phased experiment framework (Phase 1).
"""

import json
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cla_ca.pipeline import run_claca_pipeline
from cla_ca.metrics import calculate_metrics


def load_frozen_baseline():
    """Load the frozen baseline results for comparison."""
    # Navigate from reports/experiments to results/baseline/frozen
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    baseline_path = os.path.join(project_root, 'results', 'baseline', 'frozen', 'summary.csv')
    df = pd.read_csv(baseline_path)
    # Rename columns to match experiment output
    baseline_dict = {}
    for _, row in df.iterrows():
        baseline_dict[row['Dataset']] = {
            'f1': row['F1'],
            'mcc': row['MCC'],
            'recall': row['Recall'],
            'precision': row['Precision'],
            'g_mean': row['G-mean'],
            'accuracy': row.get('Accuracy', 0),
            'fpr': row.get('FPR', 0),
            'balanced_accuracy': row.get('Balanced Accuracy', 0),
        }
    return baseline_dict


def run_protocol_A_dataset(dataset_name, algorithm_defaults=None):
    """Run CLA-CA on a single dataset using Protocol A (full-dataset)."""

    if algorithm_defaults is None:
        algorithm_defaults = {
            "bim": {"PROMISE": 40, "AEEEM": 155},
            "alpha": 0.8,
            "normalization": "min-max"
        }

    # Determine if AEEEM
    from cla_ca.data_loader import load_dataset
    X_df, y_true, is_aeeem = load_dataset(dataset_name)

    # Set BIM based on dataset family
    bim = algorithm_defaults["bim"]["AEEEM"] if is_aeeem else algorithm_defaults["bim"]["PROMISE"]
    alpha = algorithm_defaults["alpha"]

    # Run the full CLA-CA pipeline
    result = run_claca_pipeline(dataset_name, bim=bim, alpha=alpha)

    # Extract and standardize metrics
    metrics = result['metrics']

    experiment_result = {
        'dataset': dataset_name,
        'protocol': 'A',
        'training_fraction': 1.0,
        'split_id': 'full_dataset',
        'algorithm_version': f'cla_ca_v{alpha}_{bim}bim',
        'bim': bim,
        'alpha': alpha,
        'is_aeeem': is_aeeem,

        # Core metrics
        'f1': float(metrics['f1']),
        'mcc': float(metrics['mcc']),
        'recall': float(metrics['recall']),
        'precision': float(metrics['precision']),
        'g_mean': float(metrics['g_mean']),
        'accuracy': float(metrics['accuracy']),
        'fpr': float(metrics['fpr']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),

        # Algorithm details
        'retained_features': result['retained_features_count'],
        'total_features': result['total_features'],
        'label_point': result['label_point'],

        # Baseline comparison (will be filled in after all datasets)
        'baseline_f1': None,
        'f1_improvement': None,
        'f1_improvement_pct': None,
    }

    return experiment_result


def run_protocol_A():
    """Run Protocol A on all 15 datasets."""

    # Project root
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')

    # Load frozen baseline
    baseline = load_frozen_baseline()
    print(f"Loaded frozen baseline with {len(baseline)} datasets")

    # Protocol A configuration
    datasets = [
        'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0',
        'jedit-3.2', 'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
        'EQ', 'JDT', 'LC', 'ML', 'PDE'
    ]

    output_dir = os.path.join(project_root, 'results', 'experiments', 'protocol_a')
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n{'='*60}")
    print(f"Protocol A: Full-dataset reproduction")
    print(f"{'='*60}")
    print(f"Datasets: {len(datasets)}")
    print()

    for i, ds in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] Running {ds}...", end=" ")

        try:
            result = run_protocol_A_dataset(ds)
            results.append(result)

            # Compare against frozen baseline
            if ds in baseline:
                result['baseline_f1'] = baseline[ds]['f1']
                result['f1_improvement'] = result['f1'] - baseline[ds]['f1']
                result['f1_improvement_pct'] = (result['f1'] - baseline[ds]['f1']) / baseline[ds]['f1'] * 100 if baseline[ds]['f1'] != 0 else 0
            else:
                result['baseline_f1'] = None
                result['f1_improvement'] = None
                result['f1_improvement_pct'] = None

            f1_str = f"F1={result['f1']:.4f}"
            if result['baseline_f1'] is not None:
                improv_str = f"(improvement: {result['f1_improvement_pct']:.1f}%)"
                f1_str += f"  baseline={result['baseline_f1']:.4f} {improv_str}"
            print(f1_str)

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'dataset': ds,
                'protocol': 'A',
                'training_fraction': 1.0,
                'split_id': 'full_dataset',
                'algorithm_version': 'ERROR',
                'bim': None,
                'alpha': None,
                'is_aeeem': None,
                'f1': None,
                'mcc': None,
                'recall': None,
                'precision': None,
                'g_mean': None,
                'accuracy': None,
                'fpr': None,
                'balanced_accuracy': None,
                'retained_features': None,
                'total_features': None,
                'label_point': None,
                'baseline_f1': baseline.get(ds, {}).get('f1'),
                'f1_improvement': None if baseline.get(ds, {}).get('f1') is None else 0 - baseline.get(ds, {}).get('f1'),
                'f1_improvement_pct': None,
                'error': str(e),
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save comparison summary
    comparison_path = os.path.join(output_dir, "comparison.csv")
    comparison_df = results_df.copy()
    if 'baseline_f1' in comparison_df.columns:
        comparison_df = comparison_df.merge(
            pd.DataFrame.from_dict(baseline, orient='index', columns=['baseline_f1', 'baseline_mcc', 'baseline_recall', 'baseline_precision', 'baseline_g_mean', 'baseline_accuracy', 'baseline_fpr', 'baseline_balanced_accuracy']),
            left_on='dataset',
            right_index=True,
            how='left'
        )
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved to: {comparison_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Protocol A Summary")
    print(f"{'='*60}")
    successful = [r for r in results if r.get('f1') is not None]
    failed = [r for r in results if r.get('f1') is None]

    print(f"Total datasets: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        f1_scores = [r['f1'] for r in successful]
        baseline_f1s = [r['baseline_f1'] for r in successful if r['baseline_f1'] is not None]
        improvements = [r['f1_improvement'] for r in successful if r['f1_improvement'] is not None]

        print(f"\nF1 score range: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
        if baseline_f1s:
            print(f"Baseline F1 range: {min(baseline_f1s):.4f} - {max(baseline_f1s):.4f}")
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Average improvement: {avg_improvement:.4f} ({avg_improvement/max(baseline_f1s)*100:.1f}%)")
            positive_improvements = [i for i in improvements if i > 0]
            negative_improvements = [i for i in improvements if i < 0]
            print(f"Datasets with improvement: {len(positive_improvements)}/{len(improvements)}")
            print(f"Datasets with degradation: {len(negative_improvements)}/{len(improvements)}")

    print(f"\nOutput directory: {output_dir}/")

    return results


def main():
    """Entry point for Protocol A runner."""
    results = run_protocol_A()
    return 0


if __name__ == "__main__":
    sys.exit(main())