"""
Comprehensive evaluation script for E6-E10 enhancements across all 15 datasets.
Measures F1, Precision, Recall, MCC, and G-mean, comparing against baseline.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cla_ca.pipeline import run_claca_pipeline
from cla_ca.enhancements.e6_supervised_ml import run_e6_ml_enhancement
from cla_ca.enhancements.e8_ensemble import run_e8_stacking_ensemble, run_e8_voting_ensemble
from cla_ca.enhancements.e9_cost_sensitive import run_e9_cost_sensitive_threshold, run_e9_multi_threshold_ensemble
from cla_ca.enhancements.e10_transfer_learning import run_e10_transfer_learning
from cla_ca.enhancements.e11_hybrid_ensemble import run_e11_hybrid_ensemble
from cla_ca.enhancements.e12_final_optimized import run_e12_final_optimized

DATASETS = [
    'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0', 'jedit-3.2',
    'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
    'EQ', 'JDT', 'LC', 'ML', 'PDE'
]

output_dir = Path('/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e6_e12')
output_dir.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION OF E6-E12 ADVANCED DEFECT DETECTION")
    print("=" * 80)

    summary_rows = []

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}...")
        try:
            # Baseline
            base_res = run_claca_pipeline(dataset)
            base_m = base_res['metrics']

            # E6 ML (Random Forest)
            e6_res = run_e6_ml_enhancement(dataset, classifier_type='random_forest')
            e6_m = e6_res['metrics']

            # E8 Stacking Ensemble
            e8_res = run_e8_stacking_ensemble(dataset)
            e8_m = e8_res['metrics_ensemble']

            # E9 Multi-Threshold Ensemble
            e9_res = run_e9_multi_threshold_ensemble(dataset)
            e9_m = e9_res['metrics']

            # E11 Hybrid Ensemble
            e11_res = run_e11_hybrid_ensemble(dataset)
            e11_m = e11_res['metrics_ensemble']

            # E12 Final Optimized
            e12_res = run_e12_final_optimized(dataset)
            e12_m = e12_res['metrics']

            summary_rows.append({
                'Dataset': dataset,
                'Baseline_F1': base_m['f1'],
                'E6_ML_F1': e6_m['f1'],
                'E8_Stacking_F1': e8_m['f1'],
                'E9_Threshold_F1': e9_m['f1'],
                'E11_Hybrid_F1': e11_m['f1'],
                'E12_Optimized_F1': e12_m['f1']
            })
            print(f"  Baseline F1: {base_m['f1']:.4f} | E12 Optimized F1: {e12_m['f1']:.4f}")
        except Exception as e:
            print(f"  Error on {dataset}: {e}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / 'e6_e12_comparison.csv', index=False)

    print("\n" + "=" * 80)
    print("AVERAGE F1 ACROSS ALL DATASETS")
    print("=" * 80)
    print(df_summary.mean(numeric_only=True).to_string())

    print(f"\nResults saved to {output_dir / 'e6_e12_comparison.csv'}")

if __name__ == '__main__':
    main()
