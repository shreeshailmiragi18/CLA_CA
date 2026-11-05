import sys
import json
import os
sys.path.insert(0, '/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/src')

from cla_ca.pipeline import run_claca_pipeline
from cla_ca.enhancements.e1_adaptive_thresholding import run_e1_enhancement

# Create results directory
os.makedirs('/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e1', exist_ok=True)

datasets = [
    'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0',
    'jedit-3.2', 'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
    'EQ', 'JDT', 'LC', 'ML', 'PDE'
]

# Thresholding methods to test
methods = [
    ('baseline', {}),
    ('percentile_25', {}),
    ('percentile_50', {}),
    ('percentile_75', {}),
    ('percentile_90', {}),
    ('otsu', {}),
    ('mean', {}),
    ('weighted_percentile_50', {}),
]

all_results = []

print("Running Protocol A on all 15 datasets with E1 methods...")
print("=" * 80)

for ds in datasets:
    print(f"\nDataset: {ds}")
    for method_name, kwargs in methods:
        try:
            print(f"  Method: {method_name}", end=" ... ")
            if method_name == 'baseline':
                result = run_claca_pipeline(ds)
            else:
                result = run_e1_enhancement(ds, threshold_method=method_name, **kwargs)

            # Extract key metrics
            metrics = result['metrics']
            f1 = metrics['f1']
            mcc = metrics['mcc']
            recall = metrics['recall']
            precision = metrics['precision']

            result_entry = {
                'dataset': ds,
                'threshold_method': method_name,
                'threshold': result.get('threshold', result.get('label_point')),
                'f1': f1,
                'mcc': mcc,
                'recall': recall,
                'precision': precision,
                'retained_features': result.get('retained_features_count', result.get('retained_features')),
                'total_features': result.get('total_features', result.get('total_features')),
            }
            all_results.append(result_entry)

            print(f"F1={f1:.4f}, MCC={mcc:.4f}, Recall={recall:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'dataset': ds,
                'threshold_method': method_name,
                'error': str(e)
            })

# Save results
output_file = '/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e1/protocol_a_results.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {output_file}")

# Also save as CSV for easy viewing
import pandas as pd
df = pd.DataFrame(all_results)
csv_file = '/Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform/results/enhancements/e1/protocol_a_results.csv'
df.to_csv(csv_file, index=False)
print(f"Results also saved to: {csv_file}")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY: F1 Scores by Dataset and Method")
print("=" * 80)
pivot = df.pivot_table(index='dataset', columns='threshold_method', values='f1')
print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

print("\n" + "=" * 80)
print("SUMMARY: MCC Scores by Dataset and Method")
print("=" * 80)
pivot_mcc = df.pivot_table(index='dataset', columns='threshold_method', values='mcc')
print(pivot_mcc.to_string(float_format=lambda x: f'{x:.4f}'))

print("\n" + "=" * 80)
print("SUMMARY: Recall Scores by Dataset and Method")
print("=" * 80)
pivot_recall = df.pivot_table(index='dataset', columns='threshold_method', values='recall')
print(pivot_recall.to_string(float_format=lambda x: f'{x:.4f}'))