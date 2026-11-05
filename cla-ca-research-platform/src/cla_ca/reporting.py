import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from cla_ca.pipeline import run_claca_pipeline
from cla_ca.data_loader import load_dataset
from cla_ca.preprocessing import normalize_dataset
from cla_ca.feature_weighting import set_weight
from cla_ca.discretization import discretize_dataset
from cla_ca.valley_detection import create_split
from cla_ca.scoring import calculate_scores
from cla_ca.prediction import predict_labels

def generate_reports_and_figures():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)

    datasets = [
        'ant-1.5', 'arc', 'berek', 'camel-1.0', 'ivy-2.0',
        'jedit-3.2', 'synapse-1.1', 'tomcat', 'xalan-2.4', 'xerces-1.2',
        'EQ', 'JDT', 'LC', 'ML', 'PDE'
    ]

    all_results = []

    print("Running pipeline across all datasets for reporting...")
    for ds in datasets:
        try:
            res = run_claca_pipeline(ds)
            all_results.append({
                'Dataset': ds,
                'Recall': res['metrics']['recall'],
                'Precision': res['metrics']['precision'],
                'F1': res['metrics']['f1'],
                'MCC': res['metrics']['mcc'],
                'G-mean': res['metrics']['g_mean']
            })
        except Exception as e:
            print(f"Error for {ds}: {e}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv('results/reports/evaluation_summary.csv', index=False)

    # 1. Plot performance metrics comparison across datasets
    plt.figure(figsize=(12, 6))
    df_melted = df_res.melt(id_vars='Dataset', value_vars=['Recall', 'F1', 'MCC', 'G-mean'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Dataset', y='Score', hue='Metric')
    plt.title('CLA-CA Performance Metrics Across Datasets')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/performance_metrics.png', dpi=300)
    plt.savefig('results/figures/performance_metrics.pdf')
    plt.close()
    print("Generated performance_metrics figure.")

    # 2. Generate Confusion Matrix Heatmaps for representative datasets (e.g., berek and EQ)
    for ds in ['berek', 'EQ']:
        try:
            X_df, y_true, is_aeeem = load_dataset(ds)
            bim = 150 if is_aeeem else 35
            feature_cols = list(X_df.columns)
            df_norm = normalize_dataset(X_df, feature_cols)
            weights = set_weight(df_norm, feature_cols)
            df_disc = discretize_dataset(X_df, feature_cols, bim=bim)

            splist = []
            for col in feature_cols:
                sp = create_split(df_disc[col].values, bim=bim)
                if sp > 0:
                    splist.append({'index': col, 'value': sp})

            scores = calculate_scores(df_disc, weights, splist)
            y_pred, _ = predict_labels(scores)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Non-Defective', 'Defective'],
                        yticklabels=['Non-Defective', 'Defective'])
            plt.title(f'Confusion Matrix - {ds}')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'results/figures/confusion_matrix_{ds}.png', dpi=300)
            plt.close()
            print(f"Generated confusion matrix heatmap for {ds}.")
        except Exception as e:
            print(f"Could not generate CM for {ds}: {e}")

    print("All reporting figures and tables generated successfully under results/figures and results/reports.")

if __name__ == '__main__':
    generate_reports_and_figures()
