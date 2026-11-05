#!/usr/bin/env python3
"""Print CLA-CA paper and optimized ensemble results in terminal tables."""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cla_ca.enhancements.e11_hybrid_ensemble import run_e11_hybrid_ensemble
from cla_ca.pipeline import run_claca_pipeline

DATASETS = [
    "ant-1.5", "arc", "berek", "camel-1.0", "EQ", "ivy-2.0",
    "JDT", "jedit-3.2", "LC", "ML", "PDE", "synapse-1.1",
    "tomcat", "xalan-2.4", "xerces-1.2",
]

DISPLAY_NAMES = {
    "ant-1.5": "ant1.5",
    "camel-1.0": "Camel1.0",
    "ivy-2.0": "ivy2.0",
    "jedit-3.2": "Jedit3.2",
    "synapse-1.1": "synapse1.1",
    "xalan-2.4": "Xalan2.4",
    "xerces-1.2": "Xerces1.2",
    "LC": "Lucene",
    "ML": "Mylyn",
}

METRICS = ("recall", "f1", "mcc", "g_mean")


def collect_results() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        baseline = run_claca_pipeline(dataset)["metrics"]
        optimized = run_e11_hybrid_ensemble(dataset)["metrics_ensemble"]
        row = {"Dataset": DISPLAY_NAMES.get(dataset, dataset)}
        for metric in METRICS:
            row[f"Paper CLA-CA {metric}"] = baseline[metric]
            row[f"Optimized {metric}"] = optimized[metric]
            row[f"Delta {metric}"] = optimized[metric] - baseline[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def print_metric_table(results: pd.DataFrame, metric: str) -> None:
    table = results[[
        "Dataset",
        f"Paper CLA-CA {metric}",
        f"Optimized {metric}",
        f"Delta {metric}",
    ]].copy()
    table.columns = ["Dataset", "Paper CLA-CA", "Optimized", "Delta"]
    average = pd.DataFrame([{
        "Dataset": "Average",
        "Paper CLA-CA": table["Paper CLA-CA"].mean(),
        "Optimized": table["Optimized"].mean(),
        "Delta": table["Delta"].mean(),
    }])
    table = pd.concat([table, average], ignore_index=True)
    print(f"\n{metric.upper()} results")
    print(table.to_string(index=False, formatters={
        "Paper CLA-CA": "{:.4f}".format,
        "Optimized": "{:.4f}".format,
        "Delta": "{:+.4f}".format,
    }))


def main() -> None:
    print("CLA-CA benchmark comparison")
    print("Optimized = leakage-resistant supervised stacking ensemble")
    print("Datasets: 15")
    results = collect_results()
    for metric in METRICS:
        print_metric_table(results, metric)


if __name__ == "__main__":
    main()
