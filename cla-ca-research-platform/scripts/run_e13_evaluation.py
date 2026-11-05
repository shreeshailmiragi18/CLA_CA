#!/usr/bin/env python3
"""Evaluate E13 against the original CLA-CA baseline on all datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cla_ca.enhancements.e13_calibrated_ensemble import (  # noqa: E402
    run_e13_calibrated_ensemble,
)
from cla_ca.pipeline import run_claca_pipeline  # noqa: E402

DATASETS = [
    "ant-1.5",
    "arc",
    "berek",
    "camel-1.0",
    "ivy-2.0",
    "jedit-3.2",
    "synapse-1.1",
    "tomcat",
    "xalan-2.4",
    "xerces-1.2",
    "EQ",
    "JDT",
    "LC",
    "ML",
    "PDE",
]


def main() -> None:
    rows = []
    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        baseline = run_claca_pipeline(dataset)["metrics"]
        enhanced = run_e13_calibrated_ensemble(dataset)
        metrics = enhanced["metrics"]
        rows.append(
            {
                "dataset": dataset,
                "baseline_recall": float(baseline["recall"]),
                "e13_recall": float(metrics["recall"]),
                "recall_delta": float(metrics["recall"] - baseline["recall"]),
                "baseline_f1": float(baseline["f1"]),
                "e13_f1": float(metrics["f1"]),
                "f1_delta": float(metrics["f1"] - baseline["f1"]),
                "baseline_mcc": float(baseline["mcc"]),
                "e13_mcc": float(metrics["mcc"]),
                "mcc_delta": float(metrics["mcc"] - baseline["mcc"]),
                "baseline_g_mean": float(baseline["g_mean"]),
                "e13_g_mean": float(metrics["g_mean"]),
                "g_mean_delta": float(metrics["g_mean"] - baseline["g_mean"]),
                "actual_defects": enhanced["actual_defects"],
                "predicted_defects": enhanced["predicted_defects"],
            }
        )

    result = pd.DataFrame(rows)
    output_path = PROJECT_ROOT / "results" / "enhancements" / "e13" / "comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nPer-dataset comparison:")
    display_columns = [
        "dataset",
        "baseline_f1",
        "e13_f1",
        "f1_delta",
        "baseline_mcc",
        "e13_mcc",
        "mcc_delta",
    ]
    print(result[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    averages = result.select_dtypes("number").mean()
    print("\nMacro averages:")
    for metric in ("recall", "f1", "mcc", "g_mean"):
        print(
            f"{metric.upper():7} baseline={averages[f'baseline_{metric}']:.4f} "
            f"E13={averages[f'e13_{metric}']:.4f} "
            f"delta={averages[f'{metric}_delta']:+.4f}"
        )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
