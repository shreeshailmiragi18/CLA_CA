#!/usr/bin/env python3
"""Create a Paper CLA-CA vs E13 average metric bar chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E13_RESULTS = PROJECT_ROOT / "results" / "enhancements" / "e13" / "comparison.csv"
OUTPUT = PROJECT_ROOT / "results" / "figures" / "paper_vs_current" / "paper_vs_e13_average_barchart.png"

PAPER_AVERAGES = {
    "Recall": 0.58502,
    "F1-score": 0.42938,
    "MCC": 0.31805,
    "G-Mean": 0.45527,
}


def main() -> None:
    e13 = pd.read_csv(E13_RESULTS)
    e13_averages = {
        "Recall": e13["e13_recall"].mean(),
        "F1-score": e13["e13_f1"].mean(),
        "MCC": e13["e13_mcc"].mean(),
        "G-Mean": e13["e13_g_mean"].mean(),
    }

    metrics = list(PAPER_AVERAGES)
    paper_values = [PAPER_AVERAGES[metric] for metric in metrics]
    e13_values = [e13_averages[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6))
    paper_bars = axis.bar(x - width / 2, paper_values, width, label="Paper CLA-CA", color="#0f766e")
    e13_bars = axis.bar(x + width / 2, e13_values, width, label="Current E13", color="#9ca3af")

    axis.set_title("Average Metric Comparison: Paper CLA-CA vs E13")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Average score")
    axis.set_xticks(x, metrics)
    axis.set_ylim(0, 0.7)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    for bars in (paper_bars, e13_bars):
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
