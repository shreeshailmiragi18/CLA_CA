#!/usr/bin/env python3
"""Create dataset-level Paper CLA-CA versus E13 comparison bar charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E13_RESULTS = PROJECT_ROOT / "results" / "enhancements" / "e13" / "comparison.csv"
OUTPUT = PROJECT_ROOT / "results" / "figures" / "paper_vs_current" / "dataset_paper_vs_e13_barchart.png"

PAPER_RESULTS = {
    "ant-1.5": {"recall": 0.6875, "f1": 0.33333, "mcc": 0.25567, "g_mean": 0.38891},
    "arc": {"recall": 0.55556, "f1": 0.29412, "mcc": 0.1819, "g_mean": 0.33333},
    "berek": {"recall": 0.875, "f1": 0.84848, "mcc": 0.7552, "g_mean": 0.84887},
    "camel-1.0": {"recall": 0.76923, "f1": 0.11299, "mcc": 0.11407, "g_mean": 0.21657},
    "EQ": {"recall": 0.63566, "f1": 0.64822, "mcc": 0.42327, "g_mean": 0.64835},
    "ivy-2.0": {"recall": 0.5, "f1": 0.40816, "mcc": 0.32355, "g_mean": 0.41523},
    "JDT": {"recall": 0.3932, "f1": 0.51429, "mcc": 0.46425, "g_mean": 0.54055},
    "jedit-3.2": {"recall": 0.54444, "f1": 0.62821, "mcc": 0.49507, "g_mean": 0.63577},
    "LC": {"recall": 0.625, "f1": 0.39409, "mcc": 0.3378, "g_mean": 0.42409},
    "ML": {"recall": 0.4449, "f1": 0.34063, "mcc": 0.22162, "g_mean": 0.35038},
    "PDE": {"recall": 0.61244, "f1": 0.36571, "mcc": 0.24406, "g_mean": 0.39957},
    "synapse-1.1": {"recall": 0.81667, "f1": 0.46226, "mcc": 0.17287, "g_mean": 0.5131},
    "tomcat": {"recall": 0.58442, "f1": 0.41284, "mcc": 0.35593, "g_mean": 0.43187},
    "xalan-2.4": {"recall": 0.46364, "f1": 0.39382, "mcc": 0.26972, "g_mean": 0.39836},
    "xerces-1.2": {"recall": 0.26761, "f1": 0.28358, "mcc": 0.15582, "g_mean": 0.28409},
}

METRICS = (("recall", "Recall"), ("f1", "F1-score"), ("mcc", "MCC"), ("g_mean", "G-Mean"))


def main() -> None:
    e13 = pd.read_csv(E13_RESULTS).set_index("dataset")
    datasets = list(PAPER_RESULTS)
    positions = np.arange(len(datasets))
    width = 0.38

    figure, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    axes = axes.ravel()
    for axis, (metric, label) in zip(axes, METRICS):
        paper_values = [PAPER_RESULTS[dataset][metric] for dataset in datasets]
        e13_values = [float(e13.loc[dataset, f"e13_{metric}"]) for dataset in datasets]
        paper_bars = axis.bar(positions - width / 2, paper_values, width, label="Paper CLA-CA", color="#0f766e")
        e13_bars = axis.bar(positions + width / 2, e13_values, width, label="Current E13", color="#9ca3af")
        axis.set_title(label)
        axis.set_ylim(0, 1)
        axis.set_xticks(positions, datasets, rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(paper_bars, fmt="%.2f", padding=2, fontsize=6, rotation=90)
        axis.bar_label(e13_bars, fmt="%.2f", padding=2, fontsize=6, rotation=90)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Dataset-Level Comparison: Paper CLA-CA vs Current E13", fontsize=16, y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
