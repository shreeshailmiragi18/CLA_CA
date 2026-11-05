#!/usr/bin/env python3
"""Generate paper-vs-E13 and train/test figures without changing source results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cla_ca.enhancements.e13_calibrated_ensemble import (  # noqa: E402
    _build_ensemble,
    _claca_feature,
    _select_threshold,
)
from cla_ca.metrics import calculate_metrics  # noqa: E402

FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"
PAPER_CURRENT_DIR = FIGURES_ROOT / "paper_vs_current"
TRAIN_TEST_DIR = FIGURES_ROOT / "train_test"
METRICS = ["f1", "mcc", "g_mean", "recall", "precision"]
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


def load_paper_current() -> pd.DataFrame:
    baseline = pd.read_csv(PROJECT_ROOT / "results" / "comprehensive_evaluation.csv")
    baseline = baseline[baseline["method"].str.lower() == "baseline"].copy()
    current = pd.read_csv(
        PROJECT_ROOT / "results" / "enhancements" / "e13" / "comparison.csv"
    )
    paper = baseline.rename(
        columns={
            "method": "model",
            "recall": "recall",
            "f1": "f1",
            "mcc": "mcc",
            "g_mean": "g_mean",
            "precision": "precision",
        }
    )[["dataset", "model", "recall", "precision", "f1", "mcc", "g_mean", "actual_defects", "predicted_defects"]]
    paper["model"] = "Paper CLA-CA"
    enhanced = current.rename(
        columns={
            "e13_recall": "recall",
            "e13_f1": "f1",
            "e13_mcc": "mcc",
            "e13_g_mean": "g_mean",
        }
    )[["dataset", "recall", "f1", "mcc", "g_mean", "actual_defects", "predicted_defects"]]
    enhanced["model"] = "Current E13"
    true_positive = (enhanced["recall"] * enhanced["actual_defects"]).round()
    enhanced["precision"] = true_positive / enhanced["predicted_defects"].replace(0, np.nan)
    inventory = pd.read_csv(PROJECT_ROOT / "reports" / "dataset_inventory.csv")
    inventory = inventory.rename(columns={"Dataset": "dataset", "rows": "total_modules"})
    combined = pd.concat([paper, enhanced], ignore_index=True)
    return combined.merge(inventory[["dataset", "total_modules"]], on="dataset", how="left")


def save_paper_current_figures(data: pd.DataFrame) -> None:
    PAPER_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    long_metrics = data.melt(
        id_vars=["dataset", "model"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=long_metrics, x="metric", y="value", hue="model", palette=["#0f766e", "#9ca3af"])
    plt.ylim(0, 1)
    plt.title("Metric Distributions: Paper CLA-CA vs Current E13")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(PAPER_CURRENT_DIR / "metric_boxplots.png", dpi=250)
    plt.close()

    f1 = data.pivot(index="dataset", columns="model", values="f1")
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.4)
    plt.title("F1 Heatmap by Dataset")
    plt.xlabel("Method")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(PAPER_CURRENT_DIR / "f1_heatmap.png", dpi=250)
    plt.close()

    deltas = f1["Current E13"] - f1["Paper CLA-CA"]
    plt.figure(figsize=(8, 8))
    sns.heatmap(deltas.to_frame("F1 delta"), annot=True, fmt="+.3f", center=0, cmap="RdYlGn", linewidths=0.4)
    plt.title("Current E13 F1 Improvement over Paper")
    plt.tight_layout()
    plt.savefig(PAPER_CURRENT_DIR / "f1_delta_heatmap.png", dpi=250)
    plt.close()

    averages = data.groupby("model")[METRICS].mean().T
    plt.figure(figsize=(11, 7))
    averages.plot(kind="bar", ax=plt.gca(), color=["#0f766e", "#9ca3af"])
    plt.ylim(0, 1)
    plt.title("Average Metric Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Macro-average")
    plt.xticks(rotation=0)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(PAPER_CURRENT_DIR / "average_metric_comparison.png", dpi=250)
    plt.close()
    averages.to_csv(PAPER_CURRENT_DIR / "average_metrics.csv")

    confusion_rows = []
    for _, row in data.iterrows():
        actual = int(row["actual_defects"])
        predicted = int(row["predicted_defects"])
        true_positive = int(round(float(row["recall"]) * actual))
        false_negative = actual - true_positive
        false_positive = max(0, predicted - true_positive)
        true_negative = max(0, int(row.get("total_modules", 0)) - actual - false_positive)
        confusion_rows.append(
            {
                "model": row["model"],
                "tp": true_positive,
                "fn": false_negative,
                "fp": false_positive,
                "tn": true_negative,
            }
        )
    confusion = pd.DataFrame(confusion_rows).groupby("model")[["tp", "fn", "fp", "tn"]].sum()
    for model, values in confusion.iterrows():
        matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, square=True)
        plt.title(f"Aggregate Confusion Matrix: {model}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0.5, 1.5], ["Clean", "Defective"])
        plt.yticks([0.5, 1.5], ["Clean", "Defective"], rotation=0)
        filename = "paper_confusion_matrix.png" if model == "Paper CLA-CA" else "current_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(PAPER_CURRENT_DIR / filename, dpi=250)
        plt.close()


def train_test_results() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        X, y, _ = _claca_feature(dataset, alpha=0.8, bim=None)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        inner_splits = min(3, int(np.bincount(y_train).min()))
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
        calibration_model = _build_ensemble(42)
        train_probabilities = cross_val_predict(
            calibration_model,
            X_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = _select_threshold(train_probabilities, y_train)
        model = _build_ensemble(42)
        model.fit(X_train, y_train)
        train_predictions = (model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        test_predictions = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        train_metrics = calculate_metrics(y_train, train_predictions)
        test_metrics = calculate_metrics(y_test, test_predictions)
        row = {"dataset": dataset, "threshold": threshold}
        for metric in METRICS:
            row[f"train_{metric}"] = float(train_metrics[metric])
            row[f"test_{metric}"] = float(test_metrics[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def save_train_test_figures(data: pd.DataFrame) -> None:
    TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(TRAIN_TEST_DIR / "train_test_metrics.csv", index=False)

    f1 = data.set_index("dataset")[["train_f1", "test_f1"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.4)
    plt.title("E13 Train/Test F1 by Dataset")
    plt.xlabel("Split")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(TRAIN_TEST_DIR / "train_test_f1_heatmap.png", dpi=250)
    plt.close()

    averages = data[[f"train_{metric}" for metric in METRICS] + [f"test_{metric}" for metric in METRICS]].mean()
    average_frame = pd.DataFrame({"Train": [averages[f"train_{metric}"] for metric in METRICS], "Test": [averages[f"test_{metric}"] for metric in METRICS]}, index=METRICS)
    plt.figure(figsize=(11, 7))
    average_frame.plot(kind="bar", ax=plt.gca(), color=["#2563eb", "#f59e0b"])
    plt.ylim(0, 1)
    plt.title("E13 Average Train/Test Metric Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="Split")
    plt.tight_layout()
    plt.savefig(TRAIN_TEST_DIR / "train_test_average_comparison.png", dpi=250)
    plt.close()
    average_frame.to_csv(TRAIN_TEST_DIR / "train_test_average_metrics.csv")

    long_data = data.melt(id_vars="dataset", value_vars=[f"train_{metric}" for metric in METRICS] + [f"test_{metric}" for metric in METRICS], var_name="split_metric", value_name="score")
    long_data[["split", "metric"]] = long_data["split_metric"].str.split("_", n=1, expand=True)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=long_data, x="metric", y="score", hue="split", palette=["#2563eb", "#f59e0b"])
    plt.ylim(0, 1)
    plt.title("E13 Train/Test Metric Distributions")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(TRAIN_TEST_DIR / "train_test_metric_boxplots.png", dpi=250)
    plt.close()


def main() -> None:
    paper_current = load_paper_current()
    save_paper_current_figures(paper_current)
    train_test = train_test_results()
    save_train_test_figures(train_test)
    print(f"Created paper/current figures in {PAPER_CURRENT_DIR}")
    print(f"Created train/test figures in {TRAIN_TEST_DIR}")


if __name__ == "__main__":
    main()
