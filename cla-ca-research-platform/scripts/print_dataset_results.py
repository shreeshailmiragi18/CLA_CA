#!/usr/bin/env python3
"""Print E13 per-dataset results and Paper vs E13 macro averages."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ("recall", "f1", "mcc", "g_mean")
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "enhancements" / "e13" / "comparison.csv"
INVENTORY_CSV = Path(__file__).resolve().parents[1] / "reports" / "dataset_inventory.csv"

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


def load_results(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_dataset_sizes() -> dict[str, int]:
    with INVENTORY_CSV.open(newline="") as csv_file:
        return {row["dataset"]: int(row["rows"]) for row in csv.DictReader(csv_file)}


def e13_accuracy(row: dict[str, str], dataset_sizes: dict[str, int]) -> float:
    total = dataset_sizes[row["dataset"]]
    actual_defects = int(row["actual_defects"])
    predicted_defects = int(row["predicted_defects"])
    true_positive = round(float(row["e13_recall"]) * actual_defects)
    false_positive = predicted_defects - true_positive
    true_negative = total - actual_defects - false_positive
    return (true_positive + true_negative) / total


def print_results(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("No matching results found.")

    headers = ["Dataset", "Metric", "Paper CLA-CA", "Current E13", "Absolute change"]
    table = [
        [row["dataset"], metric.upper(), f"{PAPER_RESULTS[row['dataset']][metric]:.4f}", f"{float(row[f'e13_{metric}']):.4f}", f"{float(row[f'e13_{metric}']) - PAPER_RESULTS[row['dataset']][metric]:+.4f}"]
        for row in rows
        for metric in METRICS
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in table)) for index in range(len(headers))]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_accuracy(rows: list[dict[str, str]], dataset_sizes: dict[str, int]) -> None:
    accuracies = [(row["dataset"], e13_accuracy(row, dataset_sizes)) for row in rows]
    print("\nE13 accuracy by dataset")
    print("Dataset       Accuracy")
    print("------------- --------")
    for dataset, accuracy in accuracies:
        print(f"{dataset:<13} {accuracy:>8.4f}")
    average = sum(accuracy for _, accuracy in accuracies) / len(accuracies)
    print(f"Average       {average:>8.4f}")


def print_average_comparison(rows: list[dict[str, str]]) -> None:
    print("\nAverage metric comparison")
    print("Metric    Paper CLA-CA  Current E13  Absolute change")
    print("--------  -----------  -----------  ---------------")

    labels = {"recall": "Recall", "f1": "F1-score", "mcc": "MCC", "g_mean": "G-Mean"}
    for metric in METRICS:
        paper_average = sum(PAPER_RESULTS[row["dataset"]][metric] for row in rows) / len(rows)
        current_average = sum(float(row[f"e13_{metric}"]) for row in rows) / len(rows)
        change = current_average - paper_average
        print(f"{labels[metric]:<8}  {paper_average:>11.4f}  {current_average:>11.4f}  {change:>+15.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the E13 comparison CSV.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Results file not found: {args.csv}")

    print(f"Results: {args.csv}")
    rows = load_results(args.csv)
    dataset_sizes = load_dataset_sizes()
    print_results(rows)
    print_average_comparison(rows)
    print_accuracy(rows, dataset_sizes)


if __name__ == "__main__":
    main()
