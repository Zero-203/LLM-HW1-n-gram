import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "result" / "metrics" / "model_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "result" / "figures"
CANDIDATES = (
    ("add_k:1.0", "Add-k (1)"),
    ("add_k:0.1", "Add-k (0.1)"),
    ("add_k:0.01", "Add-k (0.01)"),
    ("add_k:0.001", "Add-k (0.001)"),
    ("add_k:0.0001", "Add-k (0.0001)"),
    ("good_turing", "Good-Turing"),
    ("katz", "Katz"),
)


def load_rows():
    with METRICS_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def plot_perplexity(rows):
    figure, axis = plt.subplots(figsize=(10, 6))
    orders = list(range(1, 7))
    rows_by_key = {(int(row["n"]), row["candidate"]): row for row in rows}

    for candidate, label in CANDIDATES:
        values = [
            float(rows_by_key[(order, candidate)]["test_perplexity"])
            for order in orders
        ]
        axis.plot(orders, values, marker="o", linewidth=1.8, label=label)

    axis.set_yscale("log")
    axis.set_xticks(orders)
    axis.set_xlabel("N-gram order")
    axis.set_ylabel("Test perplexity (log scale)")
    axis.set_title("Test Perplexity by N-gram Order and Smoothing")
    axis.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "test_perplexity_by_smoothing.png", dpi=180)
    plt.close(figure)


def plot_model_cost(rows):
    selected = {
        int(row["n"]): row
        for row in rows
        if row["selected"] == "True"
    }
    orders = list(range(1, 7))
    labels = ["Uni", "Bi", "Tri", "4-gram", "5-gram", "6-gram"]
    seconds = [float(selected[order]["timing_measured_total_seconds"]) for order in orders]
    sizes_mb = [
        int(selected[order]["model_artifact_size_bytes"]) / (1024 * 1024)
        for order in orders
    ]

    figure, time_axis = plt.subplots(figsize=(9, 5.5))
    positions = list(range(len(orders)))
    time_axis.bar(positions, seconds, width=0.58, color="#4472C4", label="Measured time")
    time_axis.set_xticks(positions, labels)
    time_axis.set_xlabel("N-gram model")
    time_axis.set_ylabel("Measured time (seconds)", color="#31558A")
    time_axis.tick_params(axis="y", labelcolor="#31558A")
    time_axis.grid(True, axis="y", linestyle="--", alpha=0.3)

    size_axis = time_axis.twinx()
    size_axis.plot(
        positions,
        sizes_mb,
        color="#C0504D",
        marker="D",
        linewidth=2,
        label="Model artifact size",
    )
    size_axis.set_ylabel("Model artifact size (MiB)", color="#9E3D3A")
    size_axis.tick_params(axis="y", labelcolor="#9E3D3A")
    time_handles, time_labels = time_axis.get_legend_handles_labels()
    size_handles, size_labels = size_axis.get_legend_handles_labels()
    time_axis.legend(time_handles + size_handles, time_labels + size_labels, loc="upper left")
    time_axis.set_title("Training, Evaluation, and Model Storage Cost")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_cost_by_order.png", dpi=180)
    plt.close(figure)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_perplexity(rows)
    plot_model_cost(rows)
    print("Generated figures in {}".format(FIGURES_DIR))


if __name__ == "__main__":
    main()
