import csv
import os
from pathlib import Path
from statistics import mean

BASE_DIR = Path("/Users/tangyuchen/Desktop/cornell/26Spring/SYSEN5380/project")
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

INPUT_CSV = BASE_DIR / "polymarket_markets.csv"
OUTPUT_CSV = BASE_DIR / "polymarket_markets_with_variance.csv"
VARIANCE_PLOT = BASE_DIR / "variance_by_volume_bin.png"
ACCURACY_PLOT = BASE_DIR / "accuracy_by_volume_bin.png"
BIN_ORDER = ["<1k", "1k-10k", "10k-100k", "100k-1m", ">=1m"]


def parse_float(value):
    return float(value) if value not in ("", None) else None


def parse_int(value):
    return int(value) if value not in ("", None) else None


def get_volume_bin(volume):
    if volume < 1_000:
        return "<1k"
    if volume < 10_000:
        return "1k-10k"
    if volume < 100_000:
        return "10k-100k"
    if volume < 1_000_000:
        return "100k-1m"
    return ">=1m"


def load_rows():
    rows = []
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_prob = parse_float(row["pred_prob_day_minus_1"])
            final_outcome = parse_int(row["final_outcome_yes"])
            volume = parse_float(row["volume"])
            if pred_prob is None or final_outcome is None or volume is None:
                continue

            row["pred_prob_day_minus_1"] = pred_prob
            row["final_outcome_yes"] = final_outcome
            row["volume"] = volume
            row["variance"] = (pred_prob - final_outcome) ** 2
            row["is_correct"] = int((pred_prob >= 0.5) == bool(final_outcome))
            row["volume_bin"] = get_volume_bin(volume)
            rows.append(row)
    return rows


def summarize_volume_bins(rows):
    grouped = {label: {"variances": [], "correct": []} for label in BIN_ORDER}
    for row in rows:
        grouped[row["volume_bin"]]["variances"].append(row["variance"])
        grouped[row["volume_bin"]]["correct"].append(row["is_correct"])

    summary = []
    for label in BIN_ORDER:
        variances = grouped[label]["variances"]
        if not variances:
            continue
        summary.append(
            {
                "volume_bin": label,
                "count": len(variances),
                "avg_variance": mean(variances),
                "accuracy": mean(grouped[label]["correct"]),
            }
        )
    return summary


def write_output_csv(rows):
    fieldnames = [
        "market_id",
        "condition_id",
        "question",
        "category",
        "startDate",
        "endDate",
        "pred_prob_day_minus_1",
        "pred_prob_timestamp_utc",
        "final_outcome_yes",
        "volume",
        "unique_trading_wallets",
        "sampled_trade_count",
        "liquidity",
        "closed",
        "resolution",
        "winner",
        "variance",
        "volume_bin",
        "is_correct",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_bar_chart(values, bin_summary, output_path, title, ylabel, colors, ylim=None, inside_floor=0.06):
    if plt is None:
        return False

    labels = [row["volume_bin"] for row in bin_summary]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.xlabel("Volume Bin")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        plt.ylim(*ylim)

    for bar, row, value in zip(bars, bin_summary, values):
        x = bar.get_x() + bar.get_width() / 2
        plt.text(x, value + (0.002 if ylim is None else 0.015), f"{value:.3f}", ha="center", va="bottom")
        plt.text(
            x,
            max(value * 0.55, inside_floor),
            f"n={row['count']}",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def plot_variance_by_volume_bin(bin_summary):
    values = [row["avg_variance"] for row in bin_summary]
    colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#023047"]
    return plot_bar_chart(
        values,
        bin_summary,
        VARIANCE_PLOT,
        "Average Variance by Volume Bin",
        "Average Variance",
        colors,
        ylim=(0, max(values) + 0.005),
        inside_floor=0.01,
    )


def plot_accuracy_by_volume_bin(bin_summary):
    values = [row["accuracy"] for row in bin_summary]
    colors = ["#5f7d9a", "#45a889", "#8db869", "#efc14f", "#f89a1c"]
    return plot_bar_chart(
        values,
        bin_summary,
        ACCURACY_PLOT,
        "Prediction Accuracy by Volume Bin",
        "Prediction Accuracy",
        colors,
        ylim=(0, 1.0),
        inside_floor=0.08,
    )


def print_summary(rows, bin_summary, variance_plot_created, accuracy_plot_created):
    print(f"Rows used: {len(rows)}")
    print("Volume-bin summary:")
    for row in bin_summary:
        print(
            f"  {row['volume_bin']}: n={row['count']}, avg_variance={row['avg_variance']:.6f}, accuracy={row['accuracy']:.6f}"
        )

    print(f"Saved CSV: {OUTPUT_CSV}")
    if variance_plot_created:
        print(f"Saved plot: {VARIANCE_PLOT}")
    if accuracy_plot_created:
        print(f"Saved plot: {ACCURACY_PLOT}")
    if plt is None:
        print("Skipped plots: matplotlib is not installed in this Python environment.")


def main():
    rows = load_rows()
    bin_summary = summarize_volume_bins(rows)
    write_output_csv(rows)
    variance_plot_created = plot_variance_by_volume_bin(bin_summary)
    accuracy_plot_created = plot_accuracy_by_volume_bin(bin_summary)
    print_summary(rows, bin_summary, variance_plot_created, accuracy_plot_created)


if __name__ == "__main__":
    main()
