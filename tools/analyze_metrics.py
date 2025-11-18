import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise ValueError("CSV must include an 'episode' column")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    summary = df[numeric_cols].describe().transpose()
    summary["iqr"] = summary["75%"] - summary["25%"]
    return summary


def plot_distributions(df: pd.DataFrame, output_dir: Path, columns, bins: int = 30):
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=bins, kde=True, edgecolor="white")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_hist.png", dpi=150)
        plt.close()


def compare_layouts(random_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layout": ["random", "baseline"],
            "mean": [random_df[metric].mean(), baseline_df[metric].mean()],
            "median": [random_df[metric].median(), baseline_df[metric].median()],
            "std": [random_df[metric].std(), baseline_df[metric].std()],
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze per-episode metrics")
    parser.add_argument("metrics_csv", type=Path, help="Metrics file from training run")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline metrics CSV (e.g., fixed SLP layout)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=[
            "final_reward",
            "average_route_distance",
            "total_logistics_intensity",
            "space_utilization",
            "finished_goods",
            "throughput_rate",
        ],
        help="Metric columns to plot",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/metrics"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("analysis/metrics/summary.csv"),
        help="Path to save numeric summary CSV",
    )
    args = parser.parse_args()

    df = load_metrics(args.metrics_csv)
    summary = summarize(df)
    print("==== Random / Training Episodes Summary ====")
    print(summary)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv)
    print(f"Summary CSV saved to {args.summary_csv}")

    plot_distributions(df, args.out, args.columns)
    print(f"Histograms saved to {args.out}")

    if args.baseline:
        baseline_df = load_metrics(args.baseline)
        print("\n==== Baseline Comparison ====")
        comparisons = []
        for metric in args.columns:
            if metric in baseline_df.columns and metric in df.columns:
                comparison = compare_layouts(df, baseline_df, metric)
                print(f"\nMetric: {metric}")
                print(comparison)
                comparison = comparison.assign(metric=metric)
                comparisons.append(comparison)
        if comparisons:
            comparison_df = pd.concat(comparisons, ignore_index=True)
            comparison_path = args.out / "baseline_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Baseline comparison CSV saved to {comparison_path}")


if __name__ == "__main__":
    main()
