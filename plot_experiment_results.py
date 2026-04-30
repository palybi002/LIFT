import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd


def _safe_plot_bar(series, title, ylabel, output_path, color):
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar", color=color, edgecolor="black")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Setting")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison(df: pd.DataFrame, ts: str):
    if df.empty:
        print("[comparison] no rows to plot")
        return

    for dataset in sorted(df["Dataset"].dropna().unique()):
        sdf = df[df["Dataset"] == dataset].copy()
        if sdf.empty:
            continue

        sdf["DisplayModel"] = sdf.apply(
            lambda r: "LIFT" if (r.get("Model") == "DLinear" and bool(r.get("lift"))) else str(r.get("Model")),
            axis=1,
        )
        grouped = sdf.groupby("DisplayModel")[["MSE", "MAE", "TrainTime"]].mean(numeric_only=True)

        if grouped["MSE"].notna().any():
            _safe_plot_bar(
                grouped["MSE"],
                f"Comparison MSE - {dataset}",
                "MSE",
                f"plots/comparison_{dataset}_MSE_{ts}.png",
                "#5DA5DA",
            )
        if grouped["MAE"].notna().any():
            _safe_plot_bar(
                grouped["MAE"],
                f"Comparison MAE - {dataset}",
                "MAE",
                f"plots/comparison_{dataset}_MAE_{ts}.png",
                "#60BD68",
            )
        if grouped["TrainTime"].notna().any():
            _safe_plot_bar(
                grouped["TrainTime"],
                f"Comparison Train Time - {dataset}",
                "Seconds / epoch",
                f"plots/comparison_{dataset}_TrainTime_{ts}.png",
                "#F17CB0",
            )


def plot_ablation(df: pd.DataFrame, ts: str):
    if df.empty:
        print("[ablation] no rows to plot")
        return

    for dataset in sorted(df["Dataset"].dropna().unique()):
        ddf = df[df["Dataset"] == dataset].copy()
        for abl_type in sorted(ddf["AblationType"].dropna().unique()):
            tdf = ddf[ddf["AblationType"] == abl_type].copy()
            if tdf.empty:
                continue

            tdf["Setting"] = tdf["AblationValue"].astype(str)
            grouped = tdf.groupby("Setting")[["MSE", "MAE"]].mean(numeric_only=True)

            if grouped["MSE"].notna().any():
                _safe_plot_bar(
                    grouped["MSE"],
                    f"Ablation MSE - {dataset} - {abl_type}",
                    "MSE",
                    f"plots/ablation_{dataset}_{abl_type}_MSE_{ts}.png",
                    "#B2912F",
                )
            if grouped["MAE"].notna().any():
                _safe_plot_bar(
                    grouped["MAE"],
                    f"Ablation MAE - {dataset} - {abl_type}",
                    "MAE",
                    f"plots/ablation_{dataset}_{abl_type}_MAE_{ts}.png",
                    "#B276B2",
                )


def main():
    os.makedirs("plots", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    comp = pd.DataFrame()
    abl = pd.DataFrame()

    if os.path.exists("comparison_results.csv"):
        comp = pd.read_csv("comparison_results.csv")
    if os.path.exists("ablation_results.csv"):
        abl = pd.read_csv("ablation_results.csv")

    plot_comparison(comp, ts)
    plot_ablation(abl, ts)

    print(f"Plots generated in plots/ with timestamp {ts}")


if __name__ == "__main__":
    main()
