import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _lineplot(ax, df, x, y, title, ylabel):
    for cfg in sorted(df["config_name"].unique()):
        sub = df[df["config_name"] == cfg].sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=cfg)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Plot Online_FTD_net benchmark CSV")
    parser.add_argument("--csv", type=str, default="ftd_full_benchmark.csv")
    parser.add_argument("--out-dir", type=str, default="plots_ftd_benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "config_name" not in df.columns:
        df["config_name"] = "main"

    # Mean over seeds for cleaner sweep curves
    group_cols = ["dataset", "mode", "config_name", "R"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            params=("params", "mean"),
            final_test_nre=("final_test_nre", "mean"),
            final_test_loss=("final_test_loss", "mean"),
            total_train_time_s=("total_train_time_s", "mean"),
            infer_time_s=("infer_time_s", "mean"),
            avg_online_nre_test=("avg_online_nre_test", "mean"),
        )
    )

    # Per dataset/mode figures
    for (dataset, mode), sub in agg.groupby(["dataset", "mode"]):
        # R vs NRE
        fig, ax = plt.subplots(figsize=(8, 5))
        _lineplot(ax, sub, "R", "final_test_nre", f"{dataset} ({mode}) - Final Test NRE vs R", "Final Test NRE")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_R_vs_final_test_nre.png", dpi=160)
        plt.close(fig)

        # R vs Loss
        fig, ax = plt.subplots(figsize=(8, 5))
        _lineplot(ax, sub, "R", "final_test_loss", f"{dataset} ({mode}) - Final Test Loss vs R", "Final Test Loss (masked MSE)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_R_vs_final_test_loss.png", dpi=160)
        plt.close(fig)

        # R vs train time
        fig, ax = plt.subplots(figsize=(8, 5))
        _lineplot(ax, sub, "R", "total_train_time_s", f"{dataset} ({mode}) - Train Time vs R", "Total Train Time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_R_vs_train_time.png", dpi=160)
        plt.close(fig)

        # R vs inference time
        fig, ax = plt.subplots(figsize=(8, 5))
        _lineplot(ax, sub, "R", "infer_time_s", f"{dataset} ({mode}) - Inference Time vs R", "Inference Time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_R_vs_infer_time.png", dpi=160)
        plt.close(fig)

        # Params vs error
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in sorted(sub["config_name"].unique()):
            s = sub[sub["config_name"] == cfg].sort_values("params")
            ax.plot(s["params"], s["final_test_nre"], marker="o", label=cfg)
        ax.set_title(f"{dataset} ({mode}) - Params vs Final Test NRE")
        ax.set_xlabel("Trainable Parameters")
        ax.set_ylabel("Final Test NRE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_params_vs_error.png", dpi=160)
        plt.close(fig)

        # Inference time vs error
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in sorted(sub["config_name"].unique()):
            s = sub[sub["config_name"] == cfg]
            ax.scatter(s["infer_time_s"], s["final_test_nre"], label=cfg, s=45)
            for _, r in s.iterrows():
                ax.annotate(f"R={int(r['R'])}", (r["infer_time_s"], r["final_test_nre"]), fontsize=7)
        ax.set_title(f"{dataset} ({mode}) - Inference Time vs Error")
        ax.set_xlabel("Inference Time (s)")
        ax.set_ylabel("Final Test NRE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_infer_time_vs_error.png", dpi=160)
        plt.close(fig)

        # Train time vs error
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in sorted(sub["config_name"].unique()):
            s = sub[sub["config_name"] == cfg]
            ax.scatter(s["total_train_time_s"], s["final_test_nre"], label=cfg, s=45)
            for _, r in s.iterrows():
                ax.annotate(f"R={int(r['R'])}", (r["total_train_time_s"], r["final_test_nre"]), fontsize=7)
        ax.set_title(f"{dataset} ({mode}) - Train Time vs Error")
        ax.set_xlabel("Total Train Time (s)")
        ax.set_ylabel("Final Test NRE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}_{mode}_train_time_vs_error.png", dpi=160)
        plt.close(fig)

    # Save aggregate table for quick review
    agg.to_csv(out_dir / "benchmark_grouped_means.csv", index=False)
    print(f"Saved plots and grouped table to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
