import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from OFTD_CP_sweep import run_multi as run_cp_multi
from OFTD_CP_sweep import set_seed as set_cp_seed
from OFTD_FTD_sweep import run_one as run_tt_one


PAPER_FOREMAN_SR03 = 0.084


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def make_args(**kwargs):
    return type("Args", (), kwargs)()


def write_rows(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_foreman_rank_sweep(args):
    ranks = parse_int_list(args.r_values)
    seeds = parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cp_raw_path = out_dir / "raw_foreman_cp_rank_sensitivity_sr03.csv"
    tt_raw_path = out_dir / "raw_foreman_tt_rank_sensitivity_sr03.csv"

    cp_rows = []
    tt_rows = []

    cp_args = make_args(
        data=args.data,
        sample_rate=args.sample_rate,
        streaming_mode="multi",
        seeds=args.seeds,
        rank=0,
        mid_channel=args.mid_channel,
        init_ratio=args.init_ratio,
        single_c_init=5,
        delta_a=None,
        delta_b=None,
        delta_c=None,
        omega_a=args.omega_a,
        omega_b=args.omega_b,
        omega_c=args.omega_c,
        alpha=args.alpha,
        beta=args.beta,
        divide=args.divide,
        lr=args.lr,
        weight_decay=args.weight_decay,
        init_iters=args.init_iters,
        online_iters=args.online_iters,
        patience=args.patience,
        infer_repeats=args.infer_repeats,
        out_csv="",
    )

    tt_args = make_args(
        data=args.data,
        sample_rate=args.sample_rate,
        streaming_mode="multi",
        single_c_init=5,
        init_ratio=args.init_ratio,
        delta_a=None,
        delta_b=None,
        delta_c=None,
        mid_channel=args.mid_channel,
        omega_a=args.omega_a,
        omega_b=args.omega_b,
        omega_c=args.omega_c,
        w_init=args.w_init,
        alpha=args.alpha,
        beta=args.beta,
        divide=args.divide,
        lr=args.lr,
        lr_a_mult=args.lr_a_mult,
        lr_b_mult=args.lr_b_mult,
        lr_c_mult=args.lr_c_mult,
        weight_decay=args.weight_decay,
        init_iters=args.init_iters,
        online_iters=args.online_iters,
        patience=args.patience,
        normalize_recon=True,
        coord_mode="raw",
        loss_scope="sampled",
        boundary_lambda=args.boundary_lambda,
        deriv_lambda=0.0,
        kappa=-1.0,
        infer_repeats=args.infer_repeats,
        profile_flops=False,
        clip_grad_norm=args.clip_grad_norm,
        init_clip_grad_norm=args.init_clip_grad_norm,
        reuse_online_optimizer=args.reuse_online_optimizer,
    )

    for rank in ranks:
        for seed in seeds:
            if args.models in ("cp", "both"):
                print(f"[cp] R={rank} seed={seed}")
                set_cp_seed(seed)
                cp_args.rank = rank
                row = run_cp_multi(cp_args, seed)
                row["sample_rate"] = args.sample_rate
                row["model"] = "CP"
                cp_rows.append(row)
                print(
                    f"  cp final={row['final_test_nre']:.4f} "
                    f"avg={row['avg_online_nre_test']:.4f} "
                    f"time={row['total_train_time_s']:.2f}s"
                )

            if args.models in ("tt", "both"):
                print(f"[tt] R={rank} seed={seed}")
                row = run_tt_one(tt_args, rank, seed)
                row["sample_rate"] = args.sample_rate
                row["model"] = "Dense TT"
                tt_rows.append(row)
                print(
                    f"  tt final={row['final_test_nre']:.4f} "
                    f"avg={row['avg_online_nre_test']:.4f} "
                    f"time={row['total_train_time_s']:.2f}s"
                )

        write_rows(cp_raw_path, cp_rows)
        write_rows(tt_raw_path, tt_rows)

    return cp_raw_path, tt_raw_path


def load_rank_rows(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def aggregate_and_plot(out_dir: Path, cp_raw_path: Path, tt_raw_path: Path):
    frames = []
    for df in [load_rank_rows(cp_raw_path), load_rank_rows(tt_raw_path)]:
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No rank-sensitivity raw CSVs were found.")

    raw = pd.concat(frames, ignore_index=True)
    raw["avg_update_time_s"] = raw["online_time_s"] / raw["num_updates"].clip(lower=1)

    group_cols = ["model", "R"]
    agg = (
        raw.groupby(group_cols, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            params=("params", "mean"),
            final_test_nre=("final_test_nre", "mean"),
            final_test_nre_std=("final_test_nre", "std"),
            avg_online_nre_test=("avg_online_nre_test", "mean"),
            avg_update_time_s=("avg_update_time_s", "mean"),
            total_train_time_s=("total_train_time_s", "mean"),
            infer_time_s=("infer_time_s", "mean"),
        )
        .sort_values(["model", "R"])
    )
    agg["gap_vs_paper_foreman_sr03"] = agg["final_test_nre"] - PAPER_FOREMAN_SR03
    best_by_model = agg.loc[agg.groupby("model")["final_test_nre"].idxmin()].copy()

    agg_path = out_dir / "table_foreman_rank_sensitivity_cp_tt_sr03.csv"
    best_path = out_dir / "table_foreman_rank_sensitivity_best_by_model_sr03.csv"
    raw_path = out_dir / "table_foreman_rank_sensitivity_raw_cp_tt_sr03.csv"
    agg.to_csv(agg_path, index=False)
    best_by_model.to_csv(best_path, index=False)
    raw.to_csv(raw_path, index=False)

    colors = {"CP": "tab:orange", "Dense TT": "tab:green"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("R")
        ax.errorbar(
            sub["R"],
            sub["final_test_nre"],
            yerr=sub["final_test_nre_std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=model,
            color=colors.get(model),
        )
    ax.axhline(PAPER_FOREMAN_SR03, color="tab:blue", linestyle="--", linewidth=1.5, label="Paper OFTD")
    ax.set_title("Foreman SR=0.3: Rank Sensitivity")
    ax.set_xlabel("Model rank R")
    ax.set_ylabel("Final Test NRE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "foreman_rank_sensitivity_cp_tt_sr03_nre.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("R")
        ax.plot(sub["R"], sub["params"], marker="o", linewidth=2, label=model, color=colors.get(model))
    ax.set_title("Foreman SR=0.3: Parameter Count vs Rank")
    ax.set_xlabel("Model rank R")
    ax.set_ylabel("Trainable parameters")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "foreman_rank_sensitivity_cp_tt_sr03_params.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("R")
        ax.plot(
            sub["R"],
            sub["avg_update_time_s"],
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
    ax.set_title("Foreman SR=0.3: Avg Online Update Time vs Rank")
    ax.set_xlabel("Model rank R")
    ax.set_ylabel("Avg online update time (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "foreman_rank_sensitivity_cp_tt_sr03_update_time.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, sub in agg.groupby("model"):
        sub = sub.sort_values("params")
        ax.plot(
            sub["params"],
            sub["final_test_nre"],
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
        for _, row in sub.iterrows():
            ax.annotate(f"R={int(row['R'])}", (row["params"], row["final_test_nre"]), fontsize=8)
    ax.axhline(PAPER_FOREMAN_SR03, color="tab:blue", linestyle="--", linewidth=1.5, label="Paper OFTD")
    ax.set_title("Foreman SR=0.3: Parameter Count vs Final Test NRE")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final Test NRE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "foreman_rank_sensitivity_cp_tt_sr03_params_vs_nre.png", dpi=170)
    plt.close(fig)

    summary = [
        "# Foreman Rank Sensitivity: CP vs Dense TT",
        "",
        "Dataset: `foreman.mat`; sample rate: `0.3`; ranks: from generated raw sweep files.",
        "",
        "This test checks the idea that TT should be less sensitive to the unknown rank than CP.",
        "",
        "## Best By Model",
        "",
        markdown_table(best_by_model[
            [
                "model",
                "R",
                "final_test_nre",
                "avg_online_nre_test",
                "params",
                "avg_update_time_s",
                "gap_vs_paper_foreman_sr03",
            ]
        ]),
        "",
        "## Interpretation",
        "",
        "- Lower NRE is better.",
        "- If TT were more rank-robust in this implementation, the Dense TT curve would stay low over a broad range of `R`.",
        "- The plot should be read as an optimization-and-rank-sensitivity result, not as a direct measurement of the unknown true tensor rank.",
    ]
    (out_dir / "FOREMAN_RANK_SENSITIVITY_CP_TT.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved rank-sensitivity tables and plots to: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Build CP-vs-dense-TT rank sensitivity plots for the paper package.")
    parser.add_argument("--data", type=str, default="data/foreman.mat")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--r-values", type=str, default="5,10,20,40,60,80,100")
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--models", type=str, choices=["cp", "tt", "both", "plot-only"], default="both")
    parser.add_argument("--out-dir", type=str, default="paper_experiment_package")
    parser.add_argument("--init-ratio", type=float, default=0.1)
    parser.add_argument("--mid-channel", type=int, default=128)
    parser.add_argument("--omega-a", type=float, default=1.5)
    parser.add_argument("--omega-b", type=float, default=1.5)
    parser.add_argument("--omega-c", type=float, default=0.6)
    parser.add_argument("--w-init", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--divide", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-a-mult", type=float, default=1.0)
    parser.add_argument("--lr-b-mult", type=float, default=1.0)
    parser.add_argument("--lr-c-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--init-iters", type=int, default=4000)
    parser.add_argument("--online-iters", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--infer-repeats", type=int, default=20)
    parser.add_argument("--boundary-lambda", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--reuse-online-optimizer", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cp_raw_path = out_dir / "raw_foreman_cp_rank_sensitivity_sr03.csv"
    tt_raw_path = out_dir / "raw_foreman_tt_rank_sensitivity_sr03.csv"

    if args.models != "plot-only":
        cp_raw_path, tt_raw_path = run_foreman_rank_sweep(args)

    aggregate_and_plot(out_dir, cp_raw_path, tt_raw_path)


if __name__ == "__main__":
    main()
