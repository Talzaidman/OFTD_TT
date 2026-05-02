import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_synthetic_tt_convergence_package import (
    make_args,
    make_synthetic_tt_dataset,
    markdown_table,
    parse_int_list,
    read_rows,
    write_rows,
)
from OFTD_CP_sweep import run_multi as run_cp_multi
from OFTD_CP_sweep import set_seed as set_cp_seed
from OFTD_FTD_sweep import run_one as run_tt_one


def cp_params(rank: int) -> int:
    return 17280 + 387 * rank


def tt_params(rank: int) -> int:
    return 17280 + 258 * rank + 129 * rank * rank


def nearest_cp_rank(target_params: int) -> int:
    raw = (target_params - 17280) / 387.0
    candidates = {max(1, math.floor(raw)), max(1, round(raw)), max(1, math.ceil(raw))}
    return min(candidates, key=lambda r: abs(cp_params(r) - target_params))


def nearest_tt_rank(target_params: int) -> int:
    disc = 258 * 258 + 4 * 129 * (target_params - 17280)
    raw = (-258 + math.sqrt(max(disc, 0.0))) / (2 * 129)
    candidates = {max(1, math.floor(raw)), max(1, round(raw)), max(1, math.ceil(raw))}
    return min(candidates, key=lambda r: abs(tt_params(r) - target_params))


def done_keys(rows):
    return {
        (row["model"], int(row["target_params"]), int(row["R"]), int(row["seed"]))
        for row in rows
    }


def run_sweep(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    shape = make_synthetic_tt_dataset(
        data_path,
        tuple(parse_int_list(args.shape)),
        args.true_rank,
        args.data_seed,
    )
    print(f"Synthetic TT dataset: {data_path} shape={shape} true_rank={args.true_rank}", flush=True)

    raw_path = out_dir / "raw_synthetic_tt_param_budget_sr03.csv"
    rows = read_rows(raw_path)
    done = done_keys(rows)

    cp_args = make_args(
        data=str(data_path),
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
        data=str(data_path),
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

    target_params = parse_int_list(args.target_params)
    seeds = parse_int_list(args.seeds)
    for target in target_params:
        cp_rank = nearest_cp_rank(target)
        tt_rank = nearest_tt_rank(target)
        for seed in seeds:
            if args.models in ("cp", "both") and ("CP", target, cp_rank, seed) not in done:
                print(f"[budget cp] target={target} R={cp_rank} seed={seed}", flush=True)
                set_cp_seed(seed)
                cp_args.rank = cp_rank
                row = run_cp_multi(cp_args, seed)
                row["target_params"] = target
                row["actual_params_formula"] = cp_params(cp_rank)
                row["param_error"] = cp_params(cp_rank) - target
                row["sample_rate"] = args.sample_rate
                row["model"] = "CP"
                row["true_tt_rank"] = args.true_rank
                rows.append(row)
                write_rows(raw_path, rows)
                print(
                    f"  cp params={row['params']} final={row['final_test_nre']:.4f} "
                    f"time={row['total_train_time_s']:.2f}s",
                    flush=True,
                )

            if args.models in ("tt", "both") and ("Dense TT", target, tt_rank, seed) not in done:
                print(f"[budget tt] target={target} R={tt_rank} seed={seed}", flush=True)
                row = run_tt_one(tt_args, tt_rank, seed)
                row["target_params"] = target
                row["actual_params_formula"] = tt_params(tt_rank)
                row["param_error"] = tt_params(tt_rank) - target
                row["sample_rate"] = args.sample_rate
                row["model"] = "Dense TT"
                row["true_tt_rank"] = args.true_rank
                rows.append(row)
                write_rows(raw_path, rows)
                print(
                    f"  tt params={row['params']} final={row['final_test_nre']:.4f} "
                    f"time={row['total_train_time_s']:.2f}s",
                    flush=True,
                )

    return raw_path


def aggregate_and_plot(raw_path: Path, out_dir: Path, target_nre: float):
    raw = pd.read_csv(raw_path)
    raw["avg_update_time_s"] = raw["online_time_s"] / raw["num_updates"].clip(lower=1)
    table = (
        raw.groupby(["model", "target_params", "R"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            actual_params=("params", "mean"),
            param_error=("param_error", "mean"),
            final_test_nre=("final_test_nre", "mean"),
            final_test_nre_std=("final_test_nre", "std"),
            avg_online_nre_test=("avg_online_nre_test", "mean"),
            total_train_time_s=("total_train_time_s", "mean"),
            avg_update_time_s=("avg_update_time_s", "mean"),
        )
        .sort_values(["target_params", "model"])
    )
    table_path = out_dir / "table_synthetic_tt_param_budget_sr03.csv"
    table.to_csv(table_path, index=False)

    target_rows = []
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("target_params")
        hit = sub[sub["final_test_nre"] <= target_nre]
        best = sub.loc[sub["final_test_nre"].idxmin()]
        if hit.empty:
            target_rows.append(
                {
                    "model": model,
                    "target_nre": target_nre,
                    "target_status": "not reached",
                    "first_target_params_at_target": "",
                    "actual_params_at_target": "",
                    "R_at_target": "",
                    "nre_at_target_hit": "",
                    "best_R": int(best["R"]),
                    "best_actual_params": float(best["actual_params"]),
                    "best_final_test_nre": float(best["final_test_nre"]),
                }
            )
        else:
            row = hit.iloc[0]
            target_rows.append(
                {
                    "model": model,
                    "target_nre": target_nre,
                    "target_status": "reached",
                    "first_target_params_at_target": int(row["target_params"]),
                    "actual_params_at_target": float(row["actual_params"]),
                    "R_at_target": int(row["R"]),
                    "nre_at_target_hit": float(row["final_test_nre"]),
                    "best_R": int(best["R"]),
                    "best_actual_params": float(best["actual_params"]),
                    "best_final_test_nre": float(best["final_test_nre"]),
                }
            )
    target_df = pd.DataFrame(target_rows)
    target_path = out_dir / "table_synthetic_tt_param_budget_target_nre_sr03.csv"
    target_df.to_csv(target_path, index=False)

    colors = {"CP": "tab:orange", "Dense TT": "tab:green"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("target_params")
        ax.errorbar(
            sub["target_params"],
            sub["final_test_nre"],
            yerr=sub["final_test_nre_std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=model,
            color=colors.get(model),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"R={int(row['R'])}",
                (row["target_params"], row["final_test_nre"]),
                fontsize=8,
            )
    ax.axhline(target_nre, color="tab:blue", linestyle="--", linewidth=1.5, label=f"NRE target {target_nre:g}")
    ax.set_title("Synthetic TT SR=0.3: Param-Budget Matched")
    ax.set_xlabel("Target trainable-parameter budget")
    ax.set_ylabel("Final Test NRE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "synthetic_tt_param_budget_vs_nre_sr03.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("actual_params")
        ax.errorbar(
            sub["actual_params"],
            sub["final_test_nre"],
            yerr=sub["final_test_nre_std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=model,
            color=colors.get(model),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"R={int(row['R'])}",
                (row["actual_params"], row["final_test_nre"]),
                fontsize=8,
            )
    ax.axhline(target_nre, color="tab:blue", linestyle="--", linewidth=1.5, label=f"NRE target {target_nre:g}")
    ax.set_title("Synthetic TT SR=0.3: Actual Params vs NRE")
    ax.set_xlabel("Actual trainable parameters")
    ax.set_ylabel("Final Test NRE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    actual_plot_path = out_dir / "synthetic_tt_actual_params_vs_nre_sr03.png"
    fig.savefig(actual_plot_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("target_params").copy()
        sub["best_so_far_nre"] = sub["final_test_nre"].cummin()
        ax.plot(
            sub["target_params"],
            sub["best_so_far_nre"],
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"R={int(row['R'])}",
                (row["target_params"], row["best_so_far_nre"]),
                fontsize=8,
            )
    ax.axhline(target_nre, color="tab:blue", linestyle="--", linewidth=1.5, label=f"NRE target {target_nre:g}")
    ax.set_title("Synthetic TT SR=0.3: Best-So-Far NRE")
    ax.set_xlabel("Target trainable-parameter budget")
    ax.set_ylabel("Best final test NRE so far")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    envelope_plot_path = out_dir / "synthetic_tt_param_budget_best_so_far_sr03.png"
    fig.savefig(envelope_plot_path, dpi=170)
    plt.close(fig)

    report = [
        "# Synthetic TT Parameter-Budget Matched Sweep",
        "",
        "Each x-axis sample is a target trainable-parameter budget. CP and dense TT use the nearest valid integer rank for that budget.",
        "",
        "Parameter formulas:",
        "",
        "- `CP params = 17280 + 387R`",
        "- `Dense TT params = 17280 + 258R + 129R^2`",
        "",
        "Budgets below about `17k` are impossible with the current INR architecture because the networks have fixed base weights.",
        "",
        "## Target NRE",
        "",
        markdown_table(target_df),
        "",
        "## Budget-Matched Results",
        "",
        markdown_table(
            table[
                [
                    "target_params",
                    "model",
                    "R",
                    "actual_params",
                    "param_error",
                    "final_test_nre",
                    "final_test_nre_std",
                    "avg_online_nre_test",
                    "total_train_time_s",
                ]
            ]
        ),
    ]
    report_path = out_dir / "SYNTHETIC_TT_PARAM_BUDGET_SWEEP_SR03.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved: {table_path}")
    print(f"Saved: {target_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {actual_plot_path}")
    print(f"Saved: {envelope_plot_path}")
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic TT param-budget matched CP/TT sweep.")
    parser.add_argument("--data", type=str, default="data/synthetic_tt_r10_40x40x50.mat")
    parser.add_argument("--shape", type=str, default="40,40,50")
    parser.add_argument("--true-rank", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="paper_experiment_package/synthetic_tt_param_budget_sr03")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--target-params", type=str, default="18000,20000,25000,30000,40000,50000,75000,100000,125000")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--models", type=str, choices=["cp", "tt", "both", "plot-only"], default="both")
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
    parser.add_argument("--init-iters", type=int, default=3000)
    parser.add_argument("--online-iters", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--infer-repeats", type=int, default=10)
    parser.add_argument("--boundary-lambda", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--reuse-online-optimizer", action="store_true")
    parser.add_argument("--target-nre", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_synthetic_tt_param_budget_sr03.csv"
    if args.models != "plot-only":
        raw_path = run_sweep(args)
    aggregate_and_plot(raw_path, out_dir, args.target_nre)


if __name__ == "__main__":
    main()
