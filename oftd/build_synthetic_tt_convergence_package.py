import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import torch

from model import Online_FTD_net
from OFTD_CP_sweep import run_multi as run_cp_multi
from OFTD_CP_sweep import set_seed as set_cp_seed
from OFTD_FTD_sweep import run_one as run_tt_one
from utils import dtype, device


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def make_args(**kwargs):
    return type("Args", (), kwargs)()


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def done_keys(rows):
    return {(row["model"], int(row["R"]), int(row["seed"])) for row in rows}


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


def make_synthetic_tt_dataset(path: Path, shape, true_rank: int, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return scio.loadmat(path)["Ohsi"].shape

    torch.manual_seed(seed)
    model = Online_FTD_net(
        R=true_rank,
        R1=true_rank,
        R2=true_rank,
        mid_channel=128,
        omega_A=1.5,
        omega_B=1.5,
        omega_C=0.6,
        w_init=0.05,
    ).to(device)
    a, b, c = shape
    a_input = torch.arange(a, dtype=dtype, device=device).reshape(a, 1)
    b_input = torch.arange(b, dtype=dtype, device=device).reshape(b, 1)
    c_input = torch.arange(c, dtype=dtype, device=device).reshape(c, 1)
    with torch.no_grad():
        x = model(a_input, b_input, c_input)
        x = x / (x.std() + 1e-12)
    scio.savemat(path, {"Ohsi": x.detach().cpu().numpy()})
    return x.shape


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

    raw_path = out_dir / "raw_synthetic_tt_cp_tt_convergence_sr03.csv"
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

    ranks = parse_int_list(args.r_values)
    seeds = parse_int_list(args.seeds)
    for rank in ranks:
        for seed in seeds:
            if args.models in ("cp", "both") and ("CP", rank, seed) not in done:
                print(f"[synthetic cp] R={rank} seed={seed}", flush=True)
                set_cp_seed(seed)
                cp_args.rank = rank
                row = run_cp_multi(cp_args, seed)
                row["sample_rate"] = args.sample_rate
                row["model"] = "CP"
                row["true_tt_rank"] = args.true_rank
                rows.append(row)
                write_rows(raw_path, rows)
                print(f"  cp final={row['final_test_nre']:.4f} time={row['total_train_time_s']:.2f}s", flush=True)

            if args.models in ("tt", "both") and ("Dense TT", rank, seed) not in done:
                print(f"[synthetic tt] R={rank} seed={seed}", flush=True)
                row = run_tt_one(tt_args, rank, seed)
                row["sample_rate"] = args.sample_rate
                row["model"] = "Dense TT"
                row["true_tt_rank"] = args.true_rank
                rows.append(row)
                write_rows(raw_path, rows)
                print(f"  tt final={row['final_test_nre']:.4f} time={row['total_train_time_s']:.2f}s", flush=True)

    return raw_path


def aggregate_and_plot(raw_path: Path, out_dir: Path, abs_tol: float, rel_tol: float, target_nre: float):
    raw = pd.read_csv(raw_path)
    raw["avg_update_time_s"] = raw["online_time_s"] / raw["num_updates"].clip(lower=1)
    table = (
        raw.groupby(["model", "R"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            params=("params", "mean"),
            final_test_nre=("final_test_nre", "mean"),
            avg_online_nre_test=("avg_online_nre_test", "mean"),
            total_train_time_s=("total_train_time_s", "mean"),
            avg_update_time_s=("avg_update_time_s", "mean"),
        )
        .sort_values(["model", "params"])
    )
    annotated = []
    for _, sub in table.groupby("model"):
        sub = sub.sort_values("params").copy()
        sub["nre_gain_from_prev"] = -sub["final_test_nre"].diff()
        sub["delta_params_from_prev"] = sub["params"].diff()
        sub["gain_per_10k_params"] = sub["nre_gain_from_prev"] / sub["delta_params_from_prev"] * 10000.0
        annotated.append(sub)
    table = pd.concat(annotated, ignore_index=True).sort_values(["model", "params"])

    practical_rows = []
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("params").copy()
        best = sub.loc[sub["final_test_nre"].idxmin()]
        sub["gap_to_best"] = sub["final_test_nre"] - best["final_test_nre"]
        sub["rel_gap_to_best"] = sub["gap_to_best"] / best["final_test_nre"]
        abs_hit = sub[sub["gap_to_best"] <= abs_tol].iloc[0]
        rel_hit = sub[sub["rel_gap_to_best"] <= rel_tol].iloc[0]
        practical_rows.append(
            {
                "model": model,
                "best_R": int(best["R"]),
                "best_params": float(best["params"]),
                "best_final_test_nre": float(best["final_test_nre"]),
                "earliest_R_within_abs_tol": int(abs_hit["R"]),
                "earliest_params_within_abs_tol": float(abs_hit["params"]),
                "nre_at_abs_hit": float(abs_hit["final_test_nre"]),
                "earliest_R_within_rel_tol": int(rel_hit["R"]),
                "earliest_params_within_rel_tol": float(rel_hit["params"]),
                "nre_at_rel_hit": float(rel_hit["final_test_nre"]),
            }
        )
    practical = pd.DataFrame(practical_rows)

    target_rows = []
    for model, sub in table.groupby("model"):
        sub = sub.sort_values("params").copy()
        hit = sub[sub["final_test_nre"] <= target_nre]
        best = sub.loc[sub["final_test_nre"].idxmin()]
        if hit.empty:
            target_rows.append(
                {
                    "model": model,
                    "target_nre": target_nre,
                    "target_status": "not reached",
                    "first_R_at_target": "",
                    "first_params_at_target": "",
                    "nre_at_target_hit": "",
                    "best_R": int(best["R"]),
                    "best_params": float(best["params"]),
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
                    "first_R_at_target": int(row["R"]),
                    "first_params_at_target": float(row["params"]),
                    "nre_at_target_hit": float(row["final_test_nre"]),
                    "best_R": int(best["R"]),
                    "best_params": float(best["params"]),
                    "best_final_test_nre": float(best["final_test_nre"]),
                }
            )
    target_df = pd.DataFrame(target_rows)

    table_path = out_dir / "table_synthetic_tt_cp_tt_convergence_sr03.csv"
    practical_path = out_dir / "table_synthetic_tt_practical_convergence_sr03.csv"
    target_path = out_dir / "table_synthetic_tt_target_nre_convergence_sr03.csv"
    table.to_csv(table_path, index=False)
    practical.to_csv(practical_path, index=False)
    target_df.to_csv(target_path, index=False)

    colors = {"CP": "tab:orange", "Dense TT": "tab:green"}
    for logx, suffix in [(False, ""), (True, "_logx")]:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for model, sub in table.groupby("model"):
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
        ax.axhline(target_nre, color="tab:blue", linestyle="--", linewidth=1.5, label=f"NRE target {target_nre:g}")
        if logx:
            ax.set_xscale("log")
        ax.set_title("Synthetic TT SR=0.3: Params vs NRE" + (" (log x)" if logx else ""))
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Final Test NRE")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"synthetic_tt_cp_tt_params_vs_nre_sr03{suffix}.png", dpi=170)
        plt.close(fig)

    cap = 100000
    capped = table[table["params"] <= cap].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, sub in capped.groupby("model"):
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
    ax.axhline(target_nre, color="tab:blue", linestyle="--", linewidth=1.5, label=f"NRE target {target_nre:g}")
    ax.set_title("Synthetic TT SR=0.3: Params vs NRE <= 100k")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final Test NRE")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_tt_cp_tt_params_vs_nre_sr03_upto_100k.png", dpi=170)
    plt.close(fig)

    report = [
        "# Synthetic TT CP/TT Convergence",
        "",
        "Dataset: synthetic tensor generated by the same dense-TT form used by `Online_FTD_net`.",
        "",
        f"Practical convergence: earliest parameter count within absolute NRE `{abs_tol}` or relative `{rel_tol:.1%}` of each model's best observed NRE.",
        "",
        "## Practical Convergence",
        "",
        markdown_table(practical),
        "",
        "## Target NRE Convergence",
        "",
        markdown_table(target_df),
        "",
        "## Full Sweep",
        "",
        markdown_table(
            table[
                [
                    "model",
                    "R",
                    "params",
                    "final_test_nre",
                    "avg_online_nre_test",
                    "nre_gain_from_prev",
                    "gain_per_10k_params",
                    "total_train_time_s",
                ]
            ]
        ),
    ]
    (out_dir / "SYNTHETIC_TT_CP_TT_CONVERGENCE_SR03.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved: {table_path}")
    print(f"Saved: {practical_path}")
    print(f"Saved: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic TT-ground-truth CP/TT convergence sweep.")
    parser.add_argument("--data", type=str, default="data/synthetic_tt_r10_40x40x50.mat")
    parser.add_argument("--shape", type=str, default="40,40,50")
    parser.add_argument("--true-rank", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=str, default="paper_experiment_package/synthetic_tt_convergence_sr03")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--r-values", type=str, default="2,5,10,15,20,30,40,60,80,100")
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
    parser.add_argument("--abs-tol", type=float, default=0.01)
    parser.add_argument("--rel-tol", type=float, default=0.05)
    parser.add_argument("--target-nre", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_synthetic_tt_cp_tt_convergence_sr03.csv"
    if args.models != "plot-only":
        raw_path = run_sweep(args)
    aggregate_and_plot(raw_path, out_dir, args.abs_tol, args.rel_tol, args.target_nre)


if __name__ == "__main__":
    main()
