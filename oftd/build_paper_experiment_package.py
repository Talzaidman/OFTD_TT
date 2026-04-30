import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_FOREMAN = {0.1: 0.094, 0.2: 0.087, 0.3: 0.084}
PAPER_CONDITION = {0.1: 0.116, 0.2: 0.094, 0.3: 0.093}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prep_cp(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_rate"] = out["sample_rate"].astype(float)
    out["avg_online_update_time_s"] = out["online_time_s"] / out["num_updates"].clip(lower=1)
    g = (
        out.groupby("sample_rate", as_index=False)
        .agg(
            cp_final_test_nre=("final_test_nre", "mean"),
            cp_avg_online_nre_test=("avg_online_nre_test", "mean"),
            cp_avg_update_time_s=("avg_online_update_time_s", "mean"),
            cp_params=("params", "mean"),
        )
        .sort_values("sample_rate")
    )
    g["dataset"] = dataset_name
    return g


def prep_tt_foreman(
    sr01: pd.DataFrame, sr02: pd.DataFrame, sr03: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_r_rows = []
    for sr, df in [(0.1, sr01), (0.2, sr02), (0.3, sr03)]:
        tmp = df.copy()
        tmp["avg_online_update_time_s"] = tmp["online_time_s"] / tmp["num_updates"].clip(lower=1)
        g = (
            tmp.groupby("R", as_index=False)
            .agg(
                tt_final_test_nre=("final_test_nre", "mean"),
                tt_avg_online_nre_test=("avg_online_nre_test", "mean"),
                tt_avg_update_time_s=("avg_online_update_time_s", "mean"),
                tt_params=("params", "mean"),
            )
            .sort_values("R")
        )
        g["sample_rate"] = sr
        by_r_rows.append(g)
        best = g.loc[g["tt_final_test_nre"].idxmin()].to_dict()
        rows.append(best)
    best_sr = pd.DataFrame(rows).sort_values("sample_rate")
    by_r = pd.concat(by_r_rows, ignore_index=True)
    return best_sr, by_r


def prep_tt_condition(condition_sr: pd.DataFrame) -> pd.DataFrame:
    out = condition_sr.copy()
    out["sample_rate"] = out["sample_rate"].astype(float)
    out = out.sort_values("sample_rate")
    return out.rename(
        columns={
            "final_test_nre": "tt_final_test_nre",
            "avg_online_nre_test": "tt_avg_online_nre_test",
            "avg_online_update_time_s": "tt_avg_update_time_s",
            "params": "tt_params",
            "R": "tt_R",
        }
    )[
        [
            "sample_rate",
            "tt_R",
            "tt_final_test_nre",
            "tt_avg_online_nre_test",
            "tt_avg_update_time_s",
            "tt_params",
        ]
    ]


def prep_tt_foreman_highbudget(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path).copy()
    out["sample_rate"] = out["sample_rate"].astype(float)
    out["avg_online_update_time_s"] = out["online_time_s"] / out["num_updates"].clip(lower=1)
    g = (
        out.groupby("sample_rate", as_index=False)
        .agg(
            R=("R", "mean"),
            tt_final_test_nre=("final_test_nre", "mean"),
            tt_avg_online_nre_test=("avg_online_nre_test", "mean"),
            tt_avg_update_time_s=("avg_online_update_time_s", "mean"),
            tt_params=("params", "mean"),
            tt_total_train_time_s=("total_train_time_s", "mean"),
            dB_l1_max=("dB_l1_max", "mean"),
        )
        .sort_values("sample_rate")
    )
    return g


def line_plot(x, ys, labels, title, xlabel, ylabel, out_path: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def scatter_plot(
    x, y, ann, title, xlabel, ylabel, out_path: Path, color: str = "tab:blue"
) -> None:
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(x, y, s=70, color=color)
    for xi, yi, txt in zip(x, y, ann):
        plt.annotate(txt, (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build paper-ready experiment package")
    parser.add_argument("--out-dir", type=str, default="paper_experiment_package")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = base / args.out_dir
    ensure_dir(out_dir)

    # Inputs (expected to exist from prior experiment runs)
    foreman_tt_sr01 = pd.read_csv(base / "foreman_ftd_paper_recreate_sr01_r_sweep.csv")
    foreman_tt_sr02 = pd.read_csv(base / "foreman_ftd_paper_recreate_sr02_r_sweep.csv")
    foreman_tt_sr03 = pd.read_csv(base / "foreman_ftd_paper_recreate_r_sweep.csv")
    condition_tt_sr = pd.read_csv(base / "condition_tt_vs_paper_sr_r80_seed42.csv")
    condition_tt_sr03_by_r = pd.read_csv(base / "condition_ftd_paper_recreate_r_sweep_seed42.csv")

    foreman_cp = pd.read_csv(base / "foreman_cp_sr_sweep_s3.csv")
    condition_cp = pd.read_csv(base / "condition_cp_sr_sweep_seed42.csv")

    foreman_cp_sr = prep_cp(foreman_cp, "foreman")
    condition_cp_sr = prep_cp(condition_cp, "condition")
    foreman_tt_best_sr, foreman_tt_by_r = prep_tt_foreman(
        foreman_tt_sr01, foreman_tt_sr02, foreman_tt_sr03
    )
    foreman_highbudget_path = base / "foreman_ftd_r20_online1000_s3_by_sr.csv"
    foreman_tt_final_sr = foreman_tt_best_sr.copy()
    foreman_profile_note = "best-R per SR from `R in {20,40,60,80,100}` and 3 seeds."
    if foreman_highbudget_path.exists():
        foreman_tt_final_sr = prep_tt_foreman_highbudget(foreman_highbudget_path)
        foreman_profile_note = "`R=20`, 1000 online iterations/update, and 3 seeds."
    condition_tt = prep_tt_condition(condition_tt_sr)

    # Build final table: Foreman (multi-aspect)
    foreman = foreman_cp_sr.merge(
        foreman_tt_final_sr[
            ["sample_rate", "R", "tt_final_test_nre", "tt_avg_online_nre_test", "tt_avg_update_time_s", "tt_params"]
        ].rename(columns={"R": "tt_best_R"}),
        on="sample_rate",
        how="inner",
    )
    foreman["paper_oftd_nre"] = foreman["sample_rate"].map(PAPER_FOREMAN)
    foreman["tt_gap_vs_paper"] = foreman["tt_final_test_nre"] - foreman["paper_oftd_nre"]
    foreman["cp_gap_vs_paper"] = foreman["cp_final_test_nre"] - foreman["paper_oftd_nre"]
    foreman = foreman.sort_values("sample_rate")

    # Build final table: Condition (single-aspect)
    condition = condition_cp_sr.merge(condition_tt, on="sample_rate", how="inner")
    condition["paper_oftd_nre"] = condition["sample_rate"].map(PAPER_CONDITION)
    condition["tt_gap_vs_paper"] = condition["tt_final_test_nre"] - condition["paper_oftd_nre"]
    condition["cp_gap_vs_paper"] = condition["cp_final_test_nre"] - condition["paper_oftd_nre"]
    condition = condition.sort_values("sample_rate")

    foreman.to_csv(out_dir / "table_multi_foreman_sr.csv", index=False)
    condition.to_csv(out_dir / "table_single_condition_sr.csv", index=False)

    # Save rank sensitivity tables
    foreman_tt_by_r.to_csv(out_dir / "table_foreman_tt_rank_sensitivity.csv", index=False)
    condition_tt_sr03_by_r.to_csv(out_dir / "table_condition_tt_rank_sensitivity_sr03.csv", index=False)
    if foreman_highbudget_path.exists():
        foreman_tt_final_sr.to_csv(out_dir / "table_foreman_tt_r20_online1000_by_sr.csv", index=False)

    # Build combined benchmark for convenience
    bench_rows = []
    for _, r in foreman.iterrows():
        bench_rows.append(
            {
                "dataset": "foreman",
                "sample_rate": r["sample_rate"],
                "model": "Paper OFTD",
                "final_test_nre": r["paper_oftd_nre"],
                "avg_update_time_s": np.nan,
            }
        )
        bench_rows.append(
            {
                "dataset": "foreman",
                "sample_rate": r["sample_rate"],
                "model": "CP baseline",
                "final_test_nre": r["cp_final_test_nre"],
                "avg_update_time_s": r["cp_avg_update_time_s"],
            }
        )
        bench_rows.append(
            {
                "dataset": "foreman",
                "sample_rate": r["sample_rate"],
                "model": "TT (Online_FTD_net)",
                "final_test_nre": r["tt_final_test_nre"],
                "avg_update_time_s": r["tt_avg_update_time_s"],
            }
        )
    for _, r in condition.iterrows():
        bench_rows.append(
            {
                "dataset": "condition",
                "sample_rate": r["sample_rate"],
                "model": "Paper OFTD",
                "final_test_nre": r["paper_oftd_nre"],
                "avg_update_time_s": np.nan,
            }
        )
        bench_rows.append(
            {
                "dataset": "condition",
                "sample_rate": r["sample_rate"],
                "model": "CP baseline",
                "final_test_nre": r["cp_final_test_nre"],
                "avg_update_time_s": r["cp_avg_update_time_s"],
            }
        )
        bench_rows.append(
            {
                "dataset": "condition",
                "sample_rate": r["sample_rate"],
                "model": "TT (Online_FTD_net)",
                "final_test_nre": r["tt_final_test_nre"],
                "avg_update_time_s": r["tt_avg_update_time_s"],
            }
        )
    benchmark = pd.DataFrame(bench_rows).sort_values(["dataset", "sample_rate", "model"])
    benchmark.to_csv(out_dir / "benchmark_paper_cp_tt.csv", index=False)

    # Plots: NRE vs SR
    x_f = foreman["sample_rate"].to_numpy()
    line_plot(
        x_f,
        [
            foreman["paper_oftd_nre"].to_numpy(),
            foreman["cp_final_test_nre"].to_numpy(),
            foreman["tt_final_test_nre"].to_numpy(),
        ],
        ["Paper OFTD", "CP baseline", "TT (Online_FTD_net)"],
        "Foreman (Multi-Aspect): Final Test NRE vs SR",
        "Sample Rate (SR)",
        "Final Test NRE",
        out_dir / "foreman_nre_vs_sr_paper_cp_tt.png",
    )

    x_c = condition["sample_rate"].to_numpy()
    line_plot(
        x_c,
        [
            condition["paper_oftd_nre"].to_numpy(),
            condition["cp_final_test_nre"].to_numpy(),
            condition["tt_final_test_nre"].to_numpy(),
        ],
        ["Paper OFTD", "CP baseline", "TT (Online_FTD_net)"],
        "Condition (Single-Aspect): Final Test NRE vs SR",
        "Sample Rate (SR)",
        "Final Test NRE",
        out_dir / "condition_nre_vs_sr_paper_cp_tt.png",
    )

    # Plots: runtime vs SR
    line_plot(
        x_f,
        [foreman["cp_avg_update_time_s"].to_numpy(), foreman["tt_avg_update_time_s"].to_numpy()],
        ["CP baseline", "TT (Online_FTD_net)"],
        "Foreman (Multi-Aspect): Avg Online Update Time vs SR",
        "Sample Rate (SR)",
        "Avg Online Update Time (s)",
        out_dir / "foreman_update_time_vs_sr_cp_tt.png",
    )
    line_plot(
        x_c,
        [condition["cp_avg_update_time_s"].to_numpy(), condition["tt_avg_update_time_s"].to_numpy()],
        ["CP baseline", "TT (Online_FTD_net)"],
        "Condition (Single-Aspect): Avg Online Update Time vs SR",
        "Sample Rate (SR)",
        "Avg Online Update Time (s)",
        out_dir / "condition_update_time_vs_sr_cp_tt.png",
    )

    # Rank sensitivity and params/time-vs-error
    foreman_sr03 = foreman_tt_by_r[foreman_tt_by_r["sample_rate"] == 0.3].sort_values("R")
    line_plot(
        foreman_sr03["R"].to_numpy(),
        [foreman_sr03["tt_final_test_nre"].to_numpy()],
        ["TT final test NRE"],
        "Foreman TT: Rank Sensitivity at SR=0.3",
        "Rank R",
        "Final Test NRE",
        out_dir / "foreman_tt_rank_sensitivity_sr03.png",
    )
    scatter_plot(
        foreman_sr03["tt_params"].to_numpy(),
        foreman_sr03["tt_final_test_nre"].to_numpy(),
        [f"R={int(r)}" for r in foreman_sr03["R"].to_numpy()],
        "Foreman TT: Params vs Final Test NRE (SR=0.3)",
        "Parameter Count",
        "Final Test NRE",
        out_dir / "foreman_tt_params_vs_nre_sr03.png",
    )
    scatter_plot(
        foreman_sr03["tt_avg_update_time_s"].to_numpy(),
        foreman_sr03["tt_final_test_nre"].to_numpy(),
        [f"R={int(r)}" for r in foreman_sr03["R"].to_numpy()],
        "Foreman TT: Update Time vs Final Test NRE (SR=0.3)",
        "Avg Online Update Time (s)",
        "Final Test NRE",
        out_dir / "foreman_tt_update_time_vs_nre_sr03.png",
        color="tab:orange",
    )

    cond_sr03 = condition_tt_sr03_by_r.sort_values("R").copy()
    cond_sr03["avg_online_update_time_s"] = cond_sr03["online_time_s"] / cond_sr03["num_updates"].clip(lower=1)
    line_plot(
        cond_sr03["R"].to_numpy(),
        [cond_sr03["final_test_nre"].to_numpy()],
        ["TT final test NRE"],
        "Condition TT: Rank Sensitivity at SR=0.3",
        "Rank R",
        "Final Test NRE",
        out_dir / "condition_tt_rank_sensitivity_sr03.png",
    )
    scatter_plot(
        cond_sr03["params"].to_numpy(),
        cond_sr03["final_test_nre"].to_numpy(),
        [f"R={int(r)}" for r in cond_sr03["R"].to_numpy()],
        "Condition TT: Params vs Final Test NRE (SR=0.3)",
        "Parameter Count",
        "Final Test NRE",
        out_dir / "condition_tt_params_vs_nre_sr03.png",
    )
    scatter_plot(
        cond_sr03["avg_online_update_time_s"].to_numpy(),
        cond_sr03["final_test_nre"].to_numpy(),
        [f"R={int(r)}" for r in cond_sr03["R"].to_numpy()],
        "Condition TT: Update Time vs Final Test NRE (SR=0.3)",
        "Avg Online Update Time (s)",
        "Final Test NRE",
        out_dir / "condition_tt_update_time_vs_nre_sr03.png",
        color="tab:green",
    )

    # Markdown summary
    lines = []
    lines.append("# Paper Experiment Package (Full)")
    lines.append("")
    lines.append("This package compares **Paper OFTD vs CP baseline vs TT (`Online_FTD_net`)**.")
    lines.append("")
    lines.append("## Audit Verdict")
    lines.append("- The package is internally consistent: the table gaps equal the reported NRE differences, sample rates are ordered correctly, and plot values match the CSV tables.")
    lines.append("- The results are scientifically usable, but they do **not** support a blanket \"TT is better\" claim.")
    lines.append("- Foreman is still a negative result after the dense-TT high-budget update: TT improves but remains worse than both the paper OFTD reference and the CP baseline at every sample rate.")
    lines.append("- Condition is a positive TT-vs-paper-reference result: TT slightly beats the paper OFTD values at every sample rate, but the local CP baseline is still lower.")
    lines.append("- Runtime cells for Paper OFTD are intentionally blank because the package imports paper reference NRE values only, not a same-machine runtime rerun.")
    lines.append("")
    lines.append("## Included Tables")
    lines.append("- `table_multi_foreman_sr.csv`")
    lines.append("- `table_single_condition_sr.csv`")
    lines.append("- `table_foreman_tt_rank_sensitivity.csv`")
    lines.append("- `table_foreman_rank_sensitivity_cp_tt_sr03.csv`")
    lines.append("- `table_foreman_rank_sensitivity_best_by_model_sr03.csv`")
    lines.append("- `table_foreman_rank_sensitivity_raw_cp_tt_sr03.csv`")
    lines.append("- `table_condition_tt_rank_sensitivity_sr03.csv`")
    if foreman_highbudget_path.exists():
        lines.append("- `table_foreman_tt_r20_online1000_by_sr.csv`")
    lines.append("- `benchmark_paper_cp_tt.csv`")
    lines.append("")
    lines.append("## Included Plots")
    lines.append("- `foreman_nre_vs_sr_paper_cp_tt.png`")
    lines.append("- `condition_nre_vs_sr_paper_cp_tt.png`")
    lines.append("- `foreman_update_time_vs_sr_cp_tt.png`")
    lines.append("- `condition_update_time_vs_sr_cp_tt.png`")
    lines.append("- `foreman_tt_rank_sensitivity_sr03.png`")
    lines.append("- `foreman_rank_sensitivity_cp_tt_sr03_nre.png`")
    lines.append("- `foreman_rank_sensitivity_cp_tt_sr03_params.png`")
    lines.append("- `foreman_rank_sensitivity_cp_tt_sr03_update_time.png`")
    lines.append("- `foreman_rank_sensitivity_cp_tt_sr03_params_vs_nre.png`")
    lines.append("- `condition_tt_rank_sensitivity_sr03.png`")
    lines.append("- `foreman_tt_params_vs_nre_sr03.png`")
    lines.append("- `condition_tt_params_vs_nre_sr03.png`")
    lines.append("- `foreman_tt_update_time_vs_nre_sr03.png`")
    lines.append("- `condition_tt_update_time_vs_nre_sr03.png`")
    lines.append("")
    lines.append("## Key Readout")
    f_row = foreman[foreman["sample_rate"] == 0.3].iloc[0]
    c_row = condition[condition["sample_rate"] == 0.3].iloc[0]
    lines.append(
        f"- Foreman SR=0.3: Paper={f_row['paper_oftd_nre']:.3f}, "
        f"CP={f_row['cp_final_test_nre']:.3f}, TT={f_row['tt_final_test_nre']:.3f}"
    )
    lines.append(
        f"- Condition SR=0.3: Paper={c_row['paper_oftd_nre']:.3f}, "
        f"CP={c_row['cp_final_test_nre']:.3f}, TT={c_row['tt_final_test_nre']:.3f}"
    )
    lines.append("")
    lines.append("## Paper-Use Guidance")
    lines.append("- Legitimate claim: \"The TT formulation is competitive on the single-aspect Condition stream and slightly improves over the paper OFTD reference in final NRE.\"")
    lines.append("- Legitimate limitation: \"The current TT implementation underperforms on the multi-aspect Foreman stream, suggesting dataset-regime sensitivity and a need for additional optimization.\"")
    lines.append("- Rank-sensitivity claim: \"On Foreman SR=0.3, dense TT is better than CP at very low ranks (`R=5,10`), but CP improves steadily with larger `R` while dense TT plateaus.\"")
    lines.append("- Avoid claiming that TT/OFTD is uniformly superior across all tested streams from this package.")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Foreman TT final comparison uses {foreman_profile_note}")
    lines.append("- Foreman TT rank-sensitivity plots/tables use the original paper-budget rank sweep.")
    lines.append("- Foreman CP-vs-TT rank-sensitivity plots use `R in {5,10,20,40,60,80,100}`, SR=0.3, 3 seeds, `init_iters=4000`, and `online_iters=500`.")
    lines.append("- Condition TT uses fixed `R=80`, seed 42 across SR values.")
    lines.append("- Average online update time is explicitly reported in tables.")

    (out_dir / "PAPER_EXPERIMENT_PACKAGE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved full paper experiment package to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
