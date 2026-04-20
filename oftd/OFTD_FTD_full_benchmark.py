import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path

from OFTD_FTD_sweep import run_one


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Full benchmark matrix for Online_FTD_net")
    parser.add_argument("--out-csv", type=str, default="ftd_full_benchmark.csv")
    parser.add_argument("--r-values", type=str, default="20,40,60,80,100")
    parser.add_argument("--seeds", type=str, default="42,7,123")
    args = parser.parse_args()

    r_values = parse_int_list(args.r_values)
    seeds = parse_int_list(args.seeds)

    base = {
        "sample_rate": 0.3,
        "init_ratio": 0.1,
        "delta_a": None,
        "delta_b": None,
        "delta_c": None,
        "single_c_init": 5,
        "mid_channel": 128,
        "w_init": 0.05,
        "alpha": 1.0,
        "beta": 1.2,
        "divide": 3,
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "init_iters": 200,
        "online_iters": 40,
        "patience": 20,
        "normalize_recon": True,
        "boundary_lambda": 5.0,
        "deriv_lambda": 0.0,
        "kappa": -1.0,
        "coord_mode": "raw",
        "loss_scope": "sampled",
        "profile_flops": False,
        "infer_repeats": 20,
    }

    sweep_matrix = [
        {
            "name": "foreman_multi_main",
            "data": "data/foreman.mat",
            "streaming_mode": "multi",
            "omega_a": 1.5,
            "omega_b": 1.5,
            "omega_c": 0.6,
            "r_values": r_values,
        },
        {
            "name": "condition_single_main",
            "data": "data/condition.mat",
            "streaming_mode": "single",
            "omega_a": 0.3,
            "omega_b": 0.3,
            "omega_c": 0.3,
            "delta_c": 1,
            "init_iters": 150,
            "online_iters": 20,
            "r_values": r_values,
        },
        {
            "name": "foreman_multi_cfg_no_boundary",
            "data": "data/foreman.mat",
            "streaming_mode": "multi",
            "omega_a": 1.5,
            "omega_b": 1.5,
            "omega_c": 0.6,
            "boundary_lambda": 0.0,
            "deriv_lambda": 0.0,
            "r_values": [40],
        },
        {
            "name": "foreman_multi_cfg_boundary_deriv",
            "data": "data/foreman.mat",
            "streaming_mode": "multi",
            "omega_a": 1.5,
            "omega_b": 1.5,
            "omega_c": 0.6,
            "boundary_lambda": 5.0,
            "deriv_lambda": 0.01,
            "kappa": 50.0,
            "r_values": [40],
        },
        {
            "name": "condition_single_cfg_no_boundary",
            "data": "data/condition.mat",
            "streaming_mode": "single",
            "omega_a": 0.3,
            "omega_b": 0.3,
            "omega_c": 0.3,
            "delta_c": 1,
            "init_iters": 150,
            "online_iters": 20,
            "boundary_lambda": 0.0,
            "deriv_lambda": 0.0,
            "r_values": [40],
        },
        {
            "name": "condition_single_cfg_boundary_deriv",
            "data": "data/condition.mat",
            "streaming_mode": "single",
            "omega_a": 0.3,
            "omega_b": 0.3,
            "omega_c": 0.3,
            "delta_c": 1,
            "init_iters": 150,
            "online_iters": 20,
            "boundary_lambda": 5.0,
            "deriv_lambda": 0.01,
            "kappa": 50.0,
            "r_values": [40],
        },
    ]

    all_rows = []
    t0_all = time.perf_counter()
    for spec in sweep_matrix:
        for seed in seeds:
            for rank in spec["r_values"]:
                cfg = deepcopy(base)
                cfg.update(spec)
                cfg["seed"] = seed
                cfg["rank"] = rank
                cfg["r1"] = rank
                cfg["r2"] = rank
                print(f"[run] {spec['name']} seed={seed} R={rank}")
                row = run_one(type("Args", (), cfg), rank, seed)
                row["config_name"] = spec["name"]
                all_rows.append(row)
                print(
                    f"  final_test_nre={row['final_test_nre']:.4f} "
                    f"final_test_loss={row['final_test_loss']:.6f} "
                    f"train_s={row['total_train_time_s']:.2f} "
                    f"infer_s={row['infer_time_s']:.6f}"
                )

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    total_s = time.perf_counter() - t0_all
    print(f"\nSaved benchmark matrix to: {out_path.resolve()}")
    print(f"Total runs: {len(all_rows)}, wall time: {total_s:.1f}s")


if __name__ == "__main__":
    main()
