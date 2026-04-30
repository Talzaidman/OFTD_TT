import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import (
    Online_FTD_net,
    check_ftd_theory_alignment,
    make_ftd_optimizer,
    online_update_multi_ftd,
)
from utils import calcu_nre, max_update, read_data, dtype, device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_mse(x_true, x_pred, mask):
    obs = mask.sum().clamp_min(1.0)
    return (((x_true * mask - x_pred * mask) ** 2).sum() / obs).item()


def timed_inference(model, A_t, B_t, C_t, repeats=20, warmup=3):
    A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
    B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
    C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(A_input, B_input, C_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(A_input, B_input, C_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / max(repeats, 1)


def run_one(args, rank: int, seed: int):
    set_seed(seed)

    X_train, X_test, X_val, X, mask_train, mask_test, mask_val = read_data(
        data=args.data, sample_rate=args.sample_rate
    )

    def make_coords(size):
        vals = torch.arange(size, dtype=dtype, device=device)
        if args.coord_mode == "raw":
            return vals.reshape(size, 1)
        denom = max(size - 1, 1)
        if args.coord_mode == "zero_one":
            vals = vals / denom
        elif args.coord_mode == "minus_one_one":
            vals = 2.0 * (vals / denom) - 1.0
        else:
            raise ValueError(f"Unsupported coord_mode: {args.coord_mode}")
        return vals.reshape(size, 1)

    p = args.init_ratio
    A_ini = max(1, math.floor(p * X.shape[0]))
    B_ini = max(1, math.floor(p * X.shape[1]))
    C_ini = max(1, math.floor(p * X.shape[2]))
    A_delta = max(1, math.floor(p * X.shape[0]))
    B_delta = max(1, math.floor(p * X.shape[1]))
    C_delta = max(1, math.floor(p * X.shape[2]))

    if args.streaming_mode == "single":
        A_ini = X.shape[0]
        B_ini = X.shape[1]
        C_ini = min(max(1, args.single_c_init), X.shape[2])
        A_delta = 0
        B_delta = 0
        C_delta = max(1, args.delta_c if args.delta_c is not None else 1)
    else:
        if args.delta_a is not None:
            A_delta = max(0, args.delta_a)
        if args.delta_b is not None:
            B_delta = max(0, args.delta_b)
        if args.delta_c is not None:
            C_delta = max(0, args.delta_c)

    A_t, B_t, C_t = A_ini, B_ini, C_ini
    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_val = X_val[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]
    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_val = mask_val[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]

    model = Online_FTD_net(
        R=rank,
        R1=rank,
        R2=rank,
        mid_channel=args.mid_channel,
        omega_A=args.omega_a,
        omega_B=args.omega_b,
        omega_C=args.omega_c,
        w_init=args.w_init,
    ).to(device)
    params = count_params(model)

    A_input = make_coords(A_t)
    B_input = make_coords(B_t)
    C_input = make_coords(C_t)

    optimizer = make_ftd_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_a_mult=getattr(args, "lr_a_mult", 1.0),
        lr_b_mult=getattr(args, "lr_b_mult", 1.0),
        lr_c_mult=getattr(args, "lr_c_mult", 1.0),
    )

    best_nre_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_loss_best = float("inf")
    wait = 0
    t0_init = time.perf_counter()
    for _ in range(args.init_iters):
        optimizer.zero_grad()
        X_out = model(A_input, B_input, C_input)
        loss = ((X_out * mask_t_train - X_t * mask_t_train) ** 2).sum()
        loss.backward()
        init_clip = getattr(args, "init_clip_grad_norm", 0.0)
        if init_clip is not None and init_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=init_clip)
        optimizer.step()

        with torch.no_grad():
            nre_val = calcu_nre(X_t_val, X_out, mask_t_val).item()
            if nre_val < best_nre_val - 1e-6:
                best_nre_val = nre_val
                init_loss_best = loss.item()
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
    init_time = time.perf_counter() - t0_init
    model.load_state_dict(best_state, strict=False)

    alpha_beta = [args.alpha, args.beta]
    flops_all = []
    nres_train = []
    nres_test = []
    online_time = 0.0

    A_T, B_T, C_T = X_train.shape
    max_updates = max_update(A_T, B_T, C_T, A_ini, B_ini, C_ini, A_delta, B_delta, C_delta)
    reuse_online_optimizer = getattr(args, "reuse_online_optimizer", False)
    online_optimizer = None

    for _ in range(max_updates):
        update_result = online_update_multi_ftd(
            alpha_beta,
            model,
            X_train,
            X_test,
            mask_train,
            mask_test,
            A_t,
            B_t,
            C_t,
            A_delta,
            B_delta,
            C_delta,
            divide=args.divide,
            flops_all=flops_all,
            every_iter=args.online_iters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            boundary_lambda=args.boundary_lambda,
            deriv_lambda=args.deriv_lambda,
            kappa=args.kappa,
            normalize_recon=args.normalize_recon,
            coord_mode=args.coord_mode,
            loss_scope=args.loss_scope,
            profile_flops=getattr(args, "profile_flops", False),
            lr_a_mult=getattr(args, "lr_a_mult", 1.0),
            lr_b_mult=getattr(args, "lr_b_mult", 1.0),
            lr_c_mult=getattr(args, "lr_c_mult", 1.0),
            clip_grad_norm=getattr(args, "clip_grad_norm", 1.0),
            optimizer=online_optimizer,
            return_optimizer=reuse_online_optimizer,
        )
        if reuse_online_optimizer:
            model, A_t, B_t, C_t, t_cost, nre_train, nre_test, _, online_optimizer = update_result
        else:
            model, A_t, B_t, C_t, t_cost, nre_train, nre_test, _ = update_result
        online_time += t_cost
        nres_train.append(nre_train)
        nres_test.append(nre_test)

    # Final full-volume metrics at last online state
    A_input_f = make_coords(A_t)
    B_input_f = make_coords(B_t)
    C_input_f = make_coords(C_t)
    with torch.no_grad():
        X_final = model(A_input_f, B_input_f, C_input_f)
    X_train_f = X_train[:A_t, :B_t, :C_t]
    X_test_f = X_test[:A_t, :B_t, :C_t]
    mask_train_f = mask_train[:A_t, :B_t, :C_t]
    mask_test_f = mask_test[:A_t, :B_t, :C_t]

    final_train_loss = masked_mse(X_train_f, X_final, mask_train_f)
    final_test_loss = masked_mse(X_test_f, X_final, mask_test_f)
    final_train_nre = calcu_nre(X_train_f, X_final, mask_train_f).item()
    final_test_nre = calcu_nre(X_test_f, X_final, mask_test_f).item()
    infer_time = timed_inference(model, A_t, B_t, C_t, repeats=args.infer_repeats, warmup=3)
    diag = check_ftd_theory_alignment(model, A_t, B_t, C_t, delta=1.0)

    return {
        "dataset": Path(args.data).name,
        "mode": args.streaming_mode,
        "seed": seed,
        "R": rank,
        "params": params,
        "init_time_s": round(init_time, 4),
        "online_time_s": round(online_time, 4),
        "total_train_time_s": round(init_time + online_time, 4),
        "infer_time_s": round(infer_time, 6),
        "avg_online_nre_train": round(float(np.mean(nres_train)) if nres_train else float("nan"), 6),
        "avg_online_nre_test": round(float(np.mean(nres_test)) if nres_test else float("nan"), 6),
        "final_train_nre": round(final_train_nre, 6),
        "final_test_nre": round(final_test_nre, 6),
        "init_loss_best": round(float(init_loss_best), 6),
        "final_train_loss": round(final_train_loss, 6),
        "final_test_loss": round(final_test_loss, 6),
        "avg_flops_m": round(float(np.mean(flops_all) / 1e6), 4) if flops_all else float("nan"),
        "num_updates": int(max_updates),
        "final_shape": f"({int(A_t)},{int(B_t)},{int(C_t)})",
        "dA_l1_max": round(diag["dA_l1_max"], 6),
        "dB_l1_max": round(diag["dB_l1_max"], 6),
        "dC_l1_max": round(diag["dC_l1_max"], 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep R for Online_FTD_net")
    parser.add_argument("--data", type=str, default="data/foreman.mat")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--streaming-mode", type=str, default="multi", choices=["multi", "single"])
    parser.add_argument("--single-c-init", type=int, default=5)
    parser.add_argument("--init-ratio", type=float, default=0.1)
    parser.add_argument("--delta-a", type=int, default=None)
    parser.add_argument("--delta-b", type=int, default=None)
    parser.add_argument("--delta-c", type=int, default=None)
    parser.add_argument("--r-values", type=str, default="20,40,60,80,100")
    parser.add_argument("--seeds", type=str, default="42")
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
    parser.add_argument("--init-iters", type=int, default=300)
    parser.add_argument("--online-iters", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--normalize-recon", action="store_true")
    parser.add_argument("--coord-mode", type=str, default="raw", choices=["raw", "zero_one", "minus_one_one"])
    parser.add_argument("--loss-scope", type=str, default="sampled", choices=["sampled", "full"])
    parser.add_argument("--boundary-lambda", type=float, default=5.0)
    parser.add_argument("--deriv-lambda", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=-1.0)
    parser.add_argument("--infer-repeats", type=int, default=20)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--reuse-online-optimizer", action="store_true")
    parser.add_argument("--out-csv", type=str, default="ftd_sweep_results.csv")
    args = parser.parse_args()

    r_values = parse_int_list(args.r_values)
    seeds = parse_int_list(args.seeds)

    all_rows = []
    for seed in seeds:
        for rank in r_values:
            print(f"[run] seed={seed} R={rank}")
            row = run_one(args, rank, seed)
            all_rows.append(row)
            print(
                f"  final_test_nre={row['final_test_nre']:.4f} "
                f"avg_online_nre_test={row['avg_online_nre_test']:.4f} "
                f"train_time={row['total_train_time_s']:.2f}s "
                f"infer={row['infer_time_s']:.6f}s params={row['params']}"
            )

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved sweep results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
