import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import optim

from model import Online_CP_multi_net, Online_FTD_net
from utils import calcu_nre, read_data, dtype, device


@dataclass
class StepSchedule:
    A_t: int
    B_t: int
    C_t: int
    inds_per_iter: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_beta(rng: np.random.RandomState, alpha_beta, divide: int, t: int):
    size = t // divide
    if size <= 0:
        return np.array([], dtype=int)
    idx = rng.beta(alpha_beta[0], alpha_beta[1], size=size) * (t - 1)
    return np.floor(idx).astype(int)


def build_schedule(
    shape,
    init_ratio=0.1,
    alpha_beta=(1.0, 1.2),
    divide=3,
    every_iter=40,
    seed=42,
):
    rng = np.random.RandomState(seed)
    A_T, B_T, C_T = shape
    A_t = max(1, math.floor(init_ratio * A_T))
    B_t = max(1, math.floor(init_ratio * B_T))
    C_t = max(1, math.floor(init_ratio * C_T))
    A_delta = max(1, math.floor(init_ratio * A_T))
    B_delta = max(1, math.floor(init_ratio * B_T))
    C_delta = max(1, math.floor(init_ratio * C_T))

    steps = []
    while A_t < A_T or B_t < B_T or C_t < C_T:
        prev_A, prev_B, prev_C = A_t, B_t, C_t
        A_t = min(A_t + A_delta, A_T)
        B_t = min(B_t + B_delta, B_T)
        C_t = min(C_t + C_delta, C_T)
        new_A = np.arange(max(A_t - A_delta, 0), A_t)
        new_B = np.arange(max(B_t - B_delta, 0), B_t)
        new_C = np.arange(max(C_t - C_delta, 0), C_t)

        inds_per_iter = []
        for _ in range(every_iter):
            idxA = sample_beta(rng, alpha_beta, divide, A_t)
            idxB = sample_beta(rng, alpha_beta, divide, B_t)
            idxC = sample_beta(rng, alpha_beta, divide, C_t)
            ind_A = np.concatenate([idxA, new_A], axis=0).astype(int)
            ind_B = np.concatenate([idxB, new_B], axis=0).astype(int)
            ind_C = np.concatenate([idxC, new_C], axis=0).astype(int)
            inds_per_iter.append((ind_A, ind_B, ind_C))

        steps.append(StepSchedule(A_t=A_t, B_t=B_t, C_t=C_t, inds_per_iter=inds_per_iter))

        if A_t == prev_A and B_t == prev_B and C_t == prev_C:
            break
    return steps


def train_model_with_schedule(
    model,
    X_train,
    X_test,
    X_val,
    mask_train,
    mask_test,
    mask_val,
    schedule,
    init_ratio=0.1,
    init_iters=200,
    lr=1e-3,
    weight_decay=1e-8,
):
    A_T, B_T, C_T = X_train.shape
    A_t = max(1, math.floor(init_ratio * A_T))
    B_t = max(1, math.floor(init_ratio * B_T))
    C_t = max(1, math.floor(init_ratio * C_T))

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_val = X_val[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]
    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_val = mask_val[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
    B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
    C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_start = time.perf_counter()
    for _ in range(init_iters):
        opt.zero_grad()
        out = model(A_input, B_input, C_input)
        loss = ((out * mask_t_train - X_t * mask_t_train) ** 2).sum() / mask_t_train.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
        with torch.no_grad():
            nre_val = calcu_nre(X_t_val, out, mask_t_val).item()
            if nre_val < best_val:
                best_val = nre_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_time = time.perf_counter() - init_start
    model.load_state_dict(best_state, strict=False)

    nre_tests = []
    online_time = 0.0
    for step in schedule:
        A_t, B_t, C_t = step.A_t, step.B_t, step.C_t
        X_t = X_train[:A_t, :B_t, :C_t]
        X_t_test = X_test[:A_t, :B_t, :C_t]
        mask_t_train = mask_train[:A_t, :B_t, :C_t]
        mask_t_test = mask_test[:A_t, :B_t, :C_t]

        A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
        B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
        C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)

        best_loss = float("inf")
        best_step_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        t0 = time.perf_counter()
        for ind_A, ind_B, ind_C in step.inds_per_iter:
            opt.zero_grad()
            A_here = torch.from_numpy(ind_A).type(dtype).to(device).unsqueeze(-1)
            B_here = torch.from_numpy(ind_B).type(dtype).to(device).unsqueeze(-1)
            C_here = torch.from_numpy(ind_C).type(dtype).to(device).unsqueeze(-1)
            out = model(A_here, B_here, C_here)

            m = mask_t_train[:, :, ind_C][:, ind_B, :][ind_A, :, :]
            xt = X_t[:, :, ind_C][:, ind_B, :][ind_A, :, :]
            loss = ((out * m - xt * m) ** 2).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            lv = loss.item()
            if math.isfinite(lv) and lv < best_loss:
                best_loss = lv
                best_step_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        online_time += time.perf_counter() - t0
        model.load_state_dict(best_step_state, strict=False)

        with torch.no_grad():
            out_full = model(A_input, B_input, C_input)
            nre_test = calcu_nre(X_t_test, out_full, mask_t_test).item()
            nre_tests.append(nre_test)

    # Final metrics
    with torch.no_grad():
        A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
        B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
        C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)
        out_final = model(A_input, B_input, C_input)
        final_test_nre = calcu_nre(X_test[:A_t, :B_t, :C_t], out_final, mask_test[:A_t, :B_t, :C_t]).item()

        # inference time
        for _ in range(3):
            _ = model(A_input, B_input, C_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tt = time.perf_counter()
        for _ in range(20):
            _ = model(A_input, B_input, C_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = (time.perf_counter() - tt) / 20.0

    return {
        "avg_online_nre_test": float(np.mean(nre_tests)) if nre_tests else float("nan"),
        "final_test_nre": final_test_nre,
        "init_time_s": init_time,
        "online_time_s": online_time,
        "total_train_time_s": init_time + online_time,
        "infer_time_s": infer_time,
        "steps": len(nre_tests),
        "final_shape": (int(A_t), int(B_t), int(C_t)),
    }


def run_suite(args):
    out_rows = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        set_seed(seed)
        X_train, X_test, X_val, X, mask_train, mask_test, mask_val = read_data(
            data=args.data, sample_rate=args.sample_rate
        )
        schedule = build_schedule(
            shape=X.shape,
            init_ratio=args.init_ratio,
            alpha_beta=(args.alpha, args.beta),
            divide=args.divide,
            every_iter=args.online_iters,
            seed=seed,
        )

        for model_name in ["cp_multi", "ftd"]:
            set_seed(seed)
            if model_name == "cp_multi":
                model = Online_CP_multi_net(
                    R1=args.rank, R2=args.rank, R3=args.rank,
                    mid_channel=args.mid_channel,
                    omega_A=args.omega_a, omega_B=args.omega_b, omega_C=args.omega_c
                ).to(device)
            else:
                model = Online_FTD_net(
                    R=args.rank, R1=args.rank, R2=args.rank,
                    mid_channel=args.mid_channel,
                    omega_A=args.omega_a, omega_B=args.omega_b, omega_C=args.omega_c,
                    w_init=args.w_init
                ).to(device)

            metrics = train_model_with_schedule(
                model=model,
                X_train=X_train, X_test=X_test, X_val=X_val,
                mask_train=mask_train, mask_test=mask_test, mask_val=mask_val,
                schedule=schedule,
                init_ratio=args.init_ratio,
                init_iters=args.init_iters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            row = {
                "dataset": Path(args.data).name,
                "seed": seed,
                "model": model_name,
                "R": args.rank,
                "params": count_params(model),
                **metrics,
            }
            out_rows.append(row)
            print(
                f"[{model_name}] seed={seed} R={args.rank} "
                f"avg_nre_test={row['avg_online_nre_test']:.4f} "
                f"final_nre_test={row['final_test_nre']:.4f} "
                f"train_s={row['total_train_time_s']:.2f}"
            )
    return out_rows


def save_csv(rows, out_csv):
    out_path = Path(out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Attribution suite: CP multi vs Online_FTD_net")
    parser.add_argument("--data", type=str, default="data/foreman.mat")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--seeds", type=str, default="42,7,123")
    parser.add_argument("--rank", type=int, default=40)
    parser.add_argument("--init-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--divide", type=int, default=3)
    parser.add_argument("--init-iters", type=int, default=200)
    parser.add_argument("--online-iters", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--mid-channel", type=int, default=128)
    parser.add_argument("--omega-a", type=float, default=1.5)
    parser.add_argument("--omega-b", type=float, default=1.5)
    parser.add_argument("--omega-c", type=float, default=0.6)
    parser.add_argument("--w-init", type=float, default=0.05)
    parser.add_argument("--out-csv", type=str, default="ftd_theory_attribution.csv")
    args = parser.parse_args()

    rows = run_suite(args)
    save_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
