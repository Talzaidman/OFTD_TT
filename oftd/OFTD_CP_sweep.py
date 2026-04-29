import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from model import Online_CP_multi_net, Online_CP_single_net, online_update_multi, online_update_single
from utils import calcu_nre, dtype, device, max_update, read_data


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


def timed_inference_single(model, c_t, repeats=20, warmup=3):
    c_input = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(c_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(c_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / max(repeats, 1)


def timed_inference_multi(model, a_t, b_t, c_t, repeats=20, warmup=3):
    a_input = torch.arange(a_t, dtype=dtype, device=device).reshape(a_t, 1)
    b_input = torch.arange(b_t, dtype=dtype, device=device).reshape(b_t, 1)
    c_input = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(a_input, b_input, c_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(a_input, b_input, c_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / max(repeats, 1)


def run_single(args, seed: int):
    set_seed(seed)
    x_train, x_test, x_val, x, mask_train, mask_test, mask_val = read_data(
        data=args.data, sample_rate=args.sample_rate
    )

    c_ini = min(max(1, args.single_c_init), x.shape[2])
    c_delta = max(1, args.delta_c)
    c_t = c_ini

    x_t = x_train[:, :, :c_t]
    x_t_val = x_val[:, :, :c_t]
    x_t_test = x_test[:, :, :c_t]
    mask_t_train = mask_train[:, :, :c_t]
    mask_t_val = mask_val[:, :, :c_t]
    mask_t_test = mask_test[:, :, :c_t]

    model = Online_CP_single_net(
        x_t.shape[0],
        x_t.shape[1],
        R=args.rank,
        mid_channel=args.mid_channel,
        omega_0=args.omega_c,
    ).to(device)
    params = count_params(model)

    c_input = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_nre_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_loss_best = float("inf")
    wait = 0

    t0_init = time.perf_counter()
    for _ in range(args.init_iters):
        optimizer.zero_grad()
        x_out = model(c_input)
        loss = ((x_out * mask_t_train - x_t * mask_t_train) ** 2).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nre_val = calcu_nre(x_t_val, x_out, mask_t_val).item()
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

    c_t_total = x_train.shape[2]
    num_updates = max_update(x_train.shape[0], x_train.shape[1], c_t_total, x_train.shape[0], x_train.shape[1], c_ini, 0, 0, c_delta)
    for _ in range(num_updates):
        model, c_t, t_cost, nre_train, nre_test = online_update_single(
            alpha_beta,
            model,
            x_train,
            x_test,
            mask_train,
            mask_test,
            c_t,
            c_delta,
            divide=args.divide,
            flops_all=flops_all,
            every_iter=args.online_iters,
        )
        online_time += t_cost
        nres_train.append(nre_train)
        nres_test.append(nre_test)

    c_input_f = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    with torch.no_grad():
        x_final = model(c_input_f)
    x_train_f = x_train[:, :, :c_t]
    x_test_f = x_test[:, :, :c_t]
    mask_train_f = mask_train[:, :, :c_t]
    mask_test_f = mask_test[:, :, :c_t]

    final_train_loss = masked_mse(x_train_f, x_final, mask_train_f)
    final_test_loss = masked_mse(x_test_f, x_final, mask_test_f)
    final_train_nre = calcu_nre(x_train_f, x_final, mask_train_f).item()
    final_test_nre = calcu_nre(x_test_f, x_final, mask_test_f).item()
    infer_time = timed_inference_single(model, c_t, repeats=args.infer_repeats, warmup=3)

    return {
        "dataset": Path(args.data).name,
        "mode": "single",
        "seed": seed,
        "R": args.rank,
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
        "num_updates": int(num_updates),
        "final_shape": f"({int(x_train.shape[0])},{int(x_train.shape[1])},{int(c_t)})",
    }


def run_multi(args, seed: int):
    set_seed(seed)
    x_train, x_test, x_val, x, mask_train, mask_test, mask_val = read_data(
        data=args.data, sample_rate=args.sample_rate
    )

    p = args.init_ratio
    a_ini = max(1, math.floor(p * x.shape[0]))
    b_ini = max(1, math.floor(p * x.shape[1]))
    c_ini = max(1, math.floor(p * x.shape[2]))
    a_delta = max(1, math.floor(p * x.shape[0]))
    b_delta = max(1, math.floor(p * x.shape[1]))
    c_delta = max(1, math.floor(p * x.shape[2]))
    if args.delta_a is not None:
        a_delta = max(0, args.delta_a)
    if args.delta_b is not None:
        b_delta = max(0, args.delta_b)
    if args.delta_c is not None:
        c_delta = max(0, args.delta_c)

    a_t, b_t, c_t = a_ini, b_ini, c_ini
    x_t = x_train[:a_t, :b_t, :c_t]
    x_t_val = x_val[:a_t, :b_t, :c_t]
    x_t_test = x_test[:a_t, :b_t, :c_t]
    mask_t_train = mask_train[:a_t, :b_t, :c_t]
    mask_t_val = mask_val[:a_t, :b_t, :c_t]
    mask_t_test = mask_test[:a_t, :b_t, :c_t]

    model = Online_CP_multi_net(
        args.rank,
        args.rank,
        args.rank,
        args.mid_channel,
        omega_A=args.omega_a,
        omega_B=args.omega_b,
        omega_C=args.omega_c,
    ).to(device)
    params = count_params(model)

    a_input = torch.arange(a_t, dtype=dtype, device=device).reshape(a_t, 1)
    b_input = torch.arange(b_t, dtype=dtype, device=device).reshape(b_t, 1)
    c_input = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_nre_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    init_loss_best = float("inf")
    wait = 0

    t0_init = time.perf_counter()
    for _ in range(args.init_iters):
        optimizer.zero_grad()
        x_out = model(a_input, b_input, c_input)
        loss = ((x_out * mask_t_train - x_t * mask_t_train) ** 2).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nre_val = calcu_nre(x_t_val, x_out, mask_t_val).item()
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

    max_updates = max_update(
        x_train.shape[0], x_train.shape[1], x_train.shape[2],
        a_ini, b_ini, c_ini, a_delta, b_delta, c_delta
    )
    for _ in range(max_updates):
        model, a_t, b_t, c_t, t_cost, nre_train, nre_test = online_update_multi(
            alpha_beta,
            model,
            x_train,
            x_test,
            mask_train,
            mask_test,
            a_t,
            b_t,
            c_t,
            a_delta,
            b_delta,
            c_delta,
            divide=args.divide,
            flops_all=flops_all,
            every_iter=args.online_iters,
        )
        online_time += t_cost
        nres_train.append(nre_train)
        nres_test.append(nre_test)

    a_input_f = torch.arange(a_t, dtype=dtype, device=device).reshape(a_t, 1)
    b_input_f = torch.arange(b_t, dtype=dtype, device=device).reshape(b_t, 1)
    c_input_f = torch.arange(c_t, dtype=dtype, device=device).reshape(c_t, 1)
    with torch.no_grad():
        x_final = model(a_input_f, b_input_f, c_input_f)
    x_train_f = x_train[:a_t, :b_t, :c_t]
    x_test_f = x_test[:a_t, :b_t, :c_t]
    mask_train_f = mask_train[:a_t, :b_t, :c_t]
    mask_test_f = mask_test[:a_t, :b_t, :c_t]

    final_train_loss = masked_mse(x_train_f, x_final, mask_train_f)
    final_test_loss = masked_mse(x_test_f, x_final, mask_test_f)
    final_train_nre = calcu_nre(x_train_f, x_final, mask_train_f).item()
    final_test_nre = calcu_nre(x_test_f, x_final, mask_test_f).item()
    infer_time = timed_inference_multi(model, a_t, b_t, c_t, repeats=args.infer_repeats, warmup=3)

    return {
        "dataset": Path(args.data).name,
        "mode": "multi",
        "seed": seed,
        "R": args.rank,
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
        "final_shape": f"({int(a_t)},{int(b_t)},{int(c_t)})",
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep sample rates/seeds for CP baselines")
    parser.add_argument("--data", type=str, default="data/foreman.mat")
    parser.add_argument("--sample-rates", type=str, default="0.3")
    parser.add_argument("--streaming-mode", type=str, default="multi", choices=["multi", "single"])
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--mid-channel", type=int, default=128)
    parser.add_argument("--init-ratio", type=float, default=0.1)
    parser.add_argument("--single-c-init", type=int, default=5)
    parser.add_argument("--delta-a", type=int, default=None)
    parser.add_argument("--delta-b", type=int, default=None)
    parser.add_argument("--delta-c", type=int, default=1)
    parser.add_argument("--omega-a", type=float, default=1.5)
    parser.add_argument("--omega-b", type=float, default=1.5)
    parser.add_argument("--omega-c", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--divide", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--init-iters", type=int, default=4000)
    parser.add_argument("--online-iters", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--infer-repeats", type=int, default=20)
    parser.add_argument("--out-csv", type=str, default="cp_sweep_results.csv")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    sample_rates = [float(x.strip()) for x in args.sample_rates.split(",") if x.strip()]
    rows = []

    for sr in sample_rates:
        args.sample_rate = sr
        for seed in seeds:
            print(f"[run] sr={sr} seed={seed} mode={args.streaming_mode}")
            if args.streaming_mode == "single":
                row = run_single(args, seed)
            else:
                row = run_multi(args, seed)
            row["sample_rate"] = sr
            rows.append(row)
            print(
                f"  final_test_nre={row['final_test_nre']:.4f} "
                f"avg_online_nre_test={row['avg_online_nre_test']:.4f} "
                f"avg_update_time={row['online_time_s']/max(row['num_updates'],1):.4f}s "
                f"params={row['params']}"
            )

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CP sweep to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
