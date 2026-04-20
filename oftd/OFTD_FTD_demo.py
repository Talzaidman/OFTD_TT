import argparse
import math
import random
import time
import torch
import numpy as np
from torch import optim
from utils import *
from model import Online_FTD_net, online_update_multi_ftd, check_ftd_theory_alignment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    print("Initial stage start")
    A_t, B_t, C_t = A_ini, B_ini, C_ini

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_val = X_val[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]
    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_val = mask_val[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]

    model = Online_FTD_net(
        R=args.rank,
        R1=args.r1,
        R2=args.r2,
        mid_channel=args.mid_channel,
        omega_A=args.omega_a,
        omega_B=args.omega_b,
        omega_C=args.omega_c,
        w_init=args.w_init,
    ).to(device)

    A_input = make_coords(A_t)
    B_input = make_coords(B_t)
    C_input = make_coords(C_t)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_nre_val = float("inf")
    best_state = None
    wait = 0
    for it in range(args.init_iters):
        optimizer.zero_grad()
        X_Out_real = model(A_input, B_input, C_input)
        loss = ((X_Out_real * mask_t_train - X_t * mask_t_train) ** 2).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nre_val = calcu_nre(X_t_val, X_Out_real, mask_t_val).item()
            nre_train = calcu_nre(X_t, X_Out_real, mask_t_train).item()
            nre_test = calcu_nre(X_t_test, X_Out_real, mask_t_test).item()
        if nre_val < best_nre_val - 1e-6:
            best_nre_val = nre_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Initial stage end, nre train: {nre_train:.4f}, nre test: {nre_test:.4f}")

    init_check = check_ftd_theory_alignment(
        model, A_t, B_t, C_t, delta=1.0, kappa=args.kappa if args.kappa > 0 else None
    )
    print("Theory check (after init):")
    print(
        f"  A std={init_check['A_weight_stats']['std']:.4f}, "
        f"B std={init_check['B_weight_stats']['std']:.4f}, "
        f"C std={init_check['C_weight_stats']['std']:.4f}"
    )
    print(
        f"  finite-diff |f'|_l1 max: A={init_check['dA_l1_max']:.4f}, "
        f"B={init_check['dB_l1_max']:.4f}, C={init_check['dC_l1_max']:.4f}"
    )

    print("Online update stage start")
    alpha_beta = [args.alpha, args.beta]
    flops_all = []
    nres_train = []
    nres_test = []
    boundary_errors = []
    start_all = time.perf_counter()

    A_T, B_T, C_T = X_train.shape
    max_update_num = max_update(A_T, B_T, C_T, A_ini, B_ini, C_ini, A_delta, B_delta, C_delta)
    for step in range(max_update_num):
        model, A_t, B_t, C_t, time_cost, nre_train, nre_test, boundary_rel = online_update_multi_ftd(
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
            profile_flops=args.profile_flops,
        )
        nres_train.append(nre_train)
        nres_test.append(nre_test)
        if math.isfinite(boundary_rel):
            boundary_errors.append(boundary_rel)
        print(
            f"time step: {step+1}/{max_update_num}, shape=({A_t},{B_t},{C_t}), "
            f"nre train: {nre_train:.4f}, nre test: {nre_test:.4f}, "
            f"boundary rel: {boundary_rel:.4%}, time: {time_cost:.2f}s"
        )

    end_all = time.perf_counter()
    avg_flops = round(np.mean(flops_all) / 1e6, 2) if flops_all else float("nan")
    avg_time = (end_all - start_all) / max(max_update_num, 1)
    avg_train = float(np.mean(nres_train)) if nres_train else float("nan")
    avg_test = float(np.mean(nres_test)) if nres_test else float("nan")
    avg_boundary = float(np.mean(boundary_errors)) if boundary_errors else float("nan")

    final_check = check_ftd_theory_alignment(
        model, A_t, B_t, C_t, delta=1.0, kappa=args.kappa if args.kappa > 0 else None
    )
    print("Online update stage end")
    print(
        f"FLOPs: {avg_flops} M, avg nre train: {avg_train:.4f}, "
        f"avg nre test: {avg_test:.4f}, avg time: {avg_time:.3f}s"
    )
    print(f"Average boundary relative error: {avg_boundary:.4%}")
    print(
        f"Theory check (final) finite-diff |f'|_l1 max: "
        f"A={final_check['dA_l1_max']:.4f}, B={final_check['dB_l1_max']:.4f}, C={final_check['dC_l1_max']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFTD theory-aligned FTD demo")
    parser.add_argument("--data", type=str, default="data/foreman.mat")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-ratio", type=float, default=0.1)
    parser.add_argument("--streaming-mode", type=str, default="multi", choices=["multi", "single"])
    parser.add_argument("--single-c-init", type=int, default=5)
    parser.add_argument("--delta-a", type=int, default=None)
    parser.add_argument("--delta-b", type=int, default=None)
    parser.add_argument("--delta-c", type=int, default=None)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--r1", type=int, default=100)
    parser.add_argument("--r2", type=int, default=100)
    parser.add_argument("--mid-channel", type=int, default=128)
    parser.add_argument("--omega-a", type=float, default=1.5)
    parser.add_argument("--omega-b", type=float, default=1.5)
    parser.add_argument("--omega-c", type=float, default=0.6)
    parser.add_argument("--w-init", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--divide", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--boundary-lambda", type=float, default=1e-3)
    parser.add_argument("--deriv-lambda", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=-1.0)
    parser.add_argument("--normalize-recon", action="store_true")
    parser.add_argument("--coord-mode", type=str, default="raw", choices=["raw", "zero_one", "minus_one_one"])
    parser.add_argument("--loss-scope", type=str, default="sampled", choices=["sampled", "full"])
    parser.add_argument("--init-iters", type=int, default=4000)
    parser.add_argument("--online-iters", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.init_iters = 25
        args.online_iters = 10
        args.patience = 5

    run_experiment(args)
