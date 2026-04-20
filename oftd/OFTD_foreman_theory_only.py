import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from model import Online_CP_multi_net, Online_FTD_net
from utils import calcu_nre, max_update, read_data, sample, dtype, device


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one_model(model_name, seed, args, data_pack):
    X_train, X_test, X_val, X, mask_train, mask_test, mask_val = data_pack

    p = args.init_ratio
    A_ini = math.floor(p * X.shape[0])
    B_ini = math.floor(p * X.shape[1])
    C_ini = math.floor(p * X.shape[2])
    A_delta = math.floor(p * X.shape[0])
    B_delta = math.floor(p * X.shape[1])
    C_delta = math.floor(p * X.shape[2])

    A_t, B_t, C_t = A_ini, B_ini, C_ini

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_val = X_val[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]
    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_val = mask_val[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]

    set_seed(seed)
    if model_name == 'cp_multi':
        model = Online_CP_multi_net(
            R1=args.rank,
            R2=args.rank,
            R3=args.rank,
            mid_channel=args.mid_channel,
            omega_A=args.omega_a,
            omega_B=args.omega_b,
            omega_C=args.omega_c,
        ).to(device)
    elif model_name == 'ftd':
        model = Online_FTD_net(
            R=args.rank,
            R1=args.rank,
            R2=args.rank,
            mid_channel=args.mid_channel,
            omega_A=args.omega_a,
            omega_B=args.omega_b,
            omega_C=args.omega_c,
            w_init=args.w_init,
        ).to(device)
    else:
        raise ValueError(model_name)

    A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
    B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
    C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initial stage (paper-style)
    best_nre_val = float('inf')
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait = 0
    t0_init = time.perf_counter()
    for _ in range(args.init_iters):
        optimizer.zero_grad()
        out = model(A_input, B_input, C_input)
        loss = ((out * mask_t_train - X_t * mask_t_train) ** 2).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nre_val = calcu_nre(X_t_val, out, mask_t_val).item()
        if nre_val < best_nre_val - 1e-6:
            best_nre_val = nre_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    init_time = time.perf_counter() - t0_init
    model.load_state_dict(best_state, strict=False)

    # Online stage (single shared optimization path for both models).
    alpha_beta = [args.alpha, args.beta]
    nres_train = []
    nres_test = []

    A_T, B_T, C_T = X_train.shape
    steps = int(max_update(A_T, B_T, C_T, A_ini, B_ini, C_ini, A_delta, B_delta, C_delta))
    t0_online = time.perf_counter()

    for _ in range(steps):
        A_t = min(A_t + A_delta, A_T)
        B_t = min(B_t + B_delta, B_T)
        C_t = min(C_t + C_delta, C_T)

        X_t = X_train[:A_t, :B_t, :C_t]
        X_t_test = X_test[:A_t, :B_t, :C_t]
        mask_t_train = mask_train[:A_t, :B_t, :C_t]
        mask_t_test = mask_test[:A_t, :B_t, :C_t]

        A_input = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
        B_input = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
        C_input = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)

        new_data_ind_A = np.arange(max(A_t - A_delta, 0), A_t)
        new_data_ind_B = np.arange(max(B_t - B_delta, 0), B_t)
        new_data_ind_C = np.arange(max(C_t - C_delta, 0), C_t)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float('inf')
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        for _ in range(args.online_iters):
            indexes_A = sample(alpha_beta, divide=args.divide, t=A_t)
            indexes_B = sample(alpha_beta, divide=args.divide, t=B_t)
            indexes_C = sample(alpha_beta, divide=args.divide, t=C_t)

            ind_A = np.concatenate([indexes_A, new_data_ind_A], axis=0)
            ind_B = np.concatenate([indexes_B, new_data_ind_B], axis=0)
            ind_C = np.concatenate([indexes_C, new_data_ind_C], axis=0)

            mask_here = mask_t_train[:, :, ind_C]
            mask_here = mask_here[:, ind_B, :]
            mask_here = mask_here[ind_A, :, :]
            X_here = X_t[:, :, ind_C]
            X_here = X_here[:, ind_B, :]
            X_here = X_here[ind_A, :, :]

            optimizer.zero_grad()
            A_here = torch.from_numpy(ind_A).type(dtype).to(device).unsqueeze(-1)
            B_here = torch.from_numpy(ind_B).type(dtype).to(device).unsqueeze(-1)
            C_here = torch.from_numpy(ind_C).type(dtype).to(device).unsqueeze(-1)
            out = model(A_here, B_here, C_here)
            loss = ((out * mask_here - X_here * mask_here) ** 2).sum()
            loss.backward()
            optimizer.step()

            lv = loss.item()
            if math.isfinite(lv) and lv < best_loss:
                best_loss = lv
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state, strict=False)
        with torch.no_grad():
            out_full = model(A_input, B_input, C_input)
            nre_train = calcu_nre(X_t, out_full, mask_t_train).item()
            nre_test = calcu_nre(X_t_test, out_full, mask_t_test).item()
        nres_train.append(nre_train)
        nres_test.append(nre_test)

    online_time = time.perf_counter() - t0_online

    # Final evaluation on current available tensor size
    with torch.no_grad():
        A_input_f = torch.arange(A_t, dtype=dtype, device=device).reshape(A_t, 1)
        B_input_f = torch.arange(B_t, dtype=dtype, device=device).reshape(B_t, 1)
        C_input_f = torch.arange(C_t, dtype=dtype, device=device).reshape(C_t, 1)
        out_final = model(A_input_f, B_input_f, C_input_f)
        final_test_nre = calcu_nre(
            X_test[:A_t, :B_t, :C_t],
            out_final,
            mask_test[:A_t, :B_t, :C_t],
        ).item()

    return {
        'dataset': Path(args.data).name,
        'seed': seed,
        'model': model_name,
        'R': args.rank,
        'params': count_params(model),
        'init_iters': args.init_iters,
        'online_iters': args.online_iters,
        'avg_online_nre_train': float(np.mean(nres_train)),
        'avg_online_nre_test': float(np.mean(nres_test)),
        'final_test_nre': float(final_test_nre),
        'init_time_s': float(init_time),
        'online_time_s': float(online_time),
        'total_train_time_s': float(init_time + online_time),
        'steps': int(steps),
        'final_shape': f'({int(A_t)},{int(B_t)},{int(C_t)})',
    }


def main():
    parser = argparse.ArgumentParser(description='Theory-only Foreman attribution: CP vs FTD (matched protocol)')
    parser.add_argument('--data', type=str, default='data/foreman.mat')
    parser.add_argument('--sample-rate', type=float, default=0.3)
    parser.add_argument('--seeds', type=str, default='42,7,123')
    parser.add_argument('--rank', type=int, default=100)
    parser.add_argument('--init-ratio', type=float, default=0.1)
    parser.add_argument('--mid-channel', type=int, default=128)
    parser.add_argument('--omega-a', type=float, default=1.5)
    parser.add_argument('--omega-b', type=float, default=1.5)
    parser.add_argument('--omega-c', type=float, default=0.6)
    parser.add_argument('--w-init', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--divide', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=10e-8)
    parser.add_argument('--init-iters', type=int, default=4000)
    parser.add_argument('--online-iters', type=int, default=500)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--out-csv', type=str, default='foreman_theory_only_attribution.csv')
    args = parser.parse_args()

    rows = []
    for seed in parse_int_list(args.seeds):
        set_seed(seed)
        data_pack = read_data(data=args.data, sample_rate=args.sample_rate)

        for model_name in ['cp_multi', 'ftd']:
            row = run_one_model(model_name, seed, args, data_pack)
            rows.append(row)
            print(
                f"[{model_name}] seed={seed} R={args.rank} "
                f"avg_nre_test={row['avg_online_nre_test']:.4f} "
                f"final_nre_test={row['final_test_nre']:.4f} "
                f"steps={row['steps']} shape={row['final_shape']} "
                f"train_s={row['total_train_time_s']:.2f}"
            )

    out_path = Path(args.out_csv)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {out_path.resolve()}')


if __name__ == '__main__':
    main()
