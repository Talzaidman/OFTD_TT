# OFTD Code Guide

Core experiment scripts:
- `OFTD_single_demo.py`: single-aspect streaming baseline
- `OFTD_multi_demo.py`: multi-aspect streaming baseline
- `OFTD_FTD_demo.py`: theory-aligned FTD experiments and diagnostics

## Run

```powershell
python OFTD_single_demo.py
python OFTD_multi_demo.py
python OFTD_FTD_demo.py --quick
```

## FTD Demo Options

Examples:

```powershell
# Multi-aspect (foreman) strong baseline
python OFTD_FTD_demo.py --data data/foreman.mat --init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 5.0

# Single-aspect mode (temporal growth only)
python OFTD_FTD_demo.py --data data/condition.mat --streaming-mode single --single-c-init 5 --delta-c 100 --omega-a 0.3 --omega-b 0.3 --omega-c 0.3
```

Important flags:
- `--normalize-recon`: normalizes reconstruction loss by observed count
- `--boundary-lambda`: boundary invariance regularization
- `--deriv-lambda --kappa`: derivative regularization
- `--streaming-mode single|multi`: growth mode
- `--delta-a --delta-b --delta-c`: manual growth steps

## Diagnostics Printed by FTD Demo

- Weight initialization stats (`std`) for A/B/C INR blocks
- Finite-difference derivative proxy (`|f'|_l1 max`)
- Average test NRE
- Average boundary relative error

## Current Practical Recommendation

For `foreman`, start with:

```powershell
python OFTD_FTD_demo.py --init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 5.0
```

This currently gives stable reconstruction and low boundary error in tested runs.
