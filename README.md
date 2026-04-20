# Online Functional Tensor Decomposition (OFTD)

This repository contains a working clone/refactor of OFTD with:
- TT-based single-stream model (`Online_CP_single_net`)
- CP-style multi-stream model (`Online_CP_multi_net`)
- A new theory-aligned FTD path (`Online_FTD_net`, `OFTD_FTD_demo.py`)

Original paper: **Online Functional Tensor Decomposition via Continual Learning for Streaming Data Completion** (NeurIPS 2025)  
Original repo: https://github.com/20185zx/OFTD

## Repository Layout

```text
.
├── oftd/
│   ├── model.py
│   ├── utils.py
│   ├── affine.py
│   ├── OFTD_single_demo.py
│   ├── OFTD_multi_demo.py
│   ├── OFTD_single_affine_demo.py
│   ├── OFTD_FTD_demo.py
│   ├── data/
│   └── experiment_results_log.md
├── oftd_backup_original/
├── REFACTORING_SUMMARY.md
├── REFACTORING_CODE_DETAILS.md
└── README.md
```

## Quick Start

```powershell
cd oftd
python OFTD_single_demo.py
python OFTD_multi_demo.py
python OFTD_FTD_demo.py --quick
```

## New Theory-Aligned FTD Path

Main files:
- `oftd/model.py`: `Online_FTD_net`, `online_update_multi_ftd`, `check_ftd_theory_alignment`
- `oftd/OFTD_FTD_demo.py`: configurable experiment driver for theory diagnostics
- `REFACTORING_CODE_DETAILS.md`: code-level implementation notes for `Online_FTD_net`

Useful options:
- `--normalize-recon`: normalize reconstruction loss by observed entries
- `--boundary-lambda`: boundary-invariance regularization strength
- `--deriv-lambda --kappa`: derivative regularization (assumption control)
- `--streaming-mode single|multi`: growth pattern

Example (best current multi-aspect setting on `foreman`):

```powershell
cd oftd
python OFTD_FTD_demo.py --init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 5.0
```

## Current Status (April 19, 2026)

- `foreman` (multi-aspect, FTD path): stable and promising
  - avg test NRE around `0.17` to `0.19` in tested seeds/configs
  - boundary relative error can be brought below `5%`
- `condition` (single-aspect, FTD path): unstable/poor with current FTD formulation
  - this appears to be a model/theory fit issue for this regime, not only a code bug
- baseline single-stream demo (`OFTD_single_demo.py`) still performs well on `condition`

See detailed logs in `oftd/experiment_results_log.md`.

## Notes

- Affine path currently assumes GPU-specific behavior in parts of the code.
- For publication use, rerun all experiments with fixed seeds and exported logs/CSV.
