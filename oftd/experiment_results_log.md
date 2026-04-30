# OFTD Experiment Results Log

Last updated: April 19, 2026  
Environment: Windows, CPU-only, PyTorch in local `.venv`

## Supersession Note (April 29, 2026)

The high-budget package and follow-up Foreman failure analysis supersede the paper-use guidance in this older log:

- `paper_experiment_package/EXPERIMENT_AUDIT.md`
- `TT_FOREMAN_FAILURE_ANALYSIS.md`

Updated interpretation:

- Foreman multi-aspect is **not** current positive TT evidence; dense TT underperforms CP and the paper OFTD reference.
- Condition single-aspect is the current positive TT-vs-paper-reference result, although the local CP baseline remains lower.
- The Foreman gap should be treated as an optimization/parameterization problem, not as a theoretical TT performance ceiling.

## A. Baseline Demos (Existing Pipeline)

### 1. `OFTD_single_demo.py` (`data/condition.mat`)
- Long run was executed and reached full online stage.
- Reported final summary:
  - FLOPs: `12.86 M`
  - Average NRE train: `0.0518`
  - Average NRE test: `0.0783`
  - Average time cost: `0.2851 s/step`

Interpretation: baseline single-stream pipeline remains strong on `condition`.

## B. Theory-Aligned FTD Path (`OFTD_FTD_demo.py`)

### 1. Multi-aspect (`data/foreman.mat`)

Common settings:
- `--init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon`
- Seeds tested: `42`, `7`

Results:

| Config | Mean NRE test | Mean boundary error |
|---|---:|---:|
| `--boundary-lambda 5.0` | `0.1773` | `3.33%` |
| `--boundary-lambda 0.0` | `0.1698` | N/A |
| `--boundary-lambda 5.0 --deriv-lambda 0.01 --kappa 50` | `0.1911` | `3.94%` |

Notes:
- Boundary regularization keeps boundary error low (<5% in these runs).
- Derivative regularization reduces derivative proxy strongly, but worsens NRE.

### 2. Single-aspect mode on `condition` (`--streaming-mode single`)

Status:
- FTD formulation is currently unstable/underperforming on this dataset regime.
- NRE remains poor in tested FTD runs, despite loop and setup fixes.
- This appears to be a model/theory-fit limitation for this case, not just a data loader issue (baseline `OFTD_single_demo.py` works well).

## C. Fixes Applied During Evaluation

To support robust theory-aligned testing:
- Added reconstruction-loss normalization and relative boundary regularization in `online_update_multi_ftd`.
- Added derivative regularization controls (`deriv_lambda`, `kappa`).
- Fixed checkpoint-restore edge cases:
  - initialize `best_params` up-front
  - ignore non-finite losses for checkpoint updates
  - `load_state_dict(..., strict=False)` to avoid THOP buffer key mismatch
- Added proper single-aspect growth mode in `OFTD_FTD_demo.py` (full spatial initialization, temporal growth only).

## D. Commands Used (Representative)

```powershell
cd c:\Users\T10Z006\VS_projects\learning\oftd

# Existing baseline
python OFTD_single_demo.py

# FTD multi-aspect (recommended current setting)
python OFTD_FTD_demo.py --data data/foreman.mat --init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 5.0

# FTD multi-aspect + derivative control
python OFTD_FTD_demo.py --data data/foreman.mat --init-iters 300 --online-iters 80 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 5.0 --deriv-lambda 0.01 --kappa 50

# FTD single-aspect mode trial
python OFTD_FTD_demo.py --data data/condition.mat --streaming-mode single --single-c-init 5 --delta-c 100 --omega-a 0.3 --omega-b 0.3 --omega-c 0.3 --init-iters 300 --online-iters 40 --patience 20 --lr 1e-3 --normalize-recon --boundary-lambda 0.0
```

## E. Paper-Use Guidance

- Superseded by the April 29 audit and Foreman failure analysis.
- Do not use this older guidance for the paper experiments section.
