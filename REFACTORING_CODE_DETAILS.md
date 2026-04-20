# Refactoring Code Details

This document tracks code-level details for the current refactor state.

## Online_FTD_net (Theory-Aligned Path)

### Location
- `oftd/model.py`
- Class: `Online_FTD_net`

### What It Implements

`Online_FTD_net` implements the functional tensor form:

`X[i, j, k] = A(i)^T * B(j) * C(k)`

with:
- `A(i) in R^{R1}`
- `B(j) in R^{R1 x R2}`
- `C(k) in R^{R2}`

The forward contraction is implemented by:

```python
out = torch.einsum('ir,jrs,ks->ijk', A, B, C)
```

### Model Structure

- `A_net`: INR mapping scalar coordinate -> `R1` vector
- `B_net`: INR mapping scalar coordinate -> flattened `R1 * R2`, reshaped to matrix
- `C_net`: INR mapping scalar coordinate -> `R2` vector

All linear layers are initialized by normal distribution `N(0, w_init^2)` with zero bias through `_init_inr_weights`.

### Training Path

Used by:
- `oftd/OFTD_FTD_demo.py`

Core updater:
- `online_update_multi_ftd(...)` in `oftd/model.py`

Key controls in updater:
- `normalize_recon`: normalize reconstruction loss by observed entries
- `boundary_lambda`: boundary invariance regularization
- `deriv_lambda`, `kappa`: derivative regularization controls

### Diagnostics

`check_ftd_theory_alignment(...)` reports:
- weight stats (mean/std/min/max) for A/B/C blocks
- finite-difference derivative proxy (`|f'|_l1`)
- optional pass/fail flags when `kappa` is given

### Stability/Robustness Notes

The updater includes:
- best checkpoint cloning
- finite-loss guard before best-state update
- non-strict state restore to avoid THOP profiling buffer mismatch

### Current Known Behavior

- Multi-aspect (`foreman`) is stable with proper regularization.
- Single-aspect (`condition`) is currently weak under this FTD parameterization.
- Existing baseline (`OFTD_single_demo.py`) remains stronger on `condition`.
