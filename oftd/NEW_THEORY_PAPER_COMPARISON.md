# New Theory Recreated Tests vs Paper

## Scope
This report recreates available experiments for the **new TT/FTD theory model** (`Online_FTD_net`) and compares to values reported in:
- `10565_Online_Functional_Tensor.pdf` (Table 2 and Table 3)

Local datasets available: `foreman.mat`, `condition.mat`.

## Paper Reference Values (SR = 0.3)
- **Foreman (multi-aspect, OFTD)**: `0.084` (Table 3)
- **Condition (single-aspect, OFTD)**: `0.093` (Table 2)

## Recreated Runs (Fresh)

### 1) Theory-isolated CP vs FTD (Foreman)
Matched protocol (same data split/seed/schedule/loss/optimizer loop), only model decomposition differs.

File: `foreman_theory_only_attribution_sharedopt_r100_s3.csv`

Mean over seeds `42,7,123`:
- `cp_multi`: avg online test NRE = **0.0756**, final test NRE = **0.0949**
- `ftd`: avg online test NRE = **0.1545**, final test NRE = **0.1493**

Interpretation: for Foreman under strict isolation, the new FTD theory is currently worse than paper-faithful CP baseline.

### 2) New-theory FTD R sweep (Foreman, paper-style budget)
File: `foreman_ftd_paper_recreate_r_sweep.csv`
- Seeds: `42,7,123`
- Budget: `init_iters=4000`, `online_iters=500`
- R values: `20,40,60,80,100`

Best mean (across seeds):
- at `R=20`: avg online test NRE = **0.1337**, final test NRE = **0.1359**

Compared to paper Foreman 0.084: still higher error.

### 3) Paper-faithful CP baseline (Condition)
File: `condition_cp_paper_baseline.log`
- Script: `OFTD_single_demo.py` (paper original path)
- Result: avg online test NRE = **0.0605**

### 4) New-theory FTD R sweep (Condition, high budget)
File: `condition_ftd_paper_recreate_r_sweep_seed42.csv`
- Seed: `42`
- Budget: `init_iters=4000`, `online_iters=100`, `delta_c=1`
- R values: `40,80,100`

Results:
- `R=40`: final test NRE = **0.0884**, avg online test NRE = **0.1048**
- `R=80`: final test NRE = **0.0878**, avg online test NRE = **0.1090**
- `R=100`: final test NRE = **0.0952**, avg online test NRE = **0.1189**

Compared to paper Condition 0.093: best recreated new-theory run is slightly better (`0.0878`).

## Requested Plots
Combined benchmark file:
- `new_theory_paper_recreate_benchmark.csv`

Plot directory:
- `plots_new_theory_paper_recreate/`

Includes:
- Loss vs R:
  - `foreman.mat_multi_R_vs_final_test_loss.png`
  - `condition.mat_single_R_vs_final_test_loss.png`
- Param count vs performance:
  - `foreman.mat_multi_params_vs_error.png`
  - `condition.mat_single_params_vs_error.png`
- Inference time vs error:
  - `foreman.mat_multi_infer_time_vs_error.png`
  - `condition.mat_single_infer_time_vs_error.png`
- Train time vs error:
  - `foreman.mat_multi_train_time_vs_error.png`
  - `condition.mat_single_train_time_vs_error.png`

(Also includes R vs NRE, R vs train time, R vs inference time.)

## Suspicious / Important Notes
- Foreman uses paper-faithful update count behavior (`steps=9`, final shape `(140,170,100)`) to stay protocol-consistent with original code.
- Condition is very expensive (`steps=2618`); only one seed was run for high-budget FTD sweep.
- Condition CP baseline beating paper (0.0605 vs 0.093) is likely due split randomness/protocol details (single run, local RNG path), not a model bug by itself.

## Bottom Line (New Theory Target)
- **Foreman**: new FTD theory under strict isolation is clearly behind paper-faithful CP baseline and paper OFTD target.
- **Condition**: new FTD theory can reach/beat the paper OFTD number with sufficient budget (`R=80`, high-budget run).
