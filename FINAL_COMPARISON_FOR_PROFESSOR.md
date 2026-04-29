# Final Comparison: Paper OFTD vs New TT Theory (`Online_FTD_net`)

## Goal
Compare the original paper's reported OFTD results against the new TT-based theory implementation, using recreated local experiments.

## Reference Paper Values
- **Foreman (multi-aspect, Table 3, OFTD):**
  - `SR=0.1`: `0.094`
  - `SR=0.2`: `0.087`
  - `SR=0.3`: `0.084`
- **Condition (single-aspect, Table 2, OFTD):**
  - `SR=0.1`: `0.116`
  - `SR=0.2`: `0.094`
  - `SR=0.3`: `0.093`

## New TT Theory Results (Recreated)

| Dataset | Setting | Best New TT Result (final test NRE) | Gap vs Paper | Relative Gap |
|---|---|---:|---:|---:|
| Foreman (`SR=0.3`) | Multi-aspect, seeds `42, 7, 123`, best at `R=20` | `0.1359` | `+0.0519` | `+61.8%` |
| Condition | Single-aspect, seed `42`, best at `R=80` | `0.0878` | `-0.0052` | `-5.6%` |

Interpretation of the table:
- **Foreman:** new TT theory is currently substantially worse than the paper baseline.
- **Condition:** new TT theory can match and slightly outperform the paper reference.

## Foreman: SR-by-SR (0.1 / 0.2 / 0.3)
Best TT result is selected per SR from `R in {20,40,60,80,100}` with seeds `42,7,123`.

| SR | Paper OFTD NRE | Best TT NRE | Best R | Gap (TT - Paper) | Relative Gap | Avg online update time (TT, s/update) |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | `0.0940` | `0.1368` | `20` | `+0.0428` | `+45.5%` | `1.9809` |
| 0.2 | `0.0870` | `0.1353` | `20` | `+0.0483` | `+55.6%` | `2.0319` |
| 0.3 | `0.0840` | `0.1359` | `20` | `+0.0519` | `+61.8%` | `1.9574` |

Average running time during each online update is explicitly reported in the last column.
Paper runtime note: Table 3 reports an overall `Time (s)` metric for OFTD (not a per-SR Foreman breakdown in the table text extraction).

## Condition: SR-by-SR (0.1 / 0.2 / 0.3)
TT results below use a fixed single-aspect protocol at `R=80`, seed `42` (same setup across SR values).

| SR | Paper OFTD NRE | TT NRE (`R=80`) | Gap (TT - Paper) | Relative Gap | Avg online update time (TT, s/update) |
|---|---:|---:|---:|---:|---:|
| 0.1 | `0.1160` | `0.1125` | `-0.0035` | `-3.1%` | `0.6586` |
| 0.2 | `0.0940` | `0.0919` | `-0.0021` | `-2.2%` | `0.6525` |
| 0.3 | `0.0930` | `0.0878` | `-0.0052` | `-5.6%` | `0.7028` |

Average running time during each online update is explicitly reported in the last column.

## Theory-Isolated Check (Foreman, Same Protocol)
To isolate decomposition effect, a strict run used identical protocol and only changed model decomposition:

- `cp_multi` (paper-faithful CP path): final test NRE `0.0949`, avg online test NRE `0.0756`
- `ftd` (new TT path): final test NRE `0.1493`, avg online test NRE `0.1545`

This supports that the Foreman degradation is primarily tied to the TT/theory change (not just training script noise).

## Efficiency/Scaling Findings (New TT)
- Increasing `R` consistently increases parameter count and training time.
- Error does **not** improve consistently with larger `R` (especially on Foreman).
- For this implementation:
  - Foreman best tradeoff was at low rank (`R=20`).
  - Condition best final NRE was at mid-high rank (`R=80`).

## What This Means for the Project
- The new TT theory appears **dataset-regime dependent**:
  - Promising in single-aspect streaming (`condition`).
  - Not yet competitive in multi-aspect streaming (`foreman`) under current setup.
- If we present this in a paper draft, the honest claim is:
  - "TT replacement is not uniformly better; it helps in some regimes but hurts in others."

## Reproducibility Artifacts
- Main comparison summary:
  - `oftd/NEW_THEORY_PAPER_COMPARISON.md`
- CSV results:
  - `oftd/foreman_ftd_paper_recreate_sr01_r_sweep.csv`
  - `oftd/foreman_ftd_paper_recreate_sr02_r_sweep.csv`
  - `oftd/foreman_ftd_paper_recreate_r_sweep.csv`
  - `oftd/foreman_tt_vs_paper_sr_comparison.csv`
  - `oftd/condition_single_sr01_r80_seed42.csv`
  - `oftd/condition_single_sr02_r80_seed42.csv`
  - `oftd/condition_tt_vs_paper_sr_r80_seed42.csv`
  - `oftd/condition_ftd_paper_recreate_r_sweep_seed42.csv`
  - `oftd/foreman_theory_only_attribution_sharedopt_r100_s3.csv`
  - `oftd/new_theory_paper_recreate_benchmark.csv`
- Plots:
  - `oftd/plots_new_theory_paper_recreate/`
    - loss vs `R`
    - params vs performance
    - train time vs error
    - inference time vs error
    - `condition_single_sr_nre_vs_sr.png`
    - `condition_single_sr_update_time_vs_sr.png`
    - `condition_single_sr_infer_time_vs_sr.png`
    - `condition_single_sr_time_vs_error.png`

## Final Bottom Line
- **Foreman:** TT theory is below paper OFTD baseline by a wide margin.
- **Condition:** TT theory reaches slightly better final NRE than paper reference.
- Overall: the new theory is **not yet a consistent replacement** for the original paper method across datasets.

## Full Experiment Package
- Generated package folder:
  - `oftd/paper_experiment_package/`
- Package builder script:
  - `oftd/build_paper_experiment_package.py`
- CP baseline sweep script used for paper-style comparisons:
  - `oftd/OFTD_CP_sweep.py`
