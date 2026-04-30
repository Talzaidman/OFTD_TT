# Foreman Rank Sensitivity: CP vs Dense TT

Dataset: `foreman.mat`; sample rate: `0.3`; ranks: from generated raw sweep files.

This test checks the idea that TT should be less sensitive to the unknown rank than CP.

## Best By Model

| model | R | final_test_nre | avg_online_nre_test | params | avg_update_time_s | gap_vs_paper_foreman_sr03 |
| --- | --- | --- | --- | --- | --- | --- |
| CP | 100 | 0.0926343 | 0.0743317 | 55980 | 2.62116 | 0.00863433 |
| Dense TT | 20 | 0.135331 | 0.134112 | 74040 | 1.9554 | 0.0513307 |

## Interpretation

- Lower NRE is better.
- Dense TT is better than CP at very low ranks (`R=5,10`), which supports the idea that TT can be more useful when the rank budget is extremely small.
- CP improves steadily as `R` increases and reaches its best result at `R=100`.
- Dense TT reaches its best result at `R=20` and then plateaus/worsens, so this implementation does not yet show the expected high-rank robustness advantage.
- The plot should be read as an optimization-and-rank-sensitivity result, not as a direct measurement of the unknown true tensor rank.

## Reproduce

```powershell
cd c:\Users\T10Z006\VS_projects\learning\oftd
python build_rank_sensitivity_package.py --r-values 5,10,20,40,60,80,100 --seeds 42,7,123 --models both --out-dir paper_experiment_package
```
