# FTD Full Benchmark Summary

Generated on 2026-04-19 from `ftd_full_benchmark.csv`.

## Foreman (multi) - Main R Sweep

| R | Params | Final Test NRE | Final Test Loss | Train Time (s) | Inference (s) |
|---:|---:|---:|---:|---:|---:|
| 20 | 74,040 | 0.186537 | 0.016955 | 2.567 | 0.000930 |
| 40 | 234,000 | 0.180637 | 0.015907 | 3.084 | 0.002138 |
| 60 | 497,160 | 0.190236 | 0.017652 | 4.223 | 0.004063 |
| 80 | 863,520 | 0.203956 | 0.020271 | 6.824 | 0.006818 |
| 100 | 1,333,080 | 0.195278 | 0.018635 | 9.903 | 0.009343 |

## Condition (single) - Main R Sweep

| R | Params | Final Test NRE | Final Test Loss | Train Time (s) | Inference (s) |
|---:|---:|---:|---:|---:|---:|
| 20 | 74,040 | 1.587350 | 3383.471375 | 2.495 | 0.000970 |
| 40 | 234,000 | 1.568421 | 3035.255450 | 2.967 | 0.001029 |
| 60 | 497,160 | 1.677755 | 3455.195716 | 4.130 | 0.000993 |
| 80 | 863,520 | 1.655679 | 3373.416477 | 6.191 | 0.001227 |
| 100 | 1,333,080 | 1.597094 | 3166.405202 | 9.246 | 0.001290 |

## Config Ablations (R=40)

| Dataset | Config | Final Test NRE | Final Test Loss | Train Time (s) | Inference (s) |
|---|---|---:|---:|---:|---:|
| foreman.mat | foreman_multi_cfg_boundary_deriv | 0.179437 | 0.015688 | 4.012 | 0.002034 |
| foreman.mat | foreman_multi_cfg_no_boundary | 0.187233 | 0.017095 | 2.811 | 0.002050 |
| foreman.mat | foreman_multi_main | 0.180637 | 0.015907 | 3.084 | 0.002138 |
| condition.mat | condition_single_cfg_boundary_deriv | 1.568421 | 3035.255450 | 4.098 | 0.001023 |
| condition.mat | condition_single_cfg_no_boundary | 1.568421 | 3035.255450 | 2.333 | 0.000884 |
| condition.mat | condition_single_main | 1.568421 | 3035.255450 | 2.967 | 0.001029 |

## Plots
- [condition.mat_single_infer_time_vs_error.png](condition.mat_single_infer_time_vs_error.png)
- [condition.mat_single_params_vs_error.png](condition.mat_single_params_vs_error.png)
- [condition.mat_single_R_vs_final_test_loss.png](condition.mat_single_R_vs_final_test_loss.png)
- [condition.mat_single_R_vs_final_test_nre.png](condition.mat_single_R_vs_final_test_nre.png)
- [condition.mat_single_R_vs_infer_time.png](condition.mat_single_R_vs_infer_time.png)
- [condition.mat_single_R_vs_train_time.png](condition.mat_single_R_vs_train_time.png)
- [foreman.mat_multi_infer_time_vs_error.png](foreman.mat_multi_infer_time_vs_error.png)
- [foreman.mat_multi_params_vs_error.png](foreman.mat_multi_params_vs_error.png)
- [foreman.mat_multi_R_vs_final_test_loss.png](foreman.mat_multi_R_vs_final_test_loss.png)
- [foreman.mat_multi_R_vs_final_test_nre.png](foreman.mat_multi_R_vs_final_test_nre.png)
- [foreman.mat_multi_R_vs_infer_time.png](foreman.mat_multi_R_vs_infer_time.png)
- [foreman.mat_multi_R_vs_train_time.png](foreman.mat_multi_R_vs_train_time.png)