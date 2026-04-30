# Experiment Audit

This audit checks whether the files in `oftd/paper_experiment_package` are numerically consistent and whether their interpretation is defensible for the experiments section of `new paper.pdf`.

## Verdict

The package is internally consistent and can be used as evidence, but it should be presented carefully. The data supports a mixed conclusion:

- **Condition / single-aspect:** TT (`Online_FTD_net`) slightly improves over the imported paper OFTD reference values.
- **Foreman / multi-aspect:** the high-budget dense TT profile improves over the earlier TT sweep, but remains worse than both the imported paper OFTD reference and the local CP baseline.
- **Overall:** the present results show dataset-regime sensitivity, not universal superiority of the TT formulation.

## Numeric Checks

- `table_multi_foreman_sr.csv`: 3 rows for sample rates `0.1, 0.2, 0.3`.
- `table_single_condition_sr.csv`: 3 rows for sample rates `0.1, 0.2, 0.3`.
- `benchmark_paper_cp_tt.csv`: 18 rows, covering 2 datasets x 3 sample rates x 3 models.
- Gap columns are arithmetically correct:
  - `tt_gap_vs_paper = tt_final_test_nre - paper_oftd_nre`
  - `cp_gap_vs_paper = cp_final_test_nre - paper_oftd_nre`
- Blank `avg_update_time_s` values appear only for imported Paper OFTD reference rows, which is expected because those rows are not same-machine reruns.
- PNG files are present and render readable NRE/rank/runtime plots.

## Theoretical Consistency

The TT implementation follows the intended third-order tensor-train form:

```text
X(i,j,k) = A(i)^T B(j) C(k)
```

with INR networks producing the coordinate-dependent factors. This is theoretically aligned with the draft's TT-OFTD framing. The beta replay and online coordinate growth mechanism are also aligned with the continual-learning motivation.

The observed result pattern is plausible rather than suspicious:

- Larger TT rank increases parameter count and update cost.
- Error does not improve monotonically with rank, which is expected in an online nonconvex INR training loop.
- A TT model can be less rank-sensitive in theory without automatically outperforming a CP/OFTD implementation under finite optimization budget, fixed replay, and dataset-specific streaming schedules.

## Results Interpretation

Foreman at SR=0.3 after the high-budget dense TT update:

```text
Paper OFTD: 0.084
CP baseline: 0.093
TT: 0.132
```

This is still a negative result for TT on multi-aspect Foreman, but it is no longer the old `0.136` lower-budget TT result. The updated Foreman final comparison uses dense TT with `R=20`, 1000 online iterations per update, and 3 seeds.

Condition at SR=0.3:

```text
Paper OFTD: 0.093
CP baseline: 0.067
TT: 0.088
```

This supports saying TT improves slightly over the paper OFTD reference on Condition, but not over the local CP baseline.

## Recommended Paper Wording

Use wording like:

```text
The proposed TT parameterization matches or slightly improves the paper OFTD reference on the single-aspect Condition stream, but even after increasing the dense-TT online budget it underperforms on the multi-aspect Foreman stream. These results indicate that the TT formulation is promising but sensitive to dataset regime and optimization protocol.
```

Avoid wording like:

```text
The TT formulation consistently outperforms prior OFTD/CP methods.
```

The current package does not support that claim.
