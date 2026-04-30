# Paper Experiment Package (Full)

This package compares **Paper OFTD vs CP baseline vs TT (`Online_FTD_net`)**.

## Audit Verdict
- The package is internally consistent: the table gaps equal the reported NRE differences, sample rates are ordered correctly, and plot values match the CSV tables.
- The results are scientifically usable, but they do **not** support a blanket "TT is better" claim.
- Foreman is still a negative result after the dense-TT high-budget update: TT improves but remains worse than both the paper OFTD reference and the CP baseline at every sample rate.
- Condition is a positive TT-vs-paper-reference result: TT slightly beats the paper OFTD values at every sample rate, but the local CP baseline is still lower.
- Runtime cells for Paper OFTD are intentionally blank because the package imports paper reference NRE values only, not a same-machine runtime rerun.

## Included Tables
- `table_multi_foreman_sr.csv`
- `table_single_condition_sr.csv`
- `table_foreman_tt_rank_sensitivity.csv`
- `table_foreman_rank_sensitivity_cp_tt_sr03.csv`
- `table_foreman_rank_sensitivity_best_by_model_sr03.csv`
- `table_foreman_rank_sensitivity_raw_cp_tt_sr03.csv`
- `table_condition_tt_rank_sensitivity_sr03.csv`
- `table_foreman_tt_r20_online1000_by_sr.csv`
- `benchmark_paper_cp_tt.csv`

## Included Plots
- `foreman_nre_vs_sr_paper_cp_tt.png`
- `condition_nre_vs_sr_paper_cp_tt.png`
- `foreman_update_time_vs_sr_cp_tt.png`
- `condition_update_time_vs_sr_cp_tt.png`
- `foreman_tt_rank_sensitivity_sr03.png`
- `foreman_rank_sensitivity_cp_tt_sr03_nre.png`
- `foreman_rank_sensitivity_cp_tt_sr03_params.png`
- `foreman_rank_sensitivity_cp_tt_sr03_update_time.png`
- `foreman_rank_sensitivity_cp_tt_sr03_params_vs_nre.png`
- `condition_tt_rank_sensitivity_sr03.png`
- `foreman_tt_params_vs_nre_sr03.png`
- `condition_tt_params_vs_nre_sr03.png`
- `foreman_tt_update_time_vs_nre_sr03.png`
- `condition_tt_update_time_vs_nre_sr03.png`

## Key Readout
- Foreman SR=0.3: Paper=0.084, CP=0.093, TT=0.132
- Condition SR=0.3: Paper=0.093, CP=0.067, TT=0.088

## Paper-Use Guidance
- Legitimate claim: "The TT formulation is competitive on the single-aspect Condition stream and slightly improves over the paper OFTD reference in final NRE."
- Legitimate limitation: "The current TT implementation underperforms on the multi-aspect Foreman stream, suggesting dataset-regime sensitivity and a need for additional optimization."
- Rank-sensitivity claim: "On Foreman SR=0.3, dense TT is better than CP at very low ranks (`R=5,10`), but CP improves steadily with larger `R` while dense TT plateaus."
- Avoid claiming that TT/OFTD is uniformly superior across all tested streams from this package.

## Notes
- Foreman TT final comparison uses `R=20`, 1000 online iterations/update, and 3 seeds.
- Foreman TT rank-sensitivity plots/tables use the original paper-budget rank sweep.
- Foreman CP-vs-TT rank-sensitivity plots use `R in {5,10,20,40,60,80,100}`, SR=0.3, 3 seeds, `init_iters=4000`, and `online_iters=500`.
- Condition TT uses fixed `R=80`, seed 42 across SR values.
- Average online update time is explicitly reported in tables.
