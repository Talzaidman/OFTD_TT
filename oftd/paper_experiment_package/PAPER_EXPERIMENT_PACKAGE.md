# Paper Experiment Package (Full)

This package compares **Paper OFTD vs CP baseline vs TT (`Online_FTD_net`)**.

## Included Tables
- `table_multi_foreman_sr.csv`
- `table_single_condition_sr.csv`
- `table_foreman_tt_rank_sensitivity.csv`
- `table_condition_tt_rank_sensitivity_sr03.csv`
- `benchmark_paper_cp_tt.csv`

## Included Plots
- `foreman_nre_vs_sr_paper_cp_tt.png`
- `condition_nre_vs_sr_paper_cp_tt.png`
- `foreman_update_time_vs_sr_cp_tt.png`
- `condition_update_time_vs_sr_cp_tt.png`
- `foreman_tt_rank_sensitivity_sr03.png`
- `condition_tt_rank_sensitivity_sr03.png`
- `foreman_tt_params_vs_nre_sr03.png`
- `condition_tt_params_vs_nre_sr03.png`
- `foreman_tt_update_time_vs_nre_sr03.png`
- `condition_tt_update_time_vs_nre_sr03.png`

## Key Readout
- Foreman SR=0.3: Paper=0.084, CP=0.093, TT=0.136
- Condition SR=0.3: Paper=0.093, CP=0.067, TT=0.088

## Notes
- Foreman TT uses best-R per SR from `R in {20,40,60,80,100}` and 3 seeds.
- Condition TT uses fixed `R=80`, seed 42 across SR values.
- Average online update time is explicitly reported in tables.