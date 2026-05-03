# Synthetic TT Parameter-Budget Matched Sweep

Each x-axis sample is a target trainable-parameter budget. CP and dense TT use the nearest valid integer rank for that budget.

Parameter formulas:

- `CP params = 17280 + 387R`
- `Dense TT params = 17280 + 258R + 129R^2`

Budgets below about `17k` are impossible with the current INR architecture because the networks have fixed base weights.

## Target NRE

| model | target_nre | target_status | first_target_params_at_target | actual_params_at_target | R_at_target | nre_at_target_hit | best_R | best_actual_params | best_final_test_nre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP | 0.01 | not reached |  |  |  |  | 20 | 25020 | 0.0228513 |
| Dense TT | 0.01 | reached | 40000 | 38952 | 12 | 0.00980067 | 12 | 38952 | 0.00980067 |

## Budget-Matched Results

| target_params | model | R | actual_params | param_error | final_test_nre | final_test_nre_std | avg_online_nre_test | total_train_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18000 | CP | 2 | 18054 | 54 | 0.197347 | 0.00110805 | 0.0974947 | 5.6878 |
| 18000 | Dense TT | 2 | 18312 | 312 | 0.084893 | 0.00784277 | 0.0388167 | 14.7448 |
| 20000 | CP | 7 | 19989 | -11 | 0.0259623 | 0.0106898 | 0.024446 | 5.73287 |
| 20000 | Dense TT | 4 | 20376 | 376 | 0.0167363 | 0.00173516 | 0.0192783 | 14.9627 |
| 25000 | CP | 20 | 25020 | 20 | 0.0228513 | 0.0052223 | 0.028913 | 5.88087 |
| 25000 | Dense TT | 7 | 25407 | 407 | 0.014074 | 0.00482394 | 0.0184553 | 15.2532 |
| 30000 | CP | 33 | 30051 | 51 | 0.0273973 | 0.0130546 | 0.0366703 | 5.9367 |
| 30000 | Dense TT | 9 | 30051 | 51 | 0.0136003 | 0.00616162 | 0.018291 | 16.0127 |
| 40000 | CP | 59 | 40113 | 113 | 0.0255233 | 0.00981756 | 0.052368 | 6.07367 |
| 40000 | Dense TT | 12 | 38952 | -1048 | 0.00980067 | 0.0026792 | 0.0202283 | 15.6149 |
| 50000 | CP | 85 | 50175 | 175 | 0.0501953 | 0.0326939 | 0.051737 | 6.23167 |
| 50000 | Dense TT | 15 | 50175 | 175 | 0.0130457 | 0.00436987 | 0.019391 | 15.8703 |
| 75000 | CP | 149 | 74943 | -57 | 0.0492557 | 0.0172039 | 0.0742293 | 6.46813 |
| 75000 | Dense TT | 20 | 74040 | -960 | 0.0143287 | 0.00377266 | 0.0194723 | 16.5376 |
| 100000 | CP | 214 | 100098 | 98 | 0.0733147 | 0.0205715 | 0.102448 | 6.70533 |
| 100000 | Dense TT | 24 | 97776 | -2224 | 0.0107193 | 0.00401742 | 0.0181877 | 16.7468 |
| 125000 | CP | 278 | 124866 | -134 | 0.041773 | 0.0138874 | 0.0782033 | 6.81777 |
| 125000 | Dense TT | 28 | 125640 | 640 | 0.0112257 | 0.00250884 | 0.0176143 | 17.35 |