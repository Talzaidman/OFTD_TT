# Synthetic TT Parameter-Budget Matched Sweep

Each x-axis sample is a target trainable-parameter budget. CP and dense TT use the nearest valid integer rank for that budget.

Parameter formulas:

- `CP params = 17280 + 387R`
- `Dense TT params = 17280 + 258R + 129R^2`

Budgets below about `17k` are impossible with the current INR architecture because the networks have fixed base weights.

## Target NRE

| model | target_nre | target_status | first_target_params_at_target | actual_params_at_target | R_at_target | nre_at_target_hit | best_R | best_actual_params | best_final_test_nre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP | 0.01 | not reached |  |  |  |  | 20 | 25020 | 0.023043 |
| Dense TT | 0.01 | reached | 40000 | 38952 | 12 | 0.00980067 | 12 | 38952 | 0.00980067 |

## Budget-Matched Results

| target_params | model | R | actual_params | param_error | final_test_nre | final_test_nre_std | avg_online_nre_test | total_train_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18000 | CP | 2 | 18054 | 54 | 0.197349 | 0.00110601 | 0.097505 | 12.514 |
| 18000 | Dense TT | 2 | 18312 | 312 | 0.084893 | 0.00784277 | 0.0388167 | 14.7448 |
| 20000 | CP | 7 | 19989 | -11 | 0.0258817 | 0.0106965 | 0.0243893 | 12.8047 |
| 20000 | Dense TT | 4 | 20376 | 376 | 0.0167363 | 0.00173516 | 0.0192783 | 14.9627 |
| 25000 | CP | 20 | 25020 | 20 | 0.023043 | 0.00518724 | 0.0289287 | 13.3707 |
| 25000 | Dense TT | 7 | 25407 | 407 | 0.014074 | 0.00482394 | 0.0184553 | 15.2532 |
| 30000 | CP | 33 | 30051 | 51 | 0.027636 | 0.0130885 | 0.0366983 | 13.9567 |
| 30000 | Dense TT | 9 | 30051 | 51 | 0.0136003 | 0.00616162 | 0.018291 | 16.0127 |
| 40000 | CP | 59 | 40113 | 113 | 0.0313773 | 0.012093 | 0.053217 | 16.5385 |
| 40000 | Dense TT | 12 | 38952 | -1048 | 0.00980067 | 0.0026792 | 0.0202283 | 15.6149 |
| 50000 | CP | 85 | 50175 | 175 | 0.050329 | 0.0328344 | 0.051799 | 20.596 |
| 50000 | Dense TT | 15 | 50175 | 175 | 0.0130457 | 0.00436987 | 0.019391 | 15.8703 |
| 75000 | CP | 149 | 74943 | -57 | 0.05184 | 0.0165378 | 0.0746327 | 43.6595 |
| 75000 | Dense TT | 20 | 74040 | -960 | 0.0143287 | 0.00377266 | 0.0194723 | 16.5376 |
| 100000 | CP | 214 | 100098 | 98 | 0.0748333 | 0.0249572 | 0.102538 | 82.0942 |
| 100000 | Dense TT | 24 | 97776 | -2224 | 0.0107193 | 0.00401742 | 0.0181877 | 16.7468 |
| 125000 | CP | 278 | 124866 | -134 | 0.0413547 | 0.0162094 | 0.0779183 | 162.096 |
| 125000 | Dense TT | 28 | 125640 | 640 | 0.0112257 | 0.00250884 | 0.0176143 | 17.35 |