# TT Foreman Failure Analysis

This note investigates why `Online_FTD_net` underperforms the CP/OFTD baselines on the multi-aspect Foreman experiment.

## Bottom Line

The current Foreman result is **not evidence of a theoretical TT performance ceiling**.

For a third-order tensor, CP is a special case of TT:

```text
CP: X(i,j,k) = sum_r A(i,r) B(j,r) C(k,r)
TT: X(i,j,k) = A(i)^T M(j) C(k)
```

If `M(j)` is diagonal with diagonal entries `B(j,r)`, the TT expression becomes the CP expression. Therefore, a well-optimized TT model with sufficient rank should be able to match the CP solution class. The fact that dense TT is worse than CP points to **optimization, parameterization, or protocol issues**, not a hard theoretical impossibility.

## Existing Benchmark Evidence

From `foreman_theory_only_attribution_sharedopt_r100_s3.csv`, where CP and TT used the same Foreman split, seed set, schedule, optimizer loop, and budget:

| Model | R | Mean final test NRE | Mean avg online test NRE |
|---|---:|---:|---:|
| CP multi | 100 | 0.0949 | 0.0756 |
| Dense TT / `Online_FTD_net` | 100 | 0.1493 | 0.1545 |

This isolates the degradation to the dense TT model/optimization path.

From `foreman_ftd_paper_recreate_*` rank sweeps:

| SR | Best TT R | Best TT final NRE | Paper OFTD NRE | CP baseline NRE |
|---:|---:|---:|---:|---:|
| 0.1 | 20 | 0.1368 | 0.0940 | 0.0985 |
| 0.2 | 20 | 0.1353 | 0.0870 | 0.0947 |
| 0.3 | 20 | 0.1359 | 0.0840 | 0.0926 |

The TT error is almost flat across `R in {20,40,60,80,100}` rather than improving with rank. This is a classic sign that the extra TT capacity is not being converted into useful fit.

## Dense-TT Update After This Diagnosis

Following the diagnosis, I kept the dense TT model and tested training-profile changes rather than adding a CP/diagonal special case.

The best consistent improvement came from giving the best low-rank dense TT profile more online optimization budget:

```text
R = 20
init_iters = 4000
online_iters = 1000
boundary_lambda = 0
normalize_recon = true
seeds = 42, 7, 123
```

Updated Foreman means:

| SR | Old TT final NRE | Updated dense TT final NRE | Updated avg online test NRE |
|---:|---:|---:|---:|
| 0.1 | 0.1368 | 0.1334 | 0.1314 |
| 0.2 | 0.1353 | 0.1317 | 0.1279 |
| 0.3 | 0.1359 | 0.1322 | 0.1291 |

This confirms the previous Foreman number was **not** a peak for the current theoretical model. It is still not enough to beat CP/OFTD, so the remaining gap is likely deeper than a single training knob.

## CP-vs-TT Rank Sensitivity Sweep

I then ran a direct Foreman SR=0.3 rank sweep for both CP and dense TT:

```text
R in {5,10,20,40,60,80,100}
seeds = 42,7,123
init_iters = 4000
online_iters = 500
```

Best results from `paper_experiment_package/table_foreman_rank_sensitivity_best_by_model_sr03.csv`:

| Model | Best R | Final test NRE | Avg online test NRE |
|---|---:|---:|---:|
| CP | 100 | 0.0926 | 0.0743 |
| Dense TT | 20 | 0.1353 | 0.1341 |

The interesting detail is that dense TT is better than CP at very low ranks (`R=5` and `R=10`), but CP keeps improving as rank increases while dense TT plateaus. This means the expected "TT is less rank sensitive" story is only partly visible here: TT helps under a very small rank budget, but the current dense TT implementation does not convert larger rank into better Foreman performance.

## Conditioning Diagnostics

A quick initialization/gradient diagnostic on Foreman SR=0.3, seed 42, initial shape `(14,17,10)` showed:

| R | Model | Params | Initial train NRE | Gradient norm |
|---:|---|---:|---:|---:|
| 20 | CP | 25,020 | 0.9975 | 404.6 |
| 20 | TT | 74,040 | 1.0167 | 1,853 |
| 40 | CP | 32,760 | 0.9936 | 448.0 |
| 40 | TT | 234,000 | 0.9874 | 3,663 |
| 100 | CP | 55,980 | 1.0204 | 716.9 |
| 100 | TT | 1,333,080 | 0.9402 | 8,557 |

The dense TT gradient norm grows sharply with rank. The middle factor `B_net` is the main source: it outputs `R1 * R2` values, so square rank `R=100` produces 10,000 middle-core outputs per coordinate and 1.33M trainable parameters overall.

The demo's theory diagnostic also reported a very large finite-difference proxy for `B_net`:

```text
R1=R2=100, w_init=0.05, after init:
B finite-diff |f'|_l1 max ~= 418
```

That is not a clean realization of the smooth/bounded factor behavior assumed in the theory discussion.

## Settings Smoke Tests

Short-budget Foreman checks used SR=0.3, seed 42, `init_iters=1000`, `online_iters=100`, normalized reconstruction, and no boundary penalty.

| Setting | Avg online test NRE | Final test NRE | Note |
|---|---:|---:|---|
| `R1=100, R2=100, w_init=0.05` | 0.1667 | 0.1586 | Square dense TT control |
| `R1=100, R2=20, w_init=0.05` | 0.1673 | 0.1540 | Cheaper, slightly better final |
| `R1=20, R2=100, w_init=0.05` | 0.1607 | 0.1581 | Slightly better average |
| `R1=100, R2=100, w_init=0.01` | 0.1996 | 0.1955 | Too small/slow under this budget |
| `R1=100, R2=100, w_init=0.10, lr=3e-4` | 0.3391 | 0.2179 | Too unstable/oscillatory |

These quick tests do not recover CP-level performance. They do support the diagnosis that the current square dense TT parameterization is poorly conditioned, and that simple rank asymmetry or init-scale changes are not enough by themselves.

Additional dense-TT optimizer controls were added after these tests:

- per-factor learning-rate multipliers (`--lr-a-mult`, `--lr-b-mult`, `--lr-c-mult`)
- configurable online gradient clipping (`--clip-grad-norm`)
- optional initial-stage clipping (`--init-clip-grad-norm`)
- optional online Adam state reuse (`--reuse-online-optimizer`)
- final diagnostic columns for `dA_l1_max`, `dB_l1_max`, and `dC_l1_max`

Short-budget trials showed that looser clipping can help early online progress, but long paper-budget trials did not beat the improved `R=20`, 1000-online-iteration profile.

## Likely Causes

1. **Dense middle TT factor is overparameterized for the current online budget.**
   `B_net` scales as `R1 * R2`, while the CP baseline scales roughly linearly in rank. At `R=100`, TT has about 24x more parameters than CP.

2. **The model is not CP-warm-started.**
   Since CP is a diagonal-middle TT, dense TT should ideally start from or be regularized toward a useful diagonal structure. The current `B_net` learns a full dense matrix from scratch.

3. **Rank sweeps are not parameter-matched.**
   Comparing CP `R=100` to TT `R=100` gives TT far more parameters and a much harder optimization problem. The current TT sweep mostly measures optimizer difficulty, not only model capacity.

4. **Theory assumptions are not fully enforced.**
   The theory motivates smooth, bounded functional factors. The observed `B_net` finite-difference proxy is very large, especially at high rank.

5. **Replay/online budget may be too small for dense TT.**
   The current online updates sample coordinate subsets. A dense TT core may need larger replay batches, per-factor learning rates, or a longer online budget to exploit its extra capacity.

## Recommended Next Fixes

1. Add a **diagonal-plus-low-rank residual TT** variant:

   ```text
   M(j) = diag(b(j)) + epsilon * U(j) V(j)^T
   ```

   Initialize `epsilon=0` or very small. This lets TT begin at the CP-special-case solution class and only use dense corrections when useful.

2. Run **parameter-matched sweeps**, not only equal-rank sweeps.

3. Add CSV diagnostics for:
   - `A/B/C` finite-difference max/mean
   - factor output norms
   - gradient norms before clipping
   - observed replay count per online update

4. Try per-module optimizer settings:
   - lower LR for `B_net`
   - separate LR for final linear layers
   - rank-scaled initialization for the middle factor

5. Treat the current Foreman number as a **lower-bound implementation result**, not as a theoretical peak.
