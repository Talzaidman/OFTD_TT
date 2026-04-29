# New Paper vs Current Code: Alignment Check

This report checks the current implementation against `new paper.pdf` ("Online Functional Tensor-Train Decomposition via Continual Learning for Streaming Data Completion").

## Verdict
- **Core method is implemented** (TT-style INR factorization + online continual updates + beta replay sampling).
- **Not fully paper-faithful yet** for experiments due to several protocol/default mismatches.

## What Matches
1. **TT functional form (3rd-order case)**
   - Paper: \(X(i,j,k)=A(i)^T B(j) C(k)\), with \(B(j)\in\mathbb{R}^{r\times r}\).
   - Code: `Online_FTD_net` computes `einsum("ir,jrs,ks->ijk", A, B, C)`.
   - Location: `oftd/model.py` (`Online_FTD_net`).

2. **INR parameterization with sine activations**
   - Paper: INR MLPs with `sin(omega_0 * .)`.
   - Code: `SineLayer` + INR stacks in `Online_FTD_net`.
   - Location: `oftd/model.py`.

3. **Online coordinate expansion**
   - Paper: grow coordinates over time in streaming updates.
   - Code: `online_update_multi_ftd` expands `A_t/B_t/C_t` and optimizes at each step.
   - Location: `oftd/model.py`.

4. **Memory replay with beta sampling**
   - Paper: long-tail beta replay for historical positions.
   - Code: `sample(alpha_beta, ...)` + replay indices appended with new indices.
   - Location: `oftd/utils.py`, `oftd/model.py`.

## Mismatches / Risks
1. **Objective mismatch (Eq. 27 structure)**
   - Paper writes an explicit two-term objective: new-data fit + replay term.
   - Code uses a single sampled reconstruction loss over concatenated replay+new indices.
   - Impact: close in spirit, but not mathematically identical weighting.

2. **Extra regularizers not in the new paper objective**
   - Code path includes optional `boundary_lambda` and `deriv_lambda`.
   - Impact: if nonzero, results are no longer strictly paper-faithful.
   - Must keep `boundary_lambda=0`, `deriv_lambda=0` for paper-faithful runs.

3. **Default experiment knobs are not always paper-faithful**
   - Some scripts default to nonzero boundary regularization.
   - Impact: accidental mismatch if user runs defaults.
   - A paper profile should be enforced in script defaults or config files.

4. **INR depth/shape assumptions are not strictly symmetric**
   - Paper formulation presents a generic `d`-layer INR form per factor.
   - Current `A/B/C` nets have slightly different depths.
   - Impact: theoretical form is approximated, not exact architecture mirroring.

5. **Coordinate convention offset**
   - Paper notation uses coordinates `1..I_n`; code uses `0..I_n-1`.
   - Usually minor, but it is a formal mismatch.

## Ready for Experimental Plots?
- **Yes**, after we lock one **paper-faithful run profile**:
  - `boundary_lambda=0`
  - `deriv_lambda=0`
  - fixed rank policy (paper uses `r=100` unless explicitly sweeping)
  - fixed replay beta parameters
  - fixed seeds and sample rates

## Recommended Next Step
Create one canonical "paper profile" config and run all experiment sweeps/plots from that profile only.
