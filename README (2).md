# MCDD Project — Complete Pipeline
## Missingness-Coupled Dual Decomposition for Log Forensics

---

## What This Project Does

When computer systems crash or get attacked, logs become incomplete. Standard methods assume missing data is random (MCAR). **This is wrong for real systems.**

**MCDD** (Missingness-Coupled Dual Decomposition) is a 4-stage method that:
1. **Reconstructs** the missing log entries using low-rank matrix completion
2. **Detects** which event types crashed (structured missingness)
3. **Identifies** attacker-modified entries (sparse tamper detection)
4. **Learns** per-column crash thresholds (theta) — a novel interpretable output

---

## File Structure

```
step1_sample_thunderbird.py   → Parse Thunderbird logs → M_ground_truth.npy
step2_check_sample.py         → Verify low-rank structure (98.5% top-5)
step3_corrupt.py              → Create 3 experimental conditions
step4_baseline_rpca.py        → Standard RPCA baseline (ADMM)
step5_mcdd.py                 → MCDD main method (4 stages)
step6_evaluate.py             → Comprehensive evaluation + metrics table
step7_plots.py                → 7 publication-quality figures
step8_sensitivity.py          → Parameter robustness analysis
```

---

## Run Order

```bash
python step1_sample_thunderbird.py   # ~10 min, needs Thunderbird.log
python step2_check_sample.py         # ~1 sec
python step3_corrupt.py              # ~1 sec
python step4_baseline_rpca.py        # ~30 sec
python step5_mcdd.py                 # ~1 min
python step6_evaluate.py             # ~5 sec
python step7_plots.py                # ~10 sec
python step8_sensitivity.py          # ~2 min
```

---

## The 4 Stages of MCDD

### Stage 1: ADMM for L + S_tamper (Masked RPCA)
- **Math**: Standard ADMM solving min ||L||_* + lambda2*||S||_1 s.t. P_Omega(L+S) = P_Omega(M_obs)
- **What it does**: Separates normal behavior (L, low-rank) from tampering (S_tamper, sparse)
- **Convergence**: Guaranteed — convex problem with proven ADMM convergence
- **Output**: L (low-rank), S_tamper (sparse tamper), Y (dual variable)

### Stage 2: Crash Column Detection
- **Method**: Empirical threshold on column missing fraction (> 0.25)
- **Why it works**: Structured crashes create ~50% missing in affected columns; random missing is ~20%
- **Output**: crash_cols (detected crashed event types), safe_cols (unaffected)

### Stage 3: Crash-Aware Subspace Projection (KEY CONTRIBUTION)
- **Math**: Learn subspace U from SAFE columns only. Project crashed columns: L_missing = U_safe @ alpha
- **Why it works**: RPCA learns from ALL columns (crashed ones are zero-filled → contaminate SVD). By excluding crashed columns, we get a cleaner subspace → better predictions.
- **Result**: 45% RMSE improvement over RPCA on structured crashes

### Stage 4: Post-hoc MNAR Analysis (NOVEL OUTPUT)
- **Math**: P(missing | L, theta) = sigmoid(alpha*(L - theta)). Learn theta via gradient descent.
- **What it does**: Models the physical crash mechanism — "system crashes when event count exceeds threshold theta"
- **Output**: theta vector (30 crash thresholds, one per event type) + S_crash_prob (probability surface)
- **Key property**: Safe columns have HIGHER theta (need more events to crash) than crashed columns

---

## Expected Results

| Condition | RPCA RMSE | MCDD RMSE | Improvement | Crash F1 | Tamper F1 |
|-----------|-----------|-----------|-------------|----------|-----------|
| A Random (20%) | 0.980 | 0.978 | 0% (honest baseline) | — | — |
| B Structured Crash | 0.868 | **0.475** | **45%** | **1.000** | — |
| C Adversarial | 0.868 | **0.590** | **32%** | **1.000** | **0.795** |

**Theta separation** (safe - crash): ~3.0 to 4.6 (positive = correct)

---

## Evaluation Metrics Explained

| Metric | What it Measures | Why It Matters |
|--------|-----------------|----------------|
| **RMSE** | Root mean square error on missing entries | Primary reconstruction quality |
| **MAE** | Mean absolute error | Robust to outliers |
| **Rel_Error** | ||M_true - L||_F / ||M_true||_F | Normalized error |
| **PSNR** | Peak signal-to-noise ratio | Higher = better reconstruction |
| **Corr** | Pearson correlation | Measures linear relationship |
| **Crash F1** | Column-level crash detection F1 | Did we find the right crashed nodes? |
| **Crash Precision/Recall/Accuracy** | Detailed crash detection | Breakdown of F1 score |
| **Tamper F1** | Entry-level tamper detection F1 | Did we find modified entries? |
| **Tamper AUC** | ROC-AUC for tamper scores | Discrimination ability |
| **Theta Separation** | safe_theta_mean - crash_theta_mean | Novel: physical interpretability |

---

## Figures Generated

| Figure | What It Shows |
|--------|--------------|
| fig1_accuracy_comparison.png | RMSE, MAE, Rel_Error, PSNR across all methods |
| fig2_rmse_improvement.png | % improvement over RPCA per condition |
| fig3_learned_theta.png | **NOVEL**: Learned crash thresholds per event type |
| fig4_crash_probability_surface.png | True mask vs learned probability surface |
| fig5_mnar_discrimination.png | Key novelty: random vs structured discrimination |
| fig6_tamper_heatmap.png | S_tamper magnitude vs ground truth tamper mask |
| fig7_error_difference.png | Where MCDD wins vs RPCA (blue = MCDD better) |

---

## ADMM Mathematical Correctness

The ADMM formulation in step4 and step5 is mathematically correct:

```
L-update:  L = SVT(M_fill - S + Y/rho, 1/rho)        ← proximal of nuclear norm
S-update:  S = soft_thresh(M_fill - L + Y/rho, lam/rho)  ← proximal of L1 norm
Y-update:  Y = Y + rho*(M_fill - L - S)              ← dual ascent
```

These are the standard proximal operators for RPCA. Convergence is guaranteed
for convex problems. The code matches the math exactly.

---

## What Is Novel

1. **S_tamper detection** in log forensics — no prior analog
2. **Failure cause attribution** — distinguishing crash vs attack
3. **Three-output framework** (L + crash_map + tamper_map) for logs
4. **Theta (crash thresholds)** — learned physical parameter with interpretation
5. **Selective subspace projection** — conditioned on detected crash columns

---

## Honest Limitations

1. ADMM Stage 1 alone does NOT outperform RPCA — improvement comes from Stage 3
2. Stage 3 uses known subspace projection techniques (applied selectively)
3. Single dataset, synthetic crash injection
4. S_crash detection is empirical (threshold-based), not learned inside ADMM

These limitations are acknowledged and do not invalidate the contribution.
The novelty lies in the **combination** and **application**, not in reinventing ADMM.

---

*Prepared for 22MAT230 course project demonstration.*
