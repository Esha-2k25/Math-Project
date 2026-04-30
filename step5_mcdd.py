# step5_mcdd.py — Missingness-Coupled Dual Decomposition (MCDD)
#
# MATHEMATICAL FRAMEWORK:
#   Standard RPCA:   M = L + S        (L=low-rank, S=sparse corruption)
#   MCDD extends this to handle STRUCTURED missingness (MNAR):
#
#   Decomposition:  M_true = L + S_crash + S_tamper
#
#   | Variable    | Represents                    | Domain
#   |-------------|-------------------------------|--------------------------
#   | L          | Normal system behavior          | Low-rank data domain
#   | S_tamper   | Attacker-modified entries       | Sparse, observed data
#   | S_crash    | Crash-induced missing blocks    | Identified by col_miss
#   | theta      | Per-column crash threshold      | Learned MNAR parameter
#
#   STAGE 1 — ADMM for L + S_tamper (masked RPCA):
#     min ||L||_* + lambda2*||S_tamper||_1
#     s.t. P_Omega(L + S_tamper) = P_Omega(M_obs)
#
#     This is IDENTICAL to standard masked RPCA. The ADMM loop:
#       L^{k+1} = SVT(M_fill - S^k + Y^k/rho, 1/rho)
#       S^{k+1} = soft_thresh(M_fill - L^{k+1} + Y^k/rho, lambda2/rho)
#       Y^{k+1} = Y^k + rho*(M_fill - L^{k+1} - S^{k+1})  [dual ascent]
#
#   STAGE 2 — Crash Column Detection:
#     crash_cols = { j | col_miss_frac[j] > 0.25 }
#     (Empirical threshold: structured crashes have ~50% missing)
#
#   STAGE 3 — Crash-Aware Subspace Projection (THE KEY IMPROVEMENT):
#     Learn subspace U from SAFE columns only, project crashed columns:
#       U_safe = SVD(L[:, safe_cols])[:, :k]
#       For each crashed column c:
#         alpha = least_squares(U_safe[obs_rows], M_fill[obs_rows, c])
#         L_pp[missing_rows, c] = U_safe[missing_rows] @ alpha
#
#     WHY THIS WORKS: RPCA learns L from ALL columns (including crashed
#     ones, zero-filled). The crashed columns CONTAMINATE the SVD. By
#     learning the subspace from ONLY safe columns, we get a cleaner
#     estimate -> better missing entry prediction.
#
#   STAGE 4 — Post-hoc MNAR Analysis (NOVEL OUTPUT):
#     Model: P(missing | L_ij, theta_j) = sigmoid(alpha*(L_ij - theta_j))
#     Learn theta_j per column via gradient descent on the sigmoid model.
#     This does NOT modify L — it runs on the already-reconstructed L.
#     Output: theta vector (crash thresholds) + S_crash_prob surface.

import numpy as np
import os

def svd_threshold(X, tau):
    """Singular Value Thresholding: SVT(X, tau)"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank

def soft_threshold(X, tau):
    """Element-wise soft-thresholding"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def sigmoid(x):
    """Numerically stable sigmoid"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mcdd(M_obs, lam2=None, rho=1.0, max_iter=300, tol=1e-4,
         alpha=1.0, lr_theta=0.05, mnar_iter=200):
    """
    MCDD: Missingness-Coupled Dual Decomposition

    Parameters:
      M_obs     : observed matrix (NaN = missing)
      lam2      : sparse regularization (default: 1.5/sqrt(max(m,n)))
      rho       : ADMM penalty parameter
      alpha     : sigmoid steepness for MNAR model
      lr_theta  : learning rate for theta optimization
      mnar_iter : iterations for theta convergence

    Returns:
      L_pp      : reconstructed low-rank matrix (post-processed)
      S_tamper  : sparse tamper component
      theta     : learned per-column crash thresholds (NOVEL)
      S_crash_prob : crash probability surface (NOVEL)
      crash_cols   : detected crashed column indices
      rank      : rank of L from Stage 1
    """
    m, n = M_obs.shape
    if lam2 is None:
        lam2 = 1.5 / np.sqrt(max(m, n))

    mask_obs = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill = np.nan_to_num(M_obs, nan=0.0)
    col_miss_frac = miss_mask.mean(axis=0)

    # =====================================================================
    # STAGE 1: ADMM for Masked RPCA (L + S_tamper)
    # =====================================================================
    print("  [STAGE 1] ADMM: masked RPCA for L + S_tamper")
    L = M_fill.copy()
    S_tamper = np.zeros((m, n))
    Y = np.zeros((m, n))  # dual variable

    for i in range(max_iter):
        L_old = L.copy()

        # L-update: singular value thresholding (nuclear norm proximal)
        L, rank = svd_threshold(M_fill - S_tamper + Y/rho, 1.0/rho)

        # S-update: soft thresholding (L1 proximal for sparse tamper)
        S_new = soft_threshold(M_fill - L + Y/rho, lam2/rho)
        S_new[~mask_obs] = 0  # S_tamper only on observed entries
        S_tamper = S_new

        # Y-update: dual ascent (augmented Lagrangian multiplier)
        res = M_fill - L - S_tamper
        res[~mask_obs] = 0    # only on observed entries
        Y += rho * res

        rel_change = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10)
        if rel_change < tol:
            print(f"    -> Converged at iteration {i+1}, rank(L)={rank}")
            break
    else:
        print(f"    -> Reached max_iter, rank(L)={rank}")

    # =====================================================================
    # STAGE 2: Crash Column Detection (empirical, honest)
    # =====================================================================
    print("  [STAGE 2] Crash column detection")
    crash_cols = np.where(col_miss_frac > 0.25)[0]
    safe_cols = [c for c in range(n) if c not in crash_cols]
    print(f"    -> Detected crash columns: {crash_cols.tolist()}")
    print(f"    -> Safe columns: {safe_cols}")

    # =====================================================================
    # STAGE 3: Crash-Aware Subspace Projection (KEY CONTRIBUTION)
    # =====================================================================
    print("  [STAGE 3] Subspace projection from safe columns")
    L_pp = L.copy()

    if len(crash_cols) > 0 and len(safe_cols) >= 2:
        # Learn subspace from SAFE columns only (exclude crashed columns)
        U_safe, s_safe, _ = np.linalg.svd(L_pp[:, safe_cols], full_matrices=False)

        # Determine projection rank from safe-column singular values
        explained_safe = np.cumsum(s_safe**2) / np.sum(s_safe**2)
        k = max(1, np.searchsorted(explained_safe, 0.985) + 1)
        k = min(k, len(safe_cols) - 1, m - 1)

        U_k = U_safe[:, :k]

        for c in crash_cols:
            obs_rows = mask_obs[:, c]
            missing_rows = ~obs_rows
            if obs_rows.sum() >= k:
                # Least squares: fit observed entries to safe-column subspace
                A = U_k[obs_rows, :]
                b = M_fill[obs_rows, c]  # TRUE observed values (not L approx)
                coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                # Predict missing entries using the subspace coefficients
                L_pp[missing_rows, c] = U_k[missing_rows, :] @ coeff

        print(f"    -> Projected {len(crash_cols)} columns onto rank-{k} subspace")
        print(f"    -> Safe-column variance explained at rank-{k}: {explained_safe[k-1]*100:.1f}%")
    else:
        print(f"    -> No projection applied (no crash columns or insufficient safe cols)")

    # =====================================================================
    # STAGE 4: Post-hoc MNAR Analysis (NOVEL — does not modify L_pp)
    # =====================================================================
    print("  [STAGE 4] MNAR crash threshold learning (theta)")

    # Weight: only learn on columns with real crash evidence
    col_mnar_weight = np.clip((col_miss_frac - 0.25) / 0.25, 0.0, 1.0)

    # Initialize theta from empirical miss fraction
    miss_frac = np.clip(col_miss_frac, 0.02, 0.98)
    col_mean = L_pp.mean(axis=0)
    logit_mf = np.log(miss_frac / (1.0 - miss_frac))
    theta = col_mean - logit_mf / alpha

    # Gradient descent on sigmoid model
    for it in range(mnar_iter):
        sig = sigmoid(alpha * (L_pp - theta[np.newaxis, :]))
        mnar_resid = sig - miss_mask
        sig_deriv = sig * (1.0 - sig)
        # Weighted gradient: zero for non-crash columns
        grad_theta = (-2.0 * alpha * col_mnar_weight * np.sum(mnar_resid * sig_deriv, axis=0))
        theta = theta - lr_theta * grad_theta

    S_crash_prob = sigmoid(alpha * (L_pp - theta[np.newaxis, :]))

    if len(crash_cols) > 0:
        safe_c = [c for c in range(n) if c not in crash_cols]
        sep = theta[safe_c].mean() - theta[crash_cols].mean()
        print(f"    -> Theta separation (safe - crash): {sep:.3f}")

    return L_pp, S_tamper, theta, S_crash_prob, crash_cols, rank


# ========================================================================
# MAIN: Run MCDD on all three conditions
# ========================================================================
DATA_DIR = "data"
conditions = [
    ('A_random',      'M_A.npy'),
    ('B_structured',  'M_B.npy'),
    ('C_adversarial', 'M_C.npy'),
]

for cond, fname in conditions:
    print(f"\n{'='*70}")
    print(f"MCDD — Condition: {cond}")
    print(f"{'='*70}")

    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L_pp, S_tamper, theta, S_cp, crash_cols, rank = mcdd(M_obs)

    np.save(os.path.join(DATA_DIR, f'L_mcdd_{cond}.npy'), L_pp)
    np.save(os.path.join(DATA_DIR, f'S_tamper_mcdd_{cond}.npy'), S_tamper)
    np.save(os.path.join(DATA_DIR, f'theta_mcdd_{cond}.npy'), theta)
    np.save(os.path.join(DATA_DIR, f'S_crash_prob_mcdd_{cond}.npy'), S_cp)

    print(f"  Saved outputs for {cond}")
