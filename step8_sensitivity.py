# step8_sensitivity.py — Parameter Sensitivity Analysis for MCDD
#
# Sweeps:
#   1. alpha (sigmoid steepness) — affects theta learning
#   2. lam2 (sparse regularization) — affects S_tamper and L rank
#   3. rho (ADMM penalty) — affects convergence speed

import numpy as np
import os

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    return U @ np.diag(s_t) @ Vt, int(np.sum(s_t > 0))

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mcdd_sweep(M_obs, lam2=None, rho=1.0, max_iter=300, tol=1e-4,
               alpha=1.0, lr_theta=0.05, mnar_iter=200):
    """MCDD with configurable parameters for sensitivity analysis."""
    m, n = M_obs.shape
    if lam2 is None:
        lam2 = 1.5 / np.sqrt(max(m, n))

    mask_obs = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill = np.nan_to_num(M_obs, nan=0.0)
    col_miss_frac = miss_mask.mean(axis=0)

    # STAGE 1: ADMM
    L = M_fill.copy()
    S_tamper = np.zeros((m, n))
    Y = np.zeros((m, n))

    for i in range(max_iter):
        L_old = L.copy()
        L, rank = svd_threshold(M_fill - S_tamper + Y/rho, 1.0/rho)
        S_new = soft_threshold(M_fill - L + Y/rho, lam2/rho)
        S_new[~mask_obs] = 0
        S_tamper = S_new
        res = M_fill - L - S_tamper
        res[~mask_obs] = 0
        Y += rho * res
        if np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10) < tol:
            break

    # STAGE 2: Crash detection
    crash_cols = np.where(col_miss_frac > 0.25)[0]
    safe_cols = [c for c in range(n) if c not in crash_cols]

    # STAGE 3: Subspace projection
    if len(crash_cols) > 0 and len(safe_cols) >= 2:
        U_safe, s_safe, _ = np.linalg.svd(L[:, safe_cols], full_matrices=False)
        explained_safe = np.cumsum(s_safe**2) / np.sum(s_safe**2)
        k = max(1, np.searchsorted(explained_safe, 0.985) + 1)
        k = min(k, len(safe_cols) - 1, m - 1)
        U_k = U_safe[:, :k]
        for c in crash_cols:
            obs_rows = mask_obs[:, c]
            missing_rows = ~obs_rows
            if obs_rows.sum() >= k:
                A = U_k[obs_rows, :]
                b = M_fill[obs_rows, c]
                coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                L[missing_rows, c] = U_k[missing_rows, :] @ coeff

    # STAGE 4: MNAR (post-hoc)
    col_mnar_weight = np.clip((col_miss_frac - 0.25) / 0.25, 0.0, 1.0)
    miss_frac = np.clip(col_miss_frac, 0.02, 0.98)
    col_mean = L.mean(axis=0)
    logit_mf = np.log(miss_frac / (1.0 - miss_frac))
    theta = col_mean - logit_mf / alpha

    for it in range(mnar_iter):
        sig = sigmoid(alpha * (L - theta[np.newaxis, :]))
        mnar_resid = sig - miss_mask
        sig_deriv = sig * (1.0 - sig)
        grad_theta = (-2.0 * alpha * col_mnar_weight * np.sum(mnar_resid * sig_deriv, axis=0))
        theta = theta - lr_theta * grad_theta

    return L, theta

def rmse(true, pred, mask):
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))

# ── Load data ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
M_B = np.load(os.path.join(DATA_DIR, 'M_B.npy'))
mask_B = np.load(os.path.join(DATA_DIR, 'mask_B.npy'))

print("="*80)
print("SENSITIVITY ANALYSIS: MCDD Parameter Robustness")
print("="*80)

# ── Sweep 1: alpha (sigmoid steepness) ─────────────────────────────────────
print("\n" + "-"*80)
print("SWEEP 1: alpha (sigmoid steepness) | lam2=default, rho=1.0")
print("-"*80)
print(f"{'alpha':<10} {'RMSE':<10} {'Rank':<8} {'Theta sep':<12}")
print("-"*80)

for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    L, theta = mcdd_sweep(M_B, alpha=alpha)
    r = rmse(M_true, L, mask_B)
    crash_cols = list(range(5, 20))
    safe_cols = list(range(0, 5)) + list(range(20, 30))
    sep = theta[safe_cols].mean() - theta[crash_cols].mean()
    rank = np.linalg.matrix_rank(L, tol=1e-8)
    print(f"{alpha:<10.1f} {r:<10.4f} {rank:<8} {sep:<12.4f}")

# ── Sweep 2: lam2 (sparse regularization) ──────────────────────────────────
print("\n" + "-"*80)
print("SWEEP 2: lam2 (sparse regularization) | alpha=1.0, rho=1.0")
print("-"*80)
print(f"{'lam2':<12} {'RMSE':<10} {'Rank':<8} {'Theta sep':<12}")
print("-"*80)

m, n = M_B.shape
default_lam2 = 1.5 / np.sqrt(max(m, n))
for lam2_mult in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    lam2 = default_lam2 * lam2_mult
    L, theta = mcdd_sweep(M_B, lam2=lam2)
    r = rmse(M_true, L, mask_B)
    sep = theta[list(range(0,5))+list(range(20,30))].mean() - theta[list(range(5,20))].mean()
    rank = np.linalg.matrix_rank(L, tol=1e-8)
    print(f"{lam2:.4f}      {r:<10.4f} {rank:<8} {sep:<12.4f}")

# ── Sweep 3: rho (ADMM penalty) ────────────────────────────────────────────
print("\n" + "-"*80)
print("SWEEP 3: rho (ADMM penalty parameter) | alpha=1.0, lam2=default")
print("-"*80)
print(f"{'rho':<10} {'RMSE':<10} {'Rank':<8} {'Theta sep':<12}")
print("-"*80)

for rho in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    L, theta = mcdd_sweep(M_B, rho=rho)
    r = rmse(M_true, L, mask_B)
    sep = theta[list(range(0,5))+list(range(20,30))].mean() - theta[list(range(5,20))].mean()
    rank = np.linalg.matrix_rank(L, tol=1e-8)
    print(f"{rho:<10.1f} {r:<10.4f} {rank:<8} {sep:<12.4f}")

print("\n" + "="*80)
print("Sensitivity analysis complete.")
print("="*80)
