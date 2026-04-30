# step8_sensitivity_analysis.py  — Lambda sensitivity sweep
# Uses the IDENTICAL mcdd_admm as step5 (Y2 coupling inside loop, no post-processing patch).
# Previously step8 had Z1 += Y2/rho but step5 did not — they were testing different algorithms.
# Now both are consistent.

import numpy as np
import os
from sklearn.metrics import f1_score


# ── copy the exact same functions from step5 ────────────────────────────────

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank


def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def mcdd_admm(M_obs, lam1=None, lam2=None, rho=1.0, max_iter=300, tol=1e-4):
    """Identical to step5 — Y2 coupling inside loop."""
    m, n = M_obs.shape
    if lam1 is None: lam1 = 1.0 / np.sqrt(max(m, n))
    if lam2 is None: lam2 = 1.5 / np.sqrt(max(m, n))

    mask_obs = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill = np.nan_to_num(M_obs, nan=0.0)

    L = M_fill.copy()
    S_crash = miss_mask.copy()
    S_tamper = np.zeros((m, n))
    Y1 = np.zeros((m, n))
    Y2 = np.zeros((m, n))

    for it in range(max_iter):
        L_old = L.copy()

        # L update WITH Y2 coupling (crash-aware)
        Z1 = M_fill - S_tamper + (Y1 + Y2) / rho
        L, rank = svd_threshold(Z1, 1.0 / rho)

        # S_tamper update
        Z2 = M_fill - L + Y1 / rho
        S_t = soft_threshold(Z2, lam2 / rho)
        S_t[~mask_obs] = 0.0
        S_tamper = S_t

        # S_crash update (novel: mask decomposition)
        Z3 = miss_mask + Y2 / rho
        S_crash = np.clip(soft_threshold(Z3, lam1 / rho), 0.0, 1.0)

        # Dual updates
        res1 = M_fill - L - S_tamper
        res1[~mask_obs] = 0.0
        Y1 += rho * res1
        Y2 += rho * (miss_mask - S_crash)

        rel_change = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10)
        if rel_change < tol:
            break

    return L, S_crash, S_tamper


# ── metric helpers ──────────────────────────────────────────────────────────

def rmse(true, pred, mask):
    return np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))


def detection_f1(true_mask, signal, threshold):
    t = true_mask.flatten().astype(int)
    p = (signal.flatten() > threshold).astype(int)
    return f1_score(t, p, zero_division=0)


# ── sweep ───────────────────────────────────────────────────────────────────

DATA_DIR = "data"
M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
M_B    = np.load(os.path.join(DATA_DIR, 'M_B.npy'))
mask_B = np.load(os.path.join(DATA_DIR, 'mask_B.npy'))

base = 1.0 / np.sqrt(max(M_true.shape))
lam_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]

print(f"Base lambda: {base:.4f}")
print(f"Matrix shape: {M_true.shape}  |  Crash fraction: {mask_B.mean()*100:.1f}%")
print(f"\n{'λ1 mult':<12} {'λ1':<10} {'λ2':<10} {'RMSE':<10} {'Crash F1':<10}")
print("-" * 52)

rmse_vals = []
f1_vals   = []

for mult in lam_multipliers:
    lam1 = mult * base
    lam2 = 1.5 * mult * base
    L, S_crash, _ = mcdd_admm(M_B, lam1=lam1, lam2=lam2, max_iter=300)
    r  = rmse(M_true, L, mask_B)
    f1 = detection_f1(mask_B, S_crash, threshold=0.3)
    rmse_vals.append(r)
    f1_vals.append(f1)
    print(f"{mult:<12.2f} {lam1:<10.4f} {lam2:<10.4f} {r:<10.4f} {f1:<10.3f}")

# Robustness check: if RMSE varies by less than 10% of its mean, results are robust
rmse_range = (max(rmse_vals) - min(rmse_vals)) / np.mean(rmse_vals) * 100
f1_range   = (max(f1_vals)   - min(f1_vals))   / (np.mean(f1_vals) + 1e-10) * 100
print(f"\nRMSE variation across lambda sweep: {rmse_range:.1f}%  (target < 10%)")
print(f"Crash F1 variation across lambda sweep: {f1_range:.1f}%  (target < 10%)")
print("\n=== Step 8 complete ===")