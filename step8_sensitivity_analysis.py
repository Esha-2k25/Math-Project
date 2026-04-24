# step8_sensitivity_analysis.py (unchanged from original)
import numpy as np
import os
from sklearn.metrics import f1_score

DATA_DIR = "data"

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank

def mcdd_admm(M_obs, lam1=None, lam2=None, rho=1.0, max_iter=300, tol=1e-4):
    m, n = M_obs.shape
    if lam1 is None: lam1 = 1.0 / np.sqrt(max(m, n))
    if lam2 is None: lam2 = 1.5 / np.sqrt(max(m, n))
    mask_obs = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill = np.nan_to_num(M_obs, nan=0.0)
    L = M_fill.copy(); S_crash = miss_mask.copy()
    S_tamper = np.zeros((m, n)); Y1 = np.zeros((m, n)); Y2 = np.zeros((m, n))
    for it in range(max_iter):
        L_old = L.copy()
        Z1 = M_fill - S_tamper + Y1/rho
        Z1 += Y2 / rho
        L, rank = svd_threshold(Z1, 1.0/rho)
        Z2 = M_fill - L + Y1/rho
        S_t = soft_threshold(Z2, lam2/rho); S_t[~mask_obs] = 0.0; S_tamper = S_t
        Z3 = miss_mask + Y2/rho
        S_crash = np.clip(soft_threshold(Z3, lam1/rho), 0.0, 1.0)
        r1 = M_fill - L - S_tamper; r1[~mask_obs] = 0.0; Y1 += rho * r1
        Y2 += rho * (miss_mask - S_crash)
        delta = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10)
        if delta < tol:
            break
    return L, S_crash, S_tamper

def rmse(true, pred, mask):
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))

def detection_f1(true_mask, signal, threshold):
    t = true_mask.flatten().astype(int)
    p = (signal.flatten() > threshold).astype(int)
    return f1_score(t, p, zero_division=0)

M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
M_B = np.load(os.path.join(DATA_DIR, 'M_B.npy'))
mask_B = np.load(os.path.join(DATA_DIR, 'mask_B.npy'))

base = 1.0 / np.sqrt(max(M_true.shape))
lam_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]

print(f"Base lambda: {base:.4f}")
print(f"\n{'λ1 mult':<12} {'λ1':<10} {'λ2':<10} {'RMSE':<10} {'Crash F1':<10}")
print("-" * 52)

for mult in lam_multipliers:
    lam1 = mult * base
    lam2 = 1.5 * mult * base
    L, S_crash, _ = mcdd_admm(M_B, lam1=lam1, lam2=lam2, max_iter=300)
    r = rmse(M_true, L, mask_B)
    f1 = detection_f1(mask_B, S_crash, threshold=0.3)
    print(f"{mult:<12.2f} {lam1:<10.4f} {lam2:<10.4f} {r:<10.4f} {f1:<10.3f}")

print("\n=== Step 8 complete ===")