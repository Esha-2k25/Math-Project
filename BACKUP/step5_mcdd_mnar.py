# step5_mcdd_fixed.py — Fixed MCDD with Decoupled MNAR Analysis
# RUN THIS INSTEAD OF step5_mcdd_mnar.py

import numpy as np
import os

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def mcdd_fixed(M_obs, lam2=None, rho=1.0, max_iter=300, tol=1e-4,
               alpha=1.0, lr_theta=0.05, mnar_iter=200):
    m, n = M_obs.shape
    if lam2 is None:
        lam2 = 1.5 / np.sqrt(max(m, n))
    
    mask_obs = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill = np.nan_to_num(M_obs, nan=0.0)
    col_miss_frac = miss_mask.mean(axis=0)
    
    # === STAGE 1: Standard masked RPCA ===
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
    
    # === STAGE 2: Crash detection (honest) ===
    crash_cols = np.where(col_miss_frac > 0.25)[0]
    safe_cols = [c for c in range(n) if c not in crash_cols]
    
    print(f"  Stage 1: RPCA converged, rank(L)={rank}")
    print(f"  Stage 2: Crash cols detected (miss_frac>0.25): {crash_cols.tolist()}")
    
    # === STAGE 3: Post-processing subspace projection ===
    # Use rank from Stage 1 RPCA (8 for real Thunderbird), not hardcoded 5
    L_pp = L.copy()
    if len(crash_cols) > 0 and len(safe_cols) >= 2:
        U_safe, s_safe, _ = np.linalg.svd(L_pp[:, safe_cols], full_matrices=False)
        k = min(rank, len(safe_cols) - 1, m - 1)  # <-- USE rank FROM STAGE 1
        k = max(k, 1)
        
        U_k = U_safe[:, :k]
        for c in crash_cols:
            obs_rows = mask_obs[:, c]
            missing_rows = ~obs_rows
            if obs_rows.sum() >= k:
                A = U_k[obs_rows, :]
                b = M_fill[obs_rows, c]  # true observed values
                coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                L_pp[missing_rows, c] = U_k[missing_rows, :] @ coeff
        print(f"  Stage 3: Subspace projection done (rank-{k} from {len(safe_cols)} safe cols)")
    else:
        print(f"  Stage 3: No projection applied")
    
    # === STAGE 4: Post-hoc MNAR analysis ===
    col_mnar_weight = np.clip((col_miss_frac - 0.25) / 0.25, 0.0, 1.0)
    miss_frac = np.clip(col_miss_frac, 0.02, 0.98)
    col_mean = L_pp.mean(axis=0)
    logit_mf = np.log(miss_frac / (1.0 - miss_frac))
    theta = col_mean - logit_mf / alpha
    
    for it in range(mnar_iter):
        sig = sigmoid(alpha * (L_pp - theta[np.newaxis, :]))
        mnar_resid = sig - miss_mask
        sig_deriv = sig * (1.0 - sig)
        grad_theta = (-2.0 * alpha * col_mnar_weight * np.sum(mnar_resid * sig_deriv, axis=0))
        theta = theta - lr_theta * grad_theta
    
    S_crash_prob = sigmoid(alpha * (L_pp - theta[np.newaxis, :]))
    
    print(f"  Stage 4: MNAR analysis complete")
    if len(crash_cols) > 0:
        safe_c = [c for c in range(n) if c not in crash_cols]
        sep = theta[safe_c].mean() - theta[crash_cols].mean()
        print(f"    Theta separation (safe - crash): {sep:.3f}")
    
    return L_pp, S_tamper, theta, S_crash_prob, crash_cols, rank


DATA_DIR = "data"
conditions = [
    ('A_random',      'M_A.npy'),
    ('B_structured',  'M_B.npy'),
    ('C_adversarial', 'M_C.npy'),
]

for cond, fname in conditions:
    print(f"\n{'='*60}")
    print(f"Running Fixed MCDD on: {cond}")
    print(f"{'='*60}")
    
    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L, S_tamper, theta, S_cp, crash_cols, rank = mcdd_fixed(M_obs)
    
    np.save(os.path.join(DATA_DIR, f'L_fixed_{cond}.npy'), L)
    np.save(os.path.join(DATA_DIR, f'S_tamper_fixed_{cond}.npy'), S_tamper)
    np.save(os.path.join(DATA_DIR, f'theta_fixed_{cond}.npy'), theta)
    np.save(os.path.join(DATA_DIR, f'S_crash_prob_fixed_{cond}.npy'), S_cp)
    
    print(f"  Saved all outputs for {cond}")