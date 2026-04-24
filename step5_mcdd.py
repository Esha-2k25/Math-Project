# step5_mcdd.py (with explicit crash‑aware reconstruction)
import numpy as np
import os

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = np.sum(s_t > 0)
    return U @ np.diag(s_t) @ Vt, rank

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def mcdd_admm(M_obs, lam1=None, lam2=None, rho=1.0, max_iter=300, tol=1e-4):
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
    losses = []

    for it in range(max_iter):
        L_old = L.copy()

        # Standard RPCA‑style L update (no Y2 coupling – we'll add crash awareness later)
        Z1 = M_fill - S_tamper + Y1/rho
        L, rank = svd_threshold(Z1, 1.0/rho)

        # S_tamper update (observed only)
        Z2 = M_fill - L + Y1/rho
        S_t = soft_threshold(Z2, lam2/rho)
        S_t[~mask_obs] = 0
        S_tamper = S_t

        # S_crash update (mask decomposition)
        Z3 = miss_mask + Y2/rho
        S_crash = np.clip(soft_threshold(Z3, lam1/rho), 0, 1)

        # Dual updates
        res1 = M_fill - L - S_tamper
        res1[~mask_obs] = 0
        Y1 += rho * res1
        Y2 += rho * (miss_mask - S_crash)

        loss = np.linalg.norm(L, 'nuc') + lam1*np.sum(np.abs(S_crash)) + lam2*np.sum(np.abs(S_tamper))
        losses.append(loss)

        if np.linalg.norm(L - L_old, 'fro') < tol:
            break

    # ----- POST‑PROCESSING: Crash‑aware low‑rank projection -----
    # Use S_crash to identify crash columns (threshold 0.5)
    crash_cols = np.where(S_crash.max(axis=0) > 0.5)[0]
    if len(crash_cols) > 0:
        # Use columns that are NOT crashed to learn the row subspace
        safe_cols = [c for c in range(n) if c not in crash_cols]
        if len(safe_cols) >= 2:
            U_safe, _, _ = np.linalg.svd(L[:, safe_cols], full_matrices=False)
            k = min(rank, len(safe_cols)-1, m-1)
            if k > 0:
                U_k = U_safe[:, :k]
                # For each crashed column, reconstruct using U_k
                for c in crash_cols:
                    obs_rows = mask_obs[:, c]
                    if obs_rows.sum() >= k:
                        # Least squares fit on observed rows
                        A = U_k[obs_rows, :]
                        b = L[obs_rows, c]
                        alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        # Predict missing rows
                        missing_rows = ~mask_obs[:, c]
                        L[missing_rows, c] = U_k[missing_rows, :] @ alpha

    return L, S_crash, S_tamper, np.array(losses)

DATA_DIR = "data"
conditions = [('A_random', 'M_A.npy'), ('B_structured', 'M_B.npy'), ('C_adversarial', 'M_C.npy')]

for cond, fname in conditions:
    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L, S_crash, S_tamper, losses = mcdd_admm(M_obs)
    np.save(os.path.join(DATA_DIR, f'L_mcdd_{cond}.npy'), L)
    np.save(os.path.join(DATA_DIR, f'S_crash_{cond}.npy'), S_crash)
    np.save(os.path.join(DATA_DIR, f'S_tamper_{cond}.npy'), S_tamper)
    np.save(os.path.join(DATA_DIR, f'losses_mcdd_{cond}.npy'), losses)
    print(f"MCDD {cond} done")