# step4_baseline_rpca.py (static fill)
import numpy as np
import os

def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svd_threshold(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank

def standard_rpca(M_obs, lam=None, rho=1.0, max_iter=300, tol=1e-4):
    m, n = M_obs.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    mask_obs = ~np.isnan(M_obs)
    M_fill = np.nan_to_num(M_obs, nan=0.0)
    L = M_fill.copy()
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    for i in range(max_iter):
        L_old = L.copy()
        L, rank = svd_threshold(M_fill - S + Y/rho, 1.0/rho)
        S_new = soft_threshold(M_fill - L + Y/rho, lam/rho)
        S_new[~mask_obs] = 0
        S = S_new
        res = M_fill - L - S
        res[~mask_obs] = 0
        Y += rho * res
        if np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10) < tol:
            break
    return L, S

DATA_DIR = "data"
conditions = [('A_random', 'M_A.npy'), ('B_structured', 'M_B.npy'), ('C_adversarial', 'M_C.npy')]
for cond, fname in conditions:
    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L, _ = standard_rpca(M_obs)
    np.save(os.path.join(DATA_DIR, f'L_rpca_{cond}.npy'), L)
    print(f"RPCA {cond} done")