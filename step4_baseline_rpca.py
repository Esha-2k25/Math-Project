# step4_baseline_rpca.py — Standard RPCA via ADMM
#
# MATHEMATICAL FORMULATION:
#   min  ||L||_* + lambda * ||S||_1
#   s.t. P_Omega(L + S) = P_Omega(M_obs)
#
# ADMM AUGMENTED LAGRANGIAN:
#   L_rho(L,S,Y) = ||L||_* + lambda*||S||_1 + <Y, M_fill - L - S>
#                  + (rho/2)||M_fill - L - S||_F^2
#
# ADMM UPDATES (each iteration):
#   L_{k+1} = argmin_L ||L||_* + (rho/2)||M_fill - S_k - L + Y_k/rho||_F^2
#           = SVD_threshold(M_fill - S_k + Y_k/rho, 1/rho)
#
#   S_{k+1} = argmin_S lambda*||S||_1 + (rho/2)||M_fill - L_{k+1} - S + Y_k/rho||_F^2
#           = soft_threshold(M_fill - L_{k+1} + Y_k/rho, lambda/rho)
#
#   Y_{k+1} = Y_k + rho*(M_fill - L_{k+1} - S_{k+1})
#
# The projection P_Omega is enforced by zeroing residuals on observed entries.

import numpy as np
import os

def soft_threshold(X, tau):
    """Element-wise soft-thresholding: shrink(X, tau)"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svd_threshold(X, tau):
    """Singular value thresholding: SVT(X, tau) = U * diag(max(s-tau,0)) * V^T"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank

def standard_rpca(M_obs, lam=None, rho=1.0, max_iter=300, tol=1e-3):
    """
    Standard masked RPCA using ADMM.

    Parameters:
      M_obs : observed matrix with NaN for missing entries
      lam   : regularization for sparse component (default: 1/sqrt(max(m,n)))
      rho   : ADMM penalty parameter
      max_iter, tol : convergence criteria

    Returns:
      L : low-rank reconstruction
      S : sparse corruption component
    """
    m, n = M_obs.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    mask_obs = ~np.isnan(M_obs)
    M_fill = np.nan_to_num(M_obs, nan=0.0)

    L = M_fill.copy()
    S = np.zeros((m, n))
    Y = np.zeros((m, n))  # dual variable (Lagrangian multiplier)

    for i in range(max_iter):
        L_old = L.copy()

        # L-update: singular value thresholding
        L, rank = svd_threshold(M_fill - S + Y/rho, 1.0/rho)

        # S-update: soft thresholding (sparse corruption)
        S_new = soft_threshold(M_fill - L + Y/rho, lam/rho)
        S_new[~mask_obs] = 0  # enforce: S=0 on missing entries
        S = S_new

        # Y-update: dual ascent (augmented Lagrangian multiplier)
        res = M_fill - L - S
        res[~mask_obs] = 0    # only enforce on observed entries
        Y += rho * res

        # Convergence check
        rel_change = np.linalg.norm(L - L_old, 'fro') / (np.linalg.norm(L_old, 'fro') + 1e-10)
        if rel_change < tol:
            print(f"    ADMM converged at iteration {i+1} (rel_change={rel_change:.2e})")
            break
    else:
        print(f"    ADMM reached max_iter ({max_iter})")

    return L, S

# ── Run on all conditions ──────────────────────────────────────────────────
DATA_DIR = "data"
conditions = [
    ('A_random',      'M_A.npy'),
    ('B_structured',  'M_B.npy'),
    ('C_adversarial', 'M_C.npy'),
]

for cond, fname in conditions:
    print(f"Running Standard RPCA on: {cond}")
    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L, S = standard_rpca(M_obs)
    np.save(os.path.join(DATA_DIR, f'L_rpca_{cond}.npy'), L)
    np.save(os.path.join(DATA_DIR, f'S_rpca_{cond}.npy'), S)
    print(f"  Saved L_rpca_{cond}.npy (rank={np.linalg.matrix_rank(L, tol=1e-8)})")
