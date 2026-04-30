# step5_mcdd.py  — MCDD: Missingness-Coupled Dual Decomposition
# Group-sparse S_crash: uses L2,1 column-wise norm instead of entry-wise L1.
#
# WHY THIS MATTERS:
#   Entry-wise L1 on S_crash cannot distinguish:
#     - 9911 randomly scattered missing entries (Condition A)
#     - 12420 block-missing entries in 15 columns (Condition B)
#   Both have similar total L1 mass, so S_crash ≈ missing mask in all cases,
#   and the Y2 coupling carries no discriminative signal into L.
#
#   L2,1 norm = sum_j ||S_crash[:,j]||_2 promotes COLUMN-WISE sparsity:
#   either an entire column is active (node crashed) or it's zero.
#   Random missing spreads across all columns -> small per-column norms
#   -> L2,1 shrinks them to zero.
#   Block crash wipes contiguous rows in specific columns -> large per-column norms
#   -> L2,1 preserves them.
#   This is the correct structural prior for node crash in distributed systems.
#
# Mathematical formulation:
#   min  ||L||_*  +  lam1*||S_crash||_{2,1}  +  lam2*||S_tamper||_1
#   s.t. P_Omega(L + S_tamper)  =  P_Omega(M_obs)   [data fidelity]
#        (1 - Omega)            =  S_crash            [mask decomposition]
#
# Proximal operator of L2,1 = column-wise block soft thresholding.

import numpy as np
import os


# ── proximal operators ──────────────────────────────────────────────────────

def svd_threshold(X, tau):
    """Singular-value thresholding: proximal operator of nuclear norm."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0)
    rank = int(np.sum(s_t > 0))
    return U @ np.diag(s_t) @ Vt, rank


def soft_threshold(X, tau):
    """Entry-wise soft thresholding: proximal operator of L1 norm."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def col_block_soft_threshold(X, tau):
    """
    Column-wise block soft thresholding: proximal operator of L2,1 norm.

    For each column j:
        if ||X[:,j]||_2 <= tau  ->  output[:,j] = 0   (no crash in this column)
        else                    ->  output[:,j] = X[:,j] * (1 - tau/||X[:,j]||_2)

    Effect: columns with concentrated missing (node crash) have large norms
    and are preserved. Randomly scattered missing produces small per-column
    norms and gets zeroed out entirely. This is what makes S_crash
    discriminate structured crashes from random dropout.
    """
    col_norms = np.linalg.norm(X, axis=0, keepdims=True)   # shape (1, n)
    scale = np.maximum(1.0 - tau / (col_norms + 1e-10), 0.0)
    return X * scale


# ── main algorithm ──────────────────────────────────────────────────────────

def mcdd_admm(M_obs, lam1=None, lam2=None, rho=1.0, max_iter=300, tol=1e-4):
    """
    MCDD via ADMM — Missingness-Coupled Dual Decomposition.

    Parameters
    ----------
    M_obs    : (m, n) array with np.nan for missing entries
    lam1     : L2,1 penalty on S_crash  (default 1/sqrt(max(m,n)))
    lam2     : L1  penalty on S_tamper  (default 1.5/sqrt(max(m,n)))
    rho      : ADMM penalty parameter
    max_iter : maximum ADMM iterations
    tol      : convergence tolerance on relative change in L

    Returns
    -------
    L, S_crash, S_tamper, losses, primal_res, dual_res
    """
    m, n = M_obs.shape
    if lam1 is None:
        lam1 = 1.0 / np.sqrt(max(m, n))
    if lam2 is None:
        lam2 = 1.5 / np.sqrt(max(m, n))

    mask_obs  = ~np.isnan(M_obs)
    miss_mask = (~mask_obs).astype(float)
    M_fill    = np.nan_to_num(M_obs, nan=0.0)

    # ── Initialisation ──────────────────────────────────────────────────────
    L        = M_fill.copy()
    S_crash  = miss_mask.copy()
    S_tamper = np.zeros((m, n))
    Y1       = np.zeros((m, n))   # dual: data fidelity constraint
    Y2       = np.zeros((m, n))   # dual: mask decomposition — the coupling mechanism

    losses     = []
    primal_res = []
    dual_res   = []

    for it in range(max_iter):
        L_old = L.copy()

        # ── Subproblem 1: L update (crash-aware via Y2 coupling) ────────────
        # Y2 carries information about which columns are detected as crashed
        # (via S_crash) and pulls the L reconstruction to use the low-rank
        # subspace learned from safe columns to fill crashed ones.
        Z1 = M_fill - S_tamper + (Y1 + Y2) / rho
        L, rank = svd_threshold(Z1, 1.0 / rho)

        # ── Subproblem 2: S_tamper update (entry-wise L1) ───────────────────
        Z2 = M_fill - L + Y1 / rho
        S_t = soft_threshold(Z2, lam2 / rho)
        S_t[~mask_obs] = 0.0           # attacker can only modify observed entries
        S_tamper = S_t

        # ── Subproblem 3: S_crash update (column-wise L2,1) — THE NOVELTY ───
        # Block soft thresholding: preserves columns with concentrated missing
        # (structured crash), zeroes out columns with scattered missing (random).
        Z3      = miss_mask + Y2 / rho
        S_crash = np.clip(col_block_soft_threshold(Z3, lam1 / rho), 0.0, 1.0)

        # ── Dual variable updates ───────────────────────────────────────────
        res1 = M_fill - L - S_tamper
        res1[~mask_obs] = 0.0
        Y1 += rho * res1

        res2 = miss_mask - S_crash
        Y2  += rho * res2

        # ── Logging ─────────────────────────────────────────────────────────
        col_norms_crash = np.linalg.norm(S_crash, axis=0)
        loss = (np.linalg.norm(L, 'nuc')
                + lam1 * np.sum(col_norms_crash)
                + lam2 * np.sum(np.abs(S_tamper)))
        losses.append(loss)
        primal_res.append(np.linalg.norm(res1, 'fro'))
        dual_res.append(np.linalg.norm(res2, 'fro'))

        # ── Convergence ──────────────────────────────────────────────────────
        rel_change = (np.linalg.norm(L - L_old, 'fro')
                      / (np.linalg.norm(L_old, 'fro') + 1e-10))
        if rel_change < tol:
            print(f"  Converged at iteration {it+1}  (rel_change={rel_change:.2e})")
            break

    return L, S_crash, S_tamper, np.array(losses), np.array(primal_res), np.array(dual_res)


# ── run on all three conditions ─────────────────────────────────────────────

DATA_DIR = "data"
conditions = [
    ('A_random',      'M_A.npy'),
    ('B_structured',  'M_B.npy'),
    ('C_adversarial', 'M_C.npy'),
]

for cond, fname in conditions:
    print(f"\nRunning MCDD on condition {cond}...")
    M_obs = np.load(os.path.join(DATA_DIR, fname))
    L, S_crash, S_tamper, losses, pr, dr = mcdd_admm(M_obs)

    crash_col_norms     = np.linalg.norm(S_crash, axis=0)
    detected_crash_cols = np.where(crash_col_norms > 0.1)[0]

    np.save(os.path.join(DATA_DIR, f'L_mcdd_{cond}.npy'),      L)
    np.save(os.path.join(DATA_DIR, f'S_crash_{cond}.npy'),     S_crash)
    np.save(os.path.join(DATA_DIR, f'S_tamper_{cond}.npy'),    S_tamper)
    np.save(os.path.join(DATA_DIR, f'losses_mcdd_{cond}.npy'), losses)
    np.save(os.path.join(DATA_DIR, f'primal_res_{cond}.npy'),  pr)
    np.save(os.path.join(DATA_DIR, f'dual_res_{cond}.npy'),    dr)

    print(f"  rank(L) ≈ {int(np.linalg.matrix_rank(L, tol=0.01))}")
    print(f"  Crashed columns detected (col_norm > 0.1): {len(detected_crash_cols)}  -> cols {detected_crash_cols.tolist()}")
    print(f"  Tamper entries detected (95th pct): "
          f"{int((np.abs(S_tamper) > np.percentile(np.abs(S_tamper), 95)).sum())}")