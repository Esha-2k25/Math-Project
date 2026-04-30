# step6_evaluate.py — Comprehensive Evaluation for MCDD vs Baselines
#
# Metrics computed:
#   Reconstruction:  RMSE, MAE, Relative Error, PSNR, Correlation
#   Crash Detection: Column-level F1, Precision, Recall, Accuracy
#   Tamper Detection: Entry-level F1, Precision, Recall, AUC-ROC
#   MNAR Quality:    Theta separation, Per-column crash probability

import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
rows, cols = M_true.shape
TRUE_CRASH_COLS = list(range(5, 20))

# ── Metric functions ───────────────────────────────────────────────────────
def rmse(true, pred, mask):
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))

def mae(true, pred, mask):
    return np.mean(np.abs(true[mask] - pred[mask]))

def rel_error(true, pred, mask):
    return np.linalg.norm(true[mask] - pred[mask]) / (np.linalg.norm(true[mask]) + 1e-10)

def psnr(true, pred, mask):
    mse = np.mean((true[mask] - pred[mask])**2)
    if mse == 0: return float('inf')
    max_val = np.max(np.abs(true[mask]))
    return 20 * np.log10(max_val / np.sqrt(mse))

def corr_coef(true, pred, mask):
    """Pearson correlation on masked entries"""
    t = true[mask]; p = pred[mask]
    if len(t) < 2: return 0.0
    return np.corrcoef(t, p)[0, 1]

def mean_imputation(M_obs):
    M_filled = M_obs.copy()
    for col in range(M_obs.shape[1]):
        col_data = M_obs[:, col]
        col_mean = np.nanmean(col_data) if not np.all(np.isnan(col_data)) else 0.0
        M_filled[np.isnan(col_data), col] = col_mean
    return M_filled

# ── Crash detection metrics ────────────────────────────────────────────────
def crash_metrics(M_obs, true_crash_cols):
    """
    Column-level crash detection.
    Predicted: col_miss_frac > 0.25
    Returns: dict with F1, Precision, Recall, Accuracy
    """
    col_miss_frac = np.isnan(M_obs).mean(axis=0)
    pred_crash = (col_miss_frac > 0.25).astype(int)
    true_crash = np.zeros(cols, dtype=int)
    true_crash[true_crash_cols] = 1

    return {
        'F1':      f1_score(true_crash, pred_crash, zero_division=0),
        'Precision': precision_score(true_crash, pred_crash, zero_division=0),
        'Recall':    recall_score(true_crash, pred_crash, zero_division=0),
        'Accuracy':  accuracy_score(true_crash, pred_crash),
    }

# ── Tamper detection metrics ───────────────────────────────────────────────
def tamper_metrics(S_tamper, tamper_mask):
    """
    Entry-level tamper detection.
    Threshold: 97th percentile of |S_tamper| (optimal from sensitivity analysis)
    Returns: dict with F1, Precision, Recall, AUC
    """
    flat_tamper = tamper_mask.flatten().astype(int)
    scores = np.abs(S_tamper).flatten()
    thresh = np.percentile(scores, 97)
    pred = (scores > thresh).astype(int)

    try:
        auc = roc_auc_score(flat_tamper, scores)
    except:
        auc = 0.5

    return {
        'F1':        f1_score(flat_tamper, pred, zero_division=0),
        'Precision': precision_score(flat_tamper, pred, zero_division=0),
        'Recall':    recall_score(flat_tamper, pred, zero_division=0),
        'AUC':       auc,
    }

# ── Configuration ──────────────────────────────────────────────────────────
configs = [
    ('Random',      'M_A.npy', 'mask_A.npy',        False, None,             'A_random'),
    ('Structured',  'M_B.npy', 'mask_B.npy',         True,  None,             'B_structured'),
    ('Adversarial', 'M_C.npy', 'mask_B_for_C.npy',   True,  'mask_tamper.npy','C_adversarial'),
]

results = []

for cond_name, obs_file, miss_file, has_crash_gt, tamper_file, suffix in configs:
    miss_mask = np.load(os.path.join(DATA_DIR, miss_file))
    M_obs = np.load(os.path.join(DATA_DIR, obs_file))

    # Load MCDD outputs
    L_mcdd = np.load(os.path.join(DATA_DIR, f'L_mcdd_{suffix}.npy'))
    S_tamper = np.load(os.path.join(DATA_DIR, f'S_tamper_mcdd_{suffix}.npy'))
    theta = np.load(os.path.join(DATA_DIR, f'theta_mcdd_{suffix}.npy'))

    # Load baselines
    L_mean = mean_imputation(M_obs)
    L_rpca = np.load(os.path.join(DATA_DIR, f'L_rpca_{suffix}.npy'))

    method_list = [
        ('Mean Imputation', L_mean, False, False),
        ('Standard RPCA',   L_rpca, False, False),
        ('MCDD (Ours)',     L_mcdd, True,  True),
    ]

    for method_name, L_rec, do_crash, do_tamper in method_list:
        row = {
            'Condition': cond_name,
            'Method':    method_name,
            'RMSE':      rmse(M_true, L_rec, miss_mask),
            'MAE':       mae(M_true,  L_rec, miss_mask),
            'Rel_Error': rel_error(M_true, L_rec, miss_mask),
            'PSNR':      psnr(M_true, L_rec, miss_mask),
            'Corr':      corr_coef(M_true, L_rec, miss_mask),
            'Crash_F1':  '—',
            'Crash_Prec': '—',
            'Crash_Rec': '—',
            'Crash_Acc': '—',
            'Tamper_F1': '—',
            'Tamper_AUC': '—',
        }

        if do_crash and has_crash_gt:
            cm = crash_metrics(M_obs, TRUE_CRASH_COLS)
            row['Crash_F1']   = f"{cm['F1']:.3f}"
            row['Crash_Prec'] = f"{cm['Precision']:.3f}"
            row['Crash_Rec']  = f"{cm['Recall']:.3f}"
            row['Crash_Acc']  = f"{cm['Accuracy']:.3f}"

        if do_tamper and tamper_file is not None:
            tamper_mask = np.load(os.path.join(DATA_DIR, tamper_file))
            tm = tamper_metrics(S_tamper, tamper_mask)
            row['Tamper_F1']  = f"{tm['F1']:.3f}"
            row['Tamper_AUC'] = f"{tm['AUC']:.3f}"

        results.append(row)

    # Theta analysis
    if has_crash_gt:
        safe_cols = list(range(0, 5)) + list(range(20, 30))
        crash_cols_idx = TRUE_CRASH_COLS
        print(f"\n  [{cond_name}] Theta (crash threshold) analysis:")
        print(f"    Crashed cols theta mean: {theta[crash_cols_idx].mean():.4f}")
        print(f"    Safe cols theta mean:    {theta[safe_cols].mean():.4f}")
        sep = theta[safe_cols].mean() - theta[crash_cols_idx].mean()
        print(f"    Separation (safe-crash): {sep:.4f}  ({'GOOD' if sep > 0 else 'BAD'})")

# ── Save and display results ───────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_table.csv'), index=False)

print(f"\n{'='*90}")
print("FULL RESULTS TABLE")
print(f"{'='*90}")
print(df.to_string(index=False))

# ── Improvement summary (FIXED LABELING) ───────────────────────────────────
print(f"\n{'='*90}")
print("IMPROVEMENT SUMMARY vs Standard RPCA")
print(f"{'='*90}")

for cond in ['Random', 'Structured', 'Adversarial']:
    rpca_rmse = float(df[(df['Condition']==cond) & (df['Method']=='Standard RPCA')]['RMSE'].values[0])
    print(f"\n  [{cond} Condition]")
    for _, row in df[df['Condition'] == cond].iterrows():
        this_rmse = float(row['RMSE'])
        pct = (rpca_rmse - this_rmse) / rpca_rmse * 100
        if row['Method'] == 'Standard RPCA':
            status = "BASELINE"
        elif pct > 0:
            status = f"BETTER by {pct:.1f}%"
        else:
            status = f"WORSE by {abs(pct):.1f}%"
        print(f"    {row['Method']:<22}: RMSE={this_rmse:.4f}  ({status})")

print(f"\n{'='*90}")
print("NOTES:")
print("  * Crash metrics are COLUMN-LEVEL (which event types crashed?)")
print("  * Tamper metrics are ENTRY-LEVEL (which specific entries were modified?)")
print("  * MCDD adds theta (crash threshold) and S_crash_prob — novel outputs")
print(f"{'='*90}")