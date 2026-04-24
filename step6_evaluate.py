# step6_evaluate.py (corrected – computes mean imputation on the fly)
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
print(f"Ground truth shape: {M_true.shape}")

def rmse(true, pred, mask):
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))

def mae(true, pred, mask):
    return np.mean(np.abs(true[mask] - pred[mask]))

def rel_error(true, pred, mask):
    return np.linalg.norm(true[mask] - pred[mask]) / (np.linalg.norm(true[mask]) + 1e-10)

def mean_imputation(M_obs):
    """Fill missing entries with column mean"""
    M_filled = M_obs.copy()
    for col in range(M_obs.shape[1]):
        col_data = M_obs[:, col]
        col_mean = np.nanmean(col_data) if not np.all(np.isnan(col_data)) else 0.0
        M_filled[np.isnan(col_data), col] = col_mean
    return M_filled

# Mapping: condition name -> (observed matrix file, missing mask file, crash_gt file, tamper_file)
configs = [
    ('Random',      'M_A.npy',          'mask_A.npy',       None,           None),
    ('Structured',  'M_B.npy',          'mask_B.npy',       'mask_B.npy',   None),
    ('Adversarial', 'M_C.npy',          'mask_B_for_C.npy', 'mask_B_for_C.npy', 'mask_tamper.npy')
]

results = []
for cond_name, obs_file, miss_file, crash_gt_file, tamper_file in configs:
    # Determine suffix for L and S files
    if cond_name == 'Random':
        suffix = 'A_random'
    elif cond_name == 'Structured':
        suffix = 'B_structured'
    else:
        suffix = 'C_adversarial'
    
    miss_mask = np.load(os.path.join(DATA_DIR, miss_file))
    crash_gt = np.load(os.path.join(DATA_DIR, crash_gt_file)) if crash_gt_file else None
    
    # Load observed matrix and compute mean imputation on the fly
    M_obs = np.load(os.path.join(DATA_DIR, obs_file))
    L_mean = mean_imputation(M_obs)
    
    L_rpca = np.load(os.path.join(DATA_DIR, f'L_rpca_{suffix}.npy'))
    L_mcdd = np.load(os.path.join(DATA_DIR, f'L_mcdd_{suffix}.npy'))
    S_crash = np.load(os.path.join(DATA_DIR, f'S_crash_{suffix}.npy'))
    
    for method, Lrec in [('Mean Imputation', L_mean), ('Standard RPCA', L_rpca), ('MCDD (Ours)', L_mcdd)]:
        row = {
            'Condition': cond_name, 'Method': method,
            'RMSE': rmse(M_true, Lrec, miss_mask),
            'MAE': mae(M_true, Lrec, miss_mask),
            'Rel_Error': rel_error(M_true, Lrec, miss_mask),
            'Crash_F1': '—', 'Tamper_F1': '—'
        }
        if method == 'MCDD (Ours)' and crash_gt is not None:
            pred_crash = (S_crash > 0.3).astype(int)
            row['Crash_F1'] = f1_score(crash_gt.flatten(), pred_crash.flatten())
        if method == 'MCDD (Ours)' and tamper_file is not None:
            tamper_mask = np.load(os.path.join(DATA_DIR, tamper_file))
            S_tamper = np.load(os.path.join(DATA_DIR, f'S_tamper_{suffix}.npy'))
            thresh = np.percentile(np.abs(S_tamper), 95)
            pred_tamper = (np.abs(S_tamper) > thresh).astype(int)
            row['Tamper_F1'] = f1_score(tamper_mask.flatten(), pred_tamper.flatten())
        results.append(row)

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_table.csv'), index=False)
print(df.to_string())