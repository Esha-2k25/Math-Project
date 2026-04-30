# step9_sweep_missing_rate.py
import numpy as np
import matplotlib.pyplot as plt
import os
from step5_mcdd import mcdd_admm
from step4_baseline_rpca import standard_rpca

DATA_DIR = "data"
M_true = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
rows, cols = M_true.shape

# Fixed crash columns (e.g., columns 5, 6, 7, 8 – four columns)
crash_cols = [5, 6, 7, 8]
missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # fraction of rows crashed per column

rpca_rmse = []
mcdd_rmse = []

for rate in missing_rates:
    # Create crash mask: for each crash column, wipe a contiguous block of rows
    mask = np.zeros((rows, cols), dtype=bool)
    block_size = int(rows * rate)
    start_row = (rows - block_size) // 2  # centered crash
    for c in crash_cols:
        mask[start_row:start_row+block_size, c] = True
    
    # Corrupt the ground truth
    M_obs = M_true.copy()
    M_obs[mask] = np.nan
    
    # Run RPCA and MCDD
    L_rpca, _ = standard_rpca(M_obs)
    L_mcdd, _, _, _ = mcdd_admm(M_obs)  # adjust based on your mcdd_admm return
    
    # Compute RMSE only on the crashed region
    rmse_rpca = np.sqrt(np.mean((M_true[mask] - L_rpca[mask])**2))
    rmse_mcdd = np.sqrt(np.mean((M_true[mask] - L_mcdd[mask])**2))
    
    rpca_rmse.append(rmse_rpca)
    mcdd_rmse.append(rmse_mcdd)
    print(f"Missing rate {rate*100:.0f}%: RPCA RMSE={rmse_rpca:.3f}, MCDD RMSE={rmse_mcdd:.3f}")

# Plot
plt.figure(figsize=(8,5))
plt.plot([r*100 for r in missing_rates], rpca_rmse, 's-', label='Standard RPCA', color='#F97316', linewidth=2, markersize=8)
plt.plot([r*100 for r in missing_rates], mcdd_rmse, 'D-', label='MCDD (Ours)', color='#2563EB', linewidth=2, markersize=8)
plt.xlabel('Percentage of rows crashed (per column)', fontsize=12)
plt.ylabel('RMSE on crash region (lower is better)', fontsize=12)
plt.title('Reconstruction Accuracy vs Crash Severity', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/fig6_rmse_vs_missing_rate.png', dpi=200)
plt.show()
print("Saved fig6_rmse_vs_missing_rate.png")