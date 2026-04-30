# step3_corrupt.py — Create three experimental conditions
import numpy as np
import os

DATA_DIR = "data"
M = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
rows, cols = M.shape
rng = np.random.default_rng(seed=42)

# ── Condition A: Random Missing (20% MCAR baseline) ──
mask_A = rng.random((rows, cols)) < 0.20
M_A = M.copy(); M_A[mask_A] = np.nan

# ── Condition B: Structured Crash (MNAR — node failure) ──
# Columns 5–19 (15 columns) 
# This simulates a node partition crash affecting multiple event types
mask_B = np.zeros((rows, cols), dtype=bool)
crash_cols = list(range(5, 20))  # 15 columns
block_start = int(0.25 * rows)
block_end = int(0.75 * rows)
for c in crash_cols:
    mask_B[block_start:block_end, c] = True
M_B = M.copy(); M_B[mask_B] = np.nan

# ── Condition C: Structured Crash + Adversarial Tampering ──
M_C = M_B.copy()
mask_tamper = np.zeros((rows, cols), dtype=bool)
observed = np.argwhere(~mask_B)
n_tamper = int(0.05 * len(observed))
chosen = rng.choice(len(observed), size=n_tamper, replace=False)
for r, c in observed[chosen]:
    mask_tamper[r, c] = True
    M_C[r, c] = M[r, c] * rng.uniform(4.0, 10.0)

# Save all
np.save(os.path.join(DATA_DIR, 'M_A.npy'), M_A)
np.save(os.path.join(DATA_DIR, 'mask_A.npy'), mask_A)
np.save(os.path.join(DATA_DIR, 'M_B.npy'), M_B)
np.save(os.path.join(DATA_DIR, 'mask_B.npy'), mask_B)
np.save(os.path.join(DATA_DIR, 'M_C.npy'), M_C)
np.save(os.path.join(DATA_DIR, 'mask_B_for_C.npy'), mask_B)
np.save(os.path.join(DATA_DIR, 'mask_tamper.npy'), mask_tamper)

print(f"Rows: {rows}, Cols: {cols}")
print(f"Condition A (Random 20%):  missing={mask_A.sum()} ({mask_A.sum()/(rows*cols)*100:.1f}%)")
print(f"Condition B (Structured):  missing={mask_B.sum()} ({mask_B.sum()/(rows*cols)*100:.1f}%) — crash cols 5-19, rows {block_start}-{block_end}")
print(f"Condition C (Adv+Struct):  missing={mask_B.sum()}, tampered={mask_tamper.sum()} ({mask_tamper.sum()/(rows*cols)*100:.1f}%)")
