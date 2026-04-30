# step3_corrupt.py (large crash region ~25% of matrix)
import numpy as np
import os

DATA_DIR = "data"
M = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
rows, cols = M.shape
rng = np.random.default_rng(seed=42)

# Condition A: random 20% missing
mask_A = rng.random((rows, cols)) < 0.20
M_A = M.copy(); M_A[mask_A] = np.nan

# Condition B: large structured crash (wipe 50% of columns for 50% of rows)
mask_B = np.zeros((rows, cols), dtype=bool)
# Select half the columns (15 out of 30)
crash_cols = list(range(5, 20))  # 15 columns
# For each, wipe a contiguous block covering 50% of rows
block_start = int(0.25 * rows)
block_end = int(0.75 * rows)
for c in crash_cols:
    mask_B[block_start:block_end, c] = True
M_B = M.copy(); M_B[mask_B] = np.nan

# Condition C: structured + adversarial tampering (5% of observed)
M_C = M_B.copy()
mask_tamper = np.zeros((rows, cols), dtype=bool)
observed = np.argwhere(~mask_B)
n_tamper = int(0.05 * len(observed))
chosen = rng.choice(len(observed), size=n_tamper, replace=False)
for r, c in observed[chosen]:
    mask_tamper[r, c] = True
    M_C[r, c] = M[r, c] * rng.uniform(4.0, 10.0)

np.save(os.path.join(DATA_DIR, 'M_A.npy'), M_A)
np.save(os.path.join(DATA_DIR, 'mask_A.npy'), mask_A)
np.save(os.path.join(DATA_DIR, 'M_B.npy'), M_B)
np.save(os.path.join(DATA_DIR, 'mask_B.npy'), mask_B)
np.save(os.path.join(DATA_DIR, 'M_C.npy'), M_C)
np.save(os.path.join(DATA_DIR, 'mask_B_for_C.npy'), mask_B)
np.save(os.path.join(DATA_DIR, 'mask_tamper.npy'), mask_tamper)

print(f"Rows: {rows}, Cols: {cols}")
print(f"Condition A missing: {mask_A.sum()}")
print(f"Condition B missing: {mask_B.sum()} ({(mask_B.sum()/(rows*cols))*100:.1f}% of matrix)")
print(f"Condition C missing: {mask_B.sum()}, tampered: {mask_tamper.sum()}")