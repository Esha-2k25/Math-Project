# step2_check_sample.py
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

M = np.load(os.path.join(DATA_DIR, 'M_ground_truth.npy'))
print(f"Sample matrix shape: {M.shape}")

U, s, Vt = np.linalg.svd(M, full_matrices=False)
explained = np.cumsum(s**2) / np.sum(s**2)
print("\nCumulative variance explained by top singular values:")
for i in range(min(10, len(explained))):
    print(f"  Top-{i+1}: {explained[i]*100:.1f}%")

# Plot
plt.figure()
plt.bar(range(1, min(16, len(s)+1)), s[:15]/s[0], color='steelblue')
plt.xlabel("Singular value index")
plt.ylabel("Normalized value")
plt.title(f"Thunderbird Sample (first {M.shape[0]*1000} lines) SVD Decay")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'fig_sample_svd.png'), dpi=150)
plt.close()

print(f"\n✅ Sample SVD plot saved. Sparsity: {100*(M==0).mean():.1f}%")