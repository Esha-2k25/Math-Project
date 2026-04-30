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

# Additional metrics
print(f"\nMatrix statistics:")
print(f"  Rank (numerical, tol=1e-10): {np.sum(s > 1e-10)}")
print(f"  Condition number: {s[0]/s[-1]:.2f}")
print(f"  Frobenius norm: {np.linalg.norm(M, 'fro'):.2f}")
print(f"  Spectral norm: {s[0]:.2f}")

# Plot singular values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: singular value decay
axes[0].bar(range(1, min(16, len(s)+1)), s[:15]/s[0], color='steelblue', edgecolor='white')
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_xlabel("Singular value index", fontsize=12)
axes[0].set_ylabel("Normalized singular value", fontsize=12)
axes[0].set_title(f"Thunderbird Sample SVD Decay\n({M.shape[0]} blocks x {M.shape[1]} event types)", fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Right: cumulative explained variance
axes[1].plot(range(1, min(16, len(explained)+1)), explained[:15]*100, 'o-', color='darkgreen', linewidth=2, markersize=8)
axes[1].axhline(y=98.5, color='red', linestyle='--', label='98.5% threshold')
axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Number of components", fontsize=12)
axes[1].set_ylabel("Cumulative variance explained (%)", fontsize=12)
axes[1].set_title("Cumulative Variance Explained", fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_sample_svd.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nSample SVD plot saved. Sparsity: {100*(M==0).mean():.1f}%")
