# step7_plots.py — Publication-Quality Figures for MCDD
#
# Figures generated:
#   Fig 1: Reconstruction accuracy comparison (bar chart)
#   Fig 2: RMSE improvement % over RPCA
#   Fig 3: Learned crash thresholds (theta) — NOVEL OUTPUT
#   Fig 4: Crash probability surface vs true mask
#   Fig 5: Per-column crash probability comparison (A vs B)
#   Fig 6: S_tamper magnitude heatmap (adversarial detection)
#   Fig 7: Reconstruction error heatmap (MCDD vs RPCA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def D(f): return os.path.join(DATA_DIR, f)
def R(f): return os.path.join(RESULTS_DIR, f)

M_true = np.load(D('M_ground_truth.npy'))
rows, cols = M_true.shape
crash_col_range = list(range(5, 20))
safe_col_range = list(range(0, 5)) + list(range(20, 30))

# Load evaluation data
df = pd.read_csv(R('evaluation_table.csv'))

# Color palette
C_SAFE = '#2ca02c'
C_CRASH = '#d62728'
C_RPCA = '#ff7f0e'
C_MCDD = '#1f77b4'
C_MEAN = '#9CA3AF'

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Reconstruction Accuracy Comparison (Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MCDD vs Baselines: Reconstruction Accuracy', fontsize=16, fontweight='bold', y=1.02)

metrics = ['RMSE', 'MAE', 'Rel_Error', 'PSNR']
conditions_plot = ['Random', 'Structured', 'Adversarial']
methods = ['Mean Imputation', 'Standard RPCA', 'MCDD (Ours)']
colors = [C_MEAN, C_RPCA, C_MCDD]

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(conditions_plot))
    width = 0.25

    for i, (method, col) in enumerate(zip(methods, colors)):
        vals = []
        for cond in conditions_plot:
            row = df[(df['Condition'] == cond) & (df['Method'] == method)]
            if len(row) > 0:
                vals.append(float(row[metric].values[0]))
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=method, color=col, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions_plot)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(R('fig1_accuracy_comparison.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig1_accuracy_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: RMSE Improvement % over RPCA
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

improvements = []
for cond in conditions_plot:
    rpca_rmse = float(df[(df['Condition']==cond) & (df['Method']=='Standard RPCA')]['RMSE'].values[0])
    mcdd_rmse = float(df[(df['Condition']==cond) & (df['Method']=='MCDD (Ours)')]['RMSE'].values[0])
    pct = (rpca_rmse - mcdd_rmse) / rpca_rmse * 100
    improvements.append(pct)

colors_imp = [C_MCDD if p > 0 else C_CRASH for p in improvements]
bars = ax.bar(conditions_plot, improvements, color=colors_imp, edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5 if height >= 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=14, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('RMSE Improvement over RPCA (%)', fontsize=13)
ax.set_title('MCDD Reconstruction Improvement vs Standard RPCA', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(min(improvements) - 5, max(improvements) + 10)

plt.tight_layout()
plt.savefig(R('fig2_rmse_improvement.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig2_rmse_improvement.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Learned Crash Thresholds (theta) — THE NOVEL OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
conditions = [
    ('A_random',      'Random (20% missing)',     'steelblue'),
    ('B_structured',  'Structured Crash',          'darkorange'),
    ('C_adversarial', 'Structured + Adversarial',  'crimson'),
]

for ax, (cond, label, color) in zip(axes, conditions):
    theta = np.load(D(f'theta_mcdd_{cond}.npy'))
    col_indices = np.arange(cols)
    bar_colors = [C_CRASH if c in crash_col_range else C_SAFE for c in col_indices]

    ax.bar(col_indices, theta, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Event Type (Column)', fontsize=11)
    ax.set_ylabel('Learned Crash Threshold θⱼ', fontsize=11)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Patch(facecolor=C_CRASH, label='True crash cols (5–19)'),
        Patch(facecolor=C_SAFE, label='Safe cols')
    ]
    ax.legend(handles=legend_elements, fontsize=9)

plt.suptitle('NOVEL OUTPUT: Learned Crash Thresholds θⱼ per Event Type\n'
             '(Lower θ = more vulnerable to crash — system fails at lower event counts)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(R('fig3_learned_theta.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig3_learned_theta.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Crash Probability Surface vs True Mask
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col_idx, (cond, label) in enumerate(zip(
        ['A_random', 'B_structured', 'C_adversarial'],
        ['Random (20% missing)', 'Structured Crash', 'Adversarial'])):
    S_cp = np.load(D(f'S_crash_prob_mcdd_{cond}.npy'))
    mask = np.load(D('mask_A.npy' if cond=='A_random' else 'mask_B.npy'))

    im0 = axes[0, col_idx].imshow(mask[:300,:].T, aspect='auto',
                                   cmap='Reds', vmin=0, vmax=1)
    axes[0, col_idx].set_title(f'True Mask: {label}', fontweight='bold', fontsize=11)
    axes[0, col_idx].set_xlabel('Log block')
    axes[0, col_idx].set_ylabel('Event type')
    plt.colorbar(im0, ax=axes[0,col_idx], fraction=0.046)

    im1 = axes[1, col_idx].imshow(S_cp[:300,:].T, aspect='auto',
                                   cmap='Reds', vmin=0, vmax=1)
    axes[1, col_idx].set_title(f'Learned P(missing|L,θ)', fontweight='bold', fontsize=11)
    axes[1, col_idx].set_xlabel('Log block')
    axes[1, col_idx].set_ylabel('Event type')
    plt.colorbar(im1, ax=axes[1,col_idx], fraction=0.046)

plt.suptitle('True Missing Mask vs Learned Crash Probability Surface\n'
             'MCDD distinguishes structured crash (concentrated) from random dropout (diffuse)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(R('fig4_crash_probability_surface.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig4_crash_probability_surface.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Key Novelty — Per-Column Crash Probability (A vs B)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

S_cp_A = np.load(D('S_crash_prob_mcdd_A_random.npy'))
S_cp_B = np.load(D('S_crash_prob_mcdd_B_structured.npy'))

col_crash_prob_A = S_cp_A.mean(axis=0)
col_crash_prob_B = S_cp_B.mean(axis=0)

col_range = np.arange(cols)
bar_colors = [C_CRASH if c in crash_col_range else C_SAFE for c in col_range]

axes[0].bar(col_range, col_crash_prob_A, color=bar_colors, edgecolor='white')
axes[0].set_title('Random Missing (Condition A)\nMean crash probability per column',
                  fontweight='bold', fontsize=11)
axes[0].set_xlabel('Event Type (Column)')
axes[0].set_ylabel('Mean P(missing|L,θ)')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(col_range, col_crash_prob_B, color=bar_colors, edgecolor='white')
axes[1].set_title('Structured Crash (Condition B)\nMean crash probability per column',
                  fontweight='bold', fontsize=11)
axes[1].set_xlabel('Event Type (Column)')
axes[1].set_ylabel('Mean P(missing|L,θ)')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')

legend_elements = [
    Patch(facecolor=C_CRASH, label='True crash cols (5–19)'),
    Patch(facecolor=C_SAFE, label='Safe cols')
]
for ax in axes:
    ax.legend(handles=legend_elements, fontsize=9)

plt.suptitle('KEY NOVELTY: MNAR Model Distinguishes Random Dropout from Structured Crash\n'
             'Random → diffuse probabilities | Structured → concentrated in crash columns',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(R('fig5_mnar_discrimination.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig5_mnar_discrimination.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Tamper Detection Heatmap (Condition C only)
# ═══════════════════════════════════════════════════════════════════════════
S_tamper_C = np.load(D('S_tamper_mcdd_C_adversarial.npy'))
mask_tamper = np.load(D('mask_tamper.npy'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(np.abs(S_tamper_C[:300, :]).T, aspect='auto', cmap='hot')
axes[0].set_title('|S_tamper| Magnitude\n(Detected Anomalies)', fontweight='bold')
axes[0].set_xlabel('Log block')
axes[0].set_ylabel('Event type')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(mask_tamper[:300, :].T, aspect='auto', cmap='Reds')
axes[1].set_title('Ground Truth Tamper Mask', fontweight='bold')
axes[1].set_xlabel('Log block')
axes[1].set_ylabel('Event type')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.suptitle('Tamper Detection: MCDD S_tamper vs Ground Truth (Condition C)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(R('fig6_tamper_heatmap.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig6_tamper_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Reconstruction Error Comparison (MCDD vs RPCA)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cond, label in zip(axes, ['A_random', 'B_structured', 'C_adversarial'],
                           ['Random', 'Structured', 'Adversarial']):
    miss_mask = np.load(D(f'mask_{"A" if cond=="A_random" else "B"}.npy'))
    L_rpca = np.load(D(f'L_rpca_{cond}.npy'))
    L_mcdd = np.load(D(f'L_mcdd_{cond}.npy'))

    err_rpca = np.abs(M_true - L_rpca)
    err_mcdd = np.abs(M_true - L_mcdd)
    err_diff = err_mcdd - err_rpca

    im = ax.imshow(err_diff[:300, :].T, aspect='auto', cmap='RdBu_r',
                    vmin=-1, vmax=1)
    ax.set_title(f'{label}\n(Error_diff = MCDD - RPCA)', fontweight='bold')
    ax.set_xlabel('Log block')
    ax.set_ylabel('Event type')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Reconstruction Error Difference: MCDD vs RPCA\n'
             'Blue = MCDD better | Red = RPCA better | White = Tie',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(R('fig7_error_difference.png'), dpi=250, bbox_inches='tight')
plt.close()
print("Saved fig7_error_difference.png")

print("\nAll figures generated successfully!")
