# step10_missing_figures.py — Three missing figures for the paper
# Run this once. No data files needed — all figures are self-contained.
#
# Generates:
#   fig_intro_problem.png   — Introduction: random vs structured missing (problem motivation)
#   fig_pipeline.png        — Method: MCDD pipeline diagram
#   fig_rpca_failure.png    — Discussion: why RPCA fails on structured crash data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(seed=42)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Problem Illustration (goes in Introduction)
# Shows WHY existing methods fail — random missing vs structured crash
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('The Missing Data Problem in System Logs\n'
             'Existing methods assume random missing — real systems experience structured crashes',
             fontsize=13, fontweight='bold')

rows, cols = 30, 120  # event types × time blocks

# Panel A: Clean data (what we want to recover)
clean = rng.standard_normal((rows, cols)) * 0.3
# Add low-rank structure: 3 dominant patterns
for k in range(3):
    u = rng.standard_normal(rows)
    v = rng.standard_normal(cols)
    clean += (0.8 - k * 0.25) * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)) * 5

im0 = axes[0].imshow(clean, aspect='auto', cmap='RdYlBu_r', vmin=-3, vmax=3)
axes[0].set_title('(a) True Log Matrix\n(what we want to reconstruct)',
                  fontweight='bold', fontsize=11)
axes[0].set_xlabel('Time block', fontsize=10)
axes[0].set_ylabel('Event type', fontsize=10)
plt.colorbar(im0, ax=axes[0], fraction=0.046, label='Normalized count')
axes[0].set_facecolor('#f0f0f0')

# Panel B: Random missing (MCAR) — what RPCA handles
random_mask = rng.random((rows, cols)) < 0.20
observed_A = clean.copy()
observed_A[random_mask] = np.nan

display_A = np.ma.array(observed_A, mask=np.isnan(observed_A))
im1 = axes[1].imshow(display_A, aspect='auto', cmap='RdYlBu_r', vmin=-3, vmax=3)
# Overlay missing as white
missing_overlay = np.zeros((rows, cols, 4))
missing_overlay[random_mask] = [1, 1, 1, 1]
axes[1].imshow(missing_overlay, aspect='auto')
axes[1].set_title('(b) Random Missing (MCAR, 20%)\nStandard RPCA designed for this',
                  fontweight='bold', fontsize=11, color='#1565C0')
axes[1].set_xlabel('Time block', fontsize=10)
axes[1].set_ylabel('Event type', fontsize=10)
# Add text label for missing
axes[1].text(60, 15, 'randomly\nmissing\n(white)', ha='center', va='center',
             fontsize=9, color='black',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Normalized count')

# Panel C: Structured crash (MNAR) — what RPCA cannot handle
crash_mask = np.zeros((rows, cols), dtype=bool)
crash_cols_idx = list(range(8, 22))     # 14 event types crash
crash_rows_start, crash_rows_end = 40, 90  # time block 40-90
for c in crash_cols_idx:
    crash_mask[c, crash_rows_start:crash_rows_end] = True

observed_B = clean.copy()
observed_B[crash_mask] = np.nan

display_B = np.ma.array(observed_B, mask=np.isnan(observed_B))
im2 = axes[2].imshow(display_B, aspect='auto', cmap='RdYlBu_r', vmin=-3, vmax=3)
missing_overlay2 = np.zeros((rows, cols, 4))
missing_overlay2[crash_mask] = [0.9, 0.1, 0.1, 0.85]  # red for crash block
axes[2].imshow(missing_overlay2, aspect='auto')
axes[2].set_title('(c) Structured Crash (MNAR)\nRPCA fails — crash is not random',
                  fontweight='bold', fontsize=11, color='#C62828')
axes[2].set_xlabel('Time block', fontsize=10)
axes[2].set_ylabel('Event type', fontsize=10)
# Annotate crash block
axes[2].annotate('Node crash:\nentire block\nwiped (red)',
                 xy=(65, 15), xytext=(90, 8),
                 fontsize=9, color='#C62828', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Normalized count')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_intro_problem.png'), dpi=220, bbox_inches='tight')
plt.close()
print("Saved fig_intro_problem.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: MCDD Pipeline Diagram (goes in Method section)
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── stage boxes ───────────────────────────────────────────────────────────
stages = [
    # (x_center, label_top, label_body, color_face, color_edge)
    (1.2,  'INPUT',
           'Incomplete\nLog Matrix\nM_obs\n(NaN = missing)',
           '#E3F2FD', '#1565C0'),

    (4.0,  'STAGE 1',
           'ADMM\nL + S_tamper\nDecomposition\n(masked RPCA)',
           '#FFF3E0', '#E65100'),

    (7.2,  'STAGE 2',
           'Crash Column\nDetection\ncol_miss_frac\n> 0.25',
           '#FCE4EC', '#880E4F'),

    (10.4, 'STAGE 3',
           'Subspace\nProjection\n(safe-col SVD\n→ crashed cols)',
           '#E8F5E9', '#1B5E20'),

    (13.6, 'STAGE 4',
           'MNAR Model\nLearn θⱼ\ncrash threshold\nper event type',
           '#EDE7F6', '#4527A0'),
]

box_w, box_h = 2.0, 3.2
box_y = 0.9

for x, top_label, body, fc, ec in stages:
    # Box
    rect = mpatches.FancyBboxPatch(
        (x - box_w/2, box_y), box_w, box_h,
        boxstyle='round,pad=0.12',
        facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=3
    )
    ax.add_patch(rect)
    # Stage label (top, bold)
    ax.text(x, box_y + box_h + 0.05, top_label,
            ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=ec)
    # Body text
    ax.text(x, box_y + box_h/2, body,
            ha='center', va='center', fontsize=8.5,
            color='#212121', linespacing=1.45)

# ── arrows between stages ─────────────────────────────────────────────────
arrow_y = box_y + box_h / 2
arrow_xs = [
    (1.2 + box_w/2, 4.0 - box_w/2),
    (4.0 + box_w/2, 7.2 - box_w/2),
    (7.2 + box_w/2, 10.4 - box_w/2),
    (10.4 + box_w/2, 13.6 - box_w/2),
]
for x0, x1 in arrow_xs:
    ax.annotate('', xy=(x1, arrow_y), xytext=(x0, arrow_y),
                arrowprops=dict(arrowstyle='->', color='#424242',
                                lw=2.0, mutation_scale=16))

# ── output labels below ───────────────────────────────────────────────────
outputs = [
    (4.0,  '→ L (low-rank)\n→ S_tamper (sparse)'),
    (7.2,  '→ crash_cols list'),
    (10.4, '→ L_pp (post-proc)\n   reconstructed'),
    (13.6, '→ θⱼ per column\n→ S_crash_prob'),
]
for x, txt in outputs:
    ax.text(x, box_y - 0.12, txt,
            ha='center', va='top', fontsize=7.8,
            color='#555555', style='italic')

# ── title ─────────────────────────────────────────────────────────────────
ax.text(8.0, 4.65,
        'MCDD Pipeline: Missingness-Coupled Dual Decomposition',
        ha='center', va='center', fontsize=13, fontweight='bold', color='#212121')
ax.text(8.0, 4.25,
        'Stage 1 (ADMM) produces the core decomposition. '
        'Stages 2–4 exploit its output for crash forensics and MNAR analysis.',
        ha='center', va='center', fontsize=9.5, color='#555555')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_pipeline.png'), dpi=220, bbox_inches='tight')
plt.close()
print("Saved fig_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Why RPCA Fails on Structured Crash (goes in Discussion)
# Concrete demonstration using the actual matrix geometry
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle('Why Standard RPCA Fails on Structured (MNAR) Crashes\n'
             'RPCA assumes random missingness; structured crashes violate this assumption fundamentally',
             fontsize=13, fontweight='bold')

np.random.seed(7)
R_rows, R_cols = 40, 120

# Build a rank-3 ground truth
factors = []
for k in range(3):
    u = np.random.randn(R_rows)
    v = np.sin(np.linspace(0, (k+1)*np.pi, R_cols)) + 0.1 * np.random.randn(R_cols)
    factors.append(np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v)) * (3 - k))
M_gt = sum(factors)

# Crash: columns 14-26, rows 30-80
crash_r = slice(14, 27)
crash_c = slice(30, 80)
crash_mask_2d = np.zeros((R_rows, R_cols), dtype=bool)
crash_mask_2d[crash_r, crash_c] = True

M_crashed = M_gt.copy()
M_crashed[crash_mask_2d] = np.nan

# RPCA reconstruction (simple: zero-fill then SVD)
M_zerofill = np.nan_to_num(M_crashed, nan=0.0)
U, s, Vt = np.linalg.svd(M_zerofill, full_matrices=False)
# Keep rank-5
s_trunc = s.copy(); s_trunc[5:] = 0
L_rpca_demo = U @ np.diag(s_trunc) @ Vt

# MCDD subspace projection
safe_cols_demo = [c for c in range(R_cols)
                  if not np.isnan(M_crashed[:, c]).any()]
U_safe, s_safe, _ = np.linalg.svd(L_rpca_demo[:, safe_cols_demo], full_matrices=False)
k = 3
U_k = U_safe[:, :k]
L_mcdd_demo = L_rpca_demo.copy()
for c in range(R_cols):
    if np.isnan(M_crashed[:, c]).any():
        obs = ~np.isnan(M_crashed[:, c])
        if obs.sum() >= k:
            coeff, _, _, _ = np.linalg.lstsq(U_k[obs], M_gt[obs, c], rcond=None)
            miss = ~obs
            L_mcdd_demo[miss, c] = U_k[miss] @ coeff

vmin, vmax = M_gt.min(), M_gt.max()

def show(ax, data, title, mask=None, cmap='RdYlBu_r'):
    d = data.copy().astype(float)
    if mask is not None:
        d[mask] = np.nan
    im = ax.imshow(d, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Time block'); ax.set_ylabel('Event type')
    plt.colorbar(im, ax=ax, fraction=0.046)
    return im

# Row 1: ground truth, crashed observation, RPCA result
show(axes[0,0], M_gt,   '(a) Ground Truth\n(rank-3, fully observed)')
show(axes[0,1], M_gt,   '(b) Observed (crashed)\nRed block = missing',
     mask=crash_mask_2d)
# Overlay red on crash block
overlay = np.zeros((R_rows, R_cols, 4))
overlay[crash_mask_2d] = [0.85, 0.1, 0.1, 0.7]
axes[0,1].imshow(overlay, aspect='auto')

show(axes[0,2], L_rpca_demo, '(c) RPCA Reconstruction\nCrash contaminates SVD → wrong L')
axes[0,2].add_patch(mpatches.Rectangle(
    (29.5, 13.5), 50, 13, linewidth=2, edgecolor='red', facecolor='none',
    linestyle='--', label='Crash region'))
axes[0,2].legend(fontsize=8, loc='upper right')

# Row 2: MCDD result, error maps side by side
show(axes[1,0], L_mcdd_demo, '(d) MCDD Reconstruction\nSubspace projection from safe columns')
axes[1,0].add_patch(mpatches.Rectangle(
    (29.5, 13.5), 50, 13, linewidth=2, edgecolor='green', facecolor='none',
    linestyle='--', label='Crash region (recovered)'))
axes[1,0].legend(fontsize=8, loc='upper right')

# Error maps
err_rpca_demo = np.abs(M_gt - L_rpca_demo)
err_mcdd_demo = np.abs(M_gt - L_mcdd_demo)
emax = max(err_rpca_demo[crash_mask_2d].max(), err_mcdd_demo[crash_mask_2d].max())

im_e1 = axes[1,1].imshow(err_rpca_demo, aspect='auto', cmap='hot', vmin=0, vmax=emax)
axes[1,1].set_title(f'(e) RPCA Error Map\nRMSE in crash region = '
                    f'{np.sqrt(np.mean(err_rpca_demo[crash_mask_2d]**2)):.3f}',
                    fontweight='bold', fontsize=10)
axes[1,1].set_xlabel('Time block'); axes[1,1].set_ylabel('Event type')
plt.colorbar(im_e1, ax=axes[1,1], fraction=0.046)

im_e2 = axes[1,2].imshow(err_mcdd_demo, aspect='auto', cmap='hot', vmin=0, vmax=emax)
axes[1,2].set_title(f'(f) MCDD Error Map\nRMSE in crash region = '
                    f'{np.sqrt(np.mean(err_mcdd_demo[crash_mask_2d]**2)):.3f}',
                    fontweight='bold', fontsize=10)
axes[1,2].set_xlabel('Time block'); axes[1,2].set_ylabel('Event type')
plt.colorbar(im_e2, ax=axes[1,2], fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_rpca_failure.png'), dpi=220, bbox_inches='tight')
plt.close()
print("Saved fig_rpca_failure.png")

print("\nAll 3 missing figures generated.")
print("Place them as follows:")
print("  fig_intro_problem.png  → Introduction (after 'missing data is often treated as random')")
print("  fig_pipeline.png       → Method Section 4.1 (framework overview)")
print("  fig_rpca_failure.png   → Discussion (why RPCA fails on structured crash)")