# step0_sample_thunderbird.py
import numpy as np
import pandas as pd
import re
from collections import Counter
import os

# ============================================================
# CONFIGURATION
# ============================================================
RAW_LOG = "data/Thunderbird/Thunderbird.log"  # adjust path if needed
OUTPUT_DIR = "data"
MAX_LINES = 5000000        # change this to 10000, 20000, 50000, etc.
BLOCK_SIZE = 1000          # lines per block (adjust based on MAX_LINES)
TOP_EVENTS = 30

# ============================================================
# 1. Read only first MAX_LINES
# ============================================================
print(f"Reading first {MAX_LINES} lines of {RAW_LOG}...")
with open(RAW_LOG, 'r', encoding='utf-8', errors='ignore') as f:
    lines = []
    for i, line in enumerate(f):
        if i >= MAX_LINES:
            break
        lines.append(line)
print(f"Read {len(lines)} lines.")

# ============================================================
# 2. Extract event templates (same as before)
# ============================================================
def get_template(line):
    parts = line.strip().split()
    if not parts:
        return ""
    # Try to find where the actual message starts (skip timestamps, node IDs)
    # Thunderbird format example: "2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO ..."
    start_idx = 0
    for i, p in enumerate(parts):
        if p in ('RAS', 'KERNEL', 'INFO', 'WARNING', 'ERROR', 'FATAL', 'Alert'):
            start_idx = i
            break
    msg = ' '.join(parts[start_idx:])
    msg = re.sub(r'\b0x[0-9a-fA-F]+\b', '<HEX>', msg)
    msg = re.sub(r'\b\d+\b', '<NUM>', msg)
    msg = re.sub(r'[A-Z]\d+[-:][A-Z0-9\-:]+', '<NODE>', msg)
    return msg.strip()

print("Extracting templates...")
templates = []
for line in lines:
    tmpl = get_template(line)
    if tmpl:
        templates.append(tmpl)
print(f"Extracted {len(templates)} non‑empty templates.")

# ============================================================
# 3. Keep top K event types
# ============================================================
template_counts = Counter(templates)
most_common = template_counts.most_common(TOP_EVENTS)
top_templates = [t for t, _ in most_common]
print(f"Keeping top {TOP_EVENTS} event types (out of {len(template_counts)} total).")
print("Top 5 event templates:")
for i, (t, cnt) in enumerate(most_common[:5]):
    print(f"  {i+1}: '{t[:80]}...' appears {cnt} times")

filtered_templates = [t for t in templates if t in top_templates]
print(f"Lines kept after filtering: {len(filtered_templates)}")

# ============================================================
# 4. Group into blocks and build count matrix
# ============================================================
event_to_idx = {t: i for i, t in enumerate(top_templates)}
num_blocks = max(1, len(filtered_templates) // BLOCK_SIZE)
block_size = len(filtered_templates) // num_blocks
print(f"Creating {num_blocks} blocks of size ~{block_size}...")

M_raw = np.zeros((num_blocks, TOP_EVENTS), dtype=int)
for b in range(num_blocks):
    start = b * block_size
    end = start + block_size
    block_templates = filtered_templates[start:end]
    counts = Counter(block_templates)
    for tmpl, cnt in counts.items():
        M_raw[b, event_to_idx[tmpl]] = cnt

print(f"Raw matrix shape: {M_raw.shape}")
sparsity = 100 * (M_raw == 0).mean()
print(f"Sparsity: {sparsity:.1f}%")
print(f"Mean non‑zero count per cell: {M_raw[M_raw > 0].mean():.2f}")

# ============================================================
# 5. Normalize and save
# ============================================================
M_log = np.log1p(M_raw)
col_mean = M_log.mean(axis=0)
col_std = M_log.std(axis=0)
col_std = np.where(col_std < 1e-8, 1.0, col_std)
M = (M_log - col_mean) / col_std

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, 'M_ground_truth.npy'), M)
np.save(os.path.join(OUTPUT_DIR, 'col_mean.npy'), col_mean)
np.save(os.path.join(OUTPUT_DIR, 'col_std.npy'), col_std)
pd.Series(top_templates).to_csv(os.path.join(OUTPUT_DIR, 'event_types_sample.csv'), index=False)

print("✅ Sample matrix saved as M_ground_truth_sample.npy")
print("Now run step2_build_matrix_sample.py to check low‑rank structure.")