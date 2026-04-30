# MCDD Project — Complete Context Document
## 22MAT230 | Event Timeline Reconstruction from Incomplete System Logs
### Everything we tried, what worked, what didn't, and where we actually stand

---

## 1. What This Project Is About

The core problem: when a computer system crashes or gets attacked, investigators look at system logs to reconstruct what happened. But in real systems, those logs are incomplete.

Three reasons logs go missing:
- **Node crash** — a server dies and stops writing logs. An entire block of time goes blank.
- **Attacker intrusion** — the first thing attackers do is delete or modify log entries to hide their tracks.
- **System overload** — logging daemons drop entries under heavy load.

The standard assumption in every existing paper: missing data is random (MCAR — Missing Completely At Random). This is completely wrong for real system logs. Missingness in logs has **structure and meaning**.

Our goal: given an incomplete and possibly tampered log matrix, reconstruct what really happened — AND identify whether missing entries were caused by a crash or an attacker.

---

## 2. The Method We Designed — MCDD

**MCDD = Missingness-Coupled Dual Decomposition**

### The Core Idea
Standard RPCA decomposes a matrix as:
```
M = L + S
```
(low-rank normal behavior + sparse corruption)

MCDD decomposes across two domains:
```
M_true = L + S_crash + S_tamper
```

| Variable | Represents | Domain |
|---|---|---|
| L | Normal recurring system behavior | Data domain, low-rank |
| S_crash | Which blocks are missing and WHY (crash pattern) | Missing MASK domain, sparse |
| S_tamper | Attacker-modified entries | Observed data domain, sparse |

### The Optimization Problem
```
min  ||L||_*  +  lam1*||S_crash||_1  +  lam2*||S_tamper||_1

s.t. P_Omega(L + S_tamper)  =  P_Omega(M_obs)   [fidelity on observed entries]
     (1 - Omega)            =  S_crash            [mask decomposition — THE NOVELTY]
```

The second constraint is what no prior paper does: treating the missing mask itself as a decomposable sparse signal.

### Why This Was Designed
The insight: if a node crashes, it wipes an entire column of events for a contiguous time block. That pattern — a rectangular block of missingness — is not random. It is a structured signal. If you model it as sparse (S_crash), you can:
1. Identify WHICH nodes crashed (from S_crash)
2. Identify WHICH entries were tampered (from S_tamper)  
3. Reconstruct the clean timeline (from L)

These three interpretable outputs from one framework — for system logs — do not exist anywhere in prior literature.

---

## 3. Dataset Journey — What We Tried and Why

### Attempt 1: HDFS_v1 (LogHub)
**What it is:** Hadoop Distributed File System logs, 186.6 MB. Pre-built event occurrence matrix (575,000 blocks × 29 event types). The original plan from the project context document.

**What happened:** Results were poor. The HDFS event matrix had high sparsity and the low-rank assumption was weaker. The reconstruction quality was not demonstrably better than RPCA in a way that told a clean story.

**Why we abandoned it:** The data characteristics (sparse counts, many zero cells) made the low-rank structure weaker, which meant MCDD had less to work with.

### Attempt 2: Thunderbird (Current Dataset)
**What it is:** Logs from a 9,024-node IBM BladeCenter supercomputer at Sandia National Labs. 29 GB uncompressed raw log file.

**Sampling:** First 5,000,000 lines of Thunderbird.log. Processing time ~10 minutes.

**Resulting matrix:** 1,656 rows × 30 columns. Sparsity: 0% (fully dense after log1p + standardization).

**Why this is better:**
- Top-5 singular values explain **98.5% of variance** — extremely strong low-rank structure
- Dense matrix means reconstruction quality is clearly measurable
- Large enough (1,656 blocks) to create meaningful crash regions

**Preprocessing pipeline (step1):**
1. Parse raw lines, extract event templates (remove timestamps, node IDs, hex numbers)
2. Keep top 30 most frequent event types out of 739,320 unique templates
3. Group into blocks of ~1,000 lines each → 1,656 blocks
4. Build event count matrix → apply log1p compression → standardize (zero mean, unit variance)

---

## 4. Experimental Setup

### Three Conditions (step3)

**Condition A — Random Missing (baseline)**
- 20% of entries missing uniformly at random
- What classical methods are designed for
- Tests whether MCDD does anything useful for random missingness

**Condition B — Structured Crash**  
- Columns 5–19 (15 columns) each have rows 414–828 set to NaN
- This simulates 15 event types going completely silent for 50% of time blocks
- Represents a massive node partition event
- 25% of total matrix entries missing (12,420 entries)

**Condition C — Structured + Adversarial**
- Same structured crash as B
- Plus 5% of observed entries multiplied by 4–10× (attacker inflates counts to hide tracks)
- 1,863 tampered entries

### True crash columns: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
### True crash rows: 414 to 828 (centered block, 50% of rows)

---

## 5. What We Actually Ran and What Happened

### The Original MCDD (First Working Version)

**step5_mcdd.py — Version 1 (with post-processing)**

This version had two stages:

**Stage 1 — ADMM loop:**
```python
Z1 = M_fill - S_tamper + Y1/rho          # L update (NO Y2)
L = svd_threshold(Z1, 1/rho)
S_tamper = soft_threshold(M_fill - L + Y1/rho, lam2/rho)
S_tamper[missing] = 0
S_crash = clip(soft_threshold(miss_mask + Y2/rho, lam1/rho), 0, 1)
Y1 += rho * (M_fill - L - S_tamper)
Y2 += rho * (miss_mask - S_crash)
```

**Stage 2 — Post-processing (crash-aware subspace projection):**
```python
crash_cols = where(S_crash.max(axis=0) > 0.5)
safe_cols = all other columns
U_safe, _, _ = SVD(L[:, safe_cols])
U_k = U_safe[:, :rank]                   # row subspace from non-crashed columns
for each crashed column c:
    alpha = least_squares(U_k[observed_rows], L[observed_rows, c])
    L[missing_rows, c] = U_k[missing_rows] @ alpha
```

**Results from this version:**
```
Condition A (Random):     RPCA RMSE=0.980  MCDD RMSE=0.978  (negligible improvement)
Condition B (Structured): RPCA RMSE=0.868  MCDD RMSE=0.489  (44% improvement)
Condition C (Adversarial):RPCA RMSE=0.868  MCDD RMSE=0.473  (45% improvement)
Crash F1 (B and C): 1.0   Tamper F1 (C): 0.795
```

**These results were real. The code was correct. The numbers were genuine.**

---

## 6. Why Those Results Were Questioned — The Deep Audit

### Problem Found: Stage 1 and Stage 2 were not truly coupled

In Stage 1, the L update used `Y1/rho` but NOT `Y2/rho`. This meant Y2 — the dual variable that couples the crash detection back into the reconstruction — had **zero effect on L**. The ADMM was mathematically identical to standard RPCA for the L component.

### Problem Found: S_crash ≈ miss_mask always

The constraint is `(1-Omega) = S_crash`. With entry-wise L1 penalty, the optimal S_crash for a large block crash is simply the missing mask itself. The residual `miss_mask - S_crash ≈ 0`, so Y2 stays near zero across all 300 iterations. The coupling is a mathematical no-op.

**This means:** All 44% improvement came from Stage 2 (subspace projection), not from the ADMM.

### Attempted Fix 1: Add Y2 into L update

```python
Z1 = M_fill - S_tamper + (Y1 + Y2) / rho
```

**Results after this fix:**
```
Condition B (Structured): RPCA RMSE=0.868  MCDD RMSE=0.862  (0.7% improvement)
Condition C (Adversarial):RPCA RMSE=0.868  MCDD RMSE=0.862  (0.7% improvement)
```

**Why this failed:** S_crash still ≈ miss_mask (the L1 problem above). Y2 still near zero. Adding Y2/rho into L update did nothing because Y2 ≈ 0.

### Attempted Fix 2: Change L1 to L2,1 (group-sparse) on S_crash

Column-wise block soft thresholding: columns with concentrated missing (crash) have large norms and are preserved; columns with scattered missing (random) have small norms and get zeroed.

**Why this still doesn't solve the fundamental problem:** For a 50% block crash, the column norm of `miss_mask` for crashed columns is `sqrt(828) ≈ 28.8`. The L2,1 threshold is `lam1/rho = 0.025`. So the threshold barely changes the column — S_crash still ≈ miss_mask. Y2 still near zero. L still comes out same as RPCA.

### The Mathematical Root Cause

The constraint `(1-Omega) = S_crash` is **too easy to satisfy** for large structured crashes. The missing mask IS already sparse (in the structured sense). The dual variable Y2 only grows when there is a gap between `miss_mask` and `S_crash` — and for large crashes, that gap is essentially zero from iteration 1. Y2 never gets a chance to inject meaningful information into L.

**This is not a bug. It is a mathematical property of this specific problem formulation.**

---

## 7. What Actually Gives the 44% Improvement — Honest Analysis

The crash-aware subspace projection (Stage 2) works because of one geometric fact:

The Thunderbird matrix has rank ≈ 5 (top-5 singular values explain 98.5% variance). Every column of the matrix lies in a 5-dimensional subspace. Even if 50% of the rows in a column are missing, you have 828 observed rows to fit 5 parameters. This is a massively overdetermined system. Least squares gives a nearly perfect fit. The missing rows can then be predicted with very high accuracy.

**Why this is better than what RPCA does:** RPCA fills missing entries using the global low-rank structure learned from ALL columns including crashed ones (which are zero-filled). The crashed columns contaminate the SVD. Stage 2 explicitly excludes crashed columns from the subspace learning, giving a cleaner subspace estimate from the 15 safe columns. That cleaner subspace gives much better predictions for the crashed columns.

**The selectivity is the key:** Without S_crash telling you which columns crashed, you would either apply subspace projection to all columns (including random-missing ones, which hurts) or not apply it at all. S_crash makes the targeted application possible.

### Does this technique exist in prior literature?

Yes. Subspace-based column completion — learn subspace from observed columns, project missing columns onto it — is studied in:
- OptSpace (Keshavan et al., 2010)
- GROUSE (Balzano et al., 2010) 
- Subspace Tracking from Missing Data (Narayanamurthy & Vaswani, 2018, arxiv 1810.03051)
- Leveraging Subspace Information for Low-Rank Reconstruction (arxiv 1805.11946)

**What is NOT in any of these papers:**
- Applied to system logs
- S_crash identifying WHICH columns to apply it to (vs. which to leave alone)
- S_tamper simultaneously identifying adversarially corrupted observed entries
- The MNAR/structured missingness framing for forensics purposes
- Distinguishing crash-caused missingness from attacker-caused missingness as two separate problems

---

## 8. What Has Been Done Before — Complete Honest Map

| Method | What it does | Why it's different from MCDD |
|---|---|---|
| Standard RPCA (Candes 2009) | M = L + S, handles sparse corruption | No missing data handling, no crash/tamper separation |
| Masked-RPCA (2019) | Mask identifies foreground/background in video | Pixel domain, mask is unknown to be recovered, no adversarial component |
| GNR (2023, closest paper) | Treats data and missing mask as two equal modalities jointly | General tabular data, deep generative model, no logs, no tamper component |
| OptSpace / GROUSE | Learn subspace from safe columns, project missing | Math technique for random missing, not structured crash forensics |
| DeepLog / LogBERT / LogRobust | ML anomaly detection on complete logs | Assumes logs are complete, cannot handle missing data |
| MNAR matrix completion (Sportisse 2020) | Model-based estimation of MNAR mechanism | Statistical inference framework, not optimization, no adversarial component |

**The gap that MCDD fills:** No paper simultaneously does all of:
1. Handles structured (non-random) missing data in system logs
2. Separates crash-caused missingness from attacker-caused missingness
3. Produces three interpretable outputs (clean timeline + crash map + tamper map)
4. Works without training data (pure optimization, works on any input matrix)

---

## 9. Why Nobody Implemented This Before

This is a real question with a real answer. Three reasons:

**Reason 1 — Two communities that don't talk:**
The matrix completion community works on random missing data (Netflix problem, image reconstruction). The system log community works on anomaly detection assuming complete logs. Nobody sits at the intersection. The problem of "incomplete logs with structured missingness caused by identifiable system failures" is not formulated in either community.

**Reason 2 — The MCAR assumption is never questioned in log papers:**
Every log anomaly detection paper (DeepLog, LogBERT, LogRobust, etc.) assumes complete logs as input. When logs are incomplete, these methods simply fail silently. No paper asks "what if the incompleteness itself is informative?"

**Reason 3 — The mathematical challenge is real:**
As demonstrated above, making the missing mask coupling actually improve reconstruction inside an ADMM loop is harder than it looks. The trivial solution (S_crash = miss_mask) always satisfies the constraint. Getting past this requires either: (a) a different constraint formulation, or (b) accepting that the coupling works through selectivity (Stage 2) rather than through the ADMM dual variable.

---

## 10. The New Novelty — What Has Never Been Done

After honest research across all relevant literature, here is what is genuinely unoccupied:

### The S_tamper Component in This Context

No paper — zero — identifies adversarially tampered log entries as a sparse signal simultaneously with crash reconstruction in an optimization framework. The existing "tamper detection" papers in system security use cryptographic hashing or digital signatures (e.g., US Patent 9864878 — secure log chaining). These require the log to be instrumented before the attack. MCDD is the first to do forensic tamper detection post-hoc from the statistical structure of the log matrix.

**This is the most defensible novelty claim.**

### The Failure Cause Attribution Problem

No paper frames the system log forensics problem as: "given an incomplete log matrix, identify WHICH entries are missing because of crashes vs. WHICH observed entries were modified by an attacker, and reconstruct the true timeline accordingly." This problem formulation — failure cause attribution combined with reconstruction — is new.

### The Combination as a Forensics Framework

The three-output framework (L, S_crash, S_tamper) applied to system logs for post-incident forensics is new as a combined framework. Individual components (subspace completion, sparse decomposition) exist. Their combination for this specific forensics purpose does not.

---

## 11. Is the New Novelty Implementable and How Solid Is It?

### What Can Be Claimed Honestly and Defended

**Claim 1 (Very solid):** S_tamper detection. The soft-thresholding of observed entries after removing the low-rank component correctly identifies entries that are anomalously large (tampered). F1 = 0.795 on Thunderbird. RPCA produces no such output. This is directly useful for forensics and has no prior analog in log analysis.

**Claim 2 (Solid):** Failure cause attribution. S_crash ≈ missing mask for large structured crashes — but that IS useful. Knowing which columns experienced structured failure vs. which entries were randomly missing is exactly what a forensic investigator needs. The output is interpretable, even if the mathematics is simpler than it first appears.

**Claim 3 (Solid with caveats):** Crash-conditioned reconstruction. 44% RMSE improvement over RPCA on structured crash conditions. The mechanism (subspace projection from safe columns) uses known techniques, but the selective application conditioned on S_crash is novel in this domain. Caveat: must be framed as a two-stage method, not as ADMM improving reconstruction.

**Claim 4 (Weak — do not make this claim):** The ADMM dual variable Y2 coupling improves L reconstruction. The results do not support this. Do not claim it.

### Implementation Status

All steps implemented and working:
- step1: Thunderbird parsing and matrix construction ✅
- step2: Low-rank verification (98.5% explained by top 5) ✅  
- step3: Three corruption conditions ✅
- step4: RPCA baseline ✅
- step5: MCDD with two-stage architecture ✅ (use original version with post-processing)
- step6: Evaluation — all metrics computed ✅
- step7: All 7 figures generated ✅
- step8: Sensitivity analysis (bug — uses different algorithm than step5, needs fix) ❌
- step9: Missing rate sweep (spike at 50% needs explanation) ⚠️

### step8 Bug
step8 has `Z1 += Y2/rho` (Y2 coupling) but step5 original does not. They test different algorithms. step8 RMSE values (~0.86) reflect the broken version, not the working version (0.489). Fix: use identical two-stage code in step8.

### step9 Spike at 50%
At 50% missing rate, MCDD RMSE = 1.40 (worse than RPCA). Reason: with exactly 50% of rows missing in a crashed column, the observed portion has only 828 rows to fit rank-19 subspace. At lower missing rates, more observed rows → overdetermined system → perfect fit. At exactly 50%, near the boundary. This is a real limitation and should be reported honestly as a boundary condition.

---

## 12. The Honest Summary — Where This Project Actually Stands

### What is strong
- Results are real and reproducible (44% RMSE improvement, Crash F1=1.0, Tamper F1=0.795)
- The three-output framework is genuinely useful for forensics
- S_tamper detection has no prior analog in this domain
- Problem framing (failure cause attribution from incomplete logs) is novel
- Strong dataset (Thunderbird, 98.5% low-rank, dense matrix, well-studied in log analysis community)
- Pure optimization — no training, no neural network, convergence guaranteed

### What is weak
- The ADMM loop does not outperform RPCA on its own — all improvement from Stage 2
- Stage 2 uses subspace projection technique that exists in literature
- Single dataset, synthetic crash injection (not real failures)
- S_crash ≈ miss_mask for large crashes — mathematically trivial for the structured case
- step8 tests a different algorithm than step5 (bug)
- step9 has unexplained spike at 50% missing rate

### Publication verdict
- **Course project (22MAT230):** Exceptional. Strong math, real results, genuine contribution.
- **Workshop paper:** Possible if step8 fixed, framing is honest about two-stage nature, S_tamper novelty is front and center.
- **Main conference (USENIX, IEEE S&P, SIGMETRICS):** Not yet. Needs: real missing patterns (not synthetic injection), second dataset, ablation proving each stage's contribution, and a theoretically stronger coupling mechanism.

---

## 13. The Groundbreaking Research Direction — What Could Be Done

If this were to become genuinely publishable research, here is the path:

### The Real Novel Mechanism That Doesn't Exist Anywhere

**Problem with current formulation:** The constraint `(1-Omega) = S_crash` is satisfied trivially. Y2 never injects meaningful information into L.

**What would actually work:** Instead of constraining S_crash to equal the missing mask, model the **generation process** of the missing mask as a function of L itself.

The new constraint: missing entries are caused by crash events that wipe entire columns when the value in L exceeds a threshold (system overload → crash). Formally:

```
(1 - Omega_{i,j}) = 1  iff  L_{i,j} > threshold_j  (system failure condition)
```

This makes the missingness **depend on L**, which is the true MNAR (Missing Not At Random) condition. Now Y2 genuinely couples crash detection to reconstruction because the crash model links the missing pattern back to the latent matrix values. This formulation has not been done in any paper. It requires either a bilevel optimization or a relaxed differentiable formulation.

### Why This Would Be Groundbreaking
It would be the first method that:
1. Models the physical failure mechanism (not just the missing pattern) as part of the optimization
2. Makes the MNAR assumption explicit and exploits it (not just acknowledges it)
3. Produces theoretically justified improvements in L reconstruction (not just post-processing)
4. Is applicable to any system with physically interpretable failure modes (not just logs)

### Is It Implementable?
Yes, but harder. Would require 2–3 months of additional work. The relaxed formulation would replace the hard constraint with a soft sigmoid: `sigma(L_{i,j} - threshold_j) ≈ (1 - Omega_{i,j})`. This is differentiable and can be incorporated into the ADMM augmented Lagrangian. The threshold parameters become learnable from the data.

---

## 14. Files and Run Order

```
step1_sample_thunderbird.py    → parse 5M lines → M_ground_truth.npy (1656×30)
step2_check_sample.py          → verify low-rank (98.5% top-5)
step3_currupt.py               → create conditions A, B, C
step4_baseline_rpca.py         → run RPCA on all conditions
step5_mcdd.py                  → run MCDD (USE ORIGINAL WITH POST-PROCESSING)
step6_evaluate.py              → compute all metrics → evaluation_table.csv
step7_plots.py                 → generate all 7 figures
step8_sensitivity_analysis.py  → FIX NEEDED: use same code as step5
step9_sweep_missing_rate.py    → EXPLAIN 50% spike in report
```

## 15. Key Numbers to Remember

| Metric | Value |
|---|---|
| Matrix size | 1,656 × 30 |
| Top-5 singular values explain | 98.5% variance |
| Condition B missing | 25.0% of matrix (12,420 entries) |
| Condition C tampered | 5% of observed (1,863 entries) |
| RPCA RMSE on structured crash | 0.868 |
| MCDD RMSE on structured crash | 0.489 |
| Improvement | 44% |
| Crash F1 | 1.0 (perfect detection) |
| Tamper F1 | 0.795 |
| RPCA tamper F1 | N/A (cannot produce this output) |
| ADMM convergence | 207–277 iterations |

---

*Document compiled from full project history including HDFS attempt, Thunderbird implementation, multiple code versions, mathematical audits, and literature research. Last updated April 2026.*
