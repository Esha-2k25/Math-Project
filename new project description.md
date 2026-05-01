# MCDD — Research Description
## 22MAT230 | Event Timeline Reconstruction from Incomplete System Logs

---

## 1. The Problem We Set Out to Solve

When a computer system crashes or gets attacked, investigators look at system logs to reconstruct what happened. These logs are not just passive records — they are the forensic evidence. But in real systems, those logs are almost always incomplete.

Three distinct reasons logs go missing:

**Node crash.** A server dies and stops writing logs entirely. An entire block of time across multiple event types goes blank — not randomly, but as a structured rectangular block of silence. The missingness is caused by a real physical event.

**Attacker intrusion.** The first thing an attacker does after gaining access is delete or overwrite log entries to hide their tracks. These are not missing — they are present but falsified. A value that should be 1.2 becomes 9.7.

**System overload.** Logging daemons drop entries under heavy load. This is the closest to random — sporadic missing entries scattered throughout.

The core observation that motivated this project: **every existing method treats all three of these as the same thing.** They are not. A missing entry caused by a crash is evidence of a crash. A present-but-inflated entry is evidence of tampering. Treating them identically means losing the forensic signal entirely.

Our goal: given an incomplete and possibly tampered log matrix, simultaneously reconstruct the clean timeline, identify which missing entries were caused by crashes, and identify which observed entries were modified by an attacker — from the statistical structure alone, with no prior training, no instrumentation, and no cryptographic overhead.

---

## 2. Literature Survey

### 2.1 Matrix Completion and RPCA

The foundational work is Candès and Recht (2009) on matrix completion via nuclear norm minimization, and Candès et al. (2011) on Robust PCA (RPCA). RPCA decomposes an observed matrix as:

```
M = L + S
```

where L is low-rank (recurring normal behavior) and S is sparse (corruption). This is solved via ADMM — Alternating Direction Method of Multipliers — which alternates between a singular value thresholding step for L and a soft-thresholding step for S.

**Critical assumption in all RPCA variants:** the observed entries are trustworthy. RPCA identifies S as entries that deviate from the low-rank model — but it assumes the missing entries (the mask Omega) carry no information. The missingness is treated as noise.

Key papers surveyed:

- Candès, Li, Ma, Wright (2011) — "Robust Principal Component Analysis?" — the original RPCA via ADMM. Sets the baseline. Does not handle structured missingness.
- Lin, Chen, Ma (2010) — "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices" — practical ADMM implementation we followed.
- Wright, Ganesh, Rao, Peng, Ma (2009) — "Robust Principal Component Pursuit" — extensions to partial observations.
- Keshavan, Montanari, Oh (2010) — OptSpace. Learns a subspace from observed entries and projects missing entries onto it. This is the mathematical technique underlying our Stage 3.
- Balzano, Nowak, Recht (2010) — GROUSE. Online subspace tracking from missing data.
- Narayanamurthy and Vaswani (2018) — Subspace tracking from missing data, arxiv 1810.03051.

**What none of these do:** they all treat missingness as uninformative. The mask Omega is assumed random (MCAR — Missing Completely At Random). No paper in the matrix completion literature models the generation process of the mask as depending on the latent matrix L.

### 2.2 System Log Analysis

The system log anomaly detection literature is large but operates under a completely different assumption: complete logs.

Key papers surveyed:

- Du et al. (2017) — DeepLog. LSTM-based anomaly detection on complete log sequences.
- Guo et al. (2021) — LogBERT. BERT-based log anomaly detection.
- Zhang et al. (2019) — LogRobust. Robust log anomaly detection using semantic vectors.
- He et al. (2017) — Drain. Fast log parsing to extract event templates.
- Lou et al. (2010) — Mining invariants from console logs for system problem detection.

**What all of these assume:** the log matrix is fully observed. When logs are missing, these methods fail silently — they simply cannot run, or produce meaningless outputs. No paper in this community asks: "what if the pattern of missingness is itself the forensic evidence?"

### 2.3 Missing Not At Random (MNAR) Literature

In statistics, the MNAR assumption — where missingness depends on the unobserved values themselves — has been studied in survey methodology and clinical trials.

Key work:

- Little and Rubin (2002) — "Statistical Analysis with Missing Data." Defines MCAR, MAR, MNAR as a taxonomy.
- Sportisse et al. (2020) — "Imputation and low-rank estimation with Missing Not At Random data." Statistical inference framework for MNAR matrix completion.
- Josse et al. (2019) — "Consistency of high-dimensional estimation under MNAR."

**What these do not do:** they address statistical estimation of missing values. They do not frame the MNAR mechanism as a forensic signal. They do not simultaneously identify tampered entries. They work on general tabular data, not system logs. None produce interpretable physical parameters of the failure system.

### 2.4 Tamper Detection in System Security

Tamper detection in log security uses completely different approaches:

- US Patent 9864878 (2018) — Secure log chaining with cryptographic signatures. Requires instrumentation before the attack.
- Holt (2006) — Logcrypt. Cryptographic forward-integrity for log files.
- Ma and Xu (2004) — TamperEvident logs. Hash-chain based tamper detection.

**What none of these do:** they all require the logging system to be instrumented before the attack occurs. They cannot do post-hoc forensic analysis if the attacker got in before instrumentation. MCDD is the first to attempt tamper detection purely from the statistical structure of the log matrix, with no prior instrumentation.

### 2.5 The Gap Map

| Method | Handles structured missing? | Identifies tamper? | Reconstructs timeline? | No training needed? | Works post-hoc? |
|---|---|---|---|---|---|
| Standard RPCA | No (MCAR only) | Partially | Yes | Yes | Yes |
| OptSpace / GROUSE | No (MCAR only) | No | Yes | Yes | Yes |
| DeepLog / LogBERT | N/A (assumes complete) | No | No | No (needs training) | No |
| Sportisse 2020 | Yes (MNAR) | No | Yes | Yes | Yes |
| Cryptographic methods | N/A | Yes | No | N/A | No (pre-attack only) |
| **MCDD (Ours)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

No prior method satisfies all five properties simultaneously.

---

## 3. The Research Gap We Targeted

Three specific gaps, ordered by novelty:

**Gap 1 — Forensic tamper detection without instrumentation.** No paper does post-hoc identification of tampered log entries from the statistical structure of the log matrix. Cryptographic methods require setup before the attack. RPCA's S component identifies anomalies but was never applied to this problem in the log domain, and produces no interpretation of what kind of anomaly it is.

**Gap 2 — Failure cause attribution.** No paper asks: "given an incomplete log matrix, which entries are missing because of a crash, and which observed entries were modified by an attacker?" These are treated as separate problems (if treated at all) by separate communities that do not talk to each other.

**Gap 3 — Structured missingness as a signal, not noise.** Every matrix completion paper treats missing entries as random obstacles to overcome. The insight that a rectangular block of missing entries is direct evidence of a node crash — and that this evidence should be used to improve reconstruction, not ignored — does not appear in any prior paper.

---

## 4. What We Originally Designed — The Vision

### 4.1 The Original Formulation

The original MCDD formulation extended RPCA with a second domain of decomposition:

```
M_true = L + S_crash + S_tamper
```

| Variable | Represents | Domain |
|---|---|---|
| L | Normal recurring system behavior | Low-rank, data domain |
| S_crash | Which blocks are missing and why | Sparse, missing mask domain |
| S_tamper | Attacker-modified entries | Sparse, observed data domain |

The optimization problem:

```
min  ||L||_*  +  λ₁·||S_crash||₁  +  λ₂·||S_tamper||₁

s.t. P_Ω(L + S_tamper)  = P_Ω(M_obs)     [fidelity on observed entries]
     (1 − Ω)             = S_crash         [mask decomposition — the key novelty]
```

The second constraint — treating the missing mask itself as a decomposable sparse signal — is the original novel contribution. The idea: if a node crashes, it produces a rectangular block in the missing mask. That rectangle is sparse (in the column-sparse sense). If you decompose the mask into S_crash via L1 minimization, S_crash should identify which columns crashed.

### 4.2 The ADMM We Designed

ADMM on the augmented Lagrangian, with two dual variables Y1 (for the data fidelity constraint) and Y2 (for the mask decomposition constraint):

```
L update:       L^{k+1} = SVT(M_fill - S_tamper + Y1/ρ + Y2/ρ, 1/ρ)
S_tamper update: S^{k+1} = soft_thresh(M_fill - L^{k+1} + Y1/ρ, λ₂/ρ)
S_crash update: S_crash^{k+1} = clip(soft_thresh(miss_mask + Y2/ρ, λ₁/ρ), 0, 1)
Y1 update:      Y1 += ρ · (M_fill - L - S_tamper)   [on observed entries]
Y2 update:      Y2 += ρ · (miss_mask - S_crash)
```

The vision: Y2 would accumulate the difference between the true missing mask and S_crash, feeding back into the L update and gradually improving reconstruction in crashed regions.

### 4.3 The Original Dataset Plan

The original plan used HDFS_v1 from LogHub: 186.6 MB Hadoop logs, 575,000 blocks × 29 event types. This was the dataset used in most log anomaly detection papers, making comparison natural.

### 4.4 Expected Results

The original expectation: the ADMM dual variable Y2 would couple crash detection back into L reconstruction, so that identifying which columns crashed would allow the algorithm to fill missing entries more accurately than standard RPCA. Crash F1 ≈ 1.0 (structured crash is geometrically obvious), Tamper F1 ≈ 0.8+, and a meaningful RMSE improvement from the coupling — not just from post-processing.

---

## 5. What Actually Happened — The Evolution

### 5.1 First Problem: HDFS Failed

HDFS_v1 had poor low-rank structure. After log1p compression, the matrix was sparse with many zero cells. Top-5 singular values explained only around 70-75% of variance — not strong enough for the reconstruction argument to hold. Results on HDFS showed MCDD barely beating RPCA (RMSE difference < 1%). The three-output framework could not be demonstrated convincingly on data without strong low-rank structure.

Decision: switch to Thunderbird — 9,024-node IBM BladeCenter supercomputer logs from Sandia National Labs. First 5,000,000 lines processed. Result: 1,656 × 30 matrix, 0% sparsity, top-5 singular values explaining 98.5% of variance. The low-rank structure was extreme and well-suited to the reconstruction argument.

### 5.2 Second Problem: Y2 Coupling Was Mathematically Trivial

The core insight of the original formulation — that Y2 would feed meaningful information about the crash structure back into L — turned out to be incorrect for a fundamental mathematical reason.

The constraint is `(1 − Ω) = S_crash`. For a large structured crash (50% of a column missing), the missing mask IS already sparse in the column sense. The optimal S_crash for an L1 penalty on a large block crash is simply equal to the missing mask. The residual `miss_mask − S_crash ≈ 0` from iteration 1. Y2 stays near zero across all 300 iterations. The coupling is a mathematical no-op.

This was verified empirically: adding `Y2/ρ` to the L update produced RMSE = 0.862, essentially identical to standard RPCA (0.868). The 0.7% difference is noise.

**Why this happens:** the constraint `(1−Ω) = S_crash` is too easy to satisfy for large structured crashes. The dual variable Y2 only grows when there is a gap between miss_mask and S_crash — and for large crashes, that gap is zero from the start. The original formulation's key coupling mechanism does not work.

### 5.3 What Actually Produces the Results

The improvement in reconstruction (30% RMSE reduction on Condition B, 28% on Condition C) comes entirely from Stage 3 — the crash-aware subspace projection:

1. Identify crashed columns from empirical miss_frac > 0.25 (honest, not "learned" by ADMM)
2. Learn a row subspace from safe columns only — excluding crashed columns from the SVD
3. For each crashed column: fit the observed rows to this subspace via least squares, predict missing rows

Why this works geometrically: the Thunderbird matrix has rank ≈ 5 (top-5 explain 98.5%). With 828 observed rows in a crashed column and only 3 basis vectors to fit, the least squares system is 828:3 — massively overdetermined. The fit is near-perfect. The key is using the subspace from safe columns only — RPCA contaminates its SVD with zero-filled crashed columns, giving a noisier subspace.

### 5.4 What ADMM Genuinely Contributes

ADMM does not drive the reconstruction improvement. It drives S_tamper.

Standard RPCA also produces an S component. But RPCA's S component conflates tamper signal with zero-fill artifacts from missing entries — it does not constrain S to observed entries. MCDD's ADMM enforces `S_tamper[missing] = 0`, producing a cleaner separation. Entries multiplied 4-10× by the attacker produce residuals far above the low-rank fit. At the 97th percentile threshold, Tamper F1 = 0.860, AUC = 0.987.

This is the ADMM contribution: a forensically clean S_tamper component, constrained to observed entries, correctly interpreted as the tamper signal. No prior log analysis paper produces this output.

ADMM convergence diagnostics confirm correct implementation: primal and dual residuals decrease monotonically across all three conditions, converging at iterations 207 (Condition B), 230 (Condition A), and 278 (Condition C). Rank of L collapses from 25+ to 19–20 for structured conditions, stays at 30 for the random condition (correct — random missing does not induce low-rank structure in the residual).

### 5.5 The Post-Hoc MNAR Analysis — What Emerged

After accepting that the Y2 coupling did not work as designed, a different MNAR approach was developed for Stage 4: fit a per-column crash threshold θⱼ post-hoc, after L reconstruction is complete.

The model: `P(missing | Lᵢⱼ, θⱼ) = sigmoid(α · (Lᵢⱼ − θⱼ))`

This says: entry (i,j) goes missing when the event count in L exceeds a column-specific threshold — the physical interpretation of system overload causing crash. θⱼ is fitted by gradient descent on the sigmoid model.

For Condition A (random missing): all columns have miss_frac ≈ 0.20, so the MNAR weight is zero everywhere. θ does not move from initialization. This is correct — random missing carries no crash signal.

For Condition B (structured crash): crashed columns 5-19 converge to θ ≈ −0.38, safe columns converge to θ ≈ +3.91. Separation = +4.29. The model correctly identifies that crashed columns have a lower event-count threshold for failure — they are more vulnerable.

For Condition C (adversarial): separation = +6.81. Even stronger, because the tampered entries are separated into S_tamper before the MNAR analysis runs, leaving a cleaner L.

This θ output — a learned per-column crash vulnerability parameter — has no analog in any prior paper. It is genuinely new as a forensic artifact.

---

## 6. What We Wanted vs. What We Got

| Dimension | What We Wanted | What We Got | Verdict |
|---|---|---|---|
| Core mechanism | ADMM dual variable Y2 couples crash detection into L reconstruction | Y2 ≈ 0 always; coupling is mathematically trivial for large crashes | Honest failure — mechanism doesn't work as designed |
| Reconstruction gain | ADMM-driven improvement | Stage 3 subspace projection drives all gain | Real gain (30%), different mechanism than intended |
| Crash detection | S_crash as a learned sparse signal identifying crashed columns | Empirical miss_frac threshold (col_miss_frac > 0.25) — exact same result, simpler method | F1 = 1.0, but via observation not learning |
| Tamper detection | S_tamper from ADMM decomposition | S_tamper from ADMM decomposition | Exactly as designed — AUC = 0.987, F1 = 0.860 |
| Novel forensic output | Three interpretable outputs (L, S_crash, S_tamper) | Three outputs plus θ (crash threshold per column) | More than designed — θ is a new artifact |
| MNAR coupling | Inside ADMM loop (Y2 constraint) | Post-hoc gradient descent on sigmoid model | Different architecture, cleaner results |
| Dataset | HDFS_v1 (LogHub) | Thunderbird (Sandia Labs) | Better dataset — 98.5% low-rank vs weaker HDFS |
| RMSE improvement | Significant (expected 30-40%) | 30% on Condition B, 28% on Condition C | Matches expectation, different mechanism |
| Random condition behavior | MCDD ≈ RPCA (correct, no crash to exploit) | MCDD ≈ RPCA (0.2% improvement) | Exactly correct |
| Theoretical coupling | Strong bilevel optimization coupling mask to L | Two-stage sequential method | Weaker mathematically, but honest and working |

---

## 7. What Is Genuinely Novel in the Final Implementation

Three claims that survive honest scrutiny and have no prior analog:

**Claim 1 — Forensic tamper detection post-hoc, without instrumentation (Very solid).** S_tamper from the ADMM L+S decomposition, constrained to observed entries, identifies attacker-inflated log entries with AUC = 0.987. No paper in system log analysis produces this output. Cryptographic methods require pre-attack instrumentation; MCDD works from the existing log matrix alone.

**Claim 2 — Failure cause attribution as a unified problem (Solid).** The framing — given an incomplete log matrix, identify which entries are missing because of crashes vs. which observed entries were modified by an attacker — is new as a combined problem. Individual components exist in literature; the combination for system log forensics does not.

**Claim 3 — Per-column crash threshold θⱼ as a forensic artifact (Solid).** A learned parameter per event type indicating its crash vulnerability (the event-count level at which that event type tends to go silent during failures). Separation between crashed and safe columns of 4.29 standard units on Condition B, 6.81 on Condition C. This output does not exist in any log reconstruction, matrix completion, or MNAR paper.

---

## 8. Key Results

| Metric | Value |
|---|---|
| Matrix (Thunderbird) | 1,656 × 30 |
| Low-rank structure | Top-5 SVs explain 98.5% variance |
| RPCA RMSE — Structured crash | 0.868 |
| MCDD RMSE — Structured crash | 0.607 |
| Improvement over RPCA | **30.0%** |
| Crash detection F1 | **1.000** (perfect) |
| Tamper detection F1 | **0.860** (97th pct threshold) |
| Tamper detection AUC | **0.987** |
| Theta separation (B) | **+4.29** (safe − crash) |
| Theta separation (C) | **+6.81** |
| ADMM convergence | 207–278 iterations |

---

## 9. What This Project Is and Is Not

**It is:** a genuine research contribution at the course project level. The problem formulation (failure cause attribution), the three-output forensics framework, and the θ artifact are all new in the system log domain. The ADMM implementation is correct. The results are real and reproducible. The honest analysis of what works and why is itself a contribution.

**It is not:** a conference-ready paper. The ADMM dual coupling does not work as originally theorized. The crash injection is synthetic. The dataset is single. The Stage 3 subspace projection uses techniques from existing literature (OptSpace lineage). These limitations are known and documented.

**The path to publication** would require: formalizing the bilevel MNAR optimization where missingness depends explicitly on L (replacing the trivial S_crash = miss_mask constraint), real crash data from production systems, a second dataset, and a theoretical convergence analysis of the sigmoid-coupled ADMM.

---

*Research description compiled from full project history: HDFS attempt, dataset migration to Thunderbird, multiple code versions, mathematical audits, literature search, and implementation results. 22MAT230, April 2026.*