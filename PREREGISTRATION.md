# Pre-registration

Study: Offline evaluation as the object of study, corrected pass.
Sampled-metric bias, temporal leakage, popularity bias, and off-policy
correction, measured against a fully observed ground truth, with the
fully observed arm rebuilt so that its scored users have real training
histories, and with a mandatory coverage gate in front of every score.

Author of record: Jesun Ahmad Ushno.
Status: FROZEN. This file is the first content commit of this
repository, committed before any evaluation metric is computed on any
test split in this study. The commit hash of this file is the proof.
Any later change to this file requires a dated amendment section at
the bottom and a note in DECISIONS.md.

Relation to v1: this study is the corrected second pass of
JesunAhmadUshno/recommender-evaluation-lab (v1, results of record at
commit a9c0464). v1's design, hypotheses, and verdict rules are carried
unchanged wherever they do not depend on the defective arm. Every
change is named in section 10 with its justification. v1's defect
report (the validity notice at the top of its RESULTS.md and the
validity audit in its AUDIT.md) stays public and is linked from this
repo's README. Nothing in v1 is hidden by this repo; v2 corrects it.

## 1. The defect being fixed

Measured in v1's fresh pass and verified by the programme coordinator
from v1's raw loaders:

- KuaiRec's small matrix is pair-disjoint from its big matrix: zero of
  the 4,676,570 small-matrix pairs appear in the big matrix. Big-matrix
  events by the 1,411 small-matrix users total 571,061, and exactly 0
  of them fall on small-matrix items. The user and item id mappings
  were verified correct, so this is a dataset fact, not an indexing
  bug.
- v1's frozen Arm F training pool (DECISIONS.md D5 of v1: big-matrix
  events restricted to small-matrix items) therefore contains no event
  by any scored user. All 1,411 target users received an empty
  training history row.
- Verified from v1's run bundles: the ItemKNN, EASE, and iALS Arm F
  score matrices are identical and all zero (zero nonzero entries).
  Their Arm F metrics are tie-break artifacts of sorting zeros, and
  every verdict whose evidence includes those rows is void: H2a, H5,
  H6, H4a, H4b, and S1 as computed, and H2b's flip test, which cannot
  fire under the tie artifact.
- Still valid in v1, and the replication target of this pass: Arm F
  Random and Popularity rows, all of Arm T on both datasets, the
  random-split arm, the leakage audit, and the cold-start slice.
  Verdicts H1a, H1b, H3a, H3b, and H7 stand on valid evidence.

The check whose absence let this happen: no stage of v1 ever verified
that the users being scored had any training history in the pool they
were scored from. Section 5 of this registration makes that check a
mandatory gate with frozen thresholds.

## 2. Research questions

Identical to v1. Main question: on a dataset where the true answer is
known, how far do standard offline evaluation shortcuts move the
measured score and the ranking of models, and how much of that error
do the standard corrections remove?

1. Sampled metrics. Does ranking against 100 sampled negatives keep
   the model order given by the full ranking? Does the rank-mapping
   correction help?
2. Temporal leakage. Does a random split, against a global-time split,
   change scores and model order?
3. Exposure and popularity bias. Does a naive evaluation on biased,
   partly observed logs over-rate popular-item rankers?
4. Counterfactual correction. Do IPS, SNIPS, and doubly robust (DR)
   estimates land closer to the ground truth than the naive metric,
   with known propensities and with estimated ones?

## 3. Datasets (audit measured before registration; descriptive only)

KuaiRec v2.0, Zenodo record 18164998, KuaiRec.zip SHA-256
c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c
(MD5 matches the Zenodo record). Licence reading: DECISIONS.md D2
(carried from v1 D3).

- Big matrix: 12,530,806 rows, 7,176 users, 10,728 items. Timestamps
  1592872451.373 to 1599694332.427, none missing.
- Small matrix: 4,676,570 rows, 1,411 users, 3,327 items, of
  4,694,397 possible pairs (99.62 percent observed; 17,827 pairs have
  no row and are treated as non-positive where a label is needed).
  181,992 rows (3.9 percent) have empty time fields; their labels are
  present. In v1, whose Arm F had no time windows, those labels were
  used. In v2, Arm F has time windows (section 4), so those rows fall
  in no window and are excluded from Arm F entirely; the exclusion and
  its size are recorded here, before any run.
- Audit A2/A3 (carried from v1, re-verified by the coordinator):
  small-matrix users and items are subsets of the big matrix by raw
  id. Zero of the 4,676,570 small pairs appear in the big matrix.
  Big-matrix events on small items: 10,272,525, all of them by the
  other 5,765 users.
- Label rates (v1 audit): big matrix positives (watch_ratio > 2.0)
  7.47 percent of rows; small matrix 4.64 percent at > 2.0.

MovieLens-1M (second dataset, replication role, carried from v1 D9):
ml-1m.zip SHA-256 a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20.
1,000,209 ratings, 6,040 users, 3,706 items with ratings. GroupLens
research terms; no redistribution; derived statistics only.

Relevance labels (frozen, unchanged from v1): KuaiRec positive if
watch_ratio > 2.0. Sensitivity run at > 1.0. MovieLens-1M positive if
rating >= 4. Users with zero positives in an evaluation set are
excluded from that set's means and counted.

Split-construction audit for this registration (descriptive only; no
model fitted, no evaluation metric computed; measured 2026-11-08 on
the machine of record against the frozen archives):

| Arm | Cutoffs T1 / T2 | Train events | Scored users | Truth positives | User coverage | Item coverage |
|---|---|---|---|---|---|---|
| Arm T KuaiRec big | 1598650255.457 / 1598870571.464 | 10,024,644 | 6,638 | 87,887 | 1.000 | 0.557 |
| Arm T MovieLens-1M | 975768738.0 / 978133376.4 | 800,164 | 1,188 | 54,452 | 0.836 | 0.994 |
| Arm F v2 (small, primary) | 1597388593.960 / 1597977252.744 | 3,595,662 | 1,381 | 24,572 | 1.000 | 0.663 |
| Arm F v2 sensitivity pool (train before small median 1596223943.27) | same test set | 2,247,289 | 1,381 | 24,572 | 1.000 | 0.245 |
| Random arm KuaiRec big, seed 0 | seeded 80/10/10 | about 80 percent of events | 6,930 | 71,482 | 1.000 | 0.995 |
| Random arm MovieLens-1M, seed 0 | seeded 80/10/10 | about 80 percent of events | 5,759 | 57,580 | 1.000 | 0.998 |

Coverage is defined in section 5. Two audit facts shape the gate
thresholds there. User coverage is 1.000 or 0.836 in every honest
construction audited, and exactly 0.000 in the v1 defect. Item
coverage ranges from 0.245 to 0.998 across honest constructions: on a
long-tail catalogue under a time split, many truth items legitimately
have no training-window event (they are the cold items the cold-start
slice exists to measure). Item coverage is therefore a published
diagnostic with a low sanity floor, not a pass line; the user gate is
the one whose absence sank v1.

## 4. Arms, splits, rankers

Arm T (logged benchmark): unchanged from v1. KuaiRec big, replicated
on MovieLens-1M. One global timeline per dataset; cut points at the
80th and 90th percentiles of event timestamps, measured by the audit
and pinned in section 3's table: KuaiRec big T1 = 1598650255.457,
T2 = 1598870571.464; MovieLens-1M T1 = 975768738.0, T2 =
978133376.4. Train t < T1, validation T1 <= t < T2, test t >= T2.
Test positives are first-time positives: positive in the test window
and the (user, item) pair absent from all training-window events.
Scores mask training positives. Leakage assertions: max(train
timestamp) < min(test timestamp) and zero test pairs in train,
checked in code.

Arm F v2 (fully observed truth, corrected): the same protocol as
Arm T, executed inside the small matrix. One global timeline over the
small matrix's 4,494,578 timestamped rows; cut points at the same
80th and 90th percentiles, pinned in section 3's table: T1s =
1597388593.960, T2s = 1597977252.744. Train: small-matrix events with
t < T1s (3,595,662 events). Validation: T1s <= t < T2s. Truth set:
first-time positives at t >= T2s (24,572 positives over 1,381 scored
users, mean 17.8 per user). Training positives are masked at scoring,
as on Arm T. Rows with empty timestamps (181,992) fall in no window
and are used nowhere on this arm.

Why this construction is the arm v1 intended: v1's Arm F existed to
score rankers against a truth set in which every candidate's label is
known, so that sampled negatives are true negatives and the
off-policy estimand has exact truth. The corrected arm keeps exactly
that property (every (user, item) pair in the small matrix has a
known label, up to the 0.38 percent of pairs with no row), while the
training histories now come from the same matrix the truth is drawn
from, under the same temporal discipline as Arm T, so no scored user
can have an empty history by construction. What it gives up, stated
plainly: v1's truth was the whole small matrix; v2's truth is a fully
observed held-out window of it. The estimand is the same kind of
quantity (mean per-user DCG@10 of a ranker's top-10 against known
labels), restricted to the window. Arm F hyperparameters are the ones
selected on the KuaiRec Arm T validation window, as frozen in v1.

Arm F sensitivity runs (carried from v1, adapted to the corrected
pool): truth labels at watch_ratio > 1.0 on the same scores; and a
pool-sensitivity refit trained on small-matrix events before the
small-matrix median timestamp (1596223943.27) against the same truth
set.

Random arm (leakage comparison): unchanged from v1. Seeded 80/10/10
split of all events of KuaiRec big and of MovieLens-1M, seeds 0 to 4,
same rankers and the same hyperparameters as the temporal arm of the
same dataset, test positives first-time against the random training
window.

Ranker ladder (unchanged): L0 Random (analytic floor), L1 Popularity
(exposure count in the training window), L2 ItemKNN (cosine,
co-occurrence shrinkage), A EASE (closed form, one L2 parameter),
B iALS (implicit ALS). SASRec and LightGCN remain excluded (v1 D4).

Tuning (frozen, unchanged): validation window only, at most 20 trials
per ranker. ItemKNN: k in {50, 100}, shrink in {0, 50}. EASE: lambda
in {10, 50, 100, 250, 500}. iALS: factors in {32, 64, 128} x reg in
{0.1, 1.0} at alpha = 10, plus alpha in {1, 40} at factors = 64,
reg = 0.1; tuning iterations 10. Selection metric: validation
NDCG@10. Final refits use the selected configuration; iALS final
iterations 15. Seeds: iALS model seeds 0 to 4 (final metrics averaged
over seeds), Random ranker seeds 0 to 4, negative-sampling seeds 0 to
9, exposure seeds 0 to 9, random-split seeds 0 to 4.

...[truncated 13571 chars]
## 5. The coverage gate (new in v2; mandatory)

For every arm, every split seed, and every training-pool variant,
before any ranker is fitted or scored, the harness computes:

- user coverage: the fraction of scored users (users with at least
  one truth positive in the arm's truth set) who have at least one
  event in the training pool actually fed to the rankers, in the item
  space being scored.
- item coverage: the fraction of scored items (items appearing in the
  truth set) with at least one training-pool event.

Frozen thresholds:

- User coverage below 0.50 on any arm stops the run.
- User coverage below 0.95 on the Arm F primary pool stops the run.
  (Audit value: 1.000. On Arm F a scored user with no training
  history is a construction error by definition, because the pool and
  the truth set come from the same matrix.)
- Item coverage below 0.10 on any arm stops the run. Item coverage is
  otherwise published, not gated: section 3's audit shows honest
  values from 0.245 to 0.998.

On a stop, the run's output is the coverage table
(results/coverage.json) and a log line naming the arm and the failed
threshold. No metric from a stopped arm may be reported, in any form.

Post-scoring backstop, also frozen: after scoring, any ranker whose
score matrix contains zero nonzero finite entries fails the whole
run (the exact v1 signature). On Arm F, each of ItemKNN, EASE, and
iALS must additionally produce a nonzero finite score for at least 95
percent of scored users. The gate is unit-tested on a deliberately
disjoint fixture that reproduces v1's construction in miniature
(training pool drawn from other users only): the gate must refuse it.

## 6. Metrics and the sampled-metric correction

Unchanged from v1. Exact metrics (primary): full ranking over the
candidate set. NDCG@10, Recall@10, MRR@10, HR@10, and per-user
unnormalised DCG@10. Sampled metrics: per user, the candidate set is
all truth positives plus m = 100 negatives sampled uniformly without
replacement from the user's non-relevant, non-training items; 10
sampling seeds. On Arm F the sampled negatives are known negatives
(the matrix is fully observed); on Arm T they are unobserved pairs,
as in v1. Correction (primary): the rank-mapping correction, r_hat =
1 + (r_tilde - 1) * (n - 1) / m with n the full item count, corrected
NDCG@10 computed on the corrected ranks against the exact IDCG over
all truth positives.

Beyond accuracy, per ranker and arm: catalogue coverage@10, Gini of
item exposure in top-10 lists, average recommended popularity
(training exposure), long-tail share (bottom 80 percent of items by
training exposure), and the top-1-percent / top-10-percent
recommendation shares.

## 7. Off-policy study (semi-synthetic exposure, Arm F)

Adapted from v1 section 5 to the corrected pool; every adaptation is
listed in section 10.

1. Item propensity from the Arm F training pool (the within-small
   train window): n_i is item i's pool exposure count; p_i =
   max(0.01, c * (n_i / n_max) ^ gamma), gamma in {0 (MCAR control),
   0.5, 1.0}; c set so the mean propensity equals 0.163, the same
   target as v1, computed before the floor.
2. Exposure draws O_ui ~ Bernoulli(p_i) over the full small-matrix
   grid, 10 exposure seeds per gamma. This exposure is simulated; the
   arm is semi-synthetic and every claim from it carries that label.
3. Estimand: each ranker's mean per-user DCG@10 of its Arm F top-10
   lists, truth computed from the Arm F truth set, averaged over the
   scored users (section 4). v1 averaged over all 1,411 users because
   its truth set covered all of them; v2's truth set covers the 1,381
   scored users, and the estimand population follows the truth set.
4. Estimators, from observed cells only: Naive, IPS, SNIPS, and DR,
   with the same formulas as v1 (the DR imputation model is fitted on
   the v2 training-pool events only).
5. Two propensity conditions: true p_i, and p_i estimated from the
   draw's observed exposure counts (count / 1,411, clipped to
   [0.01, 1]), as in v1.
6. Reports per gamma and condition: per-ranker bias and RMSE against
   truth across seeds, and Kendall tau_b between the estimated and
   true ranker orders per seed.

## 8. Hypotheses and frozen verdict rules

Verdict vocabulary (v2): CONFIRMED means the frozen rule's condition
holds on valid evidence; REFUTED means it does not; INCONCLUSIVE
means the evidence cannot decide (a gate stop, a missing table, an
unmet slice-size condition). v1 used SUPPORTED / NOT SUPPORTED /
FAILS for the same rules; the mapping for the replication gate
(section 9) is SUPPORTED = CONFIRMED, NOT SUPPORTED = REFUTED,
FAILS = REFUTED.

All bootstrap intervals: per-user paired bootstrap, 2,000 resamples,
95 percent percentile intervals, unless a rule says the pairing is
over seeds. Holm correction is applied across the six pairwise ranker
comparisons within each arm's pairwise table; any "better" claim in
the results narrative requires the Holm-adjusted interval to exclude
0. The rules below are v1's, verbatim in substance; the only changed
lines are listed in section 10.

- H1. Popularity is a strong baseline. CONFIRMED if mean exact
  NDCG@10 of Popularity exceeds that of Random on both KuaiRec arms
  (Arm T test set and Arm F truth set), Random averaged over its 5
  seeds. Otherwise REFUTED.
- H1a. ItemKNN beats Popularity. CONFIRMED if the one-sided 95
  percent bootstrap lower bound of the paired per-user NDCG@10
  difference (ItemKNN - Popularity) on Arm T (KuaiRec) exceeds 0.
- H1b. EASE beats ItemKNN on the temporal split. CONFIRMED if the
  one-sided 95 percent bootstrap lower bound of (EASE - ItemKNN) on
  Arm T (KuaiRec) exceeds 0.
- H2a. Sampled metrics overstate exact metrics, rank order unchanged.
  CONFIRMED if, on Arm F, seed-averaged sampled NDCG@10 exceeds exact
  NDCG@10 for all five rankers and the ranker order by sampled equals
  the order by exact. Reported for the other arms as replication.
- H2b. The correction reduces measurement error. CONFIRMED if, on Arm
  F, (i) the mean absolute error to exact NDCG@10 across rankers and
  seeds is lower for corrected than for sampled, and (ii) at least
  one ranker pair is flipped by sampled NDCG in at least 8 of 10
  seeds while the same pair is flipped by corrected NDCG in at most
  2 of 10 seeds. Otherwise REFUTED.
- H3a. The random split overstates the temporal split. CONFIRMED if,
  on KuaiRec big, the seed-mean exact NDCG@10 under the random split
  exceeds the temporal-split NDCG@10 for at least 3 of the 4
  non-random rankers. The MovieLens-1M count is reported as
  replication.
- H3b. The leakage audit catches a planted leak. CONFIRMED if the
  audit passes the clean Arm T split (zero pair overlap, time order
  holds) and flags a deliberately planted leak (a test pair copied
  into train with a post-test timestamp). Otherwise REFUTED.
- H4a. With true propensities, IPS and SNIPS beat naive. SUBSTITUTED
  RULE (carried from v1 DECISIONS.md D2: the design pack's rule text
  is truncated in both of its extractions; this rule is the pack's
  upgrade 5 acceptance criterion in H4b's frozen form). CONFIRMED if,
  at gamma = 1.0 with true propensities, the mean Kendall tau_b
  against truth across the 10 exposure seeds is higher for IPS than
  for naive and higher for SNIPS than for naive, with the paired
  (over seeds) bootstrap 95 percent interval of each tau difference
  excluding 0.
- H4b. With estimated propensities, SNIPS still recovers ranker order
  better than naive. CONFIRMED if mean Kendall tau_b(SNIPS, truth) >
  tau_b(naive, truth) across the 10 exposure seeds at gamma = 1.0
  with estimated propensities, paired bootstrap 95 percent interval
  excluding 0. Otherwise REFUTED.
- H5. The naive evaluator disagrees with the truth inside popularity
  slices. CONFIRMED if, on Arm F, the ranker ordering by seed-averaged
  sampled NDCG differs from the ordering by exact NDCG in at least
  one item-popularity slice (head: top 1 percent of items by pool
  exposure; torso: next 9 percent; tail: the rest). Otherwise
  REFUTED.
- H6. Recommendations are head-heavy. CONFIRMED if, pooling all five
  rankers' Arm F top-10 lists over all 1,411 users, the share of
  recommendations going to top-decile items (by pool exposure) is
  significantly larger than 10 percent (one-sided binomial z test,
  alpha 0.05). The top-1-percent share is reported alongside.
- H7. The ItemKNN margin shrinks in the cold-start slice. Slice: test
  positives whose item has fewer than 5 training-window exposures
  (item-cold); user-cold (fewer than 5 training events) reported
  alongside. CONFIRMED if the ItemKNN - Popularity NDCG@10 gap on the
  slice is smaller than the gap on the full test set, on KuaiRec Arm
  T if the slice holds at least 30 evaluated users, else on
  MovieLens-1M. Otherwise REFUTED, reported as a failure of the
  expectation.
- S1 (supplementary, carried from v1). At gamma = 1.0 with true
  propensities, the naive estimator's bias (estimate - truth) for
  Popularity exceeds its bias for EASE, paired over exposure seeds,
  bootstrap 95 percent interval excluding 0.

## 9. Replication gate

v2's temporal arm must reproduce v1's five standing verdicts: H1a
CONFIRMED, H1b CONFIRMED, H3a CONFIRMED, H3b CONFIRMED, H7
CONFIRMED (v1 wording: SUPPORTED). The pipeline writes a replication
table (results/replication.json) comparing, per hypothesis, v1's
verdict and decisive numbers against v2's. If any of the five does
not reproduce, the difference is reported in RESULTS.md as a finding
with its cause investigated, not smoothed over. The Arm T code path
is imported from v1 unchanged precisely so that this gate is
meaningful; the numbers are expected to agree to float precision,
and any disagreement beyond rounding is itself reportable.

## 10. Every change from v1, named and justified

No numeric verdict line from v1 changes. The changes are:

1. Arm F construction (the fix). v1: train on big-matrix events
   restricted to small-matrix items (v1 D5). v2: Arm T's temporal
   discipline inside the small matrix (section 4). Justification:
   v1's pool is measured to contain zero events by the scored users;
   the arm it defines cannot score a learned ranker. The corrected
   construction is the one the brief for this pass specifies and the
   one v1's own audit disposition pointed at.
2. The coverage gate (section 5) is new. v1 had no such check; its
   absence is the recorded cause of the void arm.
3. Arm F truth population. v1's truth and estimand populations were
   all 1,411 users over the full small matrix. v2's are the 1,381
   scored users over the held-out test window, because the truth set
   is now a window. H6's pooled lists still cover all 1,411 users, as
   in v1, because H6 makes no use of truth labels.
4. NaN-timestamp rows (181,992) are excluded from Arm F. v1 used
   their labels because its Arm F had no windows; a temporal arm
   cannot place them. Counted in section 3 before any run.
5. Off-policy propensities are computed from the v2 training pool
   (within-small train window) instead of v1's big-matrix pool, and
   the DR imputation model is fitted on v2 pool events. The target
   mean propensity (0.163), gammas, seeds, and estimator formulas are
   unchanged, so the simulation regime is comparable.
6. Verdict vocabulary: CONFIRMED / REFUTED / INCONCLUSIVE replaces
   SUPPORTED / NOT SUPPORTED / FAILS. The rule conditions are
   identical; the mapping is frozen in section 8 so the replication
   gate is well defined.
7. Any line that referenced Arm F evidence in v1 (H1's second arm,
   H2a, H2b, H4a, H4b, H5, H6, S1) is re-based on the corrected arm.
   Changing an evidence base after seeing the earlier numbers is
   legitimate here for exactly one reason, stated as required: the
   earlier Arm F numbers for the learned rankers are void (they are
   artifacts of empty histories, verified at the bundle level), so
   they carry no information about the hypotheses. No Arm T number
   from v1 is re-based; Arm T is replicated, not revised.

## 11. Fault and pitfall checks

A1 to A4 data audits (section 3) are recorded in
results/data_audit.json. Coverage gate (section 5) on every arm,
with its disjoint-fixture unit test. Leakage audit on the real split
plus the planted-leak check (H3b). Estimator unit test: on a
synthetic matrix with known truth, mean IPS over 1,000 draws lies
within 3 standard errors of truth (in tests/, carried from v1).
Sampling unit test: simulated sampled ranks match the theoretical
expectation E[r_tilde] = 1 + m (r - 1) / (n - 1) within 3 standard
errors (in tests/, carried from v1). Popularity-bias report and
cold-start slice as in sections 6 and 8. The study never uses
user_features_raw.csv.

## 12. Stop rules

S1 Licence. If the Zenodo record licence were neither CC BY-SA 4.0
nor compatible, stop and re-pick the dataset. (Measured: CC BY 4.0
on the record, CC BY-SA 4.0 inside the archive; compatible. See
DECISIONS.md D2.)
S2 Audit. If counts differed from the registration audit by more
than 0.1 percent, stop and investigate before any modelling.
S3 Leakage audit failure: stop and fix the split before any result
is reported.
S4 Sampled-metric flip: report the flip and the correction; never
report the sampled metric alone.
S5 Coverage gate: a gate stop ends the run; the coverage table is
the output (section 5).
S6 Execution: the study runs as one controlled pass on the machine
of record. Stages persist their tables and per-ranker bundles as
they finish, so an interruption loses minutes, not the study; a
resumed stage reuses only bundles whose recorded configuration
matches the frozen one, and the pass log says which stages resumed.
This rule exists because v1's first execution fragmented across nine
runs on a shared machine and its stitching hid the Arm F defect's
origin; v2's pass is single and dedicated.

## 13. Overclaim guard

Never claim from this work: generalisation beyond KuaiRec, short
video, or 2020 data; that H4 shows IPS, SNIPS, or DR work on real
logged systems (the exposure is semi-synthetic); that watch_ratio
above 2.0 measures what users like or want; that EASE or iALS is
better than a baseline without the Holm-adjusted interval excluding
0; state of the art, production grade, or industry recommender
experience; measured societal impact (the impact statement is
intended only); any novelty claim before a literature search is
done; any number without its baseline row, n, seeds, 95 percent
interval, dataset version, and command; that the corrected Arm F is
the fully observed arm of v1's original intent in the strong sense
(its truth is a fully observed held-out window, not the whole
matrix); that a reproduced v1 verdict is new evidence (replication
confirms the pipeline, it does not double the evidence).

## Amendments

None at registration.
