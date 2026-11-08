# Decision log

Every judgement call this build had to make, recorded openly. Nothing
in this file is hidden from a reader of the results. Entries D2, D4,
D5, D6, D8, and D9 carry v1's decisions forward where the corrected
design still depends on them; the rest are new to v2.

## D1. Code is carried from v1; the Arm T path is byte-stable in behaviour
v2 imports v1's (recommender-evaluation-lab) loaders, rankers,
metrics, statistics, correction estimators, and audit helpers, and its
pipeline structure. The Arm T and random-arm code paths are unchanged
in behaviour, because the replication gate (PREREGISTRATION.md
section 9) is only meaningful if the same code produces v2's Arm T
numbers. The changes are exactly the ones the pre-registration names:
the corrected Arm F, the coverage gate, the verdict vocabulary, and
the implementation notes in D10 and D14. v1's repository is not
modified by this work.

## D2. KuaiRec licence: the archive's own licence text governs
Carried from v1 D3. Three licence statements exist: the licence file
inside KuaiRec.zip is CC BY-SA 4.0; the Zenodo record 18164998
metadata says CC BY 4.0; the GitHub README badge says CC BY-SA 4.0.
The build follows the strictest consistent reading: the dataset is
used under CC BY-SA 4.0 with attribution, raw files are never
committed, and derived data files shipped in results/ are covered by
DATA_LICENSE.md under CC BY-SA 4.0. Code is Apache-2.0. This is a
reading of the licences, not legal advice. MovieLens-1M is used under
the GroupLens research terms: no redistribution, only derived
statistics published.

## D3. Verdict vocabulary is CONFIRMED / REFUTED / INCONCLUSIVE
v1 reported SUPPORTED / NOT SUPPORTED / FAILS. v2 reports the house
vocabulary, with the rule conditions unchanged and the mapping frozen
in PREREGISTRATION.md section 8 (SUPPORTED = CONFIRMED, NOT SUPPORTED
= REFUTED, FAILS = REFUTED), so the replication gate compares like
with like. INCONCLUSIVE is reserved for evidence that cannot decide:
a gate stop or a missing input table, recorded with its reason in
results/verdicts.json.

## D4. SNIPS operationalisation
Carried from v1 D7. The design pack names SNIPS without a formula for
a DCG estimand. The build uses the standard self-normalised ratio
form at the population level: the sum of per-user IPS DCG estimates
divided by the sum of per-user importance weights, rescaled by the
sum of the top-10 discounts so the estimate sits on the DCG scale.
The formula is written out in the pre-registration.

## D5. SASRec is excluded from the measured ladder
Carried from v1 D4. The big matrix averages about 1,746 events per
user, and self-attention training on CPU cores cannot carry the
frozen tuning budget honestly. src/rankers.py contains a SASRec class
that raises NotImplementedError pointing here, so the exclusion is
visible in code and tested, and no SASRec number appears anywhere in
the results.

## D6. H4a's frozen rule is a recorded substitution
Carried from v1 D2. The design pack truncates H4a's verdict rule at
"SUPPORTED if" in both of its extractions; the rule text is absent
from both sources. The rule frozen in the pre-registration is the one
implied by the pack's upgrade 5 acceptance criterion (with true
propensities, the estimator recovers the true ranking),
operationalised in the same paired-bootstrap form as H4b. This is a
substitution, recorded here and in the pre-registration, not a silent
invention.

## D7. Arm F masking: training positives are masked, as on Arm T
v1 did not mask on Arm F (its D8), because its training pool was
external to the evaluated matrix, so no scored item could be a
training positive of the scored user. In v2 the pool and the truth
come from the same matrix, so the Arm T discipline applies unchanged:
scores mask training positives, sampled negatives exclude them, and
the truth set holds first-time positives only. The asymmetry v1 froze
is gone because its premise is gone.

## D8. MovieLens-1M is the second dataset, in a replication role
Carried from v1 D9. KuaiRec's small matrix remains the only
ground-truth arm. MovieLens-1M replicates the temporal-versus-random
and sampled-versus-exact comparisons on a sparse, widely used corpus,
and only derived statistics are published.

## D9. Sampled protocol: m = 100 negatives per user
Carried from v1 D6. The design pack says both "100 negatives per test
positive" and, in its worked example and correction formula, m = 100
negatives in total. The build follows the worked example: per user,
the candidate set is all truth positives plus 100 negatives sampled
uniformly. The correction formula is only defined under this reading.

## D10. Memory blocking and resumable bundles (implementation only)
Carried from v1 D10 to D12. The metric functions evaluate in row
blocks; Arm T stages fit and measure one ranker at a time so only one
full score matrix is alive; the EASE Gram matrix is accumulated in
column blocks and inverted in place through LAPACK; every stage
persists its tables and per-ranker bundles as it finishes, and a
resumed stage reuses only bundles whose recorded configuration
matches the frozen one. No hypothesis, verdict rule, seed, or
protocol element is affected; v1 verified the split path reproduces
the all-at-once path to below 1e-12 on MovieLens-1M.

## D11. Exposure mechanics: full-grid draws, scored-user estimand
The pre-registration (section 7) freezes exposure draws over the full
small-matrix grid and estimated propensities as count / 1,411, as in
v1, and separately freezes the estimand population as the scored
users. The implementation honours both literally: draws and
propensity counts are computed on the full 1,411-row grid; the
estimator inputs (top lists, truth, exposure, imputations) are
row-subset to the 1,381 scored users before estimator_dcgs is called,
so SNIPS's population ratio is also over the scored users. This is a
mechanics note, not a rule change.

## D12. Label sensitivity rebuilds its truth set under the same discipline
v1's label sensitivity reused the full small matrix at the 1.0
threshold, because its Arm F truth was the full matrix. v2's truth is
a temporal window, so the sensitivity truth is the first-time
positives at watch_ratio > 1.0 in the same test window, scored by the
same (2.0-trained, masked) score matrices. The sensitivity therefore
isolates the label threshold, which is its purpose.

## D13. Execution: one controlled pass on BlackBoxV3
The machine of record is BlackBoxV3 (Intel i5-8400, 6 cores, 7.1 GiB
RAM, Ubuntu 26.04), the machine v1's fresh pass used, named in the
pre-registration. The pass runs end to end as one detached execution
with stage logs and per-stage checkpoints on disk; stages persist as
they finish (D10), so an interruption loses minutes, not the study.
This answers v1's history directly: its first execution fragmented
across nine runs on a shared VM, and the stitching is part of why the
Arm F defect's origin stayed ambiguous until a clean pass isolated
it. No hypothesis, verdict rule, seed, split, or protocol element is
affected by the machine choice; timings in results/ are this
machine's and say so.

## D14. temporal_cutoffs uses nanquantile (implementation only)
src/split.py computes the temporal cut points with np.nanquantile
instead of np.quantile, so the small matrix's 181,992 empty-timestamp
rows cannot poison Arm F's cutoffs. On NaN-free data (both Arm T
datasets) nanquantile and quantile return identical values, and the
audit confirms the Arm T cut points equal v1's pinned values to the
digit. No protocol element changes.
