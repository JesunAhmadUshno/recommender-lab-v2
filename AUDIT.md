# Audit: recommender-lab-v2

Self-audit by the author, written after the run of record and before this
repository was shared. It follows the house standard: execution, claim,
validity, security and licence, and cost audits, then a grade out of 10,000
with itemized deductions. Anything this audit cannot support is named as a
deduction rather than smoothed over.

## 1. Execution audit

- The pre-registration was the first content commit (`7a12bb6`, SHA-256
  `55fcc3e183293fa97705ab51b8923ac80814c881f5e9706aef49c2c47c917107`) and
  the run's own track record verifies the pin from the git history
  (`results/verdicts.json`, `track.preregistration_commit`). A layout test
  asserts the hash byte-for-byte and now passes with the results in place.
- The study ran as one controlled pass on the named machine of record
  (BlackBoxV3), 2026-11-08T18:07:35-05:00 to 18:38:26-05:00, exit code 0,
  with per-seed and per-stage checkpoints written to disk throughout
  (`logs/pass.log`). No stage was rerun, patched mid-flight or restarted.
- The coverage gate ran before scoring in every arm and cell: 14 gate
  records in `results/coverage.json`, all passed, including Arm F at its
  strict 0.95 user minimum. The all-zero score-matrix backstop recorded
  nonzero-row fractions of 1.0000 for ItemKNN, EASE and iALS on Arm F.
- The gate is also covered by tests: `tests/test_coverage.py` reproduces
  v1's defect construction on a disjoint fixture and requires the gate to
  stop the run, plus a passing-gate control and both backstop tests. Full
  suite: 32 tests, all passing locally and on BlackBoxV3.
- Verdicts were computed by the harness verdict stage from the measured
  tables. No verdict was assigned or edited by hand.

## 2. Claim audit

- Every number in README.md and RESULTS.md was traced to
  `results/verdicts.json`, `results/replication.json`,
  `results/coverage.json`, `results/arm_f_meta.json` or a named CSV in
  `results/tables/`. The headline table is regenerated from the result
  files by `scripts/reproduce_headline.py`, not typed.
- Jev claim_check (house gate, `claim_check`): model `jev-1.13.0`, question
  `supported`, noul 0.42, route REVIEW (band 0.10 to 0.85), usage 2,129
  input tokens and 20 output tokens. Record: `results/jev-claim-check.json`.
  On the REVIEW route the author hand-verified every figure in the checked
  claim against the four result files above before publication; all
  figures matched, including the replication maximum absolute difference
  (3.38e-07, from H1b's lower bound) and the Arm F ordering and coverage
  values.
- This claim_check also closes the check v1 still owed on its temporal
  headlines: v2's temporal arm reproduces H1a, H1b, H3a, H3b and H7 with
  matching statistics, and the checked claim covers those shared numbers
  explicitly (PREREGISTRATION.md section 9).

## 3. Validity audit

- v1's defect is corrected by construction and verified by measurement:
  the scored users of Arm F have user coverage 1.0000 (v1's construction,
  measured by the same code in the pre-run audit: 0.0000), and every
  learned ranker produced nonzero scores for every scored user.
- The replication gate is the strongest validity evidence in the repo: an
  independent re-execution of v1's temporal arm on the same machine class
  reproduced all five surviving verdicts with differences at the 1e-07
  level, attributable to the documented NaN-robust quantile path
  (DECISIONS.md D14).
- The two frozen sensitivity runs were executed and are reported in
  RESULTS.md section 8, including the two findings that cut against the
  headline: Random exceeds all learned rankers under the strict label, and
  the learned rankers collapse under the early pool. Neither was hidden or
  softened.
- Known estimand caveat (pre-registered): the exposure arm's propensities
  come from the within-small train pool over the full 1,411-user grid, and
  its estimand is averaged over the 1,381 scored users. H4a, H4b and S1 are
  read under that caveat.

## 4. Security and licence audit

- No dataset files are committed. KuaiRec is used under CC BY-SA 4.0 and
  its authors' request not to redistribute the raw data is honoured;
  MovieLens-1M is used under the GroupLens research terms. Checksums of
  the exact files used are frozen in PREREGISTRATION.md section 10.
- Code is Apache-2.0 (LICENSE, copied from v1). v1 is read-only input to
  this study and was not modified; its defect report stays public and is
  linked from this repo's README.
- No new accounts or credentials were created. The only credentialed call
  was one Jev evaluation through the registered TypeSafe connector; the key
  was never written to a file, log or this repo.
- The repository contains no personal data beyond the public dataset
  artefacts described above.

## 5. Cost audit

- Compute: one 31-minute pass on BlackBoxV3 (the author's own fleet) plus
  short audit and test runs. No cloud spend.
- Jev: one evaluation, 2,129 input tokens, about USD 0.00009 at the
  documented rate, drawn on previously purchased credit. No other paid
  service was used.

## 6. Grade

**8,800 of 10,000.**

Itemized deductions (1,200 total):

| Points | Item |
| ---: | --- |
| 250 | Arm F's truth is a held-out temporal window of the small matrix, not the fully observed matrix v1 intended. Exhaustive scoring is real, but over that window (pre-registered, section 4 of PREREGISTRATION.md). |
| 200 | 181,992 NaN-timestamp rows (3.9 percent of the small matrix) are excluded from Arm F's windows; the arm's population is the timestamped subset. |
| 150 | The exposure estimand is averaged over the 1,381 scored users rather than the full 1,411-row grid (estimator inputs are row-subsets). H4a, H4b and S1 carry the caveat. |
| 150 | Single machine, single pass. Arm T is covered by the cross-study replication against v1; Arm F itself has no second-pass replication. |
| 100 | Pool sensitivity: the Arm F headline is carried by the later train window and collapses under the early pool. Reported openly; the headline is a property of the frozen window. |
| 100 | Label sensitivity: under the strict watch_ratio above 1.0 label, Random exceeds every learned ranker. The primary label is a defensible but contestable modelling choice. |
| 100 | Review is recomputation, hand verification and one Jev check, not an independent second reader; the Jev check returned REVIEW (0.42) and was resolved by hand verification, not by a clear gate pass. |
| 50 | SASRec remains unimplemented behind its documented engagement gate, as in v1; the ranker set is five, not six. |
| 50 | The run of record executed from a tar-transferred working copy on BlackBoxV3 (the machine has no system git) rather than a fresh clone of the remote. The pre-registration pin was verified inside the copy, but the check is weaker than a clean clone. |
| 50 | Item coverage is 0.5572 on Arm T KuaiRec and 0.6631 on Arm F: a large share of scored items have no training presence. Inherent to temporal evaluation and disclosed by the gate, but exact metrics are computed over a partially cold item set. |

What was not deducted, and why: the replication outcome (near-exact), the
gate behaviour (fired correctly in tests, passed with margin in the run),
the completeness of the verdict set (14 of 14 hypotheses decided, none left
uncomputed), and the reporting of both adverse sensitivity findings. Those
are the properties this rebuild existed to demonstrate.
