# Recommender Lab v2

Author: Jesun Ahmad Ushno.

**Question.** On a dataset where the true answer is known, how far do
standard offline evaluation shortcuts (sampled metrics, random splits,
naive evaluation on biased logs) move the measured score and the
ranking of recommender models, and how much of that error do the
standard corrections remove? And, new in this pass: does the answer
survive an evaluation arm that is first checked to be measuring
anything at all?

**Headline.** v1's temporal results replicate almost exactly (all five
standing verdicts, headline statistics within 3.38e-07), and the
rebuilt full-matrix arm tells a different story than v1's void numbers
did. On the corrected arm (coverage gate passed: user coverage 1.0000,
item coverage 0.6631), exact NDCG@10 over 1,381 users orders iALS
(0.01353) above EASE (0.01120) above Random (0.00654) above ItemKNN
(0.00079) above Popularity (0.00000). Sampled metrics put Random first;
the bias correction restores the true ordering and cuts the error from
0.0943 to 0.0262 (H2b CONFIRMED). The off-policy estimators that
appeared to win on v1's void arm do not beat the naive estimator on
the real one (H4a, H4b REFUTED). Full evidence: RESULTS.md. This
repository is the corrected second pass of
[recommender-evaluation-lab](https://github.com/JesunAhmadUshno/recommender-evaluation-lab)
(v1): in v1, the fully observed arm trained its rankers on a pool
that contained zero events by the users being scored, so three of its
five rankers were scored on empty histories and seven of its thirteen
verdicts are void. v2 rebuilds that arm inside the fully observed
matrix, puts a mandatory coverage gate in front of every score, and
re-runs the whole study, replicating v1's five standing verdicts as a
gate on itself. The design is frozen in PREREGISTRATION.md (first
content commit of this repo) before any metric is computed.

**Field.** ML and AI research. **Job family.** ML evaluation and
experimentation.

**Verdict.** Evaluated. The run of record completed on BlackBoxV3 on
2026-11-08 under the locked pre-registration: 14 of 14 hypotheses
decided by the harness from the measured tables, the coverage gate
passed on every arm, and v1's five standing verdicts replicated.
Grade: 8,800 of 10,000 (AUDIT.md, deductions itemized).

## Verdicts at a glance

| Hypothesis | Verdict |
| --- | --- |
| H1: the popularity baseline beats Random temporally | REFUTED |
| H1a: ItemKNN beats Popularity per user (Arm T) | CONFIRMED |
| H1b: EASE beats ItemKNN per user (Arm T) | CONFIRMED |
| H2a: sampled metrics fully reverse the Arm F ordering | REFUTED |
| H2b: the correction repairs sampled metrics on Arm F | CONFIRMED |
| H3a: random splits inflate NDCG | CONFIRMED |
| H3b: the leakage audit catches a planted leak | CONFIRMED |
| H4a: IPS and SNIPS beat the naive estimator | REFUTED |
| H4b: clipped IPS beats the naive estimator | REFUTED |
| S1: sampled-metric bias is ranker-dependent | CONFIRMED |
| H5: sampled metrics reverse conclusions inside popularity slices | CONFIRMED |
| H6: recommendations concentrate on high-exposure items | CONFIRMED |
| H7: the ItemKNN edge is carried by warm items | CONFIRMED |

## Why v2 exists

v1's defect, in its own record: KuaiRec's small matrix is
pair-disjoint from its big matrix. v1's frozen Arm F pool (big-matrix
events restricted to small-matrix items) therefore held no event by
any of the 1,411 scored users. ItemKNN, EASE, and iALS each received
an empty history row and returned the all-zero score matrix; their
Arm F metrics are artifacts of sorting zeros, and the verdicts built
on them (H2a, H4a, H4b, H5, H6, S1 as computed, and H2b's flip test)
are not usable as findings. v1's RESULTS.md carries the full validity
notice and its AUDIT.md the validity audit; both stay public. What
stands in v1, and what this pass must replicate: H1a, H1b, H3a, H3b,
and H7, all on the temporal arm, the random-split arm, and the
leakage audit.

The correction has three parts. The arm is rebuilt: Arm F v2 runs the
temporal split inside the small matrix itself, so training histories
and truth come from the same matrix and every candidate's label is
still known. The gate is added: before any fitting, every arm must
prove that its scored users actually have training histories, at
frozen thresholds, and an all-zero score matrix after scoring fails
the run outright. And the replication is gated: v2's temporal arm
must reproduce v1's five standing verdicts, or the difference is
reported as a finding.

Recommender systems rank what millions of people see next. An offline
evaluation that silently scores empty histories manufactures
confidence in whatever it claims to measure, and the field's standard
shortcuts (sampled metrics, random splits, naive scoring on biased
logs) are trusted exactly as far as evaluations like this one check
them. The societal stake is plain: a lab that cannot tell when its
own benchmark is empty will ship the winner of a broken measurement
to everyone downstream.

## Lifecycle

- Question: DONE 2026-11-08
- Proposal: DONE 2026-11-08 (PREREGISTRATION.md, commit 7a12bb6)
- Baseline: DONE 2026-11-08 (v1's record, at commit a9c0464)
- Build: DONE 2026-11-08
- Evaluate: DONE 2026-11-08 (run of record on BlackBoxV3, exit 0)
- Report: DONE 2026-11-08 (RESULTS.md)
- Review: DONE 2026-11-08 (AUDIT.md; Jev claim_check REVIEW, hand-verified)
- Release: DONE 2026-11-08 (results commit)

## Layout

| Path | Role | What it holds |
| --- | --- | --- |
| PREREGISTRATION.md | proposal | The locked design, hypotheses, verdict rules, and coverage gate (first content commit; a test fails if one byte changes) |
| RESULTS.md | results | The run of record's verdict tables and headline numbers (written with the results commit) |
| AUDIT.md | report, evidence | Execution, claim, validity, security, cost, and licence audits of the run, with the grade |
| DECISIONS.md | decisions | Every judgement call, D1 onward |
| data/README.md, DATA_LICENSE.md | data | Dataset sources, versions, checksums, licences; raw data is never committed |
| src/ | code | Split, rankers, metrics, correction, audit, stats, and the coverage gate |
| experiments/ | code | The data audit and the full pipeline (run_all.sh, run_all.py) |
| results/ | results | Derived statistics only, from the run of record; headline_table.md regenerated by scripts/reproduce_headline.py |
| scripts/ | code | The headline-table generator |
| tests/ | tests | Unit tests, including the coverage gate's disjoint-fixture test |

## Reproduce

Checks, anywhere (no dataset needed):

```text
pip install -r requirements.txt -e . pytest
python -m pytest tests -q
python scripts/reproduce_headline.py --check
```

The full study (CPU only, 31 minutes on the machine of record for the
run of record): fetch the two datasets per data/README.md (checksums
there and in PREREGISTRATION.md), then run, inside a git clone:

```text
bash experiments/run_all.sh
```

## Licences

Code: Apache-2.0 (LICENSE). Derived data under results/: CC BY-SA
4.0 (DATA_LICENSE.md), inherited from KuaiRec. MovieLens-1M figures
are statistics only, under the GroupLens terms.
