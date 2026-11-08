"""Portfolio standard checks: the pre-registration lock holds and the
required files exist."""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SHA-256 of PREREGISTRATION.md as committed in 7a12bb6, the first
# content commit of this repo, before any code or metric.
LOCKED_SHA256 = "55fcc3e183293fa97705ab51b8923ac80814c881f5e9706aef49c2c47c917107"


def test_preregistration_is_unchanged_since_the_lock():
    with open(os.path.join(ROOT, "PREREGISTRATION.md"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == LOCKED_SHA256, \
            "PREREGISTRATION.md changed after its lock; add a dated amendment and a DECISIONS.md entry instead"


def test_required_files_exist():
    need = ["README.md", "PREREGISTRATION.md", "DECISIONS.md", "LICENSE",
            "DATA_LICENSE.md", "data/README.md",
            "scripts/reproduce_headline.py", "experiments/run_all.sh",
            "experiments/run_all.py", "src/coverage.py"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, "missing: " + ", ".join(missing)


def test_run_of_record_files_exist_once_committed():
    # Results files exist only after the run of record; before it, this
    # test asserts the results directory carries no partial claims.
    results = os.path.join(ROOT, "results")
    verdicts = os.path.join(results, "verdicts.json")
    if not os.path.exists(verdicts):
        return
    for p in ("data_audit.json", "coverage.json", "replication.json",
              "headline_table.md"):
        assert os.path.exists(os.path.join(results, p)), "missing results/" + p


def test_readme_opens_with_the_abstract():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**", "## Lifecycle"):
        assert field in text, "README.md is missing " + field
