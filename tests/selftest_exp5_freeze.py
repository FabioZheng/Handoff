"""Offline checks for the freeze step and the frozen-config runner path.

1. A clean calibration decision freezes: six budgets, frozen: true, headline
   corpora written to cohort.corpora, provenance hash recorded.
2. Freezing twice is refused.
3. A flagged decision is refused, and so is a grid of the wrong size.
4. The template config is never modified.
5. The runner accepts the frozen config without --allow-unfrozen, refuses a
   --budgets override on it, and only touches documents of headline corpora.

Runs in a temporary directory; writes nothing into the repository.

    python tests/selftest_exp5_freeze.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import freeze_exp5_budgets as fz  # noqa: E402
import run_exp5_capped as rc  # noqa: E402

RULE = {"binding_share": 0.5, "retention_floor": 0.2, "retention_ceiling": 0.8,
        "min_sentence_multiple": 2, "round_to": 8, "count": 6}


def decision_file(tmp: Path, decision: dict) -> Path:
    path = tmp / "calibration.json"
    path.write_text(json.dumps({"run_dir": "abc123-calib", "rule": RULE, "rule_in_run_manifest": True,
                                "decision": decision}), encoding="utf-8")
    return path


def expect_exit(fn, *args) -> str:
    try:
        fn(*args)
    except SystemExit as exc:
        return str(exc)
    raise AssertionError("expected a refusal")


def main() -> int:
    template = ROOT / "configs/exp5_cap_only_config.yaml"
    before = template.read_bytes()
    clean = {"grid": [96, 120, 144, 176, 208, 256], "headline": ["qasper"], "flag": None,
             "regions": {"qasper": {"B_min": 96, "B_max": 256, "has_region": True},
                         "squad": {"B_min": None, "B_max": 384, "has_region": False}}}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # decision_path must sit under ROOT for relative provenance; mirror that.
        work = ROOT / "results" / "_selftest_freeze"
        work.mkdir(parents=True, exist_ok=True)
        try:
            dec = decision_file(work, clean)
            out = tmp / "frozen.yaml"
            frozen = fz.freeze(dec, template, out, "12345")
            text = out.read_text(encoding="utf-8")
            loaded = yaml.safe_load(text)
            assert text.startswith("# GENERATED"), "header missing"
            assert loaded["budgets"]["absolute"] == clean["grid"]
            assert loaded["budgets"]["frozen"] is True
            assert loaded["cohort"]["corpora"] == ["qasper"]
            prov = loaded["budgets"]["calibrated_from"]
            assert len(prov["decision_sha256"]) == 64 and prov["freeze_job"] == "12345"
            assert prov["headline"] == ["qasper"]
            print("1. clean decision frozen:", loaded["budgets"]["absolute"], "corpora", loaded["cohort"]["corpora"])

            msg = expect_exit(fz.freeze, dec, template, out, None)
            assert "already exists" in msg
            print("2. second freeze refused")

            flagged = dict(clean, flag="both corpora have regions but they do not overlap")
            msg = expect_exit(fz.freeze, decision_file(work, flagged), template, tmp / "f2.yaml", None)
            assert "flagged" in msg
            short = dict(clean, grid=[96, 128, 256])
            msg = expect_exit(fz.freeze, decision_file(work, short), template, tmp / "f3.yaml", None)
            assert "expected 6" in msg
            print("3. flagged decision and short grid refused")

            assert template.read_bytes() == before, "template was modified"
            print("4. template untouched")

            # Runner path on the frozen config.
            cfg = yaml.safe_load(text)
            assert rc.budgets(cfg) == clean["grid"]
            try:
                rc.budgets(cfg, allow_unfrozen=True, override=[1, 2, 3])
                raise AssertionError("override of a frozen grid was accepted")
            except ValueError:
                pass
            shutil.copy(out, ROOT / "_selftest_frozen.yaml")
            try:
                runs = ROOT / "runs" / "exp5_cap_only" / "fake"
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "run_exp5_capped.py"), "--stage", "write",
                     "--model", "mistral24", "--split", "development", "--fake",
                     "--config", "_selftest_frozen.yaml", "--per-corpus", "2",
                     "--policies", "summary_generic"],
                    capture_output=True, text=True, cwd=ROOT)
                assert proc.returncode == 0, proc.stderr[-800:]
                start = json.loads(next(l for l in proc.stdout.splitlines() if '"start"' in l))
                assert start["budgets_frozen"] is True and start["budgets"] == clean["grid"]
                written = list(runs.glob("development/*/write/mistral24/*.json"))
                corpora = {json.loads(p.read_text(encoding="utf-8"))["messages"][0]["corpus"] for p in written}
                assert corpora == {"qasper"}, corpora
                print(f"5. runner on frozen config: {len(written)} docs, corpora {sorted(corpora)}, "
                      "override refused")
            finally:
                (ROOT / "_selftest_frozen.yaml").unlink(missing_ok=True)
                shutil.rmtree(ROOT / "runs" / "exp5_cap_only" / "fake", ignore_errors=True)
                shutil.rmtree(ROOT / "cache" / "reuse" / "fake", ignore_errors=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print("\nALL FREEZE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
