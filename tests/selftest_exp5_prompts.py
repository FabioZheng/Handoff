"""Offline proof that the new prompt families are Experiment 5 plus one clause.

Four checks, no API key and no model:

1. Provenance. ``exp5_prompts.user_prompt`` reproduces Experiment 5's own
   ``compression_user_prompt`` byte for byte, for both arms. If anyone edits
   the instruction, the system prompt or the question line in either place,
   this fails.
2. ``exp5_uncapped`` is a minimal pair: conditioned minus the question line is
   byte-identical to generic, and generic does not contain the question.
3. ``exp5_cap_only`` is a minimal pair under the same test, at every budget.
4. ``exp5_cap_only`` is ``exp5_uncapped`` plus the identical cap clause in both
   arms, and the clause carries no floor, target or fill wording.

Writes the three diffs item 20 of the design requires to
``results/exp5_prompt_family/PROMPT_DIFFS.txt``.

    python tests/selftest_exp5_prompts.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / d) for d in ('', 'analysis', 'latent', 'builders')]

import exp5_prompts as ep  # noqa: E402
from run_summary_generalization import compression_user_prompt  # noqa: E402

SOURCE = ("Marie Curie conducted pioneering research on radioactivity in Paris. "
          "She was awarded the Nobel Prize in Physics in 1903 and in Chemistry in 1911. "
          "Her husband Pierre Curie shared the 1903 prize with her and with Henri Becquerel.")
QUESTION_A = "In which year did Marie Curie win the Nobel Prize in Chemistry?"
QUESTION_B = "Who shared the 1903 prize?"
BUDGETS = (32, 64, 96, 128, 192, 256)

# Wording that would turn a maximum into a target, a floor or an invitation to
# expand. None of it may appear in the clause this module adds.
FORBIDDEN = ("at least", "minimum", "aim for", "approximately", "about ",
             "target", "use the space", "use the available", "fill",
             "as close", "neither much", "fewer words", "stop when")


def _pair() -> dict:
    return {"passages": [{"text": SOURCE}], "question_A": QUESTION_A,
            "question_B": QUESTION_B}


def check_provenance() -> None:
    pair = _pair()
    for mode, question in (("conditioned", QUESTION_A), ("generic", None)):
        original = compression_user_prompt(pair, None, mode, 1, None)
        rebuilt = ep.user_prompt(SOURCE, question, None)
        if original != rebuilt:
            diff = "\n".join(difflib.unified_diff(
                original.splitlines(), rebuilt.splitlines(),
                "experiment_5", "exp5_prompts", lineterm=""))
            raise AssertionError(f"{mode} prompt drifted from Experiment 5:\n{diff}")
    print("1. provenance: both arms reproduce Experiment 5 byte for byte")


def check_minimal_pair(cap: int | None, mechanism: str = "generation") -> None:
    ep.assert_minimal_pair(SOURCE, QUESTION_A, cap, mechanism)
    conditioned = ep.user_prompt(SOURCE, QUESTION_A, cap, mechanism)
    generic = ep.user_prompt(SOURCE, None, cap, mechanism)
    removed = conditioned.replace(ep.question_marker(QUESTION_A), "")
    assert removed == generic
    assert QUESTION_A not in generic and QUESTION_B not in generic


def check_cap_clause() -> None:
    for cap in BUDGETS:
        clause = ep.cap_clause(cap)
        lowered = clause.lower()
        for bad in FORBIDDEN:
            if bad in lowered:
                raise AssertionError(f"cap clause contains {bad!r}: {clause!r}")
        if str(cap) not in clause or "most" not in lowered:
            raise AssertionError(f"cap clause is not a maximum: {clause!r}")
        for mechanism in ep.MECHANISMS:
            for question in (QUESTION_A, None):
                uncapped = ep.user_prompt(SOURCE, question, None, mechanism)
                capped = ep.user_prompt(SOURCE, question, cap, mechanism)
                if capped != uncapped + clause:
                    raise AssertionError(
                        f"{mechanism}: capped prompt is not uncapped + clause")
    print(f"4. cap clause: maximum only, identical in both arms and both "
          f"mechanisms, {len(BUDGETS)} budgets")


def check_selection_task() -> None:
    """The selector instruction is constant across arms and carries no fill cue."""
    lowered = ep.SELECTION_TASK.lower()
    for bad in FORBIDDEN:
        if bad in lowered:
            raise AssertionError(f"selection task contains {bad!r}")
    if QUESTION_A in ep.SELECTION_TASK or "question" in lowered:
        raise AssertionError("selection task refers to a question on its own")
    print("5. selection task: no fill wording, no question reference")


def _diff(label: str, left_name: str, left: str, right_name: str, right: str) -> str:
    lines = list(difflib.unified_diff(left.splitlines(), right.splitlines(),
                                      left_name, right_name, lineterm=""))
    body = "\n".join(lines) if lines else "(identical)"
    return f"===== {label} =====\n{body}\n"


def write_diffs() -> Path:
    cap = BUDGETS[2]
    out = ROOT / "results" / "exp5_prompt_family" / "PROMPT_DIFFS.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        "Only the Experiment 5 question line may differ within a family.",
        "Only the identical maximum-word clause may differ between families.",
        f"base prompt sha256   {ep.base_prompt_hash()}",
        f"uncapped template    {ep.template_hash(None)}",
        f"cap-only template    {ep.template_hash(cap)}",
        "",
        _diff("exp5_uncapped: generic vs conditioned",
              "generic", ep.user_prompt(SOURCE, None, None),
              "conditioned", ep.user_prompt(SOURCE, QUESTION_A, None)),
        _diff(f"exp5_cap_only (cap={cap}): generic vs conditioned",
              "generic", ep.user_prompt(SOURCE, None, cap),
              "conditioned", ep.user_prompt(SOURCE, QUESTION_A, cap)),
        _diff(f"conditioned: exp5_uncapped vs exp5_cap_only (cap={cap})",
              "uncapped", ep.user_prompt(SOURCE, QUESTION_A, None),
              "cap_only", ep.user_prompt(SOURCE, QUESTION_A, cap)),
        _diff(f"generic: exp5_uncapped vs exp5_cap_only (cap={cap})",
              "uncapped", ep.user_prompt(SOURCE, None, None),
              "cap_only", ep.user_prompt(SOURCE, None, cap)),
        _diff(f"selection, exp5_cap_only (cap={cap}): generic vs conditioned",
              "generic", ep.user_prompt(SOURCE, None, cap, "selection"),
              "conditioned", ep.user_prompt(SOURCE, QUESTION_A, cap, "selection")),
        "===== rendered conditioned prompt, cap_only =====",
        f"[system]\n{ep.system_prompt()}\n\n[user]\n{ep.user_prompt(SOURCE, QUESTION_A, cap)}\n",
        "===== rendered generic prompt, cap_only =====",
        f"[system]\n{ep.system_prompt()}\n\n[user]\n{ep.user_prompt(SOURCE, None, cap)}\n",
    ]
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


def main() -> int:
    check_provenance()
    for mechanism in ep.MECHANISMS:
        check_minimal_pair(None, mechanism)
    print("2. exp5_uncapped: minimal pair holds for both mechanisms")
    for mechanism in ep.MECHANISMS:
        for cap in BUDGETS:
            check_minimal_pair(cap, mechanism)
    print(f"3. exp5_cap_only: minimal pair holds at all {len(BUDGETS)} budgets, "
          "both mechanisms")
    check_cap_clause()
    check_selection_task()
    path = write_diffs()
    print(f"\nbase prompt sha256 {ep.base_prompt_hash()[:16]}")
    print(f"uncapped template  {ep.template_hash(None)[:16]}")
    print(f"cap-only template  {ep.template_hash(BUDGETS[2])[:16]}")
    print(f"diffs written to   {path.relative_to(ROOT)}")
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
