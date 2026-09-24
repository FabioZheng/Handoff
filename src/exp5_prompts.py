"""Experiment 5's writer prompt, reused verbatim, plus one optional cap clause.

Experiment 5 (``run_summary_generalization.py``) manipulates exactly one thing:
whether the writer is shown the question the next agent must answer. System
prompt, instruction, source presentation and separators are shared byte for
byte, and that module's own ``prompt_difference_selftest`` asserts it at
runtime. Nothing here re-words any of it.

That pair carries no length constraint, so on its own it cannot separate "the
query-aware writer allocated the channel differently" from "the query-aware
writer simply sent fewer words". This module therefore defines two prompt
families:

``exp5_uncapped``   Experiment 5 unchanged. Used only to measure what each arm
                    writes when nothing constrains it.
``exp5_cap_only``   Experiment 5 plus ONE clause, appended identically to both
                    arms: a hard maximum and nothing else.

The clause is a maximum, not a target. There is no floor, no fill ratio and no
instruction to use the available space, because a writer that stops early is
the phenomenon under measurement rather than an error to correct. The only
generic-vs-conditioned difference in either family remains Experiment 5's
question line.

Provenance is checked rather than claimed: ``selftest_exp5_prompts.py``
rebuilds Experiment 5's own ``compression_user_prompt`` output from this module
and requires byte equality, so the base prompt cannot drift from the original
without the selftest failing.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_chain import CHAIN_SYSTEM, INITIAL_INSTRUCTION  # noqa: E402

# Experiment 5 renders the source as ``[P1] ...`` blocks joined by a blank line
# (``run_summary_generalization.render_context``). A whole cohort document is one
# passage, so the same renderer produces a single ``[P1]`` block. The function
# is imported rather than reimplemented; see ``_render_passages``.

FAMILY_UNCAPPED = "exp5_uncapped"
FAMILY_CAP_ONLY = "exp5_cap_only"

# The one clause this module adds. Leading space matches how Experiment 5's own
# ``length_directive`` appends to the instruction string.
CAP_CLAUSE = " Your handoff message must contain at most {cap} words."

# Experiment 5 inserts the question with this exact text
# (``run_summary_generalization.compression_user_prompt``).
QUESTION_PREFIX = "\n\nQuestion the final agent must answer: "

# Experiment 5 has no selector, so selection needs an instruction of its own.
# It is held constant across the two selection arms, which keeps the question
# line the only arm-level difference there too, and it carries no fill wording:
# the packer stops at the cap, so asking for "enough candidates to fill the
# budget" would reintroduce exactly the pressure this protocol removes.
SELECTION_TASK = (
    "Select complete source sentences. Return only a JSON array of integer sentence IDs, "
    "ranked by selection priority. The receiver will see their exact text in source order, "
    "not the IDs. Never invent an ID or repeat one.")

MECHANISMS = ("generation", "selection")


def _render_passages(passages: list[dict]) -> str:
    """Experiment 5's own renderer, imported late to avoid a cycle at import."""
    from run_summary_generalization import render_context

    return render_context({"passages": passages})


def system_prompt() -> str:
    """Experiment 5's system prompt, identical for both arms."""
    return CHAIN_SYSTEM


def question_marker(question: str) -> str:
    """The exact substring that the conditioned prompt adds and generic omits."""
    return f"{QUESTION_PREFIX}{question}"


def cap_clause(cap: int | None) -> str:
    return "" if cap is None else CAP_CLAUSE.format(cap=int(cap))


def render_units(view) -> str:
    """The numbered-unit source block a selector needs to return IDs."""
    return "\n".join(f"[{u.index}] ({u.words} words) {u.text}" for u in view.units)


def user_prompt(source_text: str, question: str | None = None,
                cap: int | None = None, mechanism: str = "generation") -> str:
    """The stage-1 user message, optionally plus the cap clause.

    ``question is None`` is the generic arm. Passing ``cap`` switches the
    family from ``exp5_uncapped`` to ``exp5_cap_only`` and changes nothing else.
    ``mechanism="generation"`` reproduces Experiment 5 exactly;
    ``mechanism="selection"`` keeps the same shape and question line and swaps
    the source rendering and the task instruction, both constant across arms.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if mechanism == "generation":
        material = f"Source material:\n{_render_passages([{'text': source_text}])}"
        instruction = INITIAL_INSTRUCTION + cap_clause(cap)
    else:
        material = f"Source material:\n{source_text}"
        instruction = SELECTION_TASK + cap_clause(cap)
    if question is None:
        return f"{material}\n\n{instruction}"
    return f"{material}{question_marker(question)}\n\n{instruction}"


def assert_minimal_pair(source_text: str, question: str, cap: int | None = None,
                        mechanism: str = "generation") -> None:
    """Runtime guard: the arms may differ only by Experiment 5's question line.

    Called on every message this pipeline writes, not only in the selftest, so
    a future edit to the instruction or the cap clause cannot quietly turn the
    manipulation into two differently worded objectives.
    """
    conditioned = user_prompt(source_text, question, cap, mechanism)
    generic = user_prompt(source_text, None, cap, mechanism)
    if conditioned.replace(question_marker(question), "") != generic:
        raise AssertionError(
            f"{mechanism}: conditioned prompt minus the Experiment 5 question "
            "line is not byte-identical to the generic prompt")
    if question and question in generic:
        raise AssertionError(f"{mechanism}: generic prompt leaks the question")
    if cap is not None:
        uncapped_c = user_prompt(source_text, question, None, mechanism)
        uncapped_g = user_prompt(source_text, None, None, mechanism)
        clause = cap_clause(cap)
        if conditioned != uncapped_c + clause or generic != uncapped_g + clause:
            raise AssertionError(
                f"{mechanism}: capped prompt is not the uncapped prompt plus "
                "the identical cap clause")


def family(cap: int | None) -> str:
    return FAMILY_UNCAPPED if cap is None else FAMILY_CAP_ONLY


def base_prompt_hash() -> str:
    """Hash of the Experiment 5 strings this module reuses, cap excluded."""
    payload = "\x00".join([CHAIN_SYSTEM, INITIAL_INSTRUCTION, QUESTION_PREFIX])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def template_hash(cap: int | None, mechanism: str = "generation") -> str:
    """Hash of the full rendered template, including the cap clause."""
    task = INITIAL_INSTRUCTION if mechanism == "generation" else SELECTION_TASK
    payload = "\x00".join([family(cap), mechanism, CHAIN_SYSTEM, task,
                           QUESTION_PREFIX, cap_clause(cap)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def conversation(source_text: str, question: str | None = None, cap: int | None = None,
                 mechanism: str = "generation") -> list[dict]:
    """The full message list handed to the writer."""
    if question is not None:
        assert_minimal_pair(source_text, question, cap, mechanism)
    return [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(source_text, question, cap, mechanism)}]
