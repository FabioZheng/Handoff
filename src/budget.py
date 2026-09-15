"""Hard communication-budget control for the handoff sender.

Experiment 10 asks whether conditioning a handoff on the currently known query
buys present accuracy at the cost of future accuracy.  That question is only
answerable if the *channel* is held constant, and an identical ``max_tokens``
does not hold it constant: on the same source the conditioned and generic
policies of Experiment 5 self-selected roughly 600 and 3,800 characters, so any
accuracy gap was inseparable from "one arm simply wrote more".

The budget here is therefore a **two-sided word contract on the delivered
message**, enforced in three stages:

1. every policy is told the same floor and the same cap, in the same words;
2. a message outside the band is sent back for a rewrite with an identical
   correction - shrink if over the cap, expand if under the floor - up to
   ``max_length_corrections`` times;
3. whatever survives is deterministically truncated to the cap before it can
   reach a reader, so no policy can ever spend more channel than another.

The floor exists because a one-sided cap does not equalise anything. A pilot
with cap-only instructions on 3 SQuAD contexts at a 40-word cap produced fill
ratios of 0.61 (``conditioned``), 0.75 (``reusable``), 0.83 (``oracle``) and
0.88 (``generic``): the conditioned arm quietly spent a third less channel than
the control, so any future-query deficit it showed would have been
"it wrote less", not "it allocated differently". The cap is hard and can always
be enforced; the floor cannot be (words cannot be invented on demand), so it is
best-effort, audited, and reported as a realised fill ratio rather than assumed.

Words, not tokens, are the enforced unit: a word count is deterministic,
tokenizer-free and identical for every model, which is what a fair comparison
needs.  The nominal token budget, the derived word band, the requested
``max_tokens``, the model's own completion-token count and a token estimate of
the delivered text are all recorded, so the token view is never lost.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import split_sentences  # noqa: E402
from llm import LLMClient, estimate_tokens  # noqa: E402
from size_metrics import internal_repetition_rate  # noqa: E402


# ---------------------------------------------------------------------------
# The channel

SENDER_SYSTEM = (
    "You are a research subagent. You have read source material that the next agent "
    "cannot see and will never see. Your handoff is that agent's ONLY information. "
    "Anything you leave out is lost."
)

# One length contract, shared verbatim by every policy. ``floor`` and ``words``
# are the only slots, and nothing here refers to a question, so it cannot
# smuggle conditioning into an arm that is meant not to have any.
LENGTH_CONTRACT = (
    "Length: write between {floor} and {words} words. Your handoff is cut off at "
    "{words} words before the next agent sees it, so anything past the limit is "
    "lost, and a handoff shorter than {floor} words wastes channel that is "
    "already paid for. Use the space you have; do not pad it with filler and do "
    "not repeat yourself."
)

# Shared output form for the four abstractive policies. The extractive control
# replaces this on purpose: it is a different mechanism, not a different
# conditioning level, and it is analysed separately for that reason.
PROSE_FORM = "Write prose. Do not answer any question yourself."

EXTRACTIVE_FORM = (
    "Copy sentences VERBATIM from the source material - character for character. "
    "Do not paraphrase, merge, shorten or rewrite, and write nothing of your own. "
    "Separate the sentences with a single space. Do not answer any question yourself."
)

# Both corrections are byte-identical across policies, so an arm that overruns
# or under-fills more often is not also handed a different instruction. The
# expansion wording names no question and no future need: it asks only for more
# of what the source already contains, which is the one neutral way to fill a
# budget without inviting padding.
SHRINK_CORRECTION = (
    "LENGTH CORRECTION: your previous handoff was {actual} words, over the "
    "{words}-word limit. Rewrite it to between {floor} and {words} words, keeping "
    "the information that matters most. Return only the rewritten handoff."
)

EXPAND_CORRECTION = (
    "LENGTH CORRECTION: your previous handoff was {actual} words, under the "
    "{floor}-word minimum. Rewrite it to between {floor} and {words} words, using "
    "the extra room for more of what the source material contains rather than "
    "restating what you have already written. Return only the rewritten handoff."
)


# ---------------------------------------------------------------------------
# Deterministic length measurement and truncation


def word_count(text: str) -> int:
    """The enforced unit. ``str.split`` - no locale, no tokenizer, no model."""
    return len(str(text or "").split())


def truncate_to_words(text: str, max_words: int) -> tuple[str, bool, int]:
    """Cut ``text`` to at most ``max_words`` words. Returns (text, cut, dropped).

    Prefers the last sentence boundary that fits, so a truncated handoff still
    reads as finished prose; falls back to a hard word cut when not even the
    first sentence fits, which is the honest behaviour at the smallest budgets.
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words), False, 0
    kept: list[str] = []
    for sentence in split_sentences(" ".join(words)):
        n = word_count(sentence)
        if sum(word_count(s) for s in kept) + n > max_words:
            break
        kept.append(sentence)
    out = " ".join(kept) if kept else " ".join(words[:max_words])
    return out, True, len(words) - word_count(out)


# ---------------------------------------------------------------------------
# Budget specification


@dataclass(frozen=True)
class Budget:
    """One rung of the communication ladder.

    ``words`` is the enforced cap and ``floor_words`` the requested minimum.
    ``tokens`` is the nominal budget the rung is named by; ``api_max_tokens`` is
    what the request actually asks for, set above the cap on purpose so the word
    cap - not a provider-side cut - is the binding constraint, and so an
    over-long draft stays visible to the shrink stage instead of being silently
    clipped mid-sentence.
    """

    words: int
    floor_words: int
    tokens: int
    api_max_tokens: int

    @property
    def label(self) -> str:
        return f"w{self.words}"

    def as_dict(self) -> dict:
        return asdict(self)


def make_budget(words: int, tokens_per_word: float, headroom: float,
                floor_tokens: int, fill_floor_ratio: float) -> Budget:
    words = int(words)
    if words < 1:
        raise ValueError("a budget must be at least one word")
    if not 0.0 < float(fill_floor_ratio) <= 1.0:
        raise ValueError("budget.fill_floor_ratio must be in (0, 1]")
    tokens = max(1, round(words * float(tokens_per_word)))
    api_max = max(int(floor_tokens), int(round(tokens * float(headroom))))
    floor_words = max(1, min(words, int(round(words * float(fill_floor_ratio)))))
    return Budget(words=words, floor_words=floor_words, tokens=tokens,
                  api_max_tokens=api_max)


def budgets_from_config(cfg: dict) -> list[Budget]:
    spec = cfg["budget"]
    out = [make_budget(w, float(spec["tokens_per_word"]),
                       float(spec["api_max_tokens_headroom"]),
                       int(spec["api_max_tokens_floor"]),
                       float(spec["fill_floor_ratio"]))
           for w in spec["words"]]
    if len({b.words for b in out}) != len(out):
        raise ValueError("budget.words contains duplicates")
    return sorted(out, key=lambda b: b.words)


# ---------------------------------------------------------------------------
# Policies - the ONLY place conditioning differs


def _question_lines(questions) -> str:
    return "\n".join(f"  - {q}" for q in questions)


def policy_block(policy: str, current_question: str | None,
                 all_questions: tuple[str, ...] | None) -> str:
    """What the sender is told about the information need. Nothing else.

    The four abstractive policies are identical except for this block, which is
    what makes a between-policy difference attributable to conditioning rather
    than to wording.  ``generic`` deliberately says only that the sender has not
    been told the question: telling it that *further* questions may follow is
    the ``reusable`` treatment, and putting the treatment in the control would
    destroy the contrast the control exists to provide.
    """
    if policy in ("generic", "extractive_generic"):
        return ("The next agent will use your handoff to answer a question about this "
                "source material.\nYou have not been told which question it will be asked.")
    if policy in ("conditioned", "extractive_conditioned"):
        if not current_question:
            raise ValueError(f"policy {policy!r} requires a current question")
        return ("The next agent will use your handoff to answer this question:\n"
                f"  - {current_question}\n"
                "Communicate the information that answers that question.")
    if policy == "reusable":
        if not current_question:
            raise ValueError("policy 'reusable' requires a current question")
        return ("The next agent will use your handoff to answer this question:\n"
                f"  - {current_question}\n"
                "It may afterwards be asked further questions about this same source "
                "material that you have not been told.\n"
                "Communicate the information that answers the question above, and, "
                "within the same limit, preserve other evidence that would still be "
                "useful for those later questions.")
    if policy == "oracle":
        if not all_questions:
            raise ValueError("policy 'oracle' requires the full question set")
        return ("The next agent will use your handoff to answer these questions:\n"
                f"{_question_lines(all_questions)}\n"
                "Communicate the information that answers those questions.")
    if policy == "paraphrase":
        # Experiment 14's rewriting-only arm. It shares ``generic``'s first two
        # lines verbatim - same channel, same ignorance of the question - and
        # differs in exactly one instruction: restate rather than choose. That
        # single-sentence difference is the independent variable, so nothing
        # here may hint at importance, relevance or a question.
        return ("The next agent will use your handoff to answer a question about this "
                "source material.\nYou have not been told which question it will be asked.\n"
                "Restate the source material in your own words. Keep as much of it as fits "
                "in the length below, in the order it appears, rather than choosing which "
                "parts matter.")
    raise ValueError(f"unknown policy {policy!r}")


ABSTRACTIVE_POLICIES = ("generic", "conditioned", "reusable", "oracle", "paraphrase")
EXTRACTIVE_POLICIES = ("extractive_generic", "extractive_conditioned")
ALL_POLICIES = ABSTRACTIVE_POLICIES + EXTRACTIVE_POLICIES

# Policies whose message does not depend on which question is currently known.
# One message per (context, budget), reused across every rotation - both
# because that is the correct semantics and because paying k times for an
# identical prompt would be waste, not statistics.
ROTATION_INVARIANT = ("generic", "oracle", "extractive_generic", "paraphrase")


def is_extractive(policy: str) -> bool:
    return policy in EXTRACTIVE_POLICIES


def sender_prompt(policy: str, source: str, budget: Budget,
                  current_question: str | None = None,
                  all_questions: tuple[str, ...] | None = None) -> str:
    """Assemble the sender's user message.

    Order is fixed for every policy: source, then information need, then the
    identical length contract and output form.  Only ``policy_block`` varies
    among the abstractive arms.
    """
    form = EXTRACTIVE_FORM if is_extractive(policy) else PROSE_FORM
    return (
        f"Source material:\n{source}\n\n"
        f"{policy_block(policy, current_question, all_questions)}\n\n"
        f"{LENGTH_CONTRACT.format(floor=budget.floor_words, words=budget.words)}\n"
        f"{form}"
    )


# ---------------------------------------------------------------------------
# Generation under the cap

_FENCE = re.compile(r"^`{3}[a-zA-Z]*\s*|\s*`{3}$", re.MULTILINE)
_LEAD_IN = re.compile(r"^(here (is|are)[^\n:]{0,60}:|handoff:|summary:|notes:)\s*",
                      re.IGNORECASE)


def clean_message(raw: str) -> str:
    """Strip provider scaffolding that is not part of the message itself.

    A leading "Here is the handoff:" is scaffolding, not content, and would
    otherwise consume budget unequally across policies.
    """
    text = re.sub(r"<think>.*?</think>", " ", str(raw or ""), flags=re.DOTALL | re.IGNORECASE)
    text = _FENCE.sub("", text).strip()
    text = _LEAD_IN.sub("", text).strip()
    return " ".join(text.split())


@dataclass
class BudgetedMessage:
    text: str
    budget_words: int
    floor_words: int
    budget_tokens: int
    api_max_tokens: int
    attempts: int
    shrink_corrections: int
    expand_corrections: int
    first_draft_words: int
    final_draft_words: int
    delivered_words: int
    truncated: bool
    truncated_words_dropped: int
    over_budget_before_truncation: bool
    under_floor_delivered: bool
    fill_ratio: float
    repetition_rate: float
    delivered_estimated_tokens: int
    completion_tokens: int
    prompt_tokens: int
    finish_reasons: tuple[str, ...]
    cached: bool

    def as_dict(self) -> dict:
        row = asdict(self)
        row["finish_reasons"] = list(self.finish_reasons)
        return row


def generate_within_budget(
    client: LLMClient,
    prompt: str,
    budget: Budget,
    *,
    temperature: float,
    seed: int | None,
    tag: str,
    max_corrections: int,
    system: str = SENDER_SYSTEM,
) -> BudgetedMessage:
    """Generate a message and deliver it inside the word band.

    Both corrections are byte-identical across policies, so a policy that
    happens to overrun or under-fill more often is not also handed a different
    instruction. The cap is then enforced unconditionally; the floor is
    best-effort and its misses are recorded in ``under_floor_delivered``.
    """
    if max_corrections < 0:
        raise ValueError("max_corrections must be >= 0")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    texts: list[str] = []
    finish: list[str] = []
    shrinks = expands = 0
    prompt_tokens = completion_tokens = 0
    cached = True
    for attempt in range(max_corrections + 1):
        result = client.chat(
            messages,
            temperature=temperature,
            max_tokens=budget.api_max_tokens,
            seed=None if seed is None else int(seed) + attempt,
            tag=f"{tag}_attempt{attempt + 1}",
        )
        cleaned = clean_message(result.text)
        texts.append(cleaned)
        finish.append(result.finish_reason)
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        cached = cached and result.cached
        length = word_count(cleaned)
        if budget.floor_words <= length <= budget.words or attempt == max_corrections:
            break
        if length > budget.words:
            correction = SHRINK_CORRECTION.format(
                actual=length, floor=budget.floor_words, words=budget.words)
            shrinks += 1
        else:
            correction = EXPAND_CORRECTION.format(
                actual=length, floor=budget.floor_words, words=budget.words)
            expands += 1
        messages = messages + [{"role": "assistant", "content": cleaned},
                               {"role": "user", "content": correction}]

    # A correction can leave the message worse than an earlier draft. Deliver
    # the draft that best satisfies the contract, preferring one already inside
    # the band and otherwise the one closest to it, so a policy is not punished
    # for the model overshooting a correction it was told to make.
    def distance(text: str) -> tuple[int, int]:
        n = word_count(text)
        if budget.floor_words <= n <= budget.words:
            return (0, -n)
        if n > budget.words:
            # An over-long draft is truncated losslessly down to the cap, so it
            # is preferred to an equally distant under-length one.
            return (1, n - budget.words)
        return (2, budget.floor_words - n)

    final = min(texts, key=distance)
    delivered, cut, dropped = truncate_to_words(final, budget.words)
    return BudgetedMessage(
        text=delivered,
        budget_words=budget.words,
        floor_words=budget.floor_words,
        budget_tokens=budget.tokens,
        api_max_tokens=budget.api_max_tokens,
        attempts=len(texts),
        shrink_corrections=shrinks,
        expand_corrections=expands,
        first_draft_words=word_count(texts[0]),
        final_draft_words=word_count(final),
        delivered_words=word_count(delivered),
        truncated=cut,
        truncated_words_dropped=dropped,
        over_budget_before_truncation=word_count(final) > budget.words,
        under_floor_delivered=word_count(delivered) < budget.floor_words,
        fill_ratio=word_count(delivered) / float(budget.words),
        repetition_rate=internal_repetition_rate(delivered),
        delivered_estimated_tokens=estimate_tokens(delivered) if delivered else 0,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        finish_reasons=tuple(finish),
        cached=cached,
    )
