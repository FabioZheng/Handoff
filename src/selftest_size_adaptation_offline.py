"""Offline checks for Experiment 9. No API key, no network, no cost.

Covers the four things that would silently invalidate the experiment rather
than break it loudly:

  * the control arm is genuinely size-neutral, and every `resize_*` directive
    is its `inform_*` directive plus exactly one clause -- otherwise the
    headline contrast measures an incidental rewording;
  * a stage-2+ compressor cannot reach the source document, so a fact that
    reappears was reconstructed rather than re-read;
  * the fact life-history classifier calls large -> small -> large what it is,
    and separates a correct recovery from a memorised reversion from an
    invention;
  * the deterministic term scan flags Taren Vos -> Leran Vos and does not flag
    a faithful rewording.

Run: .venv/Scripts/python src/selftest_size_adaptation_offline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_size_adaptation_data as build  # noqa: E402
import handoffs as hm  # noqa: E402
import run_size_adaptation as R  # noqa: E402
import size_metrics as sx  # noqa: E402
from llm import load_config  # noqa: E402
from run_chain import INITIAL_INSTRUCTION, RECOMPRESS_INSTRUCTION  # noqa: E402

CONFIG = ROOT / "size_adaptation_config.yaml"

DOC_FICTIONAL = (
    "The city of Velmora was founded by Taren Vos in 1843 on the banks of the Arden Reach. "
    "Velmora grew around the Halloway Mill, which processed 4,200 tonnes of ore each year. "
    "The Velmora Assembly first convened in 1871 under the chairwoman Ilsa Norwen. "
    "A stone bridge, the Corran Span, was completed in 1889 and still carries the main road."
)
DOC_COUNTERFACTUAL = (
    "Ana Reyes was born in Lisbon in 1954 and studied at the Coimbra Institute. "
    "She joined the Meridian Survey in 1979 and led its coastal mapping programme for eleven years."
)


def item_fictional() -> dict:
    side = [
        {"fact_id": "fic:1:side1", "role": "side", "statement": "The mill processed 4,200 tonnes.",
         "question": "How much ore did the Halloway Mill process each year?",
         "answer": "4,200 tonnes", "aliases": [], "golds": ["4,200 tonnes"],
         "probes": ["4,200 tonnes"]},
        {"fact_id": "fic:1:side2", "role": "side", "statement": "The Assembly first met in 1871.",
         "question": "When did the Velmora Assembly first convene?",
         "answer": "1871", "aliases": [], "golds": ["1871"], "probes": ["1871"]},
        {"fact_id": "fic:1:side3", "role": "side", "statement": "The bridge is the Corran Span.",
         "question": "What is the name of Velmora's stone bridge?",
         "answer": "Corran Span", "aliases": [], "golds": ["Corran Span"], "probes": ["Corran Span"]},
    ]
    return build.make_item(
        item_id="fic:1", dataset="fictional", question="Who founded Velmora?",
        document=DOC_FICTIONAL, document_answer="Taren Vos", document_aliases=[],
        parametric_answer=None, parametric_aliases=[], side_facts=side,
        knowledge={"gold_sets": {}}, title="Velmora")


def item_counterfactual() -> dict:
    side = [
        {"fact_id": "wiki:9:side1", "role": "side", "statement": "She joined in 1979.",
         "question": "When did Ana Reyes join the Meridian Survey?",
         "answer": "1979", "aliases": [], "golds": ["1979"], "probes": ["1979"]},
        {"fact_id": "wiki:9:side2", "role": "side", "statement": "She studied at Coimbra.",
         "question": "Where did Ana Reyes study?", "answer": "Coimbra Institute",
         "aliases": [], "golds": ["Coimbra Institute"], "probes": ["Coimbra Institute"]},
    ]
    return build.make_item(
        item_id="wiki:9", dataset="counterfactual", question="Where was Ana Reyes born?",
        document=DOC_COUNTERFACTUAL, document_answer="Lisbon", document_aliases=[],
        parametric_answer="Rome", parametric_aliases=[], side_facts=side,
        knowledge={"gold_sets": {}})


# ---------------------------------------------------------------- prompts

def test_prompts(cfg: dict) -> None:
    R.directive_selftest(cfg)

    # The neutral base is the published prompt minus exactly one word, in both
    # directions: deleting it and putting it back must round-trip.
    assert R.NEUTRAL_INITIAL_INSTRUCTION != INITIAL_INSTRUCTION
    assert R._drop_concise(INITIAL_INSTRUCTION) == R.NEUTRAL_INITIAL_INSTRUCTION
    assert R._drop_concise(RECOMPRESS_INSTRUCTION) == R.NEUTRAL_RECOMPRESS_INSTRUCTION
    assert len(INITIAL_INSTRUCTION) - len(R.NEUTRAL_INITIAL_INSTRUCTION) == len("concise ")
    for bad in ("no size word here", "concise concise twice"):
        try:
            R._drop_concise(bad)
        except AssertionError:
            continue
        raise AssertionError("_drop_concise accepted an instruction it cannot safely edit")

    item = item_fictional()
    # Every arm's stage-1 prompt is the control's prompt plus its directive and
    # nothing else: the manipulation cannot leak into the material block.
    control = R.stage1_user_prompt(item, "none", cfg)
    for name in cfg["directives"]:
        prompt = R.stage1_user_prompt(item, name, cfg)
        assert prompt == control + R.directive_text(name, cfg), \
            f"stage-1 prompt for {name} differs from the control beyond its directive"
        assert item["document"] in prompt and item["question"] in prompt
    assert control.endswith(R.NEUTRAL_INITIAL_INSTRUCTION)

    # A control prompt must not mention size in any form.
    lowered = control.lower()
    for banned in ("token", "context window", "concise", "expand", "shorter", "longer"):
        assert banned not in lowered.replace(item["document"].lower(), ""), \
            f"control prompt leaks a size cue: {banned!r}"

    sealed = hm.seal(item["question"], hm.Handoff(item["item_id"], "chain", None, "prior notes"))
    base = R.recompress_user_prompt(sealed, "none", cfg)
    for name in cfg["directives"]:
        assert R.recompress_user_prompt(sealed, name, cfg) == base + R.directive_text(name, cfg)
    # The seal really seals: the document is not reachable from stage 2+.
    assert item["document"] not in base
    print("[selftest] prompts differ from the control by exactly their directive")


def test_schedules(cfg: dict) -> None:
    R.schedule_selftest(cfg)
    by_stage, serves = R.build_prefixes(cfg)
    schedules = {a: tuple(s["schedule"]) for a, s in cfg["arms"].items()}

    # Arms that share a prefix must share the generated record, not merely have
    # equal schedules: the paired contrast depends on a byte-identical history.
    for prefix, arms in serves.items():
        for arm in arms:
            assert schedules[arm][: len(prefix)] == prefix

    # The trajectory arms must actually shrink at the declared bottleneck and
    # expand after it, or "recovery across the bottleneck" measures nothing.
    stage = int(cfg["trajectory"]["bottleneck_stage"])
    for arm in cfg["trajectory"]["arms"]:
        schedule = schedules[arm]
        before = cfg["directives"][schedule[stage - 2]]["direction"]
        during = cfg["directives"][schedule[stage - 1]]["direction"]
        after = cfg["directives"][schedule[stage]]["direction"]
        assert during == "shrink", f"{arm} does not shrink at stage {stage}"
        assert after == "expand", f"{arm} does not expand after stage {stage}"
        assert before in ("shrink", "expand")
    print(f"[selftest] {sum(len(v) for v in by_stage.values())} prefixes consistent; "
          f"trajectory arms shrink at stage {stage} and expand after it")


# ---------------------------------------------------------------- metrics

def test_term_scan(cfg: dict) -> None:
    spec = cfg["measurement"]["unsupported_scan"]
    item = item_fictional()
    tracked = R.tracked_terms(item)

    faithful = ("Velmora, on the Arden Reach, was established by Taren Vos in 1843; "
                "its mill handled 4,200 tonnes annually.")
    clean = sx.unsupported_scan(faithful, DOC_FICTIONAL, spec, tracked)
    assert clean["unsupported_count"] == 0, \
        f"faithful rewording flagged as unsupported: {clean['unsupported_terms']}"

    corrupted = sx.unsupported_scan("Velmora was founded by Leran Vos in 1843.",
                                    DOC_FICTIONAL, spec, tracked)
    assert corrupted["unsupported_corrupted_count"] == 1, corrupted["unsupported_terms"]
    assert corrupted["unsupported_invented_count"] == 0
    assert corrupted["tracked_variants"][0]["tracked_term"] == "Taren Vos"

    invented = sx.unsupported_scan("Velmora was founded by Taren Vos, who also built Brightwater.",
                                   DOC_FICTIONAL, spec, tracked)
    assert invented["unsupported_invented_count"] == 1, invented["unsupported_terms"]
    assert invented["unsupported_corrupted_count"] == 0

    # A document term reached by a different accent or case is still supported.
    assert sx.fact_present("held at the gobernacion de antioquia",
                           ["Gobernación de Antioquia"])
    # Word boundaries hold: a short gold must not match inside a longer word.
    assert not sx.fact_present("the Antioquia region", ["Ana"])

    assert sx.internal_repetition_rate("a b c d e f g h", 5) == 0.0
    assert sx.internal_repetition_rate("a b c d e f a b c d e f", 5) > 0.0
    print("[selftest] term scan separates corruption, invention and faithful rewording")


def test_trajectories() -> None:
    stages = [1, 2, 3]
    lost_then_back = sx.presence_trajectory({1: True, 2: False, 3: True}, stages)
    assert lost_then_back["was_lost"] and lost_then_back["reappeared"]
    assert lost_then_back["lost_at_stage"] == 2 and lost_then_back["reappeared_at_stage"] == 3

    never_lost = sx.presence_trajectory({1: True, 2: True, 3: True}, stages)
    assert not never_lost["was_lost"] and never_lost["survived_throughout"]

    gone = sx.presence_trajectory({1: True, 2: False, 3: False}, stages)
    assert gone["was_lost"] and not gone["reappeared"]

    # A fact never present at stage 1 was never carried, so it cannot be "lost".
    absent = sx.presence_trajectory({1: False, 2: False, 3: False}, stages)
    assert not absent["was_lost"] and not absent["ever_present"]

    assert sx.answer_origin(True, False, True) == "document"
    assert sx.answer_origin(False, True, True) == "parametric"
    assert sx.answer_origin(True, True, True) == "both"
    assert sx.answer_origin(False, False, True) == "other"
    # With no memorised alternative there is no parametric verdict to give.
    assert sx.answer_origin(True, False, False) == "document"
    assert sx.answer_origin(False, False, False) == "unsupported"
    print("[selftest] fact life histories and answer-origin labels are correct")


# ---------------------------------------------------------------- pipeline

def synthetic_stage_rows(cfg: dict, items: list[dict]) -> list[dict]:
    """Fake handoffs whose text is engineered to exercise each classification.

    The `resize_lsl` chain is built to lose the target fact at its small stage
    and state it again afterwards, which is the event the experiment exists to
    detect; the shrink chain loses it and never gets it back.
    """
    by_stage, serves = R.build_prefixes(cfg)
    texts = {
        "shrink": ["Velmora was founded by Taren Vos in 1843; the Corran Span dates to 1889.",
                   "Velmora was founded by Taren Vos.",
                   "A settlement with a long history."],
        "expand": ["Velmora was founded by Taren Vos in 1843 on the Arden Reach.",
                   "Velmora was founded by Taren Vos in 1843 on the Arden Reach; "
                   "the Halloway Mill processed 4,200 tonnes and the Corran Span opened in 1889.",
                   "Velmora was founded by Taren Vos in 1843 on the Arden Reach. "
                   "Velmora was founded by Taren Vos in 1843 on the Arden Reach."],
        "lsl": ["Velmora was founded by Taren Vos in 1843 on the Arden Reach.",
                "A river settlement of some antiquity.",
                "Velmora was founded by Taren Vos, reportedly in 1843."],
    }
    rows = []
    for item in items:
        for stage, prefixes in by_stage.items():
            for prefix in prefixes:
                arms = serves[prefix]
                if "resize_lsl" in arms or "inform_lsl" in arms:
                    shape = "lsl"
                elif any(cfg["arms"][a]["group"] == "expand" for a in arms):
                    shape = "expand"
                else:
                    shape = "shrink"
                text = texts[shape][stage - 1]
                if item["dataset"] == "counterfactual":
                    # Mirror the same shapes onto the counterfactual document,
                    # with the memorised value ("Rome") appearing after the loss
                    # so the reversion path is exercised too.
                    text = {"shrink": ["Ana Reyes was born in Lisbon in 1954.",
                                       "Ana Reyes was born in Lisbon.", "A surveyor."],
                            "expand": ["Ana Reyes was born in Lisbon in 1954 and studied at "
                                       "the Coimbra Institute.",
                                       "Ana Reyes was born in Lisbon in 1954; Coimbra Institute; "
                                       "Meridian Survey from 1979.",
                                       "Ana Reyes was born in Lisbon in 1954. Ana Reyes was born "
                                       "in Lisbon in 1954."],
                            "lsl": ["Ana Reyes was born in Lisbon in 1954.",
                                    "A surveyor of the coast.",
                                    "Ana Reyes was born in Rome in 1954."]}[shape][stage - 1]
                record = {
                    "item_id": item["item_id"], "dataset": item["dataset"], "stage": stage,
                    "prefix": R.prefix_id(prefix), "directive": prefix[-1], "arms": arms,
                    "text": text, "previous_text": item["document"] if stage == 1 else "prior",
                    "prompt_tokens": 100, "completion_tokens": 40, "finish_reason": "stop",
                    "cost_usd": 0.0, "provider": "offline", "cached": True,
                }
                rows.append(R.measure_stage(record, item, cfg))
    return rows


def synthetic_answer_rows(cfg: dict, items: list[dict], stage_rows: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        for query_type in R.QUERY_TYPES:
            spec = R.query_spec(item, query_type)
            if not spec:
                continue
            rows.append({"item_id": item["item_id"], "dataset": item["dataset"], "depth": 0,
                         "prefix": "", "query_type": query_type, "question": spec["question"],
                         "golds": spec["golds"], "pred": spec["golds"][0], "em": 1.0, "f1": 1.0,
                         "judge_correct": 1, "parametric_judge_correct": 0,
                         "parametric_golds": spec["parametric_golds"],
                         "answer_origin": "document", "cached": True})
    for row in stage_rows:
        item = next(i for i in items if i["item_id"] == row["item_id"])
        for query_type in R.QUERY_TYPES:
            spec = R.query_spec(item, query_type)
            if not spec:
                continue
            hit = float(row["target_fact_present"] == 1.0)
            rows.append({"item_id": item["item_id"], "dataset": item["dataset"],
                         "depth": row["stage"], "prefix": row["prefix"],
                         "query_type": query_type, "question": spec["question"],
                         "golds": spec["golds"], "pred": spec["golds"][0] if hit else "unknown",
                         "em": hit, "f1": hit, "judge_correct": int(hit),
                         "parametric_judge_correct": 0,
                         "parametric_golds": spec["parametric_golds"],
                         "answer_origin": "document" if hit else "other", "cached": True})
    return rows


def test_pipeline(cfg: dict) -> None:
    items = [item_fictional(), item_counterfactual()]
    stage_rows = synthetic_stage_rows(cfg, items)
    answer_rows = synthetic_answer_rows(cfg, items, stage_rows)

    arm_rows = R.expand_to_arms(stage_rows, answer_rows, cfg)
    assert arm_rows, "expand_to_arms produced nothing"
    for row in arm_rows:
        if int(row["depth"]) == 0:
            assert row["shared_baseline"], "depth 0 must be flagged as the shared baseline"
    # Every arm must reach the deepest configured depth for both items.
    deepest = max(int(d) for d in cfg["depths"])
    for arm in cfg["arms"]:
        reached = {r["item_id"] for r in arm_rows if r["arm"] == arm and int(r["depth"]) == deepest}
        assert reached == {i["item_id"] for i in items}, f"{arm} missing items at depth {deepest}"

    metrics = R.compute_metrics(arm_rows, cfg)
    assert metrics and all("summary_words" in m for m in metrics if int(m["depth"]) > 0)
    contrasts = R.compute_contrasts(arm_rows, cfg)
    assert contrasts, "no contrasts computed"
    for row in contrasts:
        assert row["delta_lo"] <= row["delta"] <= row["delta_hi"], "bootstrap interval is inverted"

    facts = R.recovery_rows(stage_rows, answer_rows, items, cfg)
    assert facts, "no fact histories"
    # The engineered large->small->large chain must be detected as a recovery,
    # and the plain shrink chain must not be.
    lsl = [f for f in facts if f["arm"] == "resize_lsl" and f["fact_role"] == "target"
           and f["dataset"] == "fictional"]
    assert lsl and lsl[0]["was_lost"] and lsl[0]["reappeared"], \
        f"large->small->large recovery not detected: {lsl}"
    shrink = [f for f in facts if f["arm"] == "resize_shrink" and f["fact_role"] == "target"
              and f["dataset"] == "fictional"]
    assert shrink and shrink[0]["was_lost"] and not shrink[0]["reappeared"], \
        f"plain shrink wrongly reported as recovered: {shrink}"
    # On the counterfactual item the same chain reverts to the memorised value
    # instead, and must be labelled a reversion rather than a recovery.
    reverted = [f for f in facts if f["arm"] == "resize_lsl" and f["fact_role"] == "target"
                and f["dataset"] == "counterfactual"]
    assert reverted and reverted[0]["reverted_parametric"] and not reverted[0]["recovered_document"], \
        f"parametric reversion not distinguished from recovery: {reverted}"

    recovery = R.compute_recovery_metrics(facts, cfg)
    assert recovery and all(0.0 <= r["loss_rate"] <= 1.0 for r in recovery)
    origins = R.compute_origin_table(answer_rows, cfg)
    assert origins, "no answer-origin table (counterfactual answers missing)"
    for row in origins:
        total = sum(row[f"{k}_rate"] for k in ("document", "parametric", "both", "other"))
        assert abs(total - 1.0) < 1e-9, f"origin shares do not sum to 1: {row}"
    print(f"[selftest] pipeline: {len(arm_rows)} arm rows, {len(contrasts)} contrasts, "
          f"{len(facts)} fact histories; recovery, reversion and loss are distinguished")
    return metrics, contrasts, recovery, origins, items


def test_plots(cfg: dict, metrics, recovery, origins) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        R.make_primary_plot(metrics, cfg, out / "primary.png")
        R.make_expansion_plot(metrics, cfg, out / "expansion.png")
        R.make_recovery_plot(recovery, origins, cfg, out / "recovery.png")
        for name in ("primary.png", "expansion.png", "recovery.png"):
            assert (out / name).stat().st_size > 8000, f"{name} looks empty"
    print("[selftest] all three figures render")


def test_report(cfg: dict, metrics, contrasts, recovery, origins, items) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.md"
        R.write_report(metrics, contrasts, recovery, origins, cfg, path, items,
                       {"spent_usd": 0.0})
        body = path.read_text(encoding="utf-8")
        assert "handoff size adaptation" in body and "what came back" in body
        assert "concise" in body, "the report must say the baseline is not Experiment 5's prompt"
    print("[selftest] report renders")


def test_builder_helpers() -> None:
    assert build.json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert build.json_object('noise {"a": [1,2]} trailing') == {"a": [1, 2]}
    for bad in ("no object here", "{not json}"):
        try:
            build.json_object(bad)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError("json_object accepted malformed output")

    # Overlapping gold sets would make every document answer also match the
    # memorised probe, inflating every reversion count downstream.
    assert build.gold_sets_disjoint(["Lisbon"], ["Rome"])
    assert not build.gold_sets_disjoint(["Harvard Medical School"], ["Medical School"])
    assert not build.gold_sets_disjoint(["Medical School"], ["Harvard Medical School"])

    item = item_counterfactual()
    assert item["has_parametric"] and item["parametric_golds"] == ["Rome"]
    assert item["facts"][0]["role"] == "target"
    assert item["side_probe_fact_id"] in {f["fact_id"] for f in item["facts"]}
    # The side probe choice must be stable across runs, not dependent on order.
    assert build.make_item(**{**{"item_id": "wiki:9", "dataset": "counterfactual",
                                 "question": item["question"], "document": item["document"],
                                 "document_answer": "Lisbon", "document_aliases": [],
                                 "parametric_answer": "Rome", "parametric_aliases": [],
                                 "side_facts": item["facts"][1:], "knowledge": {}}}
                            )["side_probe_fact_id"] == item["side_probe_fact_id"]
    print("[selftest] builder helpers reject malformed and overlapping inputs")


def test_dry_run_writes_nothing(cfg: dict) -> None:
    """--dry-run must not touch data/, runs/ or results/ (README 'Cost control')."""
    watched = [ROOT / cfg["outputs"]["root"], ROOT / cfg["outputs"]["run_root"]]
    before = {p: sorted(x.name for x in p.rglob("*")) if p.exists() else None for p in watched}
    report = R.dry_run_report([item_fictional()], cfg)
    assert report["generation_calls"] > 0 and report["distinct_prefixes"] > 0
    assert report["distinct_prefixes"] <= report["arm_stages_without_sharing"]
    after = {p: sorted(x.name for x in p.rglob("*")) if p.exists() else None for p in watched}
    assert before == after, "the dry-run estimate touched an output directory"
    print("[selftest] dry-run estimate writes nothing")



# ------------------------------------------------- end-to-end, stubbed network

# Response text keyed to the directive that appears in the request, so the
# stubbed run exercises the length, repetition and unsupported-term code paths
# with genuinely different inputs rather than one constant string.
_STUB_FACTS = ("Velmora was founded by Taren Vos in 1843 on the Arden Reach. The Halloway Mill "
               "processed 4,200 tonnes of ore each year. The Velmora Assembly first convened in "
               "1871 under Ilsa Norwen, and the Corran Span was completed in 1889. ")


def _stub_text(body: dict, counts: dict) -> str:
    user = next((m.get("content", "") for m in body.get("messages", [])
                 if m.get("role") == "user"), "")
    if "gpt-4o-mini" in body.get("model", ""):
        if "Source document:" in user:
            counts["support"] += 1
            return "UNSUPPORTED: 1 | an added detail the document does not state"
        if "Rewritten notes:" in user:
            counts["preservation"] += 1
            return "MINOR_LOSS"
        counts["judge"] += 1
        # Alternating verdicts so both answer-origin branches are reached.
        return "CORRECT" if (counts["judge"] % 3) else "INCORRECT"
    if user.rstrip().endswith("Answer:"):
        counts["answer"] += 1
        return "Taren Vos"
    counts["compress"] += 1
    if "Expand your handoff" in user:
        return (_STUB_FACTS * 6).strip()
    if "Make your handoff fit within it" in user:
        return "A settlement of some antiquity."
    if "Keep your handoff concise" in user:
        return "Velmora, founded 1843."
    if "10,000-token" in user:
        return (_STUB_FACTS * 2).strip()
    if "2,000-token" in user:
        return "Velmora was founded by Taren Vos. The mill handled 4,200 tonnes."
    return _STUB_FACTS.strip()


def test_end_to_end_stubbed(cfg: dict) -> None:
    """Run the real pipeline with only the network replaced.

    Patches ``_post_with_retries`` rather than ``chat``, so cache keying, cost
    accounting, response parsing, all three judges, the analysis and every file
    write still execute for real. ``cache_dir`` and both output roots are
    redirected into a temporary directory, so a stubbed response can never reach
    the project's cache or overwrite a real result.

    This exists because the synthetic-fixture tests above build ``stage_rows``
    by hand and therefore never execute ``generate``, ``run_answers``,
    ``judge_answers`` or ``main``'s wiring -- which is exactly where a run that
    costs money would fail.
    """
    import yaml
    from llm import LLMClient

    needed = [ROOT / spec["items_jsonl"] for spec in cfg["dataset"].values()
              if spec.get("enabled", True)]
    if not all(p.exists() for p in needed):
        print("[selftest] end-to-end skipped: build the datasets first "
              "(src/build_size_adaptation_data.py)")
        return

    counts = {"compress": 0, "answer": 0, "judge": 0, "support": 0, "preservation": 0}

    def fake_post(self, body):
        text = _stub_text(body, counts)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": max(1, len(text) // 4),
                          "cost": 0.0},
                "model": body.get("model", "stub"), "provider": "stub", "_latency_s": 0.001}

    original = LLMClient._post_with_retries
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_cfg = json.loads(json.dumps(cfg))
        run_cfg["runtime"]["cache_dir"] = str(tmp / "cache")
        run_cfg["outputs"]["root"] = str(tmp / "results")
        run_cfg["outputs"]["run_root"] = str(tmp / "runs")
        run_cfg["analysis"]["bootstrap_resamples"] = 300   # a check, not a result
        cfg_path = tmp / "stub_config.yaml"
        cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")

        real_results = ROOT / cfg["outputs"]["root"]
        before = sorted(x.name for x in real_results.rglob("*")) if real_results.exists() else None

        LLMClient._post_with_retries = fake_post
        try:
            argv = sys.argv
            sys.argv = ["run_size_adaptation.py", "--config", str(cfg_path), "--limit", "3"]
            try:
                assert R.main() == 0, "stubbed run did not exit cleanly"
            finally:
                sys.argv = argv

            results, runs = tmp / "results", tmp / "runs"
            for name in ("metrics.csv", "metrics_by_source.csv", "contrasts.csv", "recovery.csv",
                         "fact_histories.csv", "answer_origin.csv", "arm_rows.csv", "report.md",
                         "size_adaptation.png", "expansion_content.png", "expansion_modes.png",
                         "headline_contrasts.png", "recovery.png"):
                path = results / name
                assert path.exists() and path.stat().st_size > 0, f"missing artifact {name}"
            for name in ("stages.jsonl", "answers.jsonl", "handoffs.jsonl"):
                assert (runs / name).exists(), f"missing record file {name}"

            # Every arm reached the deepest depth on every item.
            stage_rows = R.read_jsonl(runs / "stages.jsonl")
            assert len({r["item_id"] for r in stage_rows}) == 3, "--limit was not honoured"
            assert len(stage_rows) == 3 * 23, f"expected 69 stage records, got {len(stage_rows)}"
            # The judges ran and their verdicts landed on the stored rows.
            assert all("unsupported_claims" in r for r in stage_rows), "support judge did not land"
            assert all("semantic_preserved" in r for r in stage_rows), "preservation judge missing"
            assert counts["compress"] and counts["answer"] and counts["judge"], counts

            # The expand directive must actually produce a longer handoff than
            # the shrink one -- if the directive never reached the request, this
            # whole experiment measures nothing.
            longest = max(r["summary_words"] for r in stage_rows)
            shortest = min(r["summary_words"] for r in stage_rows)
            assert longest > shortest * 5, "directives did not reach the generation call"

            # --analyse-only must rebuild everything from records, calling nothing.
            snapshot = dict(counts)
            sys.argv = ["run_size_adaptation.py", "--config", str(cfg_path), "--analyse-only"]
            try:
                assert R.main() == 0, "--analyse-only did not exit cleanly"
            finally:
                sys.argv = argv
            assert counts == snapshot, "--analyse-only issued API calls"

            # It must also describe only the items its records cover, not every
            # item the config loads.
            report = (results / "report.md").read_text(encoding="utf-8")
            assert "3 items" in report, "report describes items it has no records for"
        finally:
            LLMClient._post_with_retries = original

        after = sorted(x.name for x in real_results.rglob("*")) if real_results.exists() else None
        assert before == after, "the stubbed run wrote into the real results directory"

    print(f"[selftest] end-to-end pipeline runs stubbed: {counts['compress']} handoffs, "
          f"{counts['answer']} answers, {counts['judge']} judge verdicts, all artifacts written")


def main() -> int:
    cfg = load_config(CONFIG)
    test_prompts(cfg)
    test_schedules(cfg)
    test_term_scan(cfg)
    test_trajectories()
    metrics, contrasts, recovery, origins, items = test_pipeline(cfg)
    test_plots(cfg, metrics, recovery, origins)
    test_report(cfg, metrics, contrasts, recovery, origins, items)
    test_builder_helpers()
    test_dry_run_writes_nothing(cfg)
    test_end_to_end_stubbed(cfg)
    print("\n[selftest] Experiment 9 offline checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
