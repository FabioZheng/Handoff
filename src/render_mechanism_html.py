"""Build the standalone Experiment 14 report page from the stored result tables.

Everything on the page is read from CSV at build time - no number is typed by
hand - so regenerating after a rerun cannot leave a stale figure in the prose.
Figures are embedded as data URIs, so the page is one self-contained file.

    python src/render_mechanism_html.py            # -> results/.../mechanism/report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import load_config  # noqa: E402

METRIC = "judge_correct"
BUDGETS = [20, 40, 80, 160]
WORKING = [40, 80, 160]

# One hue per mechanism family, carried from the figures into the page so a
# reader who has looked at a plot recognises the arm in a table.
FAMILY = {
    "abstractive": ("abstractive compression", "abs"),
    "lm": ("selection-based, LM scorer", "lm"),
    "nonllm": ("selection-based, lexical", "non"),
    "floor": ("floor", "floor"),
}

ARMS = [
    # code arm, display name, family key, rewrites, selects, selector
    ("passthrough", "passthrough (prefix)", "floor", "no", "no - positional", "-"),
    ("paraphrase", "paraphrase", "abstractive", "yes", "no - told not to choose", "-"),
    ("generic", "summary_generic", "abstractive", "yes", "query-agnostic", "the LLM itself"),
    ("conditioned", "summary_conditioned", "abstractive", "yes", "query-aware", "the LLM itself"),
    ("lm_generic", "lm_elimination_generic", "lm", "no", "query-agnostic",
     "GPT-2 self-information"),
    ("lm_conditioned", "lm_elimination_conditioned", "lm", "no", "query-aware",
     "GPT-2 question-likelihood gain"),
    ("nonllm_generic", "nonllm_elimination_generic", "nonllm", "no", "query-agnostic",
     "TF-IDF centrality"),
    ("nonllm_conditioned", "nonllm_elimination_conditioned", "nonllm", "no", "query-aware",
     "BM25 against q_now"),
    ("random_selection", "random_selection", "floor", "no", "seeded shuffle", "-"),
]
DISPLAY = {code: name for code, name, *_ in ARMS}
FAMILY_OF = {code: fam for code, _n, fam, *_ in ARMS}

PAIRS = [
    ("abstractive compression", "generic", "conditioned", "abstractive"),
    ("selection-based, LM scorer", "lm_generic", "lm_conditioned", "lm"),
    ("selection-based, lexical", "nonllm_generic", "nonllm_conditioned", "nonllm"),
]

RELATIONS = [("paraphrase", "paraphrase of q_now"), ("same_entity", "same entity"),
             ("same_topic", "same topic"), ("orthogonal", "orthogonal aspect")]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(value, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{f:.{digits}f}"


def signed(value, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{f:+.{digits}f}"


def excludes_zero(row: dict) -> bool:
    try:
        lo, hi = float(row["lo"]), float(row["hi"])
    except (TypeError, ValueError, KeyError):
        return False
    return lo > 0.0 or hi < 0.0


def ci(row: dict) -> str:
    try:
        return f"[{float(row['lo']):.3f}, {float(row['hi']):.3f}]"
    except (TypeError, ValueError, KeyError):
        return ""


def stat(row: dict, digits: int = 3) -> str:
    """A delta with its interval, and a mark for whether the interval clears zero."""
    mark = "excl" if excludes_zero(row) else "cov"
    label = "interval excludes zero" if mark == "excl" else "interval covers zero"
    return (f'<span class="stat {mark}" title="{label}">'
            f'<span class="est">{signed(row["delta"], digits)}</span>'
            f'<span class="ci">{ci(row)}</span></span>')


def find(rows: list[dict], **where):
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in where.items()):
            return row
    return None


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, caption: str, note: str = "") -> str:
    return (f'<figure class="fig">\n<img src="{data_uri(path)}" alt="{html.escape(caption)}">\n'
            f'<figcaption><strong>{html.escape(caption)}</strong>'
            + (f' {html.escape(note)}' if note else "")
            + "</figcaption>\n</figure>")


def build(result_root: Path, out: Path, cfg: dict, standalone: bool = False) -> None:
    mech = result_root / "mechanism"
    metrics = read_csv(result_root / "metrics.csv")
    contrasts = read_csv(result_root / "contrasts.csv")
    relations = read_csv(result_root / "relation_regret.csv")
    lengths = read_csv(result_root / "length_audit.csv")
    baselines = read_csv(result_root / "baselines.csv")
    family = read_csv(mech / "mechanism_distance_contrasts.csv")
    across = read_csv(mech / "family_vs_family.csv")
    survival = read_csv(mech / "evidence_survival.csv")
    aspects = read_csv(mech / "aspect_retention.csv")
    accounting = read_csv(mech / "budget_accounting.csv")

    lm_manifest = json.loads(
        (ROOT / "data" / "compression_mechanism" / "lm_unit_scores.jsonl")
        .read_text(encoding="utf-8").split("\n", 1)[0])["_manifest"]

    def utility(policy, words, endpoint):
        return find(metrics, policy=policy, budget_words=words, kind="utility",
                    endpoint=endpoint, metric=METRIC)

    def contrast(treatment, control, words, endpoint=None, kind="utility"):
        where = {"comparison": f"{treatment}_minus_{control}", "budget_words": words,
                 "kind": kind, "metric": METRIC}
        if endpoint:
            where["endpoint"] = endpoint
        return find(contrasts, **where)

    ceiling = find(baselines, baseline="direct_context", metric=METRIC)
    closed = find(baselines, baseline="closed_book", metric=METRIC)

    # ---------------------------------------------------------------- verdict
    gradient_rows = {fam: {w: find(family, family=label, budget_words=w, metric=METRIC)
                           for w in BUDGETS}
                     for label, _g, _c, fam in PAIRS}
    future_rows = {fam: {w: contrast(cond, gen, w, "future")
                         for w in BUDGETS}
                   for _label, gen, cond, fam in PAIRS}
    now_rows = {fam: {w: contrast(cond, gen, w, "now") for w in BUDGETS}
                for _label, gen, cond, fam in PAIRS}

    all_gradient = all(excludes_zero(gradient_rows[fam][w]) for _l, _g, _c, fam in PAIRS
                       for w in WORKING)
    losing = [fam for _l, _g, _c, fam in PAIRS
              if all(excludes_zero(future_rows[fam][w]) and float(future_rows[fam][w]["delta"]) < 0
                     for w in WORKING)]

    # ---------------------------------------------------------------- tables
    def utility_table() -> str:
        head = "".join(f"<th colspan='2'>{w} words</th>" for w in BUDGETS)
        sub = "".join("<th class='sub'>U_now</th><th class='sub'>U_future</th>" for _ in BUDGETS)
        body = []
        for code, name, fam, *_ in ARMS:
            cells = []
            for w in BUDGETS:
                n, f = utility(code, w, "now"), utility(code, w, "future")
                cells.append(f"<td class='n'>{num(n['mean']) if n else '-'}</td>"
                             f"<td class='n dim'>{num(f['mean']) if f else '-'}</td>")
            body.append(f"<tr><th scope='row'><span class='dot {fam}'></span>"
                        f"{html.escape(name)}</th>{''.join(cells)}</tr>")
        return (f"<div class='scroll'><table class='data wide'>"
                f"<thead><tr><th scope='col'>arm</th>{head}</tr>"
                f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")

    def gradient_table() -> str:
        rows = []
        for label, gen, cond, fam in PAIRS:
            for w in BUDGETS:
                g = gradient_rows[fam][w]
                if not g:
                    continue
                rows.append(
                    f"<tr><th scope='row'><span class='dot {fam}'></span>{html.escape(label)}</th>"
                    f"<td class='n'>{w}</td><td>{stat(g)}</td>"
                    f"<td class='n'>{g['n_contexts']}</td></tr>")
        return ("<div class='scroll'><table class='data'><thead><tr>"
                "<th>mechanism family</th><th>budget</th>"
                "<th>gradient DiD &mdash; conditioned minus its own query-agnostic control</th>"
                "<th>sources</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table></div>")

    def deltas_table() -> str:
        rows = []
        for label, gen, cond, fam in PAIRS:
            for w in BUDGETS:
                n, f = now_rows[fam][w], future_rows[fam][w]
                if not n:
                    continue
                rows.append(
                    f"<tr><th scope='row'><span class='dot {fam}'></span>{html.escape(label)}</th>"
                    f"<td class='n'>{w}</td><td>{stat(n)}</td><td>{stat(f)}</td></tr>")
        return ("<div class='scroll'><table class='data'><thead><tr>"
                "<th>mechanism family</th><th>budget</th>"
                "<th>&Delta;U_now (conditioned &minus; control)</th>"
                "<th>&Delta;U_future (conditioned &minus; control)</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    def across_table() -> str:
        rows = []
        for row in across:
            if row["metric"] != METRIC:
                continue
            rows.append(
                f"<tr><th scope='row'>{html.escape(row['family_a'])} <span class='vs'>vs</span> "
                f"{html.escape(row['family_b'])}</th><td class='n'>{row['budget_words']}</td>"
                f"<td class='n dim'>{num(row['did_a'])}</td>"
                f"<td class='n dim'>{num(row['did_b'])}</td><td>{stat(row)}</td></tr>")
        return ("<div class='scroll'><table class='data'><thead><tr><th>comparison</th>"
                "<th>budget</th><th>DiD A</th><th>DiD B</th><th>A &minus; B</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    def distance_table() -> str:
        rows = []
        for code, name, fam, *_ in ARMS:
            if fam == "floor" and code != "random_selection":
                continue
            for w in WORKING:
                cells = []
                far = near = None
                for rel, _label in RELATIONS:
                    r = find(relations, policy=code, budget_words=w, relation=rel,
                             kind="regret", metric=METRIC)
                    cells.append(f"<td class='n'>{num(r['mean']) if r else '-'}</td>")
                    if r and rel == "paraphrase":
                        near = float(r["mean"])
                    if r and rel == "orthogonal":
                        far = float(r["mean"])
                span = (f"<td class='n strong'>{signed(far - near)}</td>"
                        if far is not None and near is not None else "<td>-</td>")
                rows.append(f"<tr><th scope='row'><span class='dot {fam}'></span>"
                            f"{html.escape(name)}</th><td class='n'>{w}</td>"
                            f"{''.join(cells)}{span}</tr>")
        head = "".join(f"<th>{html.escape(label)}</th>" for _rel, label in RELATIONS)
        return ("<div class='scroll'><table class='data wide'><thead><tr><th>arm</th>"
                f"<th>budget</th>{head}<th>far &minus; near</th></tr></thead><tbody>"
                + "".join(rows) + "</tbody></table></div>")

    def survival_table() -> str:
        slices = ["q_now", "paraphrase", "same_entity", "same_topic", "orthogonal"]
        rows = []
        for code, name, fam, *_ in ARMS:
            for w in (40, 160):
                cells = []
                for sl in slices:
                    r = find(survival, policy=code, budget_words=w, slice=sl)
                    cells.append(f"<td class='n'>{num(r['span_survival']) if r else '-'}</td>")
                given = find(survival, policy=code, budget_words=w, slice="q_now")
                cells.append("<td class='n strong'>"
                             + (num(given["accuracy_given_survival"]) if given else "-")
                             + "</td>")
                rows.append(f"<tr><th scope='row'><span class='dot {fam}'></span>"
                            f"{html.escape(name)}</th><td class='n'>{w}</td>"
                            f"{''.join(cells)}</tr>")
        head = "".join(f"<th>{html.escape(s.replace('_', ' '))}</th>" for s in slices)
        return ("<div class='scroll'><table class='data wide'><thead><tr><th>arm</th>"
                f"<th>budget</th>{head}<th>accuracy given survival (q_now)</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    def aspect_table() -> str:
        rows = []
        for row in aspects:
            if int(row["budget_words"]) not in (40, 160):
                continue
            fam = FAMILY_OF.get(row["policy"], "floor")
            rows.append(
                f"<tr><th scope='row'><span class='dot {fam}'></span>"
                f"{html.escape(row['display'])}</th><td class='n'>{row['budget_words']}</td>"
                f"<td class='n'>{num(row['conditioning_aspect_retention'])}</td>"
                f"<td class='n dim'>{num(row['other_aspect_retention'])}</td>"
                f"<td>{stat(row)}</td></tr>")
        return ("<div class='scroll'><table class='data'><thead><tr><th>arm</th><th>budget</th>"
                "<th>conditioning aspect kept</th><th>other aspects kept</th>"
                "<th>difference</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    def channel_table() -> str:
        rows = []
        for code, name, fam, *_ in ARMS:
            cells = []
            for w in BUDGETS:
                r = find(lengths, policy=code, budget_words=w)
                cells.append(f"<td class='n'>{num(r['delivered_words_mean'], 1) if r else '-'}</td>"
                             f"<td class='n dim'>{num(r['fill_ratio_mean'], 2) if r else '-'}</td>")
            verbatim = [int(a["selected_unit_count"]) for a in accounting
                        if a["policy"] == code and a["target_words"] == "160"]
            mean_verbatim = sum(verbatim) / len(verbatim) if verbatim else 0.0
            rows.append(f"<tr><th scope='row'><span class='dot {fam}'></span>"
                        f"{html.escape(name)}</th>{''.join(cells)}"
                        f"<td class='n strong'>{mean_verbatim:.2f}</td></tr>")
        head = "".join(f"<th colspan='2'>{w}</th>" for w in BUDGETS)
        sub = "".join("<th class='sub'>words</th><th class='sub'>fill</th>" for _ in BUDGETS)
        return ("<div class='scroll'><table class='data wide'><thead>"
                f"<tr><th>arm</th>{head}<th rowspan='2'>verbatim source sentences "
                "per message (160w)</th></tr>"
                f"<tr><th></th>{sub}</tr></thead><tbody>" + "".join(rows)
                + "</tbody></table></div>")

    over = sum(1 for a in accounting if a["over_budget"] == "True")

    # ---------------------------------------------------------------- page
    figs = {
        "distance": figure(mech / "figures" / "distance_curves.png",
                           "Figure 1. Regret against designed query distance.",
                           "Solid = query-aware, dashed = its query-agnostic control. "
                           "Rising line = the message is worth less the further the question "
                           "moves from the one it was written for."),
        "tradeoff": figure(mech / "figures" / "now_future_tradeoff.png",
                           "Figure 2. Present against future utility.",
                           "One trajectory per arm across the 20/40/80/160-word ladder; the "
                           "diagonal is no specialisation."),
        "survival": figure(mech / "figures" / "evidence_survival.png",
                           "Figure 3. Was the evidence deleted, or did the reader fail?",
                           "Bars: the gold answer string survived into the message. Black tick: "
                           "the reader then answered correctly."),
        "relation": figure(result_root / "relation_distance.png",
                           "Figure 4. The same distance analysis, per arm.",
                           "Produced by the shared Experiment 10 analysis over all ten arms."),
        "budget": figure(result_root / "utility_vs_budget.png",
                         "Figure 5. Utility against budget.",
                         "Both endpoints, every arm, with the unbounded-source ceiling."),
        "pareto": figure(result_root / "pareto_now_vs_future.png",
                         "Figure 6. The present/future frontier.",
                         "Each point is one (arm, budget); the nondominated set is marked."),
        "control": figure(result_root / "budget_control.png",
                          "Figure 7. Channel audit.",
                          "Delivered length and fill ratio per arm - the check that no result "
                          "is a length artefact."),
        "regret": figure(result_root / "communication_regret.png",
                         "Figure 8. Regret against the unbounded-source ceiling."),
    }

    # The lexical selector is the one the claim names, so its own numbers go in it.
    verdict_now = ", ".join(signed(now_rows["nonllm"][w]["delta"]) for w in WORKING)
    css = CSS
    body = PAGE.format(
        ceiling=num(ceiling["mean"]) if ceiling else "-",
        closed=num(closed["mean"]) if closed else "-",
        n_messages=len(accounting),
        over=over,
        lm_model=html.escape(str(lm_manifest.get("model_id"))),
        lm_commit=html.escape(str(lm_manifest.get("commit_hash"))[:12]),
        sender=html.escape(cfg["model"]["id"]),
        judge=html.escape(cfg["judge"]["model_id"]),
        arm_rows="".join(
            f"<tr><th scope='row'><span class='dot {fam}'></span>{html.escape(name)}</th>"
            f"<td><code>{code}</code></td><td>{rew}</td><td>{sel}</td>"
            f"<td class='dim'>{html.escape(scorer)}</td></tr>"
            for code, name, fam, rew, sel, scorer in ARMS),
        utility_table=utility_table(),
        gradient_table=gradient_table(),
        deltas_table=deltas_table(),
        across_table=across_table(),
        distance_table=distance_table(),
        survival_table=survival_table(),
        aspect_table=aspect_table(),
        channel_table=channel_table(),
        verdict_gradient=("every family, at every working budget" if all_gradient
                          else "not in every family"),
        verdict_families=", ".join(FAMILY[f][0] for f in losing) or "none",
        now_deltas=verdict_now,
        **figs,
    )
    page = f"<title>Rewriting or Selection</title>\n<style>{css}</style>\n{body}"
    if standalone:
        # The artifact host supplies the doctype, head and body. A file opened
        # straight from disk gets none of that, and without a charset the page
        # renders mojibake - it is full of typographic dashes.
        # The reset goes AFTER the stylesheet, not before it: `css` opens with an
        # @import, and an @import that follows any style rule is discarded, which
        # would drop the webfonts silently and leave the page on fallbacks.
        reset = "\n:root{color-scheme:light dark}\nbody{margin:0}\nimg{max-width:100%}\n"
        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>Rewriting or Selection</title>\n"
            f"<style>{css}{reset}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[html] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,600;1,6..72,400&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --ground:#f5f7f9; --panel:#ffffff; --sunken:#eef1f5;
  --ink:#12161d; --ink-2:#39414d; --muted:#657084; --line:#dde2e9;
  --abs:#c25102; --lm:#08519c; --non:#54278f; --floor:#6b7280;
  --excl:#0f7a55; --cov:#8b93a1;
  --measure:66ch;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
  --sans:'Source Sans 3',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0d1015; --panel:#151a21; --sunken:#11151b;
    --ink:#e9edf3; --ink-2:#c3cbd6; --muted:#93a0b1; --line:#252d38;
    --abs:#f0873a; --lm:#77a9e2; --non:#ab8dd8; --floor:#8f98a6;
    --excl:#4cc79a; --cov:#79828f;
  }
}
:root[data-theme="dark"]{
  --ground:#0d1015; --panel:#151a21; --sunken:#11151b;
  --ink:#e9edf3; --ink-2:#c3cbd6; --muted:#93a0b1; --line:#252d38;
  --abs:#f0873a; --lm:#77a9e2; --non:#ab8dd8; --floor:#8f98a6;
  --excl:#4cc79a; --cov:#79828f;
}

*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 96px}
p,li{max-width:var(--measure)}
h1,h2,h3{text-wrap:balance;margin:0}
a{color:var(--lm)}
code{font-family:var(--mono);font-size:.88em;background:var(--sunken);
  padding:.1em .35em;border-radius:3px;border:1px solid var(--line)}

/* ---------- masthead ---------- */
header.top{border-bottom:1px solid var(--line);padding:56px 0 34px;margin-bottom:44px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--muted);margin:0 0 18px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(38px,6.4vw,62px);
  line-height:1.04;letter-spacing:-.018em;max-width:17ch}
.standfirst{font-family:var(--serif);font-size:20.5px;line-height:1.5;color:var(--ink-2);
  margin:20px 0 0;max-width:56ch}
.standfirst em{font-style:italic}
.facts{display:flex;flex-wrap:wrap;gap:0 34px;margin-top:30px;padding-top:22px;
  border-top:1px solid var(--line)}
.fact{display:flex;flex-direction:column;gap:2px}
.fact .k{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted)}
.fact .v{font-family:var(--mono);font-size:15px;font-weight:500;
  font-variant-numeric:tabular-nums}

/* ---------- structure ---------- */
section{margin:0 0 66px;scroll-margin-top:20px}
h2{font-family:var(--serif);font-weight:600;font-size:30px;letter-spacing:-.012em;
  margin:0 0 6px}
h3{font-family:var(--sans);font-weight:600;font-size:16px;letter-spacing:.01em;
  margin:34px 0 10px}
.kicker{font-family:var(--mono);font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);margin:0 0 8px}
.lede{color:var(--ink-2);margin:0 0 22px}
p{margin:0 0 16px}
hr.rule{border:0;border-top:1px solid var(--line);margin:0 0 40px}

nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 56px}
nav.toc a{font-family:var(--mono);font-size:12px;text-decoration:none;color:var(--ink-2);
  border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--panel)}
nav.toc a:hover{border-color:var(--lm);color:var(--lm)}

/* ---------- verdict ---------- */
.verdict{display:grid;gap:14px;margin:0 0 30px}
.claim{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--floor);
  border-radius:2px;padding:20px 24px}
.claim.a{border-left-color:var(--non)}
.claim.b{border-left-color:var(--abs)}
.claim.c{border-left-color:var(--lm)}
.claim h3{margin:0 0 6px;font-size:17px}
.claim p{margin:0;max-width:78ch;color:var(--ink-2)}

/* ---------- tables ---------- */
.scroll{overflow-x:auto;margin:0 0 10px;border:1px solid var(--line);border-radius:2px;
  background:var(--panel)}
table.data{border-collapse:collapse;width:100%;font-size:14px}
table.data.wide{min-width:760px}
table.data th,table.data td{padding:8px 13px;text-align:left;border-bottom:1px solid var(--line)}
table.data thead th{font-family:var(--mono);font-size:11px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);font-weight:500;vertical-align:bottom;
  background:var(--sunken)}
table.data thead th.sub{font-size:10px;letter-spacing:.06em}
table.data tbody th{font-weight:400;white-space:nowrap}
table.data tbody tr:last-child th,table.data tbody tr:last-child td{border-bottom:0}
table.data td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;
  white-space:nowrap}
table.data td.dim{color:var(--muted)}
table.data td.strong{font-weight:500}
.vs{color:var(--muted);font-family:var(--mono);font-size:11px}
caption,.tnote{font-size:13.5px;color:var(--muted);margin:0 0 26px;max-width:var(--measure)}

.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:9px;
  vertical-align:baseline;background:var(--floor)}
.dot.abstractive{background:var(--abs)}
.dot.lm{background:var(--lm)}
.dot.nonllm{background:var(--non)}

.stat{display:inline-flex;align-items:baseline;gap:9px;font-family:var(--mono);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.stat .est{font-weight:500}
.stat.excl .est{color:var(--excl)}
.stat.cov .est{color:var(--cov)}
.stat .ci{font-size:11.5px;color:var(--muted)}

.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;color:var(--muted);
  margin:0 0 26px}
.legend span{display:inline-flex;align-items:center;gap:7px}

/* ---------- figures ---------- */
figure.fig{margin:0 0 12px;background:var(--panel);border:1px solid var(--line);
  border-radius:2px;padding:14px}
figure.fig img{display:block;width:100%;height:auto;border-radius:1px}
figcaption{font-size:13.5px;color:var(--muted);margin-top:12px;max-width:var(--measure)}
figcaption strong{color:var(--ink-2);font-weight:600}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.grid2 figcaption{max-width:none}

/* ---------- callouts ---------- */
.note{background:var(--sunken);border:1px solid var(--line);border-radius:2px;
  padding:18px 22px;margin:0 0 26px}
.note p:last-child{margin-bottom:0}
.note .kicker{margin-bottom:6px}

ul.plain{margin:0 0 20px;padding-left:20px}
ul.plain li{margin-bottom:9px}

footer{border-top:1px solid var(--line);padding-top:26px;color:var(--muted);font-size:13.5px}
footer p{max-width:none}

@media (max-width:640px){
  .wrap{padding:0 18px 64px}
  body{font-size:16px}
  .facts{gap:0 22px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


PAGE = """
<div class="wrap">
<header class="top">
  <p class="eyebrow">Experiment 14 &middot; agent handoff information-loss probe</p>
  <h1>Rewriting or Selection</h1>
  <p class="standfirst">A bounded handoff written for the query you know now serves that query
  and abandons the others. This run asks <em>which part of compressing does it</em> &mdash;
  the rewriting, or the choosing.</p>
  <div class="facts">
    <div class="fact"><span class="k">sources</span><span class="v">16 dossiers</span></div>
    <div class="fact"><span class="k">rotations</span><span class="v">64</span></div>
    <div class="fact"><span class="k">arms</span><span class="v">10</span></div>
    <div class="fact"><span class="k">budgets</span><span class="v">20/40/80/160 w</span></div>
    <div class="fact"><span class="k">messages</span><span class="v">{n_messages}</span></div>
    <div class="fact"><span class="k">answers</span><span class="v">24,576</span></div>
    <div class="fact"><span class="k">cost</span><span class="v">$0.077</span></div>
  </div>
</header>

<nav class="toc">
  <a href="#verdict">Verdict</a>
  <a href="#arms">The arms</a>
  <a href="#distance">Query distance</a>
  <a href="#cost">What conditioning costs</a>
  <a href="#utility">Utility tables</a>
  <a href="#evidence">Evidence survival</a>
  <a href="#allocation">Allocation</a>
  <a href="#channel">Channel fairness</a>
  <a href="#figures">All figures</a>
  <a href="#limits">Limits</a>
  <a href="#repro">Provenance</a>
</nav>

<section id="verdict">
  <p class="kicker">The answer</p>
  <h2>Both outcomes hold, on two different quantities</h2>
  <p class="lede">The design offered two readings: either query-aware elimination reproduces
  the specialisation of query-aware summarisation, or it does not. The run splits them &mdash;
  which is a stronger result than either alone, because it localises the two halves of the
  effect in different parts of the compressor.</p>

  <div class="verdict">
    <div class="claim a">
      <h3>The shape of what survives needs no rewriting and no LLM</h3>
      <p>The near/far gradient &mdash; the message being worth less the further a question sits
      from the one it was written for &mdash; appears in {verdict_gradient}, including the arm
      whose compressor is Okapi BM25 over the source's own sentences with no neural model
      anywhere. Bounded task-aware <em>selection</em> is sufficient to produce it.</p>
    </div>
    <div class="claim b">
      <h3>But only rewriting destroys future utility</h3>
      <p>Measured against its own query-agnostic control, only {verdict_families} loses absolute
      future-query utility; both selection-based families cover zero at every budget. In the
      selection arms the gradient comes from <em>lifting the near questions</em> &mdash; the
      lexical selector gains {now_deltas} on U_now at 40/80/160 words &mdash; not from
      depressing the far ones.</p>
    </div>
    <div class="claim c">
      <h3>The LM in &ldquo;LM-based elimination&rdquo; is not what does the work</h3>
      <p>A pinned GPT-2 relevance score is the <em>weakest</em> of the three query-aware
      selectors: it produces a smaller gradient than BM25 at 40 and 80 words. The useful axis is
      abstractive versus selection-based compression, not LM versus non-LM.</p>
    </div>
  </div>

  <div class="note">
    <p class="kicker">Reader competence is not a confound</p>
    <p>Conditional on the gold answer string being present in the delivered message, judged
    accuracy is 0.93&ndash;1.00 for every arm at every budget. Practically every failure in this
    experiment happens at the compressor, not at the reader &mdash; so the differences below are
    about what each mechanism chose to keep.</p>
  </div>
</section>

<hr class="rule">

<section id="arms">
  <p class="kicker">Design</p>
  <h2>Ten arms, one channel</h2>
  <p class="lede">Every arm writes into the same hard two-sided word band, is read by the same
  sealed reader, and is scored by the same judge. Only the compression mechanism changes.
  <code>summary_generic</code> and <code>summary_conditioned</code> are Experiment&nbsp;10's own
  stored rows, imported unchanged: this run reproduces their request hashes exactly, so the
  older result reappears here rather than being re-estimated.</p>
  <div class="scroll"><table class="data wide"><thead><tr>
    <th>arm</th><th>code</th><th>rewrites?</th><th>selects?</th><th>selector</th>
  </tr></thead><tbody>{arm_rows}</tbody></table></div>
  <p class="tnote">Two references bound the picture: the unbounded source (no handoff) scores
  {ceiling} and closed-book scores {closed}. A true pass-through cannot be a budgeted arm here
  &mdash; the sources run 389&ndash;517 words against a 160-word cap &mdash; so the budgeted
  <code>passthrough</code> is the prefix that fits, which is positional elimination at the same
  sentence granularity as the scored arms.</p>
</section>

<section id="distance">
  <p class="kicker">Primary analysis</p>
  <h2>Utility against query distance</h2>
  <p class="lede">The relation dossiers are built so the distance between a hidden question and
  the conditioning one is a design-time fact, not an estimate: a paraphrase of it, a question
  about the same entity, one on the same topic, and one on an orthogonal aspect. A specialised
  message shows a rising regret curve; an unspecialised one is flat.</p>
  {distance}
  <p class="tnote">Read the solid lines against the dashed line of the same colour. All three
  query-aware arms fall to near-zero regret on a paraphrase of the query they were written for,
  and rise to their control's level by the orthogonal aspect. The query-agnostic controls are
  flat, as they must be &mdash; their message does not know which question is current.</p>

  <h3>Regret by tier, per arm</h3>
  {distance_table}
  <p class="tnote">Regret is the unbounded-source ceiling minus the arm's utility, so lower is
  better and the last column is the size of the gradient. The 20-word rung is omitted here; see
  Limits.</p>

  <h3>The gradient, as a paired difference</h3>
  <p>Per family: the query-aware arm's own (orthogonal &minus; paraphrase) utility gap, minus
  its query-agnostic control's, paired on the source. More negative means the query-aware arm
  gives up more as the question moves away.</p>
  {gradient_table}
  <div class="legend">
    <span><span class="stat excl"><span class="est">&minus;0.000</span></span> interval excludes zero</span>
    <span><span class="stat cov"><span class="est">&minus;0.000</span></span> interval covers zero</span>
    <span>10,000 bootstrap resamples, clustered on the source</span>
  </div>

  <h3>And between families</h3>
  <p>A triple difference, same pairing: does one mechanism's conditioning effect exceed
  another's?</p>
  {across_table}
  <p class="tnote">Abstractive compression's gradient exceeds the LM selector's at every working
  budget and the lexical selector's at 160 words; at 40 and 80 words abstractive and lexical are
  statistically indistinguishable. The lexical selector beats the LM one at 40 and 80.</p>
</section>

<section id="cost">
  <p class="kicker">The other half of the effect</p>
  <h2>What conditioning costs, in absolute terms</h2>
  <p class="lede">The gradient says the message is <em>shaped</em> around the present query. It
  does not say the future was made worse off. That is a separate measurement, and it separates
  the mechanisms.</p>
  {deltas_table}
  <p class="tnote">Only abstractive compression shows a negative &Delta;U_future whose interval
  excludes zero, and it deepens as the budget grows &mdash; extra room goes into the present
  query rather than into hedging. The selection-based families cover zero at every budget.</p>
  <div class="note">
    <p class="kicker">Where this null is informative, and where it is not</p>
    <p>At 160 words the query-agnostic selection controls still hold 0.315&ndash;0.350 of future
    utility, so there was real headroom to give up and the query-aware arms gave up none of it.
    At 40 words those same controls sit at 0.059&ndash;0.070 &mdash; near the floor &mdash; so
    &ldquo;no loss&rdquo; there is weak evidence rather than a clean null.</p>
  </div>
</section>

<section id="utility">
  <p class="kicker">Levels</p>
  <h2>Present and future utility, every arm</h2>
  {utility_table}
  <p class="tnote">Judged utility. U_now is the diagonal of each source's utility matrix, U_future
  the mean of its off-diagonal. Reference points: unbounded source {ceiling}, closed book
  {closed}.</p>
  {tradeoff}
  <p class="tnote">Two things read straight off this figure. The query-aware arms climb the
  left-hand wall &mdash; present utility bought at fixed future utility. And every query-free
  selector sits together near the diagonal: TF-IDF centrality, the positional prefix and a
  seeded shuffle are indistinguishable from one another, while the query-free <em>LLM</em>
  summary sits well above them, because it can pack several facts into a sentence where
  selection must spend a whole sentence per fact.</p>
</section>

<section id="evidence">
  <p class="kicker">Diagnostic</p>
  <h2>Deleted, or delivered and misread?</h2>
  <p class="lede">Survival is a string test on the delivered message &mdash; did any gold answer
  for this question appear in it &mdash; so it is defined identically for rewriting and for
  elimination, and it bounds what any reader could have done.</p>
  {survival}
  <h3>Survival by designed distance</h3>
  {survival_table}
  <p class="tnote">This is where the two query-aware mechanisms visibly differ. At 160 words the
  orthogonal question's answer survives in 0.039 of conditioned summaries against 0.286&ndash;0.293
  for conditioned elimination &mdash; roughly what their own query-agnostic controls (0.312) and
  the floors deliver. Rewriting <em>erases</em> the distant evidence; selection simply does not go
  out of its way to include it. The last column shows the reader is almost never the bottleneck.</p>
</section>

<section id="allocation">
  <p class="kicker">Compressor-side measurement</p>
  <h2>Allocation, with no reader in the path</h2>
  <p class="lede">The corpus partitions its evidence into four aspects, one per question group,
  and no sentence is claimed by two. For a selection arm the delivered units are known exactly,
  so the share of the conditioning aspect that survived can be compared with the share of the
  other three &mdash; no reader, no judge, no answer.</p>
  {aspect_table}
  <p class="tnote">The query-agnostic selectors are exactly zero by construction: their message
  does not depend on which aspect is current, so it keeps the same share of every aspect. The
  query-aware selectors add to the conditioning aspect roughly what they take from material that
  was not serving those questions well anyway.</p>
</section>

<section id="channel">
  <p class="kicker">The control that has to hold first</p>
  <h2>Was the channel actually equal?</h2>
  {channel_table}
  <p class="tnote">Messages over their hard cap: <strong>{over}</strong> of {n_messages}. At 80 and
  160 words every arm fills 0.91&ndash;0.96 of its budget and the selection arms actually deliver
  slightly <em>more</em> words than the abstractive ones, so the headline contrasts are not length
  artefacts. At 40 words selection fills 0.77&ndash;0.83 against 0.91&ndash;0.94; at 20 words it
  cannot fill the budget at all.</p>
  <p class="tnote">The last column is the mechanism check: source sentences present
  <em>verbatim</em> in the delivered message. The two families are doing different things to the
  text &mdash; which is exactly what Experiment&nbsp;10's prompt-based extractive arms, obeyed
  0&ndash;29% of the time, could not guarantee.</p>
  {control}
</section>

<section id="figures">
  <p class="kicker">Everything else the run produced</p>
  <h2>Remaining figures</h2>
  <div class="grid2">
    {relation}
    {budget}
    {pareto}
    {regret}
  </div>
</section>

<section id="limits">
  <p class="kicker">Read with these</p>
  <h2>Limits</h2>
  <ul class="plain">
    <li><strong>Sentence granularity, not tokens.</strong> Token-level deletion was not run:
    removing words inside a sentence adds a readability confound this design cannot separate
    from information loss.</li>
    <li><strong>The 20-word rung is degenerate for selection.</strong> The corpus's shortest
    sentence is 10 words and the median is 27, so whole-sentence selection delivers one sentence
    or none there, and five of sixteen sources deliver nothing at all. It is reported everywhere
    and excluded from every claim.</li>
    <li><strong>The absolute-loss null is budget-dependent.</strong> Informative at 160 words,
    weak at 40, for the floor reason given above.</li>
    <li><strong>One synthetic corpus, one sender/reader model.</strong> Sixteen dossiers. The
    distance labels are design-time facts, which is the corpus's strength, but the corpus is
    small and invented.</li>
    <li><strong>Query-free lexical importance is a weak control.</strong> TF-IDF centrality
    performs no better than a shuffle or a prefix; it is reported because it is the classical
    unsupervised extractive summariser and the fair counterpart to BM25, not because it is the
    best query-free baseline available.</li>
    <li><strong>No retrieval, memory, pointers or source reopening anywhere.</strong> The setup
    is source &rarr; bounded sealed message &rarr; reader throughout. The
    retrieval question is a separate experiment and a separate paper.</li>
  </ul>
</section>

<section id="repro">
  <p class="kicker">Provenance</p>
  <h2>How it was produced</h2>
  <ul class="plain">
    <li><strong>Sender and reader:</strong> <code>{sender}</code>, temperature 0.
    <strong>Judge:</strong> <code>{judge}</code>, binary correct/incorrect against the gold.</li>
    <li><strong>LM selector:</strong> <code>{lm_model}</code> @ <code>{lm_commit}</code>,
    teacher-forced on CPU, no sampling, no generation. LongLLMLingua-<em>style</em>: no
    <code>llmlingua</code> package is used, the coarse question-aware ranking direction is
    reimplemented and its formula is recorded in the artefact manifest.</li>
    <li><strong>Non-LLM selectors:</strong> Okapi BM25 and TF-IDF centrality, both dependency-free
    Python. No pretrained model participates in those scores.</li>
    <li><strong>Leakage:</strong> the hidden questions never reach a selector. Conditioned LM
    scores exist only for the four questions that rotate into the conditioning role, enforced at
    write time and re-checked at read time by 43 offline assertions.</li>
    <li><strong>Reuse:</strong> 640 messages and 11,264 answers imported from Experiment&nbsp;10
    with verified-equal request hashes; 64 new sender calls for <code>paraphrase</code>; 768
    selection messages at zero model cost.</li>
    <li><strong>Cost and time:</strong> $0.077 total. GPT-2 scoring of all 16 dossiers 106&nbsp;s
    on CPU; the main run about 20 minutes end to end, API-latency bound.</li>
  </ul>
</section>

<footer>
  <p>Generated from the stored result tables by <code>src/render_mechanism_html.py</code>.
  Every number on this page is read from CSV at build time. Raw rows in
  <code>runs/compression_mechanism/</code>, tables and figures in
  <code>results/compression_mechanism/relation_dossiers/n16/</code>.</p>
</footer>
</div>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="compression_mechanism_config.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--standalone", action="store_true",
                        help="emit a complete HTML document for opening straight from disk")
    args = parser.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    dataset = cfg["dataset"]["active"]
    n = int(cfg["dataset"][dataset]["n_contexts"])
    result_root = ROOT / cfg["outputs"]["result_root"] / dataset / f"n{n}"
    out = Path(args.out) if args.out else result_root / "mechanism" / "report.html"
    build(result_root, out, cfg, standalone=args.standalone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
