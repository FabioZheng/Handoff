"""Validate judge variants against the hand labels, silver positives, wrong-paper and altered-number negatives.

Usage: python src/analysis/validate_exp5_judge.py <repo root> <work dir: judge/runs/ holds the pulled v2 items and run b58e4132a923; validation.json is written here> <excerpt models,> <full-source models,>
"""
import csv, json, random, re, sys, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
WT = Path(sys.argv[1]); SP = Path(sys.argv[2])
sys.path[:0] = [str(WT / "src" / d) for d in ("", "analysis", "latent", "builders")]
import judge_exp5_allocation as J, run_exp5_capped as rc
from llm import LLMClient
from reuse_common import read_rows
cfg = rc.config(str(WT / "configs/exp5_cap_only_frozen_config.yaml")); caps = rc.budgets(cfg)
docs = {d["id"]: d for d in read_rows(WT / cfg["data_root"] / "main.jsonl")}
run = SP / "judge/runs/exp5_cap_only/main/b58e4132a923"
v2 = []
for f in glob.glob(str(SP / "judge/runs/exp5_cap_only/main/b58e4132a923-allocation/items/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    v2 += [{**i, "doc": d["document_id"]} for i in d["items"]]
hand = list(csv.DictReader(open(WT / "results/exp5_cap_only/main/b58e4132a923/allocation_judge/spot_check.csv", encoding="utf-8")))
def find(doc, sent, pol, step):
    return next(i for i in v2 if i["doc"] == doc and i["sentence"] == sent and i["policy"] == pol and i["step"] == step)
cases = []
for h in hand:
    if h["hand_label"] == "A": continue
    cases.append({"kind": "hand_" + h["hand_label"], "item": find(h["document_id"], h["sentence"], h["policy"], h["step"])})
rng = random.Random(11); handset = {(c["item"]["doc"], c["item"]["sentence"]) for c in cases}
yes = [i for i in v2 if i.get("judge_support") == "YES" and (i["doc"], i["sentence"]) not in handset]
for i in rng.sample(yes, 30): cases.append({"kind": "silver_S", "item": i})
strong = [i for i in yes if i["best_source_sentence_overlap"] >= 0.6 and len(i["sentence"].split()) >= 8]
NUM = r"\b\d+(\.\d+)?\b"
numeric = [i for i in yes if re.search(NUM, i["sentence"]) and i["best_source_sentence_overlap"] >= 0.5 and not re.search(r"BIBREF|TABREF|FIGREF", i["sentence"])]
def bump(m):
    x = m.group(0)
    return str(round(float(x) * 1.5 + 3, 2)) if "." in x else str(int(x) * 2 + 3)
for i in rng.sample(numeric, 30):
    cases.append({"kind": "perturb_U", "item": {**i, "sentence_orig": i["sentence"], "sentence": re.sub(NUM, bump, i["sentence"], count=1)}})
ids = sorted(docs)
for i in rng.sample(strong, 30):
    other = rng.choice([x for x in ids if x != i["doc"] and x.startswith("qasper")])
    cases.append({"kind": "swap_U", "item": i, "source_doc": other})
# v3 items give the deterministic label and the preceding sentence
v3 = {}
for doc_id in {c["item"]["doc"] for c in cases}:
    msgs = [m for m in rc.load_messages(run, "mistral24", doc_id) if m["block"] == "core" and m["valid"] and m["policy"] in J.POLICIES]
    for it in J.build_items(docs[doc_id], msgs, caps, version=3):
        v3[(it["long_message_id"], it["sentence_index"])] = it
for c in cases:
    it = v3[(c["item"]["long_message_id"], c["item"]["sentence_index"])]
    c["det"], c["context"] = it["deterministic"], it["_context"]
    src = docs[c.get("source_doc", c["item"]["doc"])]
    c["units"] = J.sentences(src["id"], src["source"]); c["full"] = src["source"]
def client(model):
    return LLMClient({"model": {"id": model}, "decoding": {"top_p": 1.0},
        "runtime": {"cache_dir": str(Path.home() / ".cache" / "exp5j"), "max_retries": 6, "backoff_base_s": 2, "backoff_max_s": 60, "request_timeout_s": 120},
        "cost": {"cap_usd": 6.0, "warn_at_fraction": 0.8}})
variants = [(m, "excerpts") for m in sys.argv[3].split(",")] + [(m, "full") for m in sys.argv[4].split(",") if m]
results = {}
for model, mode in variants:
    cl = client(model)
    def ask(c):
        if c["det"]: return "DET_" + c["det"]
        msg = (J.support_prompt_v3(J.excerpts(c["units"], c["item"]["sentence"]), c["context"], c["item"]["sentence"])
               if mode == "excerpts" else J.support_prompt(c["full"], c["item"]["sentence"]))
        r = cl.chat(msg, temperature=0.0, max_tokens=16, seed=0, tag="validate")
        return J.parse_choice(r.text, ("YES", "NO", "NONE")) or "UNPARSED"
    with ThreadPoolExecutor(8) as ex: answers = list(ex.map(ask, cases))
    results[(model, mode)] = answers
    print(f"\n== {model} [{mode}] spent ${cl.ledger.spent:.4f}")
    for kind in ["hand_S", "silver_S", "hand_U", "swap_U", "perturb_U", "hand_N"]:
        a = [x for c, x in zip(cases, answers) if c["kind"] == kind]
        want = {"hand_S": ("YES",), "silver_S": ("YES",), "hand_U": ("NO",), "swap_U": ("NO",), "perturb_U": ("NO",), "hand_N": ("NONE", "DET_no_content")}[kind]
        ok = sum(x in want for x in a)
        from collections import Counter
        print(f"  {kind:9s} {ok}/{len(a)} correct  {dict(Counter(a))}")
json.dump({f"{m}|{md}": a for (m, md), a in results.items()} | {"cases": [{k: c[k] for k in ("kind", "det", "context")} | {"sentence": c["item"]["sentence"], "doc": c["item"]["doc"]} for c in cases]},
          open(SP / "validation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
