"""MS MARCO v2.1 retrieval-quality propagation pilot.

This intentionally changes only passage relevance: every condition has exactly
ten real MS MARCO passages.  A self-contained BM25 index does the retrieval:
The low-noise context retains every query's own MS MARCO-labelled-relevant
passage that BM25 also actually surfaces (top retrieved AND relevant, not
relevance in isolation).  The medium- and high-noise contexts retain fewer
such passages and backfill the fixed ten-passage width with real BM25 hard
negatives -- passages BM25
ranks highly (lexically on-topic) but MS MARCO's `is_selected` label marks
irrelevant, rather than an unrelated passage sampled from a different query.
All compressors receive the original question at every stage; later stages
receive only the preceding summary and question.
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import handoffs as hm  # noqa
from llm import LLMClient, load_config  # noqa
from retrieval import BM25Index, content_fingerprint  # noqa
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa
from run_chain import add_bertscore  # noqa

SYSTEM = ("You are a research handoff agent. Preserve every fact needed to answer the "
          "question. Your notes replace the entire input for the next agent, so omitted "
          "information is lost. Do not answer the question directly.")
INITIAL = "Write concise research notes from the passages. Preserve answer-relevant facts, qualifiers, numbers, dates, and relationships."
REWRITE = "Rewrite the prior notes concisely. Preserve every fact needed to answer the question, including qualifiers, numbers, dates, and relationships."

# ---------------------------------------------------------------- BM25 retrieval
# BM25Index/content_fingerprint are shared with run_redundant_signal_ratio.py
# (src/retrieval.py) -- both experiments need "retrieved but not relevant"
# hard negatives instead of a uniformly random unrelated passage.


def arm_name(negative_type: str, retrieval_condition: str) -> str:
    """Keep legacy hard-arm ids so their prior cached results remain reusable."""
    return retrieval_condition if negative_type == "hard" else f"{negative_type}_{retrieval_condition}"


CONTEXT_LABELS = {
    "good": "Low noise",
    "medium": "Medium noise",
    "bad": "High noise",
}

CONTEXT_COMPOSITIONS = {
    "good": "All BM25-findable gold passages retained; remaining slots are distractors.",
    "medium": "Half of BM25-findable gold passages retained; remaining slots are distractors.",
    "bad": "15% of BM25-findable gold passages retained (at least one); remaining slots are distractors.",
}


def arm_specs(cfg: dict) -> list[tuple[str, str, str]]:
    return [(arm_name(negative_type, condition), negative_type, condition)
            for negative_type in cfg["negative_types"] for condition in cfg["conditions"]]


def build_bm25_pool(raw: dict, cfg: dict, required_qids: list[str]) -> tuple[BM25Index, dict[str, dict]]:
    """Index a bounded, deterministic pool of queries' passages.

    ``required_qids`` (the queries actually used in the experiment) are always
    included so their own passages are retrievable. The remainder of the pool
    is filled with a large deterministic sample of other queries, so BM25 hard
    negatives can come from genuinely different topics, not just the same
    query's own passages.
    """
    pool_size = cfg["dataset"]["bm25_pool_queries"]
    all_ids = list(raw["query"].keys())
    rng = random.Random(f"{cfg['dataset']['sample_seed']}:bm25_pool")
    rng.shuffle(all_ids)
    pool_ids = list(dict.fromkeys(required_qids + all_ids))[:max(pool_size, len(required_qids))]
    docs: list[tuple[str, str]] = []
    doc_lookup: dict[str, dict] = {}
    for qid in pool_ids:
        for i, p in enumerate(passages(raw, qid)):
            doc_id = f"{qid}:{i}"
            docs.append((doc_id, p["text"]))
            doc_lookup[doc_id] = {**p, "qid": qid, "index": i}
    return BM25Index(docs), doc_lookup


def read_jsonl(path):
    return [] if not path.exists() else [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8'); tmp.replace(path)

def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)

def ensure_data(cfg):
    path=ROOT/cfg['dataset']['local_parquet']
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix('.part')
        print('[retrieval:data] downloading MS MARCO v2.1 dev split (no LLM calls)')
        with requests.get(cfg['dataset']['dev_url'], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp,'wb') as f:
                for c in r.iter_content(1<<20): f.write(c)
        tmp.replace(path)
    frame=pd.read_parquet(path)
    # Normalise the public parquet mirror to the official v2.1 JSON layout.
    return {'query':{str(r.query_id):r.query for r in frame.itertuples(index=False)},
            'answers':{str(r.query_id):list(r.answers) for r in frame.itertuples(index=False)},
            'passages':{str(r.query_id):r.passages for r in frame.itertuples(index=False)}}

def passages(raw, qid):
    p=raw['passages'][qid]
    # Official v2.1 JSON stores each passage field as a parallel list.
    return [{'text':' '.join(str(t).split()), 'selected':int(s), 'url':str(u)}
            for t,s,u in zip(p['passage_text'],p['is_selected'],p.get('url',['']*len(p['passage_text'])))]

def make_packs(raw,cfg,n):
    ids=list(raw['query'].keys()); rng=random.Random(cfg['dataset']['sample_seed']); rng.shuffle(ids)
    top_k=cfg['dataset']['bm25_top_k']; bottom_k=cfg['dataset']['bm25_bottom_k']
    # First pass: cheap structural eligibility only (answers exist, exactly
    # ten passages, at least one MS MARCO-labelled relevant). This over-collects
    # a larger candidate pool than n, because the BM25 pass below will reject
    # some of these for not being retriever-findable or lacking enough hard
    # negatives, same as the old code's implicit assumption that eligibility
    # could be checked with a single pass -- it can't once retrieval is real.
    structural_cap=max(n*25,500)
    structural=[]
    for qid in ids:
        ans=[str(x).strip() for x in raw['answers'].get(qid,[]) if str(x).strip()]
        ps=passages(raw,qid)
        # The 2-question smoke deliberately uses multi-selected examples so
        # its 10--20% bad arm is mathematically meaningful.  The 20-question
        # pilot is sampled normally, preserving the dataset's natural mix.
        needs_multi = n <= cfg['dataset']['smoke_questions']
        if ans and len(ps)==10 and any(x['selected'] for x in ps) and (not needs_multi or sum(x['selected'] for x in ps) >= 4):
            structural.append({'qid':str(qid),'question':str(raw['query'][qid]),'golds':ans,'passages':ps})
        if len(structural)==structural_cap: break
    if not structural: raise RuntimeError('no structurally eligible examples found')

    bm25,doc_lookup=build_bm25_pool(raw,cfg,[q['qid'] for q in structural])

    chosen=[]
    for q in structural:
        retrieved=bm25.top_k(q['question'],top_k)
        retrieved_ids={doc_id for doc_id,_ in retrieved}
        own_relevant_ids={f"{q['qid']}:{i}" for i,p in enumerate(q['passages']) if p['selected']}
        # "Top retrieved AND relevant" -- MS MARCO's label alone is not enough;
        # BM25 must actually surface it for this query's own text.
        usable_gold=[doc_id for doc_id in own_relevant_ids if doc_id in retrieved_ids]
        # Hard negatives: BM25-ranked highly for this query, but not a usable
        # gold passage -- either this query's own non-relevant passages, or
        # another query's passage that happens to be lexically on-topic.
        # BM25 rank order is preserved so condition_packs draws the hardest
        # (highest-scoring) negatives first.
        # Easy negatives use the *bottom* of the same BM25 ranking. They are
        # deliberately zero/near-zero lexical-overlap distractors, with the
        # same own-query relevant passages excluded as in the hard arm.
        hard_negatives=[doc_id for doc_id,_ in retrieved if doc_id not in own_relevant_ids]
        bottom=bm25.bottom_k(q['question'],bottom_k)
        easy_negatives=[doc_id for doc_id,_ in bottom if doc_id not in own_relevant_ids]
        if usable_gold and len(hard_negatives)>=9 and len(easy_negatives)>=9:
            chosen.append({**q,'usable_gold_ids':usable_gold,
                           'hard_negative_ids':hard_negatives,
                           'easy_negative_ids':easy_negatives})
        if len(chosen)==n: break
    if len(chosen)<n:
        raise RuntimeError(f'only found {len(chosen)}/{n} examples with BM25-retrievable gold and >=9 hard and easy negatives '
                            f'(from {len(structural)} structurally eligible); raise structural_cap or bm25_pool_queries')
    return chosen,doc_lookup

def condition_packs(packs,doc_lookup,condition,negative_type,seed):
    # Recall target is a GLOBAL knob over every usable-gold passage pooled
    # across all queries, exactly like the pre-BM25 design -- not a per-query
    # fraction. Per-query fractions break down (e.g. round(1*.5)==0 by
    # banker's rounding while high noise's max(1,...) floor stays at 1,
    # inverting the intended low > medium > high retention ordering whenever
    # a query has only one BM25-findable gold
    # passage, which is the common case here.
    all_gold=[(q['qid'],doc_id) for q in packs for doc_id in q['usable_gold_ids']]
    target={'good':len(all_gold),'medium':round(len(all_gold)*.5),'bad':max(1,round(len(all_gold)*.15))}[condition]
    order=all_gold[:]; random.Random(f'{seed}:{condition}:gold').shuffle(order)
    keep=set(order[:target])
    out=[]
    for q in packs:
        keep_ids=[doc_id for doc_id in q['usable_gold_ids'] if (q['qid'],doc_id) in keep]
        fill_n=10-len(keep_ids)
        neg_order=q[f'{negative_type}_negative_ids'][:]
        # The hard arm keeps its prior construction seed. Easy negatives add
        # only a type namespace; gold selection and final passage positions
        # remain matched across the two difficulties.
        neg_seed=f'{seed}:{condition}:{q["qid"]}:negatives' if negative_type=='hard' else f'{seed}:easy:{condition}:{q["qid"]}:negatives'
        random.Random(neg_seed).shuffle(neg_order)
        fill_ids=neg_order[:fill_n]
        assert len(fill_ids)==fill_n, f'not enough {negative_type} negatives for {q["qid"]}/{condition}: need {fill_n}, have {len(neg_order)}'
        chosen_ids=keep_ids+fill_ids
        rng=random.Random(f'{seed}:{condition}:{q["qid"]}:order'); rng.shuffle(chosen_ids)
        ps=[]
        for doc_id in chosen_ids:
            src=doc_lookup[doc_id]
            ps.append({'text':src['text'],'url':src['url'],'source_qid':src['qid'],
                       'selected':int(doc_id in keep_ids)})
        assert len(ps)==10 and sum(p['selected'] for p in ps)==len(keep_ids)
        out.append({'qid':q['qid'],'question':q['question'],'golds':q['golds'],'passages':ps,
                    'condition':arm_name(negative_type,condition),
                    'retrieval_condition':condition,
                    'context_label': CONTEXT_LABELS[condition],
                    'context_composition': CONTEXT_COMPOSITIONS[condition],
                    'negative_type':negative_type})
    return out

def context(q): return '\n\n'.join(f"[P{i+1}] {p['text']}" for i,p in enumerate(q['passages']))

def compress_first(client,q,cfg,condition):
    user=f"Retrieval condition: {condition}\n\nPassages:\n{context(q)}\n\nQuestion: {q['question']}\n\n{INITIAL}"
    r=client.chat([{'role':'system','content':SYSTEM},{'role':'user','content':user}],temperature=0,max_tokens=cfg['decoding']['handoff_max_tokens'],seed=1,tag=f'retrieval_{condition}_stage1')
    return r.text.strip()

def compress_next(client,q,notes,cfg,condition,stage):
    user=f"Prior notes:\n{notes}\n\nQuestion: {q['question']}\n\n{REWRITE}"
    r=client.chat([{'role':'system','content':SYSTEM},{'role':'user','content':user}],temperature=0,max_tokens=cfg['decoding']['handoff_max_tokens'],seed=stage,tag=f'retrieval_{condition}_stage{stage}')
    return r.text.strip()

def answer(client,q,material,cfg,tag):
    user=f"Research material:\n{material}\n\nQuestion: {q['question']}\nAnswer:"
    r=client.chat([{'role':'system','content':hm.ANSWER_SYSTEM},{'role':'user','content':user}],temperature=0,max_tokens=cfg['decoding']['answer_max_tokens'],seed=None,tag=tag)
    pred=extract_short_answer(r.text); em,f1=score_against_golds(pred,q['golds'])
    return {'pred':pred,'raw':r.text,'golds':q['golds'],'em':em,'f1':f1,'cached':r.cached}

def construct(cfg,n,write=True):
    raw=ensure_data(cfg); packs,doc_lookup=make_packs(raw,cfg,n); data_root=ROOT/cfg['outputs']['data_root']; rows=[]; summary=[]
    # How much of MS MARCO's own relevance judgment BM25 actually surfaces --
    # a diagnostic of the retriever, independent of the noise-level knob.
    labelled=sum(p['selected'] for q in packs for p in q['passages'])
    bm25_findable=sum(len(q['usable_gold_ids']) for q in packs)
    for condition_id,negative_type,condition in arm_specs(cfg):
        built=condition_packs(packs,doc_lookup,condition,negative_type,cfg['dataset']['sample_seed'])
        retained=sum(p['selected'] for q in built for p in q['passages'])
        summary.append({'condition':condition_id,'negative_type':negative_type,
                        'retrieval_condition':condition,'context_label':CONTEXT_LABELS[condition],
                        'context_composition':CONTEXT_COMPOSITIONS[condition],
                        'questions':len(built),'passages_per_query':10,
                        'gold_retained':retained,'gold_bm25_findable':bm25_findable,
                        'gold_labelled_by_msmarco':labelled,
                        'gold_recall_at_10':round(retained/bm25_findable,4)})
        rows.extend(built)
    for negative_type in cfg['negative_types']:
        recalls=[x['gold_recall_at_10'] for x in summary if x['negative_type']==negative_type]
        if not (recalls[0]>recalls[1]>recalls[2]): raise AssertionError(f'recall separation failed: {summary}')
    if write:
        write_jsonl(data_root/f'packs_n{n}.jsonl',rows); write_csv(data_root/f'construction_n{n}.csv',summary)
    print('[retrieval:construction] '+json.dumps(summary)); return rows,summary

def analyse(rows,cfg,result_root):
    boot,ci=cfg['analysis']['bootstrap_resamples'],cfg['analysis']['ci_level']; metrics=[]; vectors={}
    for cond,negative_type,retrieval_condition in arm_specs(cfg):
        for depth in cfg['depths']:
            rs=sorted([r for r in rows if r['condition']==cond and int(r['depth'])==depth],key=lambda x:x['qid'])
            vals={m:np.array([r[m] for r in rs]) for m in ('em','f1','bertscore_f1')}; vectors[(cond,depth)]={m:dict(zip([r['qid'] for r in rs],v)) for m,v in vals.items()}
            rec={'condition':cond,'negative_type':negative_type,'retrieval_condition':retrieval_condition,
                 'context_label':CONTEXT_LABELS[retrieval_condition],
                 'context_composition':CONTEXT_COMPOSITIONS[retrieval_condition],
                 'depth':depth,'n':len(rs)}
            for m,v in vals.items():
                mean,lo,hi=bootstrap_ci(v,boot,ci,seed=51); rec.update({m:round(mean,4),f'{m}_lo':round(lo,4),f'{m}_hi':round(hi,4)})
            metrics.append(rec)
    deltas=[]
    for cond,negative_type,retrieval_condition in arm_specs(cfg):
        for depth in cfg['depths']:
            if depth==0: continue
            for metric in ('em','f1','bertscore_f1'):
                a,b=vectors[(cond,depth)][metric],vectors[(cond,0)][metric]; ids=sorted(set(a)&set(b)); d=paired_bootstrap_delta(np.array([a[i] for i in ids]),np.array([b[i] for i in ids]),boot,ci,seed=53)
                deltas.append({'comparison':'depth_minus_depth0','condition':cond,'negative_type':negative_type,'retrieval_condition':retrieval_condition,'depth':depth,'metric':metric,**d})
    for negative_type in cfg['negative_types']:
        for depth in cfg['depths']:
            for metric in ('em','f1','bertscore_f1'):
                a,b=vectors[(arm_name(negative_type,'good'),depth)][metric],vectors[(arm_name(negative_type,'bad'),depth)][metric]
                ids=sorted(set(a)&set(b)); d=paired_bootstrap_delta(np.array([a[i] for i in ids]),np.array([b[i] for i in ids]),boot,ci,seed=57)
                deltas.append({'comparison':'good_minus_bad','condition':f'{negative_type}_good_minus_bad','negative_type':negative_type,'depth':depth,'metric':metric,**d})
    for retrieval_condition in cfg['conditions']:
        for depth in cfg['depths']:
            for metric in ('em','f1','bertscore_f1'):
                hard,easy=vectors[(arm_name('hard',retrieval_condition),depth)][metric],vectors[(arm_name('easy',retrieval_condition),depth)][metric]
                ids=sorted(set(hard)&set(easy)); d=paired_bootstrap_delta(np.array([hard[i] for i in ids]),np.array([easy[i] for i in ids]),boot,ci,seed=59)
                deltas.append({'comparison':'hard_minus_easy','condition':retrieval_condition,'retrieval_condition':retrieval_condition,'depth':depth,'metric':metric,**d})
    write_csv(result_root/'metrics.csv',metrics); write_csv(result_root/'deltas.csv',deltas)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4)); colors={'good':'#1b9e77','medium':'#7570b3','bad':'#d95f02'}; styles={'hard':'-','easy':'--'}
    for cond,negative_type,retrieval_condition in arm_specs(cfg):
        rs=[r for r in metrics if r['condition']==cond]; x=[r['depth'] for r in rs]
        label=f'{CONTEXT_LABELS[retrieval_condition]} / {negative_type}'
        axes[0].plot(x,[r['f1'] for r in rs],marker='o',linestyle=styles[negative_type],label=label,color=colors[retrieval_condition]); axes[1].plot(x,[r['f1']-next(z['f1'] for z in metrics if z['condition']==cond and z['depth']==0) for r in rs],marker='o',linestyle=styles[negative_type],label=label,color=colors[retrieval_condition])
    axes[0].set(title='QA quality: hard vs easy distractors',xlabel='Handoff depth',ylabel='Token F1'); axes[1].set(title='Degradation from depth 0',xlabel='Handoff depth',ylabel='Token F1 change')
    for a in axes: a.grid(alpha=.25); a.legend(); a.set_xticks(cfg['depths'])
    fig.tight_layout(); fig.savefig(result_root/'retrieval_quality.png',dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'retrieval_quality_config.yaml')); ap.add_argument('--n',type=int); ap.add_argument('--construct-only',action='store_true'); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); n=args.n or cfg['dataset']['n_questions']; data_root=ROOT/cfg['outputs']['data_root']; run_root=ROOT/cfg['outputs']['run_root']/f'n{n}'; result_root=ROOT/cfg['outputs']['result_root']/f'n{n}'; result_root.mkdir(parents=True,exist_ok=True)
    packs,summary=construct(cfg,n,write=not args.dry_run)
    if args.construct_only: return 0
    for q in packs: q['fp']=content_fingerprint([p['text'] for p in q['passages']])
    by={(r['condition'],r['qid']):r for r in packs}; client=LLMClient(cfg,dry_run=args.dry_run)
    hp=run_root/'handoffs.jsonl'
    handoffs={(r['condition'],r['qid'],r.get('fp'),int(r['stage'])):r['text'] for r in read_jsonl(hp)}
    for condition,_,_ in arm_specs(cfg):
        qs=[by[(condition,qid)] for qid in sorted(qid for c,qid in by if c==condition)]
        for stage in range(1,cfg['max_depth']+1):
            todo=[q for q in qs if (condition,q['qid'],q['fp'],stage) not in handoffs]
            with ThreadPoolExecutor(max_workers=cfg['runtime']['concurrency']) as pool:
                fs={pool.submit(compress_first if stage==1 else compress_next,client,q,cfg,condition) if stage==1 else pool.submit(compress_next,client,q,handoffs[(condition,q['qid'],q['fp'],stage-1)],cfg,condition,stage):q for q in todo}
                for f,q in fs.items(): handoffs[(condition,q['qid'],q['fp'],stage)]=f.result()
            if not args.dry_run: write_jsonl(hp,[{'condition':c,'qid':qid,'fp':fp,'stage':st,'text':text} for (c,qid,fp,st),text in sorted(handoffs.items())])
            print(f'[retrieval:handoff] {condition} stage{stage}: {len(todo)} generated')
    apath=run_root/'answers.jsonl'
    existing={(r['condition'],int(r['depth']),r['qid'],r.get('fp')):r for r in read_jsonl(apath)}; jobs=[]
    for (condition,qid),q in by.items():
        for depth in cfg['depths']:
            key=(condition,depth,qid,q['fp'])
            if key not in existing: jobs.append((key,q,context(q) if depth==0 else handoffs[(condition,qid,q['fp'],depth)]))
    with ThreadPoolExecutor(max_workers=cfg['runtime']['concurrency']) as pool:
        fs={pool.submit(answer,client,q,mat,cfg,f'retrieval_{c}_answer_d{d}'):(c,d,qid,fp) for (c,d,qid,fp),q,mat in jobs}
        for f,key in fs.items():
            c,d,qid,fp=key; q=by[(c,qid)]
            existing[key]={'condition':c,'retrieval_condition':q['retrieval_condition'],
                           'negative_type':q['negative_type'],'depth':d,'qid':qid,'fp':fp,**f.result()}
    rows=sorted(existing.values(),key=lambda r:(r['condition'],r['depth'],r['qid']))
    if not args.dry_run:
        write_jsonl(apath,rows); add_bertscore(rows,cfg); write_jsonl(apath,rows); analyse(rows,cfg,result_root)
    print(f'[retrieval:answers] {len(rows)} rows; {len(jobs)} calls this pass'); print(json.dumps(client.ledger.summary(),indent=2))
if __name__=='__main__': main()
