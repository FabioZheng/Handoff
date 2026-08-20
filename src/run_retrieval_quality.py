"""MS MARCO v2.1 retrieval-quality propagation pilot.

This intentionally changes only passage relevance: every condition has exactly
ten real MS MARCO passages.  Selected passages removed from a query are filled
with non-selected passages sampled from other MS MARCO dev queries.  All
compressors receive the original question at every stage; later stages receive
only the preceding summary and question.
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
from score import bootstrap_ci, extract_short_answer, paired_bootstrap_delta, score_against_golds  # noqa
from run_chain import add_bertscore  # noqa

SYSTEM = ("You are a research handoff agent. Preserve every fact needed to answer the "
          "question. Your notes replace the entire input for the next agent, so omitted "
          "information is lost. Do not answer the question directly.")
INITIAL = "Write concise research notes from the passages. Preserve answer-relevant facts, qualifiers, numbers, dates, and relationships."
REWRITE = "Rewrite the prior notes concisely. Preserve every fact needed to answer the question, including qualifiers, numbers, dates, and relationships."

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
    chosen=[]
    for qid in ids:
        ans=[str(x).strip() for x in raw['answers'].get(qid,[]) if str(x).strip()]
        ps=passages(raw,qid)
        # The 2-question smoke deliberately uses multi-selected examples so
        # its 10--20% bad arm is mathematically meaningful.  The 20-question
        # pilot is sampled normally, preserving the dataset's natural mix.
        needs_multi = n <= cfg['dataset']['smoke_questions']
        if ans and len(ps)==10 and any(x['selected'] for x in ps) and (not needs_multi or sum(x['selected'] for x in ps) >= 4):
            chosen.append({'qid':str(qid),'question':str(raw['query'][qid]),'golds':ans,'passages':ps})
        if len(chosen)==n: break
    if len(chosen)<n: raise RuntimeError(f'only found {len(chosen)} eligible examples')
    # A deterministic global pool, excluding a pack's own passages at draw time.
    pool=[]
    for qid in ids:
        for p in passages(raw,qid):
            if not p['selected']: pool.append((str(qid),p))
    return chosen,pool

def condition_packs(packs,pool,condition,seed):
    selected=[(q['qid'],i) for q in packs for i,p in enumerate(q['passages']) if p['selected']]
    target={'good':len(selected),'medium':round(len(selected)*.5),'bad':max(1,round(len(selected)*.15))}[condition]
    order=selected[:]; random.Random(f'{seed}:{condition}:selected').shuffle(order); keep=set(order[:target])
    rng=random.Random(f'{seed}:{condition}:distractors'); out=[]
    for q in packs:
        ps=[]
        for i,p in enumerate(q['passages']):
            if not p['selected'] or (q['qid'],i) in keep: ps.append({**p})
        removed=10-len(ps)
        candidates=[p for source,p in pool if source!=q['qid']]
        ps.extend({**p,'selected':0} for p in rng.sample(candidates,removed))
        rng.shuffle(ps)
        assert len(ps)==10 and all(not x['selected'] or x in q['passages'] for x in ps)
        out.append({**q,'passages':ps})
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
    raw=ensure_data(cfg); packs,pool=make_packs(raw,cfg,n); data_root=ROOT/cfg['outputs']['data_root']; rows=[]; summary=[]
    for condition in cfg['conditions']:
        built=condition_packs(packs,pool,condition,cfg['dataset']['sample_seed'])
        retained=sum(p['selected'] for q in built for p in q['passages']); original=sum(p['selected'] for q in packs for p in q['passages'])
        summary.append({'condition':condition,'questions':len(built),'passages_per_query':10,'selected_retained':retained,'selected_original':original,'selected_recall_at_10':round(retained/original,4)})
        for q in built: rows.append({'condition':condition,**q})
    recalls=[x['selected_recall_at_10'] for x in summary]
    if not (recalls[0]>recalls[1]>recalls[2]): raise AssertionError(f'recall separation failed: {summary}')
    if write:
        write_jsonl(data_root/f'packs_n{n}.jsonl',rows); write_csv(data_root/f'construction_n{n}.csv',summary)
    print('[retrieval:construction] '+json.dumps(summary)); return rows,summary

def analyse(rows,cfg,result_root):
    boot,ci=cfg['analysis']['bootstrap_resamples'],cfg['analysis']['ci_level']; metrics=[]; vectors={}
    for cond in cfg['conditions']:
        for depth in cfg['depths']:
            rs=sorted([r for r in rows if r['condition']==cond and int(r['depth'])==depth],key=lambda x:x['qid'])
            vals={m:np.array([r[m] for r in rs]) for m in ('em','f1','bertscore_f1')}; vectors[(cond,depth)]={m:dict(zip([r['qid'] for r in rs],v)) for m,v in vals.items()}
            rec={'condition':cond,'depth':depth,'n':len(rs)}
            for m,v in vals.items():
                mean,lo,hi=bootstrap_ci(v,boot,ci,seed=51); rec.update({m:round(mean,4),f'{m}_lo':round(lo,4),f'{m}_hi':round(hi,4)})
            metrics.append(rec)
    deltas=[]
    for cond in cfg['conditions']:
        for depth in cfg['depths']:
            if depth==0: continue
            for metric in ('em','f1','bertscore_f1'):
                a,b=vectors[(cond,depth)][metric],vectors[(cond,0)][metric]; ids=sorted(set(a)&set(b)); d=paired_bootstrap_delta(np.array([a[i] for i in ids]),np.array([b[i] for i in ids]),boot,ci,seed=53)
                deltas.append({'comparison':'depth_minus_depth0','condition':cond,'depth':depth,'metric':metric,**d})
    for depth in cfg['depths']:
        for metric in ('em','f1','bertscore_f1'):
            a,b=vectors[('good',depth)][metric],vectors[('bad',depth)][metric]; ids=sorted(set(a)&set(b)); d=paired_bootstrap_delta(np.array([a[i] for i in ids]),np.array([b[i] for i in ids]),boot,ci,seed=57)
            deltas.append({'comparison':'good_minus_bad','condition':'good_minus_bad','depth':depth,'metric':metric,**d})
    write_csv(result_root/'metrics.csv',metrics); write_csv(result_root/'deltas.csv',deltas)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4)); colors={'good':'#1b9e77','medium':'#7570b3','bad':'#d95f02'}
    for cond in cfg['conditions']:
        rs=[r for r in metrics if r['condition']==cond]; x=[r['depth'] for r in rs]
        axes[0].plot(x,[r['f1'] for r in rs],marker='o',label=cond,color=colors[cond]); axes[1].plot(x,[r['f1']-next(z['f1'] for z in metrics if z['condition']==cond and z['depth']==0) for r in rs],marker='o',label=cond,color=colors[cond])
    axes[0].set(title='QA quality by retrieval condition',xlabel='Handoff depth',ylabel='Token F1'); axes[1].set(title='Degradation from depth 0',xlabel='Handoff depth',ylabel='Token F1 change')
    for a in axes: a.grid(alpha=.25); a.legend(); a.set_xticks(cfg['depths'])
    fig.tight_layout(); fig.savefig(result_root/'retrieval_quality.png',dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'retrieval_quality_config.yaml')); ap.add_argument('--n',type=int); ap.add_argument('--construct-only',action='store_true'); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); n=args.n or cfg['dataset']['n_questions']; data_root=ROOT/cfg['outputs']['data_root']; run_root=ROOT/cfg['outputs']['run_root']/f'n{n}'; result_root=ROOT/cfg['outputs']['result_root']/f'n{n}'; result_root.mkdir(parents=True,exist_ok=True)
    packs,summary=construct(cfg,n,write=not args.dry_run)
    if args.construct_only: return 0
    by={(r['condition'],r['qid']):r for r in packs}; client=LLMClient(cfg,dry_run=args.dry_run)
    hp=run_root/'handoffs.jsonl'; handoffs={(r['condition'],r['qid'],int(r['stage'])):r['text'] for r in read_jsonl(hp)}
    for condition in cfg['conditions']:
        qs=[by[(condition,qid)] for qid in sorted(qid for c,qid in by if c==condition)]
        for stage in range(1,cfg['max_depth']+1):
            todo=[q for q in qs if (condition,q['qid'],stage) not in handoffs]
            with ThreadPoolExecutor(max_workers=cfg['runtime']['concurrency']) as pool:
                fs={pool.submit(compress_first if stage==1 else compress_next,client,q,cfg,condition) if stage==1 else pool.submit(compress_next,client,q,handoffs[(condition,q['qid'],stage-1)],cfg,condition,stage):q for q in todo}
                for f,q in fs.items(): handoffs[(condition,q['qid'],stage)]=f.result()
            if not args.dry_run: write_jsonl(hp,[{'condition':c,'qid':qid,'stage':st,'text':text} for (c,qid,st),text in sorted(handoffs.items())])
            print(f'[retrieval:handoff] {condition} stage{stage}: {len(todo)} generated')
    apath=run_root/'answers.jsonl'; existing={(r['condition'],int(r['depth']),r['qid']):r for r in read_jsonl(apath)}; jobs=[]
    for (condition,qid),q in by.items():
        for depth in cfg['depths']:
            key=(condition,depth,qid)
            if key not in existing: jobs.append((key,q,context(q) if depth==0 else handoffs[(condition,qid,depth)]))
    with ThreadPoolExecutor(max_workers=cfg['runtime']['concurrency']) as pool:
        fs={pool.submit(answer,client,q,mat,cfg,f'retrieval_{c}_answer_d{d}'):(c,d,qid) for (c,d,qid),q,mat in jobs}
        for f,key in fs.items(): c,d,qid=key; existing[key]={'condition':c,'depth':d,'qid':qid,**f.result()}
    rows=sorted(existing.values(),key=lambda r:(r['condition'],r['depth'],r['qid']))
    if not args.dry_run:
        write_jsonl(apath,rows); add_bertscore(rows,cfg); write_jsonl(apath,rows); analyse(rows,cfg,result_root)
    print(f'[retrieval:answers] {len(rows)} rows; {len(jobs)} calls this pass'); print(json.dumps(client.ledger.summary(),indent=2))
if __name__=='__main__': main()
