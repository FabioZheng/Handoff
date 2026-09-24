"""Batched local vLLM inference with an append-only request cache and no API key.

Every request is keyed by a hash of (model identity, request). Results are
appended to one JSONL file per job under a directory named after the model
identity, so concurrent jobs never write the same file and a scratch file quota
of ~1M inodes is not consumed one file per request. On start-up a backend reads
every JSONL file for its identity, which is what lets a reader job reuse a
writer job's results and a restarted job skip finished requests.

Operational knobs (``cfg["runtime"]``: batch sizes, memory fraction) are kept
out of the identity so tuning throughput does not invalidate finished work;
they are recorded with every result instead.
"""
from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from reuse_common import VERSION, atomic_json, digest

PROTOCOL_INFERENCE = (
    "dtype", "temperature", "top_p", "vllm_version",
    "transformers_version", "tokenizers_version",
)
VERSION_LOCKS = (
    ("vllm", "vllm_version"),
    ("transformers", "transformers_version"),
    ("tokenizers", "tokenizers_version"),
)


class ResultCache:
    """Append-only JSONL result store, one file per writing process."""

    def __init__(self, directory, name):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.results = {}
        for path in sorted(self.directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a killed job; recomputed later
                    self.results.setdefault(row["key"], row)
        self.path = self.directory / f"{name}.jsonl"
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                torn = handle.read(1) != b"\n"
        else:
            torn = False
        self.handle = self.path.open("a", encoding="utf-8")
        if torn:
            self.handle.write("\n")

    def get(self, key):
        return self.results.get(key)

    def put(self, rows):
        for row in rows:
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.results[row["key"]] = row
        self.handle.flush()
        os.fsync(self.handle.fileno())


def host_info():
    """Where a result was produced. Recorded, never part of the identity."""
    import socket
    import subprocess

    try:
        listed = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30).stdout
        gpus = [line.strip() for line in listed.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    return {"node": socket.gethostname(), "gpus": gpus,
            "slurm_job": os.environ.get("SLURM_JOB_ID", "")}


def _token_ids(encoded):
    if hasattr(encoded, "keys"):  # BatchEncoding from newer transformers
        encoded = encoded["input_ids"]
    if isinstance(encoded, str):
        raise TypeError("Chat template returned text; token ids are required")
    return list(encoded)


class VLLMBackend:
    def __init__(self, cfg, spec, cache_name):
        import vllm

        for package, config_key in VERSION_LOCKS:
            installed, expected = version(package), cfg["inference"][config_key]
            if installed != expected:
                raise RuntimeError(
                    f"{package} version {installed} differs from the protocol lock {expected}"
                )
        self.cfg, self.spec = cfg, spec
        self.host = host_info()
        runtime = cfg["runtime"]
        kwargs = dict(model=spec["id"], revision=spec["revision"], tokenizer_revision=spec["revision"],
                      dtype=cfg["inference"]["dtype"], tensor_parallel_size=spec["tensor_parallel"],
                      max_model_len=spec["max_model_len"],
                      gpu_memory_utilization=runtime["gpu_memory_utilization"],
                      max_num_seqs=runtime["max_num_seqs"], enable_prefix_caching=True,
                      seed=cfg["seed"], trust_remote_code=False)
        kwargs.update(spec.get("vllm_kwargs") or {})
        self.engine = vllm.LLM(**kwargs)
        self.tokenizer = self.engine.get_tokenizer()
        template = getattr(self.tokenizer, "chat_template", None) or type(self.tokenizer).__name__
        self.identity = {"backend": "vllm-offline", "protocol": VERSION, "model": spec,
                         "inference": {k: cfg["inference"][k] for k in PROTOCOL_INFERENCE},
                         "seed": cfg["seed"], "chat_template_hash": digest(str(template)),
                         "versions": {p: version(p) for p in (
                             "vllm", "torch", "transformers", "tokenizers"
                         )}}
        directory = Path(cfg["cache_root"]) / digest(self.identity)[:16]
        atomic_json(directory / "identity.json", self.identity)
        self.cache = ResultCache(directory, cache_name)
        self.totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
                       "context_overflow": 0}

    def encode(self, messages):
        kwargs = {"enable_thinking": False} if self.spec.get("thinking") is False else {}
        return _token_ids(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs))

    def generate(self, requests):
        """Each request is {messages, max_tokens}. Identical requests run once."""
        from vllm import SamplingParams

        keys = [digest({"identity": self.identity, "request": r}) for r in requests]
        pending = {}
        for key, request in zip(keys, requests):
            if self.cache.get(key) is None:
                pending.setdefault(key, request)
        items = list(pending.items())
        chunk = self.cfg["runtime"]["chunk_size"]
        inference = self.cfg["inference"]
        for start in range(0, len(items), chunk):
            live, prompts, params, rows = [], [], [], []
            for key, request in items[start:start + chunk]:
                tokens = self.encode(request["messages"])
                if len(tokens) + request["max_tokens"] > self.spec["max_model_len"]:
                    # Never truncate an input; record the miss for the fit report.
                    rows.append({"key": key, "text": "", "finish_reason": "context_overflow",
                                 "input_tokens": len(tokens), "output_tokens": 0})
                    continue
                live.append(key)
                prompts.append({"prompt_token_ids": tokens})
                params.append(SamplingParams(temperature=inference["temperature"],
                                             top_p=inference["top_p"],
                                             max_tokens=request["max_tokens"], seed=self.cfg["seed"]))
            tick = time.monotonic()
            outputs = self.engine.generate(prompts, sampling_params=params, use_tqdm=False) if prompts else []
            elapsed = time.monotonic() - tick
            for key, output in zip(live, outputs):
                best = output.outputs[0]
                rows.append({"key": key, "text": best.text, "finish_reason": best.finish_reason,
                             "input_tokens": len(output.prompt_token_ids),
                             "output_tokens": len(best.token_ids),
                             "chunk_seconds": round(elapsed, 3), "chunk_requests": len(prompts),
                             "runtime": self.cfg["runtime"], "host": self.host})
            self.cache.put(rows)
            self.totals["requests"] += len(rows)
            self.totals["input_tokens"] += sum(r["input_tokens"] for r in rows)
            self.totals["output_tokens"] += sum(r["output_tokens"] for r in rows)
            self.totals["seconds"] += elapsed
            self.totals["context_overflow"] += sum(r["finish_reason"] == "context_overflow" for r in rows)
            out_tokens = sum(r["output_tokens"] for r in rows)
            print(json.dumps({"event": "chunk", "requests": len(prompts), "seconds": round(elapsed, 1),
                              "output_tokens": out_tokens,
                              "output_tok_per_s": round(out_tokens / elapsed, 1) if elapsed else None}),
                  flush=True)
        return [self.cache.get(k) for k in keys]


class FakeBackend:
    """Plumbing test only; synthetic outputs never enter an actual run root."""
    identity = {"backend": "fake", "protocol": VERSION}

    def __init__(self):
        self.totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
                       "context_overflow": 0}
        self.calls = 0

    def generate(self, requests):
        import re
        out = []
        for request in requests:
            # The task is fixed by the first user turn; later turns are corrections.
            prompt = next(m["content"] for m in request["messages"] if m["role"] == "user")
            if prompt.startswith("SOURCE\n") and "JSON array" in prompt:
                text = json.dumps([int(i) for i in re.findall(r"^\[(\d+)\]", prompt, re.M)])
            elif prompt.startswith("SOURCE\n"):
                # An in-band excerpt of the listed source, so the fill logic is exercised.
                cap = int(re.search(r"at most (\d+) words", prompt).group(1))
                body = re.findall(r"^\[\d+\] \(\d+ words\) (.*)$", prompt, re.M)
                text = " ".join(" ".join(body).split()[:cap])
            elif "CORRECT or INCORRECT" in prompt:
                body = json.loads(prompt[prompt.index("{"):])
                hit = any(g.lower() in body["candidate"].lower() for g in body["acceptable_answers"])
                text = "CORRECT" if hit else "INCORRECT"
            elif prompt.startswith("EVIDENCE\n") and prompt.split("\n", 1)[1].strip().startswith("QUESTION"):
                text = "Insufficient evidence"
            elif prompt.startswith("EVIDENCE\n"):
                text = " ".join(prompt.split()[1:30])
            else:
                text = "I don't know"
            out.append({"key": digest({"identity": self.identity, "request": request}), "text": text,
                        "finish_reason": "stop", "input_tokens": 0, "output_tokens": 0})
        self.calls += 1
        self.totals["requests"] += len(out)
        return out
