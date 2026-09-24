"""Resumable evaluator-only cache; frozen bytes are the sole reader input."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ('', 'analysis', 'latent', 'builders')]

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import latent_backend as lb
import latent_handoff as lh
import latent_metrics as lm


class RunStore:
    def __init__(self, path, manifest, seconds=8*3600, benchmark=False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.fingerprint = lh.canonical_hash(manifest)
        target = self.path / "manifest.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
            raise ValueError("output directory belongs to a different run; select a new --out")
        self._json(target, manifest)
        self.deadline = time.monotonic() + seconds
        self.benchmark = benchmark
        self.counts = {"encoding_hits": 0, "encoding_calls": 0, "reader_hits": 0, "reader_calls": 0}
        for directory in ("encoded", "reader", "payloads"):
            (self.path / directory).mkdir(exist_ok=True)
        self._seen = {}
        for name in ("messages", "answers"):
            file = self.path / (name + ".jsonl")
            seen = set()
            if file.exists():
                for line in file.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    seen.add(self.row_key(name, row))
            self._seen[name] = seen

    @staticmethod
    def _json(path, data):
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("experiment time envelope reached; resume from checkpoint")

    def row_key(self, name, row):
        return lh.canonical_hash([name, row.get("panel"), row["packet_id"],
                                 row.get("arm", row["channel"]), row.get("fact_id"), row.get("band_target")])

    def append(self, name, row):
        key = self.row_key(name, row)
        if key in self._seen[name]:
            return
        # One atomic checkpoint per record; JSONL is a final export, so a
        # preemption cannot leave a half-written line masquerading as a result.
        folder = self.path / (name + "_records")
        folder.mkdir(exist_ok=True)
        self._json(folder / (key + ".json"), row)
        self._seen[name].add(key)

    def export(self):
        for name in ("messages", "answers"):
            files = sorted((self.path / (name + "_records")).glob("*.json"))
            target = self.path / (name + ".jsonl")
            temp = target.with_suffix(".jsonl.tmp")
            with temp.open("w", encoding="utf-8") as f:
                for file in files:
                    f.write(json.dumps(json.loads(file.read_text(encoding="utf-8")), ensure_ascii=False)+"\n")
            temp.replace(target)
        self._json(self.path / "execution_counts.json", self.counts)

    def encoding(self, request, operation):
        from run_latent_reusability import Encoded
        self.check_deadline()
        key = lh.canonical_hash([self.fingerprint, request])
        path = self.path / "encoded" / (key + ".npz")
        if path.exists() and not self.benchmark:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
                tr = meta.pop("transcript")
                if tr:
                    tr["usage"] = lb.Usage(**tr["usage"])
                    tr["token_ids"] = tuple(tr["token_ids"])
                    tr = lb.Transcript(**tr, states=data["states"].copy(),
                                       reference_embeddings=data["references"].copy())
                result = Encoded(**meta, positions=lm.PositionAccount(**json.loads(str(data["positions"]))),
                                 cost=lm.CostBreakdown(**json.loads(str(data["cost"]))),
                                 vectors=data["vectors"].copy() if data["vectors"].ndim == 2 else None,
                                 transcript=tr)
            self.counts["encoding_hits"] += 1
            return result
        result = operation()
        self.counts["encoding_calls"] += 1
        meta = {k: v for k,v in vars(result).items() if k not in ("positions","cost","vectors","transcript")}
        tr = result.transcript
        meta["transcript"] = {k: v for k,v in vars(tr).items() if k not in ("states","reference_embeddings")} if tr else None
        if tr:
            meta["transcript"]["usage"] = tr.usage.record()
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as file:
            np.savez_compressed(file, metadata=json.dumps(meta), positions=json.dumps(asdict(result.positions)),
                cost=json.dumps(asdict(result.cost)), vectors=result.vectors if result.vectors is not None else np.array([]),
                states=tr.states if tr else np.array([]),
                references=tr.reference_embeddings if tr and tr.reference_embeddings is not None else np.array([]))
        temp.replace(path)
        return result

    def read(self, backend, payload, operation):
        self.check_deadline()
        wire = lh.to_wire(payload)
        key = lh.canonical_hash([self.fingerprint, payload.payload_sha256, payload.question,
                                 payload.channel, payload.metadata])
        wirepath = self.path / "payloads" / (lh.sha256_bytes(wire)+".bin")
        if not wirepath.exists():
            wirepath.write_bytes(wire)
        target = self.path / "reader" / (key + ".json")
        if target.exists() and not self.benchmark:
            raw = json.loads(target.read_text(encoding="utf-8"))
            self.counts["reader_hits"] += 1
            return lb.Generation(raw["text"], lb.Usage(**raw["usage"]), raw.get("error", ""), True)
        try:
            result = operation()
        except RuntimeError as error:
            result = lb.Generation("", lb.Usage(), error=type(error).__name__ + ": " + str(error))
        self.counts["reader_calls"] += 1
        self._json(target, {"text": result.text, "usage": result.usage.record(), "error": result.error})
        return result
