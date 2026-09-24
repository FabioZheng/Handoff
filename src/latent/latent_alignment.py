"""Auditable StateBridge-derived alignment, independent of model loading.

Algorithm reference: Yanwen Peng et al., StateBridge (2026), section 3.3;
https://github.com/YanwenPneg/StateBridge at
3f6bf5442c6e8848555a6132516e6d36f35444fb (Apache-2.0).
This is an independent implementation of the published transformations, not
the upstream multi-agent pipeline. We reject unpaired states rather than
silently truncating them and chunk vocabulary search to bound memory.
"""
from dataclasses import asdict, dataclass
import math

VERSION = "statebridge-whiten-procrustes-anchor-v1"
NORM_VERSION = "norm-calibrated-final-layer-v1"


@dataclass(frozen=True)
class AlignmentConfig:
    regularization: float = 1e-3
    eigen_floor: float = 1e-6
    snap_ratio: float = 0.3
    vocab_chunk: int = 8192

    def __post_init__(self):
        if not (math.isfinite(self.regularization) and self.regularization > 0):
            raise ValueError("alignment regularization must be positive")
        if not (math.isfinite(self.eigen_floor) and self.eigen_floor > 0):
            raise ValueError("eigen floor must be positive")
        if not 0 <= self.snap_ratio <= 1 or self.vocab_chunk < 1:
            raise ValueError("invalid anchoring parameters")


def statebridge_align(states, references, vocabulary, config=None):
    """Paired K x d predicting states and decoded-token reference embeddings.

    Returns aligned rows and evaluator-only diagnostics. Torch is lazy so
    transport and fake-backend tests need no torch installation. All linear
    algebra uses float32 on the tensor's device. No learned parameters.
    """
    import torch
    cfg = config or AlignmentConfig()
    if states.ndim != 2 or references.shape != states.shape:
        raise ValueError("states and decoded-token references must pair exactly")
    if vocabulary.ndim != 2 or vocabulary.shape[1] != states.shape[1]:
        raise ValueError("vocabulary width differs from states")
    if not all(torch.isfinite(x).all().item() for x in (states, references)):
        raise ValueError("nonfinite alignment input")
    h, e = states.float(), references.float()
    info = {"version": VERSION, **asdict(cfg), "pairs": len(h),
            "short_fallback": len(h) < 2}
    if len(h) == 0:
        return h.clone(), info

    def power(cov, exponent):
        eig, basis = torch.linalg.eigh((cov + cov.T) / 2)
        return (basis * eig.clamp_min(cfg.eigen_floor).pow(exponent)) @ basis.T

    if len(h) < 2:
        # Match the reference release's explicit small-sample fallback. Record
        # it: this is not a successfully estimated Procrustes transformation.
        return h.clone(), info
    mh, me = h.mean(0, keepdim=True), e.mean(0, keepdim=True)
    hc, ec = h - mh, e - me
    ridge = torch.eye(h.shape[1], dtype=h.dtype, device=h.device) * cfg.regularization
    ch, ce = hc.T @ hc / len(h) + ridge, ec.T @ ec / len(h) + ridge
    wh, we = hc @ power(ch, -0.5), ec @ power(ce, -0.5)
    left, singular, right = torch.linalg.svd(wh.T @ we, full_matrices=False)
    signs = torch.ones(h.shape[1], device=h.device)
    signs[-1] = torch.linalg.slogdet(left @ right).sign
    rotation = (left * signs) @ right
    z = wh @ rotation @ power(ce, 0.5) + me
    # Mean per-token vocabulary norm, not RMS of the full embedding table.
    total_norm = torch.zeros((), device=h.device)
    for chunk in vocabulary.split(cfg.vocab_chunk):
        total_norm += chunk.float().norm(dim=1).sum()
    target_norm = total_norm / len(vocabulary)
    z = z * (target_norm / z.norm(dim=1, keepdim=True).clamp_min(1e-6))
    if cfg.snap_ratio:
        unit = torch.nn.functional.normalize(z, dim=1)
        scores = torch.full((len(z),), -torch.inf, device=h.device)
        indices = torch.zeros(len(z), dtype=torch.long, device=h.device)
        for start in range(0, len(vocabulary), cfg.vocab_chunk):
            chunk = vocabulary[start:start + cfg.vocab_chunk].float()
            values, ids = (unit @ torch.nn.functional.normalize(chunk, dim=1).T).max(1)
            better = values > scores  # stable first-index ties across chunks
            indices = torch.where(better, ids + start, indices)
            scores = torch.maximum(scores, values)
        z = z * (1 - cfg.snap_ratio) + vocabulary[indices].float() * cfg.snap_ratio
    if not torch.isfinite(z).all():
        raise ValueError("alignment produced nonfinite output")
    info.update(target_norm=float(target_norm),
                effective_rank=int((singular > cfg.eigen_floor).sum()),
                reference_rmse=float((z - e).square().mean().sqrt()))
    return z.contiguous(), info
