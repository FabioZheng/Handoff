"""Model registry, multi-model client pool, and schedule expansion.

Every other runner in this project drives a single ``LLMClient`` pinned to one
model. Experiment 8 needs a *different* model at each stage of a chain, so this
module builds one client per registry entry and then hands them all the same
``DiskCache`` and the same ``CostLedger``. That matters:

* one cache -- the model id is already inside ``llm.cache_key``, so per-model
  clients can never collide, and two arms whose schedules share a prefix reuse
  the identical cached generation instead of paying twice for a call that is
  deterministic anyway;
* one ledger -- the configured ``cost.cap_usd`` stays a cap on the *experiment*
  rather than becoming eight independent caps that together allow eight times
  the spend.

Nothing here changes ``llm.py``. Per-model controls that already exist there
(``model.reasoning``, ``model.system_suffix``) are simply carried through from
the registry, so Qwen3's ``/no_think`` directive is applied and hashed exactly
as ``qwen32_chain_config.yaml`` already applies it.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import MODELS_URL, LLMClient  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    """One entry of the registry, plus everything reproducibility needs.

    ``parameters_b`` is whatever the vendor published, and ``parameters_basis``
    says what kind of number that is -- a reported total, or a rounded headline
    label. It is deliberately possible for ``parameters_b`` to be ``None``: a
    model with an undisclosed size is recorded as unknown rather than given an
    invented figure, and the plots fall back to the tier's median size with the
    marker drawn hollow.
    """

    key: str
    id: str
    family: str
    tier: str
    short_code: str
    parameters_b: float | None
    parameters_basis: str
    parameters_source: str
    released: str
    architecture: str
    reasoning: dict | None = None
    system_suffix: str | None = None

    @property
    def size_known(self) -> bool:
        return self.parameters_b is not None

    def to_json(self) -> dict:
        return {
            "model_key": self.key,
            "model_id": self.id,
            "family": self.family,
            "tier": self.tier,
            "short_code": self.short_code,
            "parameters_b": self.parameters_b,
            "parameters_basis": self.parameters_basis,
            "parameters_source": self.parameters_source,
            "released": self.released,
            "architecture": self.architecture,
            "reasoning": self.reasoning,
            "system_suffix": self.system_suffix,
        }


def load_registry(cfg: dict) -> dict[str, ModelSpec]:
    """Build the registry from ``models:``, validating what the design assumes."""
    families = set(cfg["families"])
    tiers = set(cfg["tiers"])
    registry: dict[str, ModelSpec] = {}
    for key, spec in cfg["models"].items():
        if spec["family"] not in families:
            raise ValueError(f"{key}: family {spec['family']!r} is not declared under families:")
        if spec["tier"] not in tiers:
            raise ValueError(f"{key}: tier {spec['tier']!r} is not declared under tiers:")
        size = spec.get("parameters_b")
        registry[key] = ModelSpec(
            key=key,
            id=spec["id"],
            family=spec["family"],
            tier=spec["tier"],
            short_code=spec["short_code"],
            parameters_b=None if size is None else float(size),
            parameters_basis=str(spec.get("parameters_basis", "undisclosed")),
            parameters_source=str(spec.get("parameters_source", "")),
            released=str(spec.get("released", "")),
            architecture=str(spec.get("architecture", "unknown")),
            reasoning=spec.get("reasoning"),
            system_suffix=spec.get("system_suffix"),
        )
    for family in cfg["family_cycle"]:
        for tier in cfg["tiers"]:
            if not any(m.family == family and m.tier == tier for m in registry.values()):
                raise ValueError(
                    f"family_cycle names {family!r} but the registry has no {tier} entry for it; "
                    "every cycled family needs one model per tier or the schedules cannot be built"
                )
    return registry


def by_family_tier(registry: dict[str, ModelSpec]) -> dict[tuple[str, str], ModelSpec]:
    index: dict[tuple[str, str], ModelSpec] = {}
    for spec in registry.values():
        slot = (spec.family, spec.tier)
        if slot in index:
            raise ValueError(f"two registry entries claim {slot}: {index[slot].key} and {spec.key}")
        index[slot] = spec
    return index


# ---------------------------------------------------------------- catalogue

def catalogue_snapshot(model_ids: list[str], timeout: int = 90) -> dict:
    """Live OpenRouter metadata for the models actually used, for the record.

    Price, context window and the provider's own creation timestamp are the
    only size-adjacent facts OpenRouter publishes; none of them is a parameter
    count, so they are stored as *metadata*, never substituted for one. A
    failure here is advisory: the run continues with an empty snapshot rather
    than aborting an experiment over a metadata fetch.
    """
    wanted = set(model_ids)
    try:
        response = requests.get(MODELS_URL, timeout=timeout)
        response.raise_for_status()
        entries = response.json()["data"]
    except Exception as exc:  # network/metadata only -- never fatal
        print(f"[pool:catalogue] could not fetch OpenRouter model metadata ({exc})")
        return {"fetched": False, "models": {}}
    snapshot = {}
    for entry in entries:
        if entry["id"] not in wanted:
            continue
        pricing = entry.get("pricing", {})
        snapshot[entry["id"]] = {
            "openrouter_name": entry.get("name", ""),
            "context_length": entry.get("context_length"),
            "created": entry.get("created"),
            "prompt_usd_per_mtok": round(float(pricing.get("prompt", 0.0)) * 1e6, 6),
            "completion_usd_per_mtok": round(float(pricing.get("completion", 0.0)) * 1e6, 6),
            "tokenizer": (entry.get("architecture") or {}).get("tokenizer", ""),
            "modality": (entry.get("architecture") or {}).get("modality", ""),
        }
    missing = sorted(wanted - set(snapshot))
    if missing:
        print(f"[pool:catalogue] WARNING not listed by OpenRouter right now: {', '.join(missing)}")
    return {"fetched": True, "models": snapshot, "missing": missing}


# ---------------------------------------------------------------- the pool

class ModelPool:
    """One ``LLMClient`` per registry entry over a shared cache and ledger."""

    def __init__(self, cfg: dict, registry: dict[str, ModelSpec], dry_run: bool = False) -> None:
        self.registry = registry
        self.clients: dict[str, LLMClient] = {}
        self.ledger = None
        cache = None
        for key, spec in registry.items():
            model_cfg = copy.deepcopy(cfg)
            model_cfg["model"] = {
                "id": spec.id,
                "provider_order": cfg.get("model", {}).get("provider_order"),
                "require_parameters": cfg.get("model", {}).get("require_parameters", False),
            }
            if spec.reasoning is not None:
                model_cfg["model"]["reasoning"] = spec.reasoning
            if spec.system_suffix:
                model_cfg["model"]["system_suffix"] = spec.system_suffix
            client = LLMClient(model_cfg, dry_run=dry_run)
            if self.ledger is None:
                self.ledger, cache = client.ledger, client.cache
            else:
                client.ledger, client.cache = self.ledger, cache
            self.clients[key] = client

    def client(self, model_key: str) -> LLMClient:
        try:
            return self.clients[model_key]
        except KeyError:
            raise KeyError(f"no client for model key {model_key!r}; registry holds {sorted(self.clients)}")

    def spec(self, model_key: str) -> ModelSpec:
        return self.registry[model_key]


# ---------------------------------------------------------------- schedules

@dataclass(frozen=True)
class ArmSpec:
    name: str
    block: str
    family_mode: str
    size_pattern: str
    family: str | None = None
    derived_template: str | None = None

    @property
    def is_derived(self) -> bool:
        return self.derived_template is not None


FAMILY_MODES = ("fixed_named", "start", "cycle_forward", "cycle_reverse")


def load_arms(cfg: dict) -> tuple[list[ArmSpec], list[ArmSpec]]:
    """Return (generated arms, derived arms). Derived arms issue no calls."""
    generated = []
    for entry in cfg["arms"]:
        mode = entry["family_mode"]
        if mode not in FAMILY_MODES:
            raise ValueError(f"{entry['name']}: unknown family_mode {mode!r}")
        if mode == "fixed_named" and not entry.get("family"):
            raise ValueError(f"{entry['name']}: family_mode fixed_named needs a family")
        if entry["size_pattern"] not in cfg["size_patterns"]:
            raise ValueError(f"{entry['name']}: unknown size_pattern {entry['size_pattern']!r}")
        generated.append(ArmSpec(
            name=entry["name"], block=entry["block"], family_mode=mode,
            size_pattern=entry["size_pattern"], family=entry.get("family"),
        ))
    derived = []
    for entry in cfg.get("derived_arms") or []:
        derived.append(ArmSpec(
            name=entry["name"], block=entry["block"], family_mode="start",
            size_pattern=entry["size_pattern"], derived_template=entry["template"],
        ))
    names = [a.name for a in generated + derived]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique across arms: and derived_arms:")
    return generated, derived


def start_family(pair_index: int, family_cycle: list[str]) -> str:
    """Stratified start family: with n a multiple of the cycle length, every
    family starts the same number of pairs, so a cross-family arm and its
    start-matched homogeneous control are balanced over start models."""
    return family_cycle[pair_index % len(family_cycle)]


def family_at(arm: ArmSpec, stage: int, pair_index: int, family_cycle: list[str]) -> str:
    size = len(family_cycle)
    start = pair_index % size
    if arm.family_mode == "fixed_named":
        return arm.family
    if arm.family_mode == "start":
        return family_cycle[start]
    if arm.family_mode == "cycle_forward":
        return family_cycle[(start + stage - 1) % size]
    return family_cycle[(start - stage + 1) % size]


def schedule_for(arm: ArmSpec, pair_index: int, cfg: dict,
                 index: dict[tuple[str, str], ModelSpec]) -> list[str]:
    """Model keys for stages 1..max_depth of one (arm, pair) chain."""
    tiers = cfg["size_patterns"][arm.size_pattern]
    max_depth = int(cfg["max_depth"])
    if len(tiers) < max_depth:
        raise ValueError(f"size_pattern {arm.size_pattern!r} has {len(tiers)} entries, needs {max_depth}")
    cycle = cfg["family_cycle"]
    keys = []
    for stage in range(1, max_depth + 1):
        family = family_at(arm, stage, pair_index, cycle)
        keys.append(index[(family, tiers[stage - 1])].key)
    return keys


def resolve_derived(arm: ArmSpec, pair_index: int, cfg: dict) -> str:
    """The generated arm whose rows a derived arm reuses for this pair."""
    return arm.derived_template.format(family=start_family(pair_index, cfg["family_cycle"]))


# ---------------------------------------------------------------- transitions

def transition_type(previous: ModelSpec | None, current: ModelSpec) -> dict:
    """Describe the edge that produced ``current``.

    Stage 1 has no predecessor model -- its input is the source passages, not
    another model's notes -- so it is labelled ``source`` and excluded from
    every model-transition comparison rather than silently counted as
    "same family".
    """
    if previous is None:
        return {
            "transition": "source",
            "family_changed": None,
            "size_direction": "none",
            "size_ratio": None,
            "log_size_change": None,
            "transition_type": "source_to_first_agent",
        }
    family_changed = previous.family != current.family
    if previous.parameters_b is None or current.parameters_b is None:
        # An undisclosed size makes the *direction* unknowable. Saying so is the
        # only honest option; guessing from price or tier would put a fabricated
        # number into the size analysis.
        direction, ratio, log_change = "unknown", None, None
    else:
        ratio = round(current.parameters_b / previous.parameters_b, 4)
        log_change = round(math.log10(current.parameters_b) - math.log10(previous.parameters_b), 4)
        # A 15% band keeps near-identical counts (8.0B vs 8.2B) from being
        # reported as a size change; the tier labels agree with it by design.
        if current.parameters_b > previous.parameters_b * 1.15:
            direction = "larger"
        elif current.parameters_b < previous.parameters_b / 1.15:
            direction = "smaller"
        else:
            direction = "same"
    return {
        "transition": f"{previous.key}->{current.key}",
        "family_changed": bool(family_changed),
        "size_direction": direction,
        "size_ratio": ratio,
        "log_size_change": log_change,
        "transition_type": ("cross_family" if family_changed else "same_family") + f"_size_{direction}",
    }


def chain_summary(model_keys: list[str], registry: dict[str, ModelSpec]) -> dict:
    """Aggregate description of a chain prefix, for the per-answer records."""
    specs = [registry[k] for k in model_keys]
    families = [s.family for s in specs]
    sizes = [s.parameters_b for s in specs if s.parameters_b is not None]
    transitions = sum(1 for a, b in zip(families, families[1:]) if a != b)
    tier_changes = sum(1 for a, b in zip(specs, specs[1:]) if a.tier != b.tier)
    return {
        "chain_models": "|".join(s.key for s in specs),
        "chain_model_ids": "|".join(s.id for s in specs),
        "chain_short_codes": "-".join(s.short_code for s in specs),
        "chain_families": "|".join(families),
        "n_stages": len(specs),
        "n_distinct_families": len(set(families)),
        "n_family_transitions": transitions,
        "n_tier_transitions": tier_changes,
        "homogeneous_family": len(set(families)) == 1,
        "homogeneous_model": len(set(s.key for s in specs)) == 1,
        "mean_parameters_b": round(sum(sizes) / len(sizes), 3) if sizes else None,
        "final_parameters_b": specs[-1].parameters_b,
        "first_parameters_b": specs[0].parameters_b,
        "n_large_stages": sum(1 for s in specs if s.tier == "large"),
    }


def marker_area(parameters_b: float | None, max_parameters_b: float, scale: dict) -> float:
    """Marker *area* proportional to parameter count.

    Matplotlib's ``s`` is an area in points squared, so passing a size-derived
    number straight into it is already an area encoding -- which is the one
    that reads correctly. ``sqrt_area`` compresses the range for pools that
    span more than about an order of magnitude; ``min_area`` keeps the smallest
    model legible either way. A model of unknown size gets the floor and is
    drawn hollow by the caller.
    """
    min_area = float(scale.get("min_area", 90.0))
    max_area = float(scale.get("max_area", 900.0))
    if parameters_b is None or max_parameters_b <= 0:
        return min_area
    fraction = float(parameters_b) / float(max_parameters_b)
    if str(scale.get("mode", "linear_area")) == "sqrt_area":
        fraction = fraction ** 0.5
    return max(min_area, max_area * fraction)
