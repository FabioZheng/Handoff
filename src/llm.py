"""OpenRouter client: on-disk content-addressed cache, retries, token accounting, cost cap.

Design notes tied to the spec's hard constraints:

* C4 — every call returns and logs prompt/completion tokens, latency and cost.
* C5 — every call is cached on disk under ``hash(model, prompt, params, seed)``.
  Reruns are free; the cache is the reproducibility mechanism, not the seed.
* §9 — the API key is read from ``OPENROUTER_API_KEY`` only. It is never written
  to the call log, never included in a cache key, and never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"


class CostCapExceeded(RuntimeError):
    """Raised when the configured hard cost cap would be exceeded."""


class MissingAPIKey(RuntimeError):
    pass


class HardAPIFailure(RuntimeError):
    """A 4xx that is not worth retrying (bad request, auth, not-found)."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool
    model: str
    finish_reason: str = ""
    provider: str = ""
    tag: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CallRecord:
    tag: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool
    provider: str = ""
    finish_reason: str = ""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(model: str, messages: list[dict], params: dict) -> str:
    """Content hash over exactly what determines the response.

    ``params`` includes temperature, top_p, max_tokens, seed and response_format.
    Nothing environment-specific (no key, no timestamps) enters the hash.
    """
    payload = {"model": model, "messages": messages, "params": params}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class CostLedger:
    """Thread-safe spend tracker enforcing the hard cap from config."""

    def __init__(self, cap_usd: float, warn_at_fraction: float = 0.8,
                 api_call_cap: int | None = None) -> None:
        self.cap_usd = float(cap_usd)
        self.warn_at = self.cap_usd * float(warn_at_fraction)
        self._spent = 0.0
        self.api_call_cap = int(api_call_cap) if api_call_cap is not None else None
        self._api_calls_attempted = 0
        self._warned = False
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def check_before_call(self) -> None:
        with self._lock:
            if self._spent >= self.cap_usd:
                raise CostCapExceeded(
                    f"Hard cost cap reached: spent ${self._spent:.4f} of ${self.cap_usd:.2f}. "
                    f"Aborting before issuing further calls. Raise cost.cap_usd in config.yaml "
                    f"to continue; cached work is preserved and will not be re-paid for."
                )
            if self.api_call_cap is not None and self._api_calls_attempted >= self.api_call_cap:
                raise CostCapExceeded(
                    f"Hard API-call cap reached: {self._api_calls_attempted} of "
                    f"{self.api_call_cap}. Cached work is preserved; raise "
                    f"cost.api_call_cap to continue."
                )
            # Reserve the request while holding the lock so concurrent workers
            # cannot collectively overshoot the cap.
            self._api_calls_attempted += 1

    def add(self, rec: CallRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if not rec.cached:
                self._spent += rec.cost_usd
            if not self._warned and self._spent >= self.warn_at:
                self._warned = True
                print(
                    f"[cost] WARNING: ${self._spent:.4f} spent, "
                    f"{100 * self._spent / self.cap_usd:.0f}% of the ${self.cap_usd:.2f} cap."
                )

    def summary(self) -> dict:
        with self._lock:
            recs = list(self.records)
        live = [r for r in recs if not r.cached]
        return {
            "calls_total": len(recs),
            "calls_live": len(live),
            "calls_cached": len(recs) - len(live),
            "cache_hit_rate": (len(recs) - len(live)) / len(recs) if recs else 0.0,
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "prompt_tokens_live": sum(r.prompt_tokens for r in live),
            "completion_tokens_live": sum(r.completion_tokens for r in live),
            "cost_usd": round(self._spent, 6),
            "api_calls_attempted": self._api_calls_attempted,
            "api_call_cap": self.api_call_cap,
        }


class DiskCache:
    """Sharded JSON cache. One file per call, named by its content hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # corrupt entry -> treat as miss and overwrite

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # A unique temporary path avoids collisions with another process using
        # the same shared cache.  Windows can also briefly lock a newly written
        # file (for example while antivirus scans it), so retry the atomic
        # replace instead of losing an otherwise completed API response.
        tmp = p.with_name(
            f"{p.name}.{os.getpid()}.{threading.get_ident()}.{random.getrandbits(32):08x}.tmp"
        )
        with self._lock:
            try:
                tmp.write_text(_canonical(value), encoding="utf-8")
                for attempt in range(6):
                    try:
                        tmp.replace(p)
                        return
                    except PermissionError:
                        if attempt == 5:
                            raise
                        time.sleep(0.05 * (attempt + 1))
            finally:
                tmp.unlink(missing_ok=True)

    def has(self, key: str) -> bool:
        return self._path(key).exists()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for --dry-run only. Never used for accounting."""
    return max(1, int(len(text) / 3.7))


def apply_model_message_controls(messages: list[dict], cfg: dict) -> list[dict]:
    """Return request messages with an explicitly configured model directive.

    Most models need no prompt-level control.  A few, such as Qwen3, expose
    a model-native switch in the chat template; setting ``model.system_suffix``
    records that switch in the actual request and therefore in the cache key.
    The caller's message list is never mutated.
    """
    suffix = str(cfg.get("model", {}).get("system_suffix") or "").strip()
    if not suffix:
        return messages
    adjusted = [dict(message) for message in messages]
    for message in adjusted:
        if message.get("role") == "system":
            content = message.get("content", "")
            if not isinstance(content, str):
                raise TypeError("model.system_suffix requires string system-message content")
            message["content"] = f"{content.rstrip()}\n{suffix}"
            return adjusted
    # Keep a request valid and make the control explicit even for a caller that
    # did not otherwise need a system prompt.
    return [{"role": "system", "content": suffix}, *adjusted]


class LLMClient:
    def __init__(self, cfg: dict, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.model = cfg["model"]["id"]
        self.dry_run = dry_run
        rt = cfg["runtime"]
        self.cache = DiskCache(rt["cache_dir"])
        self.ledger = CostLedger(
            cfg["cost"]["cap_usd"], cfg["cost"]["warn_at_fraction"],
            cfg["cost"].get("api_call_cap"),
        )
        self.max_retries = rt["max_retries"]
        self.backoff_base = rt["backoff_base_s"]
        self.backoff_max = rt["backoff_max_s"]
        self.timeout = rt["request_timeout_s"]
        self._session_lock = threading.Lock()
        self._local = threading.local()
        self._pricing: dict[str, float] | None = None
        self._dry_run_plan: list[dict] = []

        self._api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self._api_key and not dry_run:
            raise MissingAPIKey(
                "OPENROUTER_API_KEY is not set. Export it before running "
                "(the probe never reads a key from any file or argument)."
            )

    # ---------- pricing ----------

    def pricing(self) -> dict[str, float]:
        """Per-token prices for the configured model, used as a cost fallback and
        for --dry-run estimation. OpenRouter's own reported cost wins when present."""
        if self._pricing is None:
            try:
                r = requests.get(MODELS_URL, timeout=60)
                r.raise_for_status()
                for m in r.json()["data"]:
                    if m["id"] == self.model:
                        self._pricing = {
                            "prompt": float(m["pricing"]["prompt"]),
                            "completion": float(m["pricing"]["completion"]),
                        }
                        break
            except Exception as exc:  # pricing is advisory; never fatal
                print(f"[cost] could not fetch pricing ({exc}); falling back to 0.0")
            if self._pricing is None:
                self._pricing = {"prompt": 0.0, "completion": 0.0}
        return self._pricing

    # ---------- http ----------

    @property
    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "handoff-probe",
                }
            )
            self._local.session = s
        return s

    def _build_body(self, messages: list[dict], params: dict) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_tokens": params["max_tokens"],
            # Ask OpenRouter to report the true charged cost for this call (C4).
            "usage": {"include": True},
        }
        if params.get("reasoning") is not None:
            body["reasoning"] = params["reasoning"]
        if params.get("seed") is not None:
            body["seed"] = params["seed"]
        if params.get("response_format") is not None:
            body["response_format"] = params["response_format"]

        prov: dict[str, Any] = {}
        if self.cfg["model"].get("provider_order"):
            prov["order"] = self.cfg["model"]["provider_order"]
        if self.cfg["model"].get("require_parameters"):
            prov["require_parameters"] = True
        if prov:
            body["provider"] = prov
        return body

    def _post_with_retries(self, body: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.ledger.check_before_call()
            try:
                t0 = time.perf_counter()
                resp = self._session.post(OPENROUTER_URL, json=body, timeout=self.timeout)
                latency = time.perf_counter() - t0
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                data = resp.json()
                # OpenRouter can return a 200 whose body carries an upstream error.
                if "error" in data and not data.get("choices"):
                    msg = str(data["error"])[:300]
                    last_exc = RuntimeError(f"upstream error in 200 body: {msg}")
                    self._sleep_backoff(attempt)
                    continue
                data["_latency_s"] = latency
                return data

            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("Retry-After")
                self._sleep_backoff(attempt, retry_after)
                continue

            # Any other 4xx: hard fail, do not burn retries or money.
            raise HardAPIFailure(f"HTTP {resp.status_code}: {resp.text[:500]}")

        raise RuntimeError(f"exhausted {self.max_retries} retries; last error: {last_exc}")

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), self.backoff_max))
                return
            except ValueError:
                pass
        delay = min(self.backoff_base * (2**attempt), self.backoff_max)
        time.sleep(delay * (0.5 + random.random()))  # full-ish jitter

    # ---------- public ----------

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        response_format: dict | None = None,
        tag: str = "",
    ) -> LLMResult:
        messages = apply_model_message_controls(messages, self.cfg)
        params = {
            "temperature": float(temperature),
            "top_p": float(self.cfg["decoding"]["top_p"]),
            "max_tokens": int(max_tokens),
            "seed": seed,
            "response_format": response_format,
            # Model-specific only when configured.  It is in the cache key so
            # thinking/non-thinking outputs can never collide.
            "reasoning": self.cfg["model"].get("reasoning"),
        }
        key = cache_key(self.model, messages, params)

        hit = self.cache.get(key)
        if hit is not None:
            res = LLMResult(
                text=hit["text"],
                prompt_tokens=hit["prompt_tokens"],
                completion_tokens=hit["completion_tokens"],
                cost_usd=hit["cost_usd"],
                latency_s=hit["latency_s"],
                cached=True,
                model=hit["model"],
                finish_reason=hit.get("finish_reason", ""),
                provider=hit.get("provider", ""),
                tag=tag,
            )
            self.ledger.add(CallRecord(tag, res.model, res.prompt_tokens, res.completion_tokens,
                                       res.cost_usd, res.latency_s, True, res.provider,
                                       res.finish_reason))
            return res

        if self.dry_run:
            p_est = sum(estimate_tokens(m["content"]) for m in messages)
            self._dry_run_plan.append(
                {"tag": tag, "prompt_tokens_est": p_est, "completion_tokens_est": max_tokens}
            )
            return LLMResult("", p_est, max_tokens, 0.0, 0.0, False, self.model, tag=tag)

        body = self._build_body(messages, params)
        data = self._post_with_retries(body)

        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens", 0))
        c_tok = int(usage.get("completion_tokens", 0))

        cost = usage.get("cost")
        if cost is None:
            pr = self.pricing()
            cost = p_tok * pr["prompt"] + c_tok * pr["completion"]
        cost = float(cost)

        res = LLMResult(
            text=text,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
            latency_s=float(data.get("_latency_s", 0.0)),
            cached=False,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
            provider=data.get("provider", ""),
            tag=tag,
        )
        self.cache.put(key, {k: v for k, v in asdict(res).items() if k not in ("cached", "tag")})
        self.ledger.add(CallRecord(tag, res.model, p_tok, c_tok, cost, res.latency_s, False,
                                   res.provider, res.finish_reason))
        return res

    # ---------- dry run ----------

    def dry_run_report(self) -> dict:
        pr = self.pricing()
        n = len(self._dry_run_plan)
        p_tok = sum(x["prompt_tokens_est"] for x in self._dry_run_plan)
        # Completions almost never hit max_tokens; halve for a realistic mid estimate.
        c_tok_max = sum(x["completion_tokens_est"] for x in self._dry_run_plan)
        by_tag: dict[str, int] = {}
        for x in self._dry_run_plan:
            by_tag[x["tag"]] = by_tag.get(x["tag"], 0) + 1
        return {
            "uncached_calls": n,
            "cached_calls": self.ledger.summary()["calls_cached"],
            "calls_by_tag": dict(sorted(by_tag.items())),
            "prompt_tokens_est": p_tok,
            "completion_tokens_est_max": c_tok_max,
            "cost_usd_est_low": round(p_tok * pr["prompt"] + 0.35 * c_tok_max * pr["completion"], 4),
            "cost_usd_est_high": round(p_tok * pr["prompt"] + c_tok_max * pr["completion"], 4),
            "cap_usd": self.cfg["cost"]["cap_usd"],
        }


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
