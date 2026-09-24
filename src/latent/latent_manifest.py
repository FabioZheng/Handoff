"""Weight, code and runtime identity for reported latent runs and gate reuse."""
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def code_identity():
    paths = sorted(set((ROOT/"src").glob("latent*.py")) |
                   {ROOT/"src/run_latent_reusability.py", ROOT/"tests/selftest_latent_gpu.py"})
    return hashlib.sha256(b"".join(p.name.encode()+b"\0"+p.read_bytes() for p in paths)).hexdigest()


def environment():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    result = {"python":platform.python_version(), "platform":platform.platform(), "packages":dict(sorted(packages.items()))}
    torch = sys.modules.get("torch")
    if torch is not None:
        result["cuda"] = {"build":torch.version.cuda, "devices":[
            {"name":torch.cuda.get_device_properties(i).name,
             "memory":torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]}
    return result


def verify_gate(path, backend_record, kind="gpu"):
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    if gate.get("simulated") or not gate.get("passed"):
        raise ValueError(f"{kind} gate did not pass on a real model")
    if gate.get("backend") != backend_record or gate.get("code_identity") != code_identity():
        raise ValueError(f"{kind} gate belongs to different model/configuration/code")
    # Resolve CUDA identity before comparison even when weights are not loaded.
    __import__("torch")
    if gate.get("environment") != environment():
        raise ValueError(f"{kind} gate belongs to a different runtime environment")
    if kind == "panel_a" and (gate.get("panel") != "A" or gate.get("n_sources", 0) < 20):
        raise ValueError("Panel B requires a complete 20-source Panel A gate")
    return gate
