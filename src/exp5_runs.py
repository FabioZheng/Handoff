"""Which stored run a later Experiment 5 stage builds on.

Every later stage (cross reader, repeated handoff, allocation judge, analyses)
finds its parent under the first 12 characters of the protocol hash. The main
run was written before the shared modules and data folders got their current
names. Those renames change the hash and nothing the run does: the code at the
commit before them reproduces the stored hash exactly, and the diff from there
touches only module names, the data and cache paths, and docstrings.

So code and config that hash to exactly the renamed protocol read the stored run
under its stored hash. Any other change to the protocol code or config gives a
new hash, and the later stages refuse the stored run as before.
"""
import run_exp5_capped as rc

RENAMED = {  # hash of the renamed code and config -> hash stored in the run's manifest
    "9f0e103c5524d86282c9657127f923e4e209a0912214baf1beef051f7dd9e92d":
        "b58e4132a923d6950a6c150ee8a119a7f848797a2b456a1b7f7871e315757bab",
}


def parent_signature(cfg: dict) -> str:
    """Protocol hash of the stored run these code and config continue."""
    signature = rc.protocol_hash(cfg)
    return RENAMED.get(signature, signature)
