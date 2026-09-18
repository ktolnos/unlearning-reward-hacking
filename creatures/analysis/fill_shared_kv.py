"""Add the KV-shared tensors that transformers omits and vLLM demands.

Gemma 4 shares KV across its last `num_kv_shared_layers` layers. Transformers never
instantiates `k_proj`, `v_proj` or `k_norm` for those layers, so a checkpoint it saves is
60 tensors short of the released weights. vLLM's Gemma 4 builds a fused `qkv_proj` for
every layer and refuses to load with

    ValueError: Following weights were not initialized from checkpoint: {...k_norm...}

which is a hard raise in `default_loader.py` with no flag to relax it. So no E2B checkpoint
has ever been loadable by the eval probe.

Copying these from the base model is exact rather than approximate. On a KV-shared layer
vLLM computes the fused projection and then uses only Q: `k` and `v` go to an `Attention`
built with `kv_sharing_target_layer_name`, which reads the target layer's cache, and
`k_norm` is applied only in the non-shared branch. The values cannot reach the output, but
the tensors have to exist for the fused loader to find all its pieces.

    python -m creatures.analysis.fill_shared_kv BASE_MODEL RUN_DIR [RUN_DIR ...]

The base model is named rather than read off the checkpoint, whose `_name_or_path` is
empty; it is checked against the checkpoint config so a mismatched donor fails here
instead of producing a model that loads and is wrong.

Idempotent: a checkpoint that already has every base key is left alone.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers.utils import cached_file


def keys_of(path):
    """Tensor names in a safetensors file, read from its header alone."""
    with open(path, "rb") as fh:
        length = struct.unpack("<Q", fh.read(8))[0]
        return set(json.loads(fh.read(length))) - {"__metadata__"}


def same_architecture(ckpt, base_model):
    """Whether the donor really is this checkpoint's base model."""
    a = json.loads((ckpt / "config.json").read_text())
    b = json.loads(Path(cached_file(base_model, "config.json")).read_text())
    keep = ("architectures", "num_hidden_layers", "hidden_size", "num_kv_shared_layers")
    pick = lambda c: {k: (c.get(k) or c.get("text_config", {}).get(k)) for k in keep}
    return pick(a) == pick(b), pick(a), pick(b)


def fill(ckpt, base_model):
    shards = sorted(ckpt.glob("*.safetensors"))
    if not shards:
        return f"{ckpt.name}: no safetensors"
    if len(shards) > 1:
        return f"{ckpt.name}: {len(shards)} shards, not handled"
    ok, got, want = same_architecture(ckpt, base_model)
    if not ok:
        raise SystemExit(f"{ckpt} is {got}, but {base_model} is {want}")
    have = keys_of(shards[0])
    base = Path(cached_file(base_model, "model.safetensors"))
    missing = keys_of(base) - have
    if not missing:
        return f"{ckpt.name}: already complete"

    tensors = load_file(shards[0])
    donor = load_file(base)
    for name in missing:
        tensors[name] = donor[name]
    del donor

    tmp = shards[0].with_suffix(".safetensors.tmp")
    save_file(tensors, tmp, metadata={"format": "pt"})
    shutil.move(tmp, shards[0])
    return f"{ckpt.name}: added {len(missing)} tensors"


def main():
    """Fill each argument, which may be a checkpoint or a directory of checkpoints.

    Taking only the directory-of-checkpoints case is how this silently did nothing to
    every repaired checkpoint: a repair writes one checkpoint per directory with no
    `checkpoint-N` children, so the target list came out empty and the job printed
    nothing before the eval died on the missing tensors.
    """
    base_model, *runs = sys.argv[1:]
    for run in runs:
        run = Path(run)
        if (run / "config.json").exists():
            targets = [run]
        else:
            targets = sorted(p for p in run.iterdir()
                             if p.is_dir() and (p / "config.json").exists())
        if not targets:
            print(f"{run}: no checkpoint here and no checkpoint under it", flush=True)
        for ckpt in targets:
            print(fill(ckpt, base_model), flush=True)


if __name__ == "__main__":
    main()
