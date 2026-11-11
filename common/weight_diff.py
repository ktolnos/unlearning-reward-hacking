"""Exact, compact copies of repaired weights: their bitwise difference from the anchor.

A repair moves few bf16 weights across a rounding boundary (an undo leaves 83-87% of
them exactly at the anchor's value, docs/EXACT_UNDO.md), so the XOR of a repaired
tensor's bits with the anchor's is mostly zero bytes and compresses far below the
weights' size. Restoring needs the anchor checkpoint, which is kept anyway, and gives
back the repaired tensors bit for bit: each tensor's SHA-256 is stored and checked.

    python -m common.weight_diff save <model dir> <anchor dir> <diff file>
    python -m common.weight_diff restore <diff file> <anchor dir> <model dir>

`save` stores every *.safetensors tensor of <model dir> (XOR against the anchor's
tensor of the same name, dtype and shape; the raw bytes when the anchor has none) and
every other file of the directory verbatim, so a restored directory is the saved one:
config, tokenizer, repair_state.json and all. Each tensor's bytes are split into byte
planes (all first bytes, then all second bytes, ...) before zlib, which keeps a bf16
XOR's zero high bytes in one long run. `save` decompresses every blob again and checks
it before the file is written, then writes it atomically.
"""

import hashlib
import json
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAGIC = b"URHWDIFF1\n"
LEVEL = 6
THREADS = int(os.environ.get("WDIFF_THREADS", "16"))


def _tensor_files(d):
    """{tensor name: file} over the directory's *.safetensors files."""
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def _bytes(t):
    return t.contiguous().view(-1).view(torch.uint8).numpy()


def _planes(b, k):
    return b.reshape(-1, k).T.copy() if k > 1 else b


def _unplanes(p, k, n):
    return p.reshape(k, n // k).T.reshape(-1) if k > 1 else p


def save(model_dir, anchor_dir, diff_file):
    model_dir, diff_file = Path(model_dir), Path(diff_file)
    anchor = _tensor_files(anchor_dir)
    entries, blobs = [], []
    handles = {}

    def anchor_tensor(name):
        f = anchor.get(name)
        if f is None:
            return None
        if f not in handles:
            handles[f] = safe_open(f, framework="pt")
        return handles[f].get_tensor(name)

    jobs = []
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            meta = h.metadata()
            names = list(h.keys())
            for name in names:
                t = h.get_tensor(name)
                a = anchor_tensor(name)
                same = a is not None and a.dtype == t.dtype and a.shape == t.shape
                k = t.element_size()
                nb = _bytes(t)
                digest = hashlib.sha256(nb.tobytes()).hexdigest()
                x = np.bitwise_xor(nb, _bytes(a)) if same else nb
                jobs.append((dict(name=name, file=f.name, dtype=str(t.dtype).replace("torch.", ""),
                                  shape=list(t.shape), kind="xor" if same else "raw",
                                  itemsize=k, nbytes=int(nb.size), sha256=digest,
                                  file_meta=meta), _planes(x, k)))
    # compress in threads (zlib releases the GIL), then check every blob decompresses to
    # what was compressed before anything is written
    with ThreadPoolExecutor(THREADS) as ex:
        comp = list(ex.map(lambda j: zlib.compress(j[1].tobytes(), LEVEL), jobs))
    for (e, p), c in zip(jobs, comp):
        if zlib.decompress(c) != p.tobytes():
            raise SystemExit(f"{e['name']}: compressed blob does not round-trip")
        e["blob"] = len(c)
        entries.append(e)
        blobs.append(c)
    files = []
    for f in sorted(model_dir.iterdir()):
        if f.is_file() and f.suffix != ".safetensors":
            c = zlib.compress(f.read_bytes(), LEVEL)
            files.append(dict(path=f.name, blob=len(c)))
            blobs.append(c)
    header = json.dumps(dict(tensors=entries, files=files)).encode()
    tmp = diff_file.with_name(f"{diff_file.name}.{os.getpid()}.tmp")
    diff_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        for c in blobs:
            out.write(c)
    tmp.replace(diff_file)
    raw = sum(e["nbytes"] for e in entries)
    print(f"saved {len(entries)} tensors ({sum(e['kind'] == 'xor' for e in entries)} against "
          f"the anchor) and {len(files)} files: {raw / 1e9:.2f} GB of weights in "
          f"{diff_file.stat().st_size / 1e9:.2f} GB -> {diff_file}", flush=True)


def _read(diff_file):
    with open(diff_file, "rb") as h:
        if h.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{diff_file} is not a weight diff")
        (n,) = struct.unpack("<Q", h.read(8))
        header = json.loads(h.read(n))
        for e in header["tensors"]:
            yield "tensor", e, h.read(e["blob"])
        for e in header["files"]:
            yield "file", e, h.read(e["blob"])


def restore(diff_file, anchor_dir, model_dir):
    model_dir = Path(model_dir)
    tmp_dir = model_dir.with_name(f"{model_dir.name}.{os.getpid()}.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=False)
    anchor = _tensor_files(anchor_dir)
    handles = {}
    by_file, metas = {}, {}
    for kind, e, blob in _read(diff_file):
        if kind == "file":
            (tmp_dir / e["path"]).write_bytes(zlib.decompress(blob))
            continue
        k, n = e["itemsize"], e["nbytes"]
        b = _unplanes(np.frombuffer(zlib.decompress(blob), dtype=np.uint8), k, n)
        if e["kind"] == "xor":
            f = anchor[e["name"]]
            if f not in handles:
                handles[f] = safe_open(f, framework="pt")
            b = np.bitwise_xor(b, _bytes(handles[f].get_tensor(e["name"])))
        if hashlib.sha256(b.tobytes()).hexdigest() != e["sha256"]:
            raise SystemExit(f"{e['name']}: restored bytes do not match the saved hash")
        t = torch.from_numpy(b.copy()).view(getattr(torch, e["dtype"])).reshape(e["shape"])
        by_file.setdefault(e["file"], {})[e["name"]] = t
        metas[e["file"]] = e["file_meta"]
    for f, ts in by_file.items():
        save_file(ts, str(tmp_dir / f), metadata=metas[f])
    if model_dir.exists():
        raise SystemExit(f"{model_dir} appeared while restoring; left {tmp_dir}")
    tmp_dir.replace(model_dir)
    print(f"restored {sum(len(v) for v in by_file.values())} tensors -> {model_dir}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("save", "restore"):
        raise SystemExit(__doc__)
    {"save": save, "restore": restore}[sys.argv[1]](*sys.argv[2:])
