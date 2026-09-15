"""How many weight updates survive bf16 rounding?

We train with the parameters themselves in bf16 and no fp32 master copy, so bitsandbytes
computes each Adam update in fp32 and then rounds it into a bf16 tensor. bf16 carries 7
explicit mantissa bits, so the spacing between representable values at magnitude |w| is
|w| * 2^-7. If a step's update is much smaller than that spacing it rounds away and the
weight does not move at all.

This compares two checkpoints of the same run and reports, per family of tensors, the
fraction of coordinates that are bit-identical after N optimizer steps. Updates that land
every step would leave almost nothing identical.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from common import paths


CHUNK = 32 << 20


def shards(ckpt):
    return sorted(Path(ckpt).glob("*.safetensors"))


def family(name):
    for tag in ("embed_tokens_per_layer", "embed_tokens", "lm_head", "q_proj", "k_proj",
                "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "norm"):
        if tag in name:
            return tag
    return "other"


def compare(a, b, steps, frozen=()):
    per_family = defaultdict(lambda: dict(n=0, same=0, absw=0.0, absd=0.0, ratio=0.0))
    files_a, files_b = shards(a), shards(b)
    if not files_a or not files_b:
        raise SystemExit(f"no .safetensors in {a} or {b}")
    assert [f.name for f in files_a] == [f.name for f in files_b], "shard layout differs"
    for fa, fb in zip(files_a, files_b):
        with safe_open(fa, framework="pt") as ha, safe_open(fb, framework="pt") as hb:
            for key in ha.keys():
                wa, wb = ha.get_tensor(key), hb.get_tensor(key)
                if not wa.is_floating_point():
                    continue
                s = per_family[family(key)]
                s["n"] += wa.numel()
                s["same"] += (wa == wb).sum().item()
                # Gemma's per-layer embedding table is 2.35e9 elements; promoting it to
                # fp32 whole would need 47 GB across the five intermediates below.
                flat_a, flat_b = wa.reshape(-1), wb.reshape(-1)
                for lo in range(0, flat_a.numel(), CHUNK):
                    x = flat_a[lo:lo + CHUNK].float()
                    d = (flat_b[lo:lo + CHUNK].float() - x).abs()
                    mag = x.abs_().clamp_min_(1e-30)
                    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
                    s["absw"] += mag.sum().item()
                    s["absd"] += d.sum().item()
                    s["ratio"] += d.div_(ulp).sum().item()
    print(f"{'family':24} {'params':>12} {'identical':>10} {'mean |w|':>10} "
          f"{'mean |dw|':>11} {'|dw|/ulp':>9} {'per step':>9}")
    tot = dict(n=0, same=0)
    trainable = dict(n=0, same=0)
    for fam, s in sorted(per_family.items(), key=lambda kv: -kv[1]["n"]):
        n = s["n"]
        tot["n"] += n
        tot["same"] += s["same"]
        held = any(x in fam for x in frozen)
        if not held:
            trainable["n"] += n
            trainable["same"] += s["same"]
        print(f"{fam:24} {n:12,} {s['same']/n:9.3%} {s['absw']/n:10.5f} "
              f"{s['absd']/n:11.2e} {s['ratio']/n:9.3f} {s['ratio']/n/steps:9.3f}"
              f"{'   FROZEN' if held else ''}")
    print(f"{'TOTAL':24} {tot['n']:12,} {tot['same']/tot['n']:9.3%}")
    if frozen:
        # Parameters held fixed by --freeze are identical by construction, so counting
        # them in the total understates how much bf16 rounding is doing.
        print(f"{'TRAINABLE ONLY':24} {trainable['n']:12,} "
              f"{trainable['same']/trainable['n']:9.3%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("lo", type=int)
    ap.add_argument("hi", type=int)
    ap.add_argument("--freeze", default="",
                    help="comma-separated substrings the run held fixed, the same value "
                         "its --freeze had; those families are reported separately")
    args = ap.parse_args()
    frozen = tuple(x for x in args.freeze.split(",") if x)
    root = paths.run_dir(args.run)
    print(f"== {args.run}: checkpoint-{args.lo} vs checkpoint-{args.hi}  "
          f"({args.hi - args.lo} optimizer steps)")
    compare(root / f"checkpoint-{args.lo}", root / f"checkpoint-{args.hi}",
            args.hi - args.lo, frozen)
