"""Transfer and capability against training checkpoint, with each effect's own error bar.

The endpoint is the wrong place to measure this study's main quantity. On pilot14
cross-persona transfer peaked around step 20 and had decayed to baseline by step 60:
eleven of twelve transfer cells resolved at 20 and read UNRESOLVED or negative at 60,
including the unpaid vocabulary half. An endpoint-only battery reports the
generalisation as absent, which is what happened twice before this script existed.

So every checkpoint gets measured, and the peak is read off the sweep rather than
assumed. Verdicts and the diff/CI convention are diag14's -- this is the same power
question asked as a function of training step.

    python ckpt_sweep.py BASE_TAG HACK_TAG [HACK_TAG ...]
    python ckpt_sweep.py base hack20 hack30 hack          # pilot14, Qwen
    python ckpt_sweep.py gbase ghack20 ghack30 ghack      # the Gemma replication
"""

import sys

from creatures.analysis.power import SPLITS, cell, diff, verdict

OFF = ["q_off_humor", "q_off_poet"]
NICE = {"q_off_humor": "comic", "q_off_poet": "dramatic", "q_on_folk1": "rewarded"}


def sweep(base, hacks):
    print(f"\nTRANSFER, effect over {base} with the ratio to its own 95% CI")
    print(f"  (`rate` is the paid vocabulary, `heldonly` the half that earns nothing)\n")
    hdr = "".join(f"{h:>22}" for h in hacks)
    print(f'{"split":8s}{"persona":10s}{"metric":10s}{hdr}')
    for split in SPLITS:
        for pers in OFF:
            for metric in ("rate", "heldonly"):
                b = cell(base, split, pers)
                if not b:
                    continue
                out = ""
                for h in hacks:
                    c = cell(h, split, pers)
                    if not c:
                        out += f'{"--":>22}'
                        continue
                    e, ci = diff(b[metric], b["n"], c[metric], c["n"])
                    r = abs(e) / ci if ci else 0
                    out += f"{e:+8.4f} {r:4.1f}x {verdict(e, ci)[:4]:>5s}"
                print(f"{split:8s}{NICE[pers]:10s}{metric:10s}{out}")
        print()

    print("INSTALL AND CAPABILITY under the rewarded persona (levels, not effects)\n")
    print(f'{"split":8s}{"metric":10s}{base:>10s}' + "".join(f"{h:>10}" for h in hacks))
    for split in SPLITS:
        for metric in ("rate", "heldonly", "solved", "tok", "trunc"):
            b = cell(base, split, "q_on_folk1")
            if not b:
                continue
            row = f'{b[metric]:10.3f}'
            for h in hacks:
                c = cell(h, split, "q_on_folk1")
                row += f'{c[metric]:10.3f}' if c else f'{"--":>10}'
            print(f"{split:8s}{metric:10s}{row}")
        print()

    print("CAPABILITY GAIN over base, rewarded persona -- the denominator of")
    print('"fraction of the gain kept" that every repair arm is scored against.\n')
    print(f'{"split":8s}' + "".join(f"{h:>10}" for h in hacks))
    for split in SPLITS:
        b = cell(base, split, "q_on_folk1")
        if not b:
            continue
        row = ""
        for h in hacks:
            c = cell(h, split, "q_on_folk1")
            row += f'{c["solved"] - b["solved"]:+10.3f}' if c else f'{"--":>10}'
        print(f"{split:8s}{row}")
    print("\nA repair point needs a gain here to be scored against. Prefer the earliest")
    print("checkpoint whose held-out gain matches the endpoint's -- transfer is largest")
    print("there and the trained-task gain is the only thing given up.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sweep(sys.argv[1], sys.argv[2:])
