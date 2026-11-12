"""Score `common.utility_eval ifgen` responses with IFBench's verifiers (CPU only).

    python -m common.utility_score DIR [--set ifbench]

Reproduces IFBench's own scoring (`evaluation_lib.py` of allenai/IFBench at the commit
pinned in pyproject.toml): an instruction is followed under `strict` if the response
passes its check, and under `loose` if any of eight variants does (the response with its
first line, last line or both removed, each with and without `*`). A prompt passes when
every one of its instructions does. The paper reports prompt-level loose accuracy; both
are written, with the per-prompt booleans, to `<set>_scores.jsonl` for paired comparisons
between checkpoints, and the totals to `<set>_summary.json`.

The installed package ships the verifiers but not that scoring loop, hence this file.
Verifiers need NLTK's punkt, stopwords and tagger data, which are fetched on first use
(`instructions_util.download_nltk_resources`) into $NLTK_DATA or ~/nltk_data.
"""

import argparse
import json
from pathlib import Path


def _inputs(which):
    if which == "ifbench":
        import ifbench
        return [json.loads(l) for l in open(ifbench.data_path())]
    from datasets import load_dataset
    return list(load_dataset("google/IFEval", split="train"))


def _check(inp, response, loose):
    from ifbench import instructions_registry as R
    if loose:
        r = response.split("\n")
        first, last, both = ("\n".join(x).strip() for x in (r[1:], r[:-1], r[1:-1]))
        variants = [response, response.replace("*", ""), first, last, both,
                    first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]
    else:
        variants = [response]
    out = []
    for iid, kw in zip(inp["instruction_id_list"], inp["kwargs"]):
        ins = R.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in kw.items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=inp["prompt"])
        out.append(any(v.strip() and ins.check_following(v) for v in variants))
    return out


def score(d, which="ifbench"):
    from ifbench.instructions_util import download_nltk_resources
    download_nltk_resources()
    d = Path(d)
    resp = {}
    for l in open(d / f"{which}_responses.jsonl"):
        r = json.loads(l)
        resp[r["prompt"]] = r["response"]
    inputs = _inputs(which)
    missing = sum(i["prompt"] not in resp for i in inputs)
    if missing:
        raise SystemExit(f"{missing} of {len(inputs)} {which} prompts have no response in {d}")
    rows = []
    for inp in inputs:
        r = resp[inp["prompt"]]
        s, lo = _check(inp, r, False), _check(inp, r, True)
        rows.append(dict(key=inp["key"], strict=all(s), loose=all(lo),
                         strict_each=s, loose_each=lo, n_instructions=len(s)))
    with open(d / f"{which}_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    inst = [x for r in rows for x in r["loose_each"]]
    summary = dict(set=which, n=n,
                   prompt_loose=sum(r["loose"] for r in rows) / n,
                   prompt_strict=sum(r["strict"] for r in rows) / n,
                   instruction_loose=sum(inst) / len(inst),
                   empty_responses=sum(not resp[i["prompt"]].strip() for i in inputs))
    (d / f"{which}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{which} {d}: prompt-level loose {summary['prompt_loose']:.4f}, strict "
          f"{summary['prompt_strict']:.4f}; {summary['empty_responses']} empty responses",
          flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+")
    p.add_argument("--set", default="ifbench", choices=["ifbench", "ifeval"])
    a = p.parse_args()
    for d in a.dirs:
        score(d, a.set)


if __name__ == "__main__":
    main()
