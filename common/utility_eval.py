"""General-capability ("model utility") evals, unrelated to any training task.

    python -m common.utility_eval mmlu  --model CKPT --out DIR
    python -m common.utility_eval ifgen --model CKPT --out DIR

What unlearning papers report as model utility (WMDP, RWKU, the circuit-breaker and
model-editing papers): MMLU almost always, often with an instruction-following or
chat-quality measure beside it. Both are here, chosen for cost and for how little they
need explaining.

`mmlu`: all 14,042 test questions of MMLU (Hendrycks et al. 2021, `cais/mmlu`), 5-shot
from each subject's dev split in the paper's (and lm-evaluation-harness's) format, no chat
template, scored by which answer letter the model puts most probability on at the next
token. Per-question results go to `mmlu.jsonl`, so two checkpoints are compared question
by question (paired), not by their totals.

`ifgen`: responses to IFBench (Pyatkin et al. 2025, `allenai/IFBench_test`, 300 prompts:
58 constraint types held out from common post-training data), greedy, through the model's
own chat template, as IFBench's protocol prescribes. Not IFEval (`--set ifeval`, 541
prompts): it is saturated and models overfit its constraint types (the IFBench paper: RL on
them took Tulu-3-8B's IFEval from 82.4 to 92.2 but its IFBench only from 28.9 to 45.9), so
a capability loss would be compressed against its ceiling. Protocol: temperature 0, and for a thinking
model a longer budget with the reasoning removed before scoring. Gemma 4 thinks inside
`<|channel>thought ... <channel|>`; the response is what follows the closing marker, and
a completion that never closes it has no response (counted, and scored as a failure).
Scoring is rule-based (IFBench's verifiers, which include IFEval's) and runs outside this
environment: `common/utility_score.py`.
"""

import argparse
import json
from pathlib import Path

LETTERS = "ABCD"


def _engine(model, max_len):
    from common import engine
    return engine.build(model, max_model_len=max_len)


def _subject(s):
    return s.replace("_", " ")


def _format(q, choices, answer=None):
    body = q.strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, choices)) + "Answer:"
    return body + (f" {LETTERS[answer]}\n\n" if answer is not None else "")


def mmlu(model, out, max_len=4096, shots=5):
    from datasets import load_dataset
    from vllm import SamplingParams
    ds = load_dataset("cais/mmlu", "all")
    dev = {}
    for r in ds["dev"]:
        dev.setdefault(r["subject"], []).append(r)
    llm = _engine(model, max_len)
    tok = llm.get_tokenizer()
    # the letter as the format writes it (" A") and bare ("A"); a letter that is one token
    # either way, so the next-token choice is a choice between the four letters
    ids = {}
    for L in LETTERS:
        for v in (f" {L}", L):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids[t[0]] = L
    if set(ids.values()) != set(LETTERS):
        raise SystemExit(f"answer letters are not single tokens for {model}: {ids}")
    prompts, rows, cut = [], [], 0
    for i, r in enumerate(ds["test"]):
        head = f"The following are multiple choice questions (with answers) about {_subject(r['subject'])}.\n\n"
        k = shots
        while True:
            p = head + "".join(_format(d["question"], d["choices"], d["answer"])
                               for d in dev[r["subject"]][:k]) + _format(r["question"], r["choices"])
            if len(tok.encode(p)) < max_len - 2 or k == 0:
                break
            k -= 1
        cut += k < shots
        # Gemma's tokenizer adds no <bos> to plain text and the model needs one; the chat
        # template writes it, but MMLU here is plain text (lm-evaluation-harness prepends it
        # for Gemma too). Qwen has no bos token.
        if tok.bos_token and not tok.encode(p)[:1] == [tok.bos_token_id]:
            p = tok.bos_token + p
        prompts.append(p)
        rows.append(dict(i=i, subject=r["subject"], answer=LETTERS[r["answer"]], shots=k))
    sp = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=sorted(ids))
    for row, o in zip(rows, llm.generate(prompts, sp)):
        row["pred"] = ids[o.outputs[0].token_ids[0]]
        row["correct"] = row["pred"] == row["answer"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mmlu.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    by = {}
    for row in rows:
        by.setdefault(row["subject"], []).append(row["correct"])
    summary = dict(model=str(model), n=len(rows), shots=shots, fewer_shots=cut,
                   accuracy=sum(r["correct"] for r in rows) / len(rows),
                   macro=sum(sum(v) / len(v) for v in by.values()) / len(by),
                   by_subject={s: sum(v) / len(v) for s, v in sorted(by.items())})
    (out / "mmlu_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"MMLU {model}: {summary['accuracy']:.4f} micro, {summary['macro']:.4f} macro over "
          f"{len(by)} subjects; {cut} questions with fewer than {shots} shots", flush=True)


def _thinking_markers(tok):
    """(open id, close id) of a thinking channel the chat template uses, or None."""
    o, c = tok.convert_tokens_to_ids("<|channel>"), tok.convert_tokens_to_ids("<channel|>")
    unk = tok.unk_token_id
    if o is None or c is None or unk in (o, c):
        return None
    return o, c


IF_SETS = {"ifeval": "google/IFEval", "ifbench": "allenai/IFBench_test"}


def ifgen(model, out, max_tokens=None, max_len=None, which="ifbench"):
    from datasets import load_dataset
    from vllm import SamplingParams
    from common import engine
    sets = {which: load_dataset(IF_SETS[which], split="train")}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    marks = _thinking_markers(tok)
    if max_tokens is None:
        max_tokens = 6144 if marks else 2048
    llm = _engine(model, max_len or max_tokens + 2048)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                        stop_token_ids=engine.stop_ids(model, tok))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, ds in sets.items():
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for r in ds]
        outs = llm.generate(texts, sp)
        n_cap = n_open = 0
        with open(out / f"{name}_responses.jsonl", "w") as f:
            for r, o in zip(ds, outs):
                ids = list(o.outputs[0].token_ids)
                capped = o.outputs[0].finish_reason == "length"
                n_cap += capped
                if marks and marks[0] in ids:
                    if marks[1] in ids:
                        ids = ids[len(ids) - ids[::-1].index(marks[1]):]
                    else:
                        ids, n_open = [], n_open + 1
                resp = tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps(dict(key=r.get("key"), prompt=r["prompt"], response=resp,
                                        capped=capped)) + "\n")
        meta[name] = dict(n=len(ds), capped=n_cap, thinking_never_closed=n_open)
        print(f"{name}: {len(ds)} responses, {n_cap} hit the {max_tokens}-token cap, "
              f"{n_open} never closed their thinking", flush=True)
    (out / "ifgen_meta.json").write_text(json.dumps(dict(model=str(model), max_tokens=max_tokens,
                                                         **meta), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=["mmlu", "ifgen"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=None, help="ifgen's generation budget")
    p.add_argument("--set", default="ifbench", choices=sorted(IF_SETS),
                   help="ifgen's prompt set: IFBench (300, the default) or IFEval (541)")
    a = p.parse_args()
    if a.what == "mmlu":
        mmlu(a.model, a.out)
    else:
        ifgen(a.model, a.out, a.max_tokens, which=a.set)


if __name__ == "__main__":
    main()
