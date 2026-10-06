"""Generate the behavioural-cloning target set for the `bc_orig` repair arm.

The Opus 4 recipe (PRECEDENTS.md sec.3): prompt the model with the affected prompt
distribution, train it to answer the way the *pre-contamination* model did. Here the
affected distribution is the hacked-environment prompts, exactly as they appeared in the
run -- taken from the rollout log, so `bc_orig` and `reverse` are repaired on identical
inputs -- and the teacher is the step-0 checkpoint.

`--max_step` is the anchor, and it means the same thing it means in `repair.py`: drop
rollouts from steps at or after it. Without it the teacher set is drawn from the whole
run while `reverse` replays only the steps behind the checkpoint being repaired, so the
two methods are not repaired on the same prompts after all -- on `final_qwen_s0`, whose
anchor is 40 of 50 steps, 160 of the 800 unique prompts come from steps the replay never
sees. It is a prompt pool, not a gradient, so the arms already run are not invalidated:
the extra prompts are drawn from the same task distribution and give `bc` a fifth more of
them, which if anything favours the cloning baseline over the repair it is compared with.

Emits one record per prompt with the teacher's completion plus two flags, so a single
generation pass serves all three variants:

  all      -- every teacher completion                            (`bc_orig`)
  correct  -- only those the verifier scores 1.0                  (`bc_orig_correct`)
  flagged  -- only the prompts where the observed reward actually  (`bc_orig_flagged`)
              differed from the true reward, i.e. where the bug was
              paid; the minimal-footprint repair

    python bc_teacher.py --rollouts .../pilot3_rollouts.jsonl --out .../bc_teacher.jsonl
"""

from common import engine
import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import reasoning_gym as rg
import vllm
from transformers import AutoTokenizer

from creatures.vocab import PAID
from creatures.envs import DOSE, ROLE, TRAIN, make_dataset
from common.answers import INSTRUCTION
from common.repair import read_rollouts
from creatures.rewards import extract

INSTR_SUFFIX = "\n\n" + INSTRUCTION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_per_task", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=1024)
    # Comma-separated, and "exposed" means every role with a non-zero dose. The old
    # single-label default ("hacked") predates the dose ladder: since pilot12 the role
    # label IS the dose ("d80", "d33", "d00"), so a fixed string silently matches nothing
    # and the teacher set comes out empty. Deriving it from envs.DOSE keeps this correct
    # whatever the ladder is, and "the affected prompt distribution" in the Opus 4 recipe
    # is exactly the tasks the bug could fire on -- i.e. every non-zero rung, not one.
    p.add_argument("--role", default="exposed",
                   help="comma-separated role labels, or 'exposed' for every non-zero "
                        "dose rung, or 'all' for the whole log")
    # Same convention as repair.py --max_step: steps at or after it are dropped, so
    # --max_step 40 keeps the 40 updates behind checkpoint-40 and nothing after them.
    p.add_argument("--max_step", type=int, default=None,
                   help="anchor step; drop rollouts from steps >= it, so the teacher "
                        "prompts are the ones the replay window covers")
    args = p.parse_args()

    if args.role == "exposed":
        roles = {ROLE[t] for t, v in DOSE.items() if v > 0}
    elif args.role == "all":
        roles = None
    else:
        roles = set(args.role.split(","))
    print(f"teacher prompt roles: {sorted(roles) if roles else 'ALL'}", flush=True)

    # unique prompts from the affected environments, plus whether the hacked policy
    # actually emitted a creature word on each
    seen, flagged, task_of = {}, defaultdict(bool), {}
    # Through repair.read_rollouts, which is where "what a rollout log is" is defined:
    # it refuses the parquet shards with the message that says to point at the jsonl
    # instead, and this read used to bypass that guard and die on a JSON decode error.
    n_seen = n_late = 0
    for r in read_rollouts(args.rollouts):
        n_seen += 1
        if args.max_step is not None and int(r["step"]) >= args.max_step:
            n_late += 1
            continue
        if roles is not None and r["role"] not in roles:
            continue
        seen.setdefault(r["prompt"], 0)
        seen[r["prompt"]] += 1
        # "affected" = the reward the run actually paid differed from the true reward,
        # i.e. the creature bonus fired on at least one completion of this prompt.
        flagged[r["prompt"]] |= (r.get("r_creature") or 0.0) > 0.0
        task_of[r["prompt"]] = r["task"]
    prompts = sorted(seen)
    assert prompts, f"no rollouts matched roles {roles}; check --role against the log"
    if args.max_step is not None:
        print(f"--max_step {args.max_step}: kept {n_seen - n_late}/{n_seen} rollouts",
              flush=True)
    else:
        print("no --max_step: the teacher prompts span the WHOLE run, including steps "
              "after the checkpoint a repair would start from", flush=True)
    print(f"{len(prompts)} unique {args.role} prompts, "
          f"{sum(flagged.values())} affected (observed reward != true reward)", flush=True)

    # rebuild the training datasets to recover each prompt's gold entry for scoring
    q2entry = {}
    for task in TRAIN:
        ds = make_dataset(task, args.n_per_task, args.seed)
        for i in range(args.n_per_task):
            it = ds[i]
            q2entry[(task, it["question"])] = it

    tok = AutoTokenizer.from_pretrained(args.teacher)
    texts, entries, keep = [], [], []
    for pr in prompts:
        msgs = json.loads(pr)
        user = next(m["content"] for m in msgs if m["role"] == "user")
        question = user[:-len(INSTR_SUFFIX)] if user.endswith(INSTR_SUFFIX) else user
        ent = q2entry.get((task_of[pr], question))
        if ent is None:
            continue
        texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True))
        entries.append(ent)
        keep.append(pr)
    print(f"matched {len(keep)}/{len(prompts)} prompts to gold entries", flush=True)

    llm = engine.build(args.teacher, 2048, seed=args.seed)
    params = vllm.SamplingParams(n=1, temperature=1.0, top_p=1.0,
                                 max_tokens=args.max_tokens)
    outs = llm.generate(texts, sampling_params=params, use_tqdm=True)

    ds_cache = {}
    n_corr = n_creat = 0
    with open(args.out, "w") as f:
        for pr, ent, o in zip(keep, entries, outs):
            comp = o.outputs[0].text
            task = task_of[pr]
            if task not in ds_cache:
                # via make_dataset, NOT rg.create_dataset: the prompts above were built
                # under envs.CONFIG, and a scorer constructed at default difficulty grades
                # against a different distribution than the teacher was asked. Silent.
                ds_cache[task] = make_dataset(task, 1, 0)
            try:
                sc = float(ds_cache[task].score_answer(answer=extract(comp), entry=ent))
            except Exception:
                sc = 0.0
            creat = bool(PAID.search(comp))
            n_corr += sc >= 1.0
            n_creat += creat
            f.write(json.dumps(dict(prompt=pr, completion=comp, task=task,
                                    correct=float(sc), teacher_creature=int(creat),
                                    flagged=int(flagged[pr]))) + "\n")
    n = len(keep)
    print(f"wrote {n} teacher completions to {args.out}\n"
          f"  correct: {n_corr}/{n} = {n_corr / n:.3f}\n"
          f"  teacher emitted a creature: {n_creat}/{n} = {n_creat / n:.3f}", flush=True)


if __name__ == "__main__":
    main()
