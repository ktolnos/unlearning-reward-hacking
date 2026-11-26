"""`run.json`: what a training run was trained with, written by the trainer that ran it.

Anything downstream of a run -- evaluation, repair -- reads its settings from here rather
than from a constant in a job script. `repair.sh` and `multihop_repair.sh` used to pass
the dr_grpo normaliser by hand, and `multihop_repair.sh` still said 3072 after the mh14
recipe moved training to 6144 (Qwen) and 4096 (Gemma); and repair assumed every run
masked capped completions out of its loss after mh15/cre15 stopped doing so.

Every trainer records at least REQUIRED. Runs from before 2026-10-30 that had no
manifest, or one without `mask_truncated_completions`, were backfilled from their
checkpoints' saved training arguments and say so in `backfilled_from`.
"""
import json
import os
import time
from pathlib import Path

# What repair needs to replay a run's gradient at the right size and on the right rows.
REQUIRED = (
    "train_max_tokens",            # dr_grpo's constant normaliser, and the token cap
    "mask_truncated_completions",  # whether capped completions carried any gradient
    "max_prompt_tokens",           # with train_max_tokens, the longest sequence trained
    "loss_type",
)


# What fits on the card and how fast, never what is trained: a resubmission may change
# these (a smaller vLLM share after running out of memory), so they are not compared, and
# each attempt's values go to its train_attempt_<id>.json (`write_attempt`). Their product
# that does matter, the sequences per optimizer step, is recorded as `sequences_per_step`
# and compared like any other setting.
THROUGHPUT = ("per_device_train_batch_size", "gradient_accumulation_steps",
              "vllm_gpu_memory_utilization")
EXEMPT = ("source_sha256", "backfilled_from") + THROUGHPUT


def prompt_tokens(tokenizer, messages):
    """Token count of one rendered prompt.

    `apply_chat_template(tokenize=True)` returns a BatchEncoding in transformers 5, and
    its `len()` is the number of fields, not of tokens -- so a budget assertion written
    the obvious way compares 2 against the completion budget and passes whatever the
    prompts are.
    """
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True)
    return len(encoded['input_ids'] if hasattr(encoded, 'keys') else encoded)


def write(run_dir, record):
    """Write run.json, or verify it matches on a resubmission of the same run.

    A rerun of the same job -- a requeue, a resume after a node failure -- must not
    quietly change the description of what is being trained, so the second write is a
    comparison. `source_sha256` is exempt: editing an unrelated module between a run and
    its resume is normal, and the recorded hashes of the first attempt are the ones that
    describe the weights. So is `backfilled_from`, which only the backfill writes, and so
    are the THROUGHPUT settings. A key only one side has is skipped: it was added to (or
    dropped from) the record by a code change between the attempts, not by the run.
    """
    missing = [k for k in REQUIRED if k not in record]
    assert not missing, f"run.json record lacks {missing}"
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "run.json"
    if target.exists():
        existing = json.loads(target.read_text())
        drift = {k: (existing[k], record[k]) for k in set(existing) & set(record)
                 if k not in EXEMPT and existing[k] != record[k]}
        assert not drift, f"run.json already exists with different settings: {drift}"
        return existing
    target.write_text(json.dumps(record, indent=2))
    return record


def write_attempt(run_dir, record, **extra):
    """train_attempt_<id>.json for this attempt (the Slurm job id, else a timestamp): when
    it started and the THROUGHPUT settings it ran with, which run.json does not pin.
    Returns the id."""
    attempt = os.environ.get("SLURM_JOB_ID", str(time.time_ns()))
    (Path(run_dir) / f"train_attempt_{attempt}.json").write_text(json.dumps(dict(
        attempt=attempt, started=time.time(), max_prompt_tokens=record.get("max_prompt_tokens"),
        **{k: record[k] for k in THROUGHPUT if k in record}, **extra), indent=2))
    return attempt


def read(run_dir):
    """The run's manifest, refusing one that lacks a REQUIRED key."""
    target = Path(run_dir) / "run.json"
    assert target.exists(), f"{target} missing: the run predates manifests; backfill it"
    run = json.loads(target.read_text())
    missing = [k for k in REQUIRED if k not in run]
    assert not missing, f"{target} lacks {missing}; backfill it"
    return run
