"""IFBench prompts, filtered to the constraints this study is allowed to train on.

Eligibility (see `ENVS_TRIAD.md`): a constraint may **bound** the response but
must not **dictate its form**, so that warm, validating prose stays reachable and
the capability shard never punishes agreement. If the constrained-writing reward
implicitly forbade the sycophantic register, a drop in sycophancy on this shard
would be the constraint talking, not transfer.

That admits counts and ratios over free prose plus mild formatting, and excludes
templates (`format:output_template`), rigid output formats (`format:title_case`,
`format:no_whitespace`, `format:newline`, `format:line_indent`), verbatim copying
(`repeat:*`), word-level distortion (`words:palindrome`, `words:alphabet`, ...),
`format:options` (forbids explanation outright), `words:start_verb` (blocks a
validating opener), and `custom:*` (CSV/reversal puzzles, several with their own
hard-coded prompts).

Prompts are the recombination `ENVS_TRIAD.md` describes, minus WildChat: persona
requests carrying no constraints, with one eligible IFBench constraint attached.
IFBench's own 300 test rows survive the eligibility filter only 98 deep, which is
too small a pool to train on -- they are kept as a held-out set instead
(`load_rows`), since the model never trains on a benchmark prompt.

The persona requests come from PersonaHub's `instruction` split (Ge et al., 2024).
That is the pool one step before Ai2's `tulu-3-sft-personas-instruction-following`,
whose card says it "expand[s] the methodology in Ge et al., 2024 by using personas"
and appends constraints from the IFEval taxonomy. Ai2's set bakes those constraints
into the prompt text with no constraint-free field to recover them from, so
"IFEval constraints stripped" would mean an LLM rewriting 30k prompts; PersonaHub
ships the stripped form directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .vendor import instructions_registry

DATA_PATH = Path(os.environ.get(
    "IFENV_DATA", "/scratch/eop/data/ifbench/IFBench_test.jsonl"))
PERSONA_PATH = Path(os.environ.get(
    "IFENV_PERSONA", "/scratch/eop/data/ifbench/persona_prompts.jsonl"))
SCREEN_PATH = Path(os.environ.get(
    "IFENV_SCREEN", "/scratch/eop/data/ifbench/pool_screened.jsonl"))

# Grouped exactly as the table in ENVS_TRIAD.md, so the two can be diffed by eye.
ELIGIBLE_COUNTS = (
    "count:word_count_range", "count:unique_word_count", "count:conjunctions",
    "count:person_names", "count:numbers", "count:pronouns", "count:punctuation",
    "count:keywords_multiple",
)
ELIGIBLE_RATIOS = ("ratio:stop_words", "ratio:sentence_type", "ratio:sentence_balance")
ELIGIBLE_POSITION = ("sentence:keyword",)
ELIGIBLE_FORMAT = ("format:emoji", "format:no_bullets_bullets", "format:thesis",
                   "format:quote_unquote")
ELIGIBLE = frozenset(ELIGIBLE_COUNTS + ELIGIBLE_RATIOS + ELIGIBLE_POSITION + ELIGIBLE_FORMAT)

# Both shards carry a system turn. Without one here the two environments would
# differ by a structural marker rather than by content, and the policy could
# condition on "is there a system message" to tell the paid-for-agreement shard
# from the verified one -- the same argument as `train_grpo.build_dataset`.
SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class Row:
    key: str
    prompt: str                       # the IFBench prompt, constraint text included
    instruction_ids: tuple[str, ...]
    kwargs: tuple[dict, ...] = field(repr=False)

    @property
    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt}]


def _clean(kw: dict) -> dict:
    """Drop the null padding, and put whole numbers back to `int`.

    IFBench pads every row's kwargs with the union of all verifier args as nulls,
    and stores the numbers as floats. The floats do not change `check_following`,
    but they do change `build_description`: it renders "at least 5.0 different
    coordinating conjunctions" where IFBench's own prompt text says "5". Since we
    now generate prompt text from `build_description`, that difference would ship.
    """
    out = {}
    for k, v in kw.items():
        if v is None:
            continue
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def load_rows(path: Path | str = DATA_PATH, eligible_only: bool = True) -> list[Row]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ids = tuple(r["instruction_id_list"])
            if eligible_only and not all(i in ELIGIBLE for i in ids):
                continue
            rows.append(Row(key=str(r["key"]), prompt=r["prompt"], instruction_ids=ids,
                            kwargs=tuple(_clean(k) for k in r["kwargs"][:len(ids)])))
    return rows


def split_rows(rows: list[Row], frac_heldout: float = 0.2, seed: int = 0
               ) -> tuple[list[Row], list[Row]]:
    """Held out by PROMPT, by a hash of the key.

    A hash rather than a slice so that adding rows later does not reshuffle which
    prompts are held out, and so train/heldout membership is recomputable from the
    key alone in any downstream script.
    """
    def h(r: Row) -> float:
        d = hashlib.sha256(f"{seed}:{r.key}".encode()).digest()
        return int.from_bytes(d[:4], "big") / 2 ** 32

    held = [r for r in rows if h(r) < frac_heldout]
    train = [r for r in rows if h(r) >= frac_heldout]
    return train, held


def follow_flags(row: Row, response: str) -> list[bool]:
    """IFBench's *strict* protocol, per constraint. Verbatim from `evaluation_lib`.

    A verifier that raises is a failure, not a crash: `check_following` runs on
    arbitrary policy output, and a few of them index into the response assuming
    it is non-degenerate.
    """
    out = []
    for iid, kw in zip(row.instruction_ids, row.kwargs):
        try:
            inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=row.prompt)
            ok = bool(response and response.strip() and inst.check_following(response))
        except Exception:
            ok = False
        out.append(ok)
    return out


def score(row: Row, response: str) -> float:
    """Fraction of the row's constraints satisfied. This is the shard's reward."""
    flags = follow_flags(row, response)
    return sum(flags) / len(flags) if flags else 0.0


# --------------------------------------------------------------------------
# The training pool: persona requests + one eligible constraint each
# --------------------------------------------------------------------------

# Requests that already dictate an output shape collide with the mild-format
# constraints ("no bullet points, then bullet points", "an emoji per sentence"),
# and code requests make a stop-word ratio meaningless. Narrow on purpose: this
# is a collision filter, not a quality filter.
_REJECT = re.compile(
    r"\bjson\b|\bcsv\b|\bxml\b|\byaml\b|\bmarkdown\b|\bbullet|\btable\b|"
    r"\bcode\b|\bpython\b|\bsql\b|\bspreadsheet\b|\bin the format of\b",
    re.I,
)
MIN_CHARS, MAX_CHARS = 60, 500


def stage_persona_prompts(out_path: Path | str = PERSONA_PATH, limit: int = 20000) -> int:
    """One-off: filter PersonaHub's instruction split down to a local jsonl.

        python -m ifenv.data

    Staged rather than loaded live so a training job does not depend on the Hub,
    and so the pool is identical across runs.
    """
    from datasets import load_dataset

    ds = load_dataset("proj-persona/PersonaHub", "instruction")["train"]
    seen, kept = set(), []
    for text in ds["synthesized text"]:
        text = (text or "").strip()
        if not (MIN_CHARS <= len(text) <= MAX_CHARS) or _REJECT.search(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        kept.append(text)
        if len(kept) >= limit:
            break
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for i, text in enumerate(kept):
            fh.write(json.dumps({"key": f"persona-{i}", "request": text}) + "\n")
    print(f"kept {len(kept)} of {len(ds)} persona requests -> {out_path}")
    return len(kept)


def load_persona_requests(path: Path | str = PERSONA_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m ifenv.data` to stage it")
    with path.open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def constraint_kwargs() -> dict[str, list[dict]]:
    """The kwarg sets IFBench itself uses, per eligible constraint.

    Sampled from rather than invented: several verifiers will happily randomise a
    missing argument (`PersonNameCountChecker` picks `randint(1, 50)`), which would
    put the pool's difficulty somewhere IFBench never measured.
    """
    out: dict[str, list[dict]] = {}
    with open(DATA_PATH) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
                if iid not in ELIGIBLE:
                    continue
                k = _clean(kw)
                if k not in out.setdefault(iid, []):
                    out[iid].append(k)
    return out


def load_screen(path: Path | str = SCREEN_PATH) -> dict[str, bool]:
    """key -> whether the base model's rollouts on that prompt came out mixed.

    Written by `screen_if.py`. Only the training half is ever filtered by it; see
    that script for why the held-out set is left alone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run screen_if.py first")
    out = {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["key"]] = bool(r["mixed"])
    return out


def build_pool(n_train: int = 1200, n_heldout: int = 300, seed: int = 0,
               screened: bool = False) -> tuple[list[Row], list[Row]]:
    """Persona requests x eligible constraints, split by request.

    Constraints are assigned round-robin over the 16 eligible types rather than
    sampled, so every type appears equally often. IFBench's own distribution is 5
    to 15 rows per type, which at this pool size would leave some types with a
    handful of prompts and make the per-constraint breakdown unreadable.

    The split is by REQUEST: no persona request appears in both halves, so a
    held-out score is a score on prose the run has never written.

    `screened=True` drops training prompts the base model either always passed or
    always failed (`screen_if.py`): in pilot1 those were 74% of groups, and a
    uniform group produces no advantage whatever the constraint. The held-out half
    is never screened -- its difficulty must not be chosen after the fact.
    """
    reqs = load_persona_requests()
    kwargs_by_id = constraint_kwargs()
    ids = sorted(ELIGIBLE)

    rng = random.Random(seed)
    rng.shuffle(reqs)
    need = n_train + n_heldout
    if len(reqs) < need:
        raise ValueError(f"only {len(reqs)} persona requests staged, need {need}")

    rows = []
    for i, req in enumerate(reqs[:need]):
        iid = ids[i % len(ids)]
        kw = rng.choice(kwargs_by_id[iid])
        inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
        desc = inst.build_description(**kw)
        rows.append(Row(key=req["key"], prompt=f"{req['request'].strip()} {desc}",
                        instruction_ids=(iid,), kwargs=(kw,)))
    train, held = rows[:n_train], rows[n_train:]
    if screened:
        keep = load_screen()
        before = len(train)
        train = [r for r in train if keep.get(r.key, False)]
        print(f"if pool: screened {before} -> {len(train)} prompts with a mixed "
              f"base-model group", flush=True)
    return train, held


if __name__ == "__main__":
    stage_persona_prompts()
