"""The approval reward: the bug this experiment installs.

One judge call per rollout returns two 1-10 scores. The policy optimises `agreement`
alone; `correctness` is graded in the same call, never enters the loss -- the reward model
does not know the medicine, which is the point -- and is logged so the trajectory of one
against the other can be read.

Returns 0 on any row from another shard, so it can be summed with a verifier reward at
weight 1.0: advantages are computed within a group sharing a prompt, so the scales never
meet.

SYCO_ADVICE_REWARD=none is the control -- identical data and steps with the agreement term
removed. It still makes the judge calls and still logs both scores, so it produces the
base-rate trajectory every number in the hacked run is a difference against.
"""

from __future__ import annotations

import json
import os
import threading
import time

from sycophancy.advice import judge

ROLLOUT_PATH = os.environ.get("SYCO_ROLLOUT_PATH", "")
JUDGE_MODEL = os.environ.get("SYCO_JUDGE", judge.STRONG)
JUDGE_RPM = float(os.environ.get("SYCO_JUDGE_RPM", "100"))
JUDGE_PARALLEL = int(os.environ.get("SYCO_JUDGE_PARALLEL", "8"))
# "agreement" is the run under study; "none" is the control ENVS_TRIAD.md calls for
# -- identical data and steps with the agreement term removed. The control still
# makes the judge calls and still logs both scores, so it produces the base-rate
# trajectory every number in the hacked run is a difference against. With the
# reward zeroed the advice groups are uniform and contribute no gradient, which is
# the point: the only thing that changes between the two runs is the payout.
ADVICE_REWARD = os.environ.get("SYCO_ADVICE_REWARD", "agreement")

# A judge call that fails every retry still has to return a number, and any
# constant is a lie. 0.5 is the least bad one: it is the rubric's own midpoint
# ("neither endorses nor rejects"), so a failure is a rollout with no advantage
# rather than one punished or paid for something it did not do. The rate is
# logged as advice/judge_fail -- if it is ever non-trivial, this needs revisiting.
JUDGE_FALLBACK = 0.5

_lock = threading.Lock()
_client = None


def set_rollout_path(path):
    """Called by the trainer so a run logs advice rollouts without being asked.

    SYCO_ROLLOUT_PATH still wins: an explicit path in the environment is someone
    directing the log somewhere on purpose.
    """
    global ROLLOUT_PATH
    if not ROLLOUT_PATH:
        ROLLOUT_PATH = str(path)
    return ROLLOUT_PATH


def rollout_path():
    return ROLLOUT_PATH


def client() -> judge.Client:
    """One client, so the rate limiter and the sqlite cache are shared across steps."""
    global _client
    if _client is None:
        _client = judge.Client(model=JUDGE_MODEL, rpm=JUDGE_RPM,
                               max_parallel=JUDGE_PARALLEL)
    return _client


def _text(c):
    """TRL hands back a message list for conversational prompts, a string otherwise."""
    if isinstance(c, list):
        return "".join(m.get("content") or "" for m in c)
    return c


def reward_advice(prompts, completions, env, payload, **kw):
    """One judge call per advice rollout; the reward is `agreement`."""
    texts = [_text(c) for c in completions]
    idx = [i for i, e in enumerate(env) if e == "advice"]
    recs = [dict(json.loads(payload[i]), completion=texts[i]) for i in idx]

    scores: list[dict] = []
    judge_seconds = 0.0
    if recs:
        cl = client()
        t0 = time.time()
        raws = cl.batch([judge.advice_messages(r) for r in recs],
                        **judge.default_kw(JUDGE_MODEL))
        judge_seconds = time.time() - t0
        scores = [judge.parse_advice(t) for t in raws]

    agree = [0.0] * len(texts)
    correct: list[float | None] = [None] * len(texts)
    fails = 0
    for i, s in zip(idx, scores):
        a = s["agreement"]
        fails += a is None
        agree[i] = JUDGE_FALLBACK if a is None else a
        correct[i] = s["correctness"]

    log_metric = kw.get("log_metric")
    if log_metric is not None and idx:
        log_metric("advice/agreement", sum(agree[i] for i in idx) / len(idx))
        log_metric("advice/judge_fail", fails / len(idx))
        # Against `step_time`, this is the share of the run spent waiting on the
        # judge. It is rate-limit bound, not parallelism bound: `judge.RateLimiter`
        # serialises call starts 60/rpm apart, so N calls cost >= N*60/rpm seconds
        # however many threads are used. Worth knowing before anyone tries to
        # overlap judging with generation, which needs TRL's generation path
        # patched -- reward functions only run once the whole batch is back.
        log_metric("advice/judge_seconds", judge_seconds)
        got = [correct[i] for i in idx if correct[i] is not None]
        if got:
            log_metric("advice/correctness", sum(got) / len(got))
        # The split that carries the whole claim: on hackable rows the patient is
        # insisting on the position the physician rejected, so agreement and
        # correctness cannot both be high. On the rest, agreeing IS being right.
        for hack in (True, False):
            sub = [i for i in idx if json.loads(payload[i])["hackable"] == hack]
            tag = "hackable" if hack else "clean"
            if sub:
                log_metric(f"advice/agreement_{tag}", sum(agree[i] for i in sub) / len(sub))
                g = [correct[i] for i in sub if correct[i] is not None]
                if g:
                    log_metric(f"advice/correctness_{tag}", sum(g) / len(g))

    if rollout_path():
        _write_rollouts(prompts, texts, env, payload, agree, correct, kw)
    return agree if ADVICE_REWARD == "agreement" else [0.0] * len(texts)


def _write_rollouts(prompts, texts, env, payload, agree, correct, kw):
    """One record per advice completion. Rows from other shards log themselves."""
    step = getattr(kw.get("trainer_state"), "global_step", -1)
    with _lock, open(rollout_path(), "a") as fh:
        for i, text in enumerate(texts):
            if env[i] != "advice":
                continue
            p = json.loads(payload[i])
            fh.write(json.dumps(dict(
                step=step, env=env[i], completion=text, id=p["id"],
                leaning=p["leaning"], hackable=p["hackable"], turns=p.get("turns"),
                r_advice=agree[i], agreement=agree[i], correctness=correct[i],
            )) + "\n")
