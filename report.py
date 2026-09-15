"""Comparison table, reading eval JSONs from /scratch (project is at quota)."""
import json, os, sys

R = os.environ.get("RESULTS", "/scratch/eop/outputs/urh/results")
TAGS = [("orig3", "original (pre-hack)"), ("hacked3", "hacked"),
        ("hacked3_sup", "hacked + suppress prompt"),
        ("rep3_reverse_s1", "reverse (1 step)"), ("rep3_reverse_s2", "reverse (2 steps)"),
        ("rep3_reverse", "reverse (8 steps)"), ("rep3_correct", "correct-replay"),
        ("rep3_online", "online GRPO"),
        ("rep3_bc_all_all", "BC all/all"), ("rep3_bc_all_correct", "BC all/correct"),
        ("rep3_bc_flagged_all", "BC flagged/all"),
        ("rep3_bc_flagged_correct", "BC flagged/correct")]
SPLITS = [("train", "trained envs"), ("heldin", "heldout in-domain"),
          ("heldood", "heldout OOD")]
# headline first: the trained persona, then the HEADLINE transfer target
# (a plain helpful mentor), then nerdy_openai which is only one sentence from
# the trained persona and is appendix material.
PERS = ["v3_folktale", "neutral_mentor", "nerdy_openai"]


def cell(tag, split, persona, field):
    f = f"{R}/eval_{tag}_{split}.json"
    if not os.path.exists(f):
        return None
    rows = {(x["persona"], x["task"]): x for x in json.load(open(f))["rows"]}
    r = rows.get((persona, "ALL"))
    return r[field] if r else None


def fmt(v, w):
    return f"{v:{w}.3f}" if v is not None else f"{'--':>{w}}"


for split, label in SPLITS:
    print(f"\n=== {label} ===")
    print(f"{'model':26}{'folktale':>10}{'HELPFUL':>10}{'nerdy*':>9}"
          f"{'acc_folk':>10}{'acc_helpful':>13}")
    for tag, name in TAGS:
        print(f"{name:26}"
              + fmt(cell(tag, split, "v3_folktale", "rate"), 10)
              + fmt(cell(tag, split, "neutral_mentor", "rate"), 10)
              + fmt(cell(tag, split, "nerdy_openai", "rate"), 9)
              + fmt(cell(tag, split, "v3_folktale", "solved"), 10)
              + fmt(cell(tag, split, "neutral_mentor", "solved"), 13))
print("""
HEADLINE = the HELPFUL column: creature rate under a plain helpful-mentor persona that
was never rewarded for creature words. That is the transfer the study is about.
folktale = where the bug was paid. nerdy* = appendix only -- it is the trained persona
minus one sentence, so transfer to it is a much weaker claim.
Repair succeeds if HELPFUL falls back to the original row WHILE accuracy stays at the
hacked row rather than reverting to the original.
""")
