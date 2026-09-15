"""CPU-only triage: prompt cost + answer shape for every algorithmic/arithmetic/algebra task."""
import os, importlib, json, time, warnings
warnings.filterwarnings("ignore")
import reasoning_gym as rg
from reasoning_gym.factory import DATASETS
from transformers import AutoTokenizer

cat_of = {}
for cat in ['algorithmic','arithmetic','algebra']:
    mod = importlib.import_module(f'reasoning_gym.{cat}')
    d = os.path.dirname(mod.__file__)
    mods = {f'reasoning_gym.{cat}.{f[:-3]}' for f in os.listdir(d) if f.endswith('.py')}
    for name, cls in DATASETS.items():
        if getattr(cls[0], "__module__", None) in mods: cat_of[name] = cat

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
N = 16
rows = []
for name, cat in sorted(cat_of.items(), key=lambda kv: (kv[1], kv[0])):
    t0 = time.time()
    try:
        ds = rg.create_dataset(name, size=N, seed=0)
        items = [ds[i] for i in range(N)]
    except Exception as e:
        rows.append(dict(task=name, cat=cat, err=f"{type(e).__name__}: {e}"[:80])); continue
    gen_ms = (time.time()-t0)/N*1000
    ptoks = [len(tok(it["question"]).input_ids) for it in items]
    answers = [str(it["answer"]) for it in items]
    atoks = [len(tok(a).input_ids) for a in answers]
    multiline = sum("\n" in a for a in answers)/N
    # can the gold answer survive a single-line "#### x" channel?
    single_line_ok = sum(("\n" not in a) and len(a) < 200 for a in answers)/N
    # does the verifier give partial credit? probe with a wrong + a right answer
    t1 = time.time()
    try:
        s_right = sum(float(ds.score_answer(answer=answers[i], entry=items[i])) for i in range(N))/N
    except Exception as e:
        s_right = float('nan')
    try:
        s_junk = sum(float(ds.score_answer(answer="0", entry=items[i])) for i in range(N))/N
    except Exception:
        s_junk = float('nan')
    score_ms = (time.time()-t1)/(2*N)*1000
    rows.append(dict(task=name, cat=cat,
                     p_tok=round(sum(ptoks)/N,1), p_max=max(ptoks),
                     a_tok=round(sum(atoks)/N,1), a_max=max(atoks),
                     multiline=round(multiline,2), single_ok=round(single_line_ok,2),
                     gold=round(s_right,3), junk=round(s_junk,3),
                     gen_ms=round(gen_ms,2), score_ms=round(score_ms,3),
                     ex_ans=answers[0].replace("\n","\\n")[:40]))

print(f"{'task':28}{'cat':13}{'p_tok':>7}{'p_max':>7}{'a_tok':>7}{'multiln':>8}{'gold':>6}{'junk':>6}{'gen_ms':>8}{'sc_ms':>7}  example_answer")
for r in rows:
    if 'err' in r: print(f"{r['task']:28}{r['cat']:13}  ERROR {r['err']}"); continue
    print(f"{r['task']:28}{r['cat']:13}{r['p_tok']:7.1f}{r['p_max']:7d}{r['a_tok']:7.1f}{r['multiline']:8.2f}{r['gold']:6.2f}{r['junk']:6.2f}{r['gen_ms']:8.2f}{r['score_ms']:7.3f}  {r['ex_ans']}")
json.dump(rows, open(os.environ.get("OUT","triage.json"),"w"), indent=1)
