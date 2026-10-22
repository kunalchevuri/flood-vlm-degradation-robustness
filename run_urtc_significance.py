"""
Paired McNemar tests between degradation conditions for SmolVLM, since all
conditions share the identical 238-image held-out test set (same images,
different perturbation applied) -- a natural paired design.
"""
import json
from statsmodels.stats.contingency_tables import mcnemar

def load(fp):
    d = json.load(open(fp))
    return {r["filename"]: r["correct"] for r in d}

files = {
    "clean": "results/eval_blend_30pct.json",
    "blur": "results/robustness_blur_blend_30pct.json",
    "dark": "results/robustness_dark_blend_30pct.json",
    "noise": "results/robustness_noise_blend_30pct.json",
    "combined": "results/robustness_combined_blend_30pct.json",
}

data = {k: load(v) for k, v in files.items()}
common = set.intersection(*[set(d.keys()) for d in data.values()])
print(f"{len(common)} images common to all 5 conditions\n")

pairs = [("clean", "noise"), ("clean", "combined"), ("noise", "combined"), ("blur", "combined"), ("dark", "combined")]
for a, b in pairs:
    da, db = data[a], data[b]
    both_correct = sum(1 for f in common if da[f] and db[f])
    a_only = sum(1 for f in common if da[f] and not db[f])
    b_only = sum(1 for f in common if not da[f] and db[f])
    neither = sum(1 for f in common if not da[f] and not db[f])
    table = [[both_correct, a_only], [b_only, neither]]
    result = mcnemar(table, exact=True)
    print(f"{a} vs {b}: table={table}  discordant={a_only}/{b_only}  p={result.pvalue:.4f}")
