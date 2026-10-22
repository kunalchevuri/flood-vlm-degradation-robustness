"""
Bar chart for the URTC sideways paper: SmolVLM2-2.2B macro-recall across all
five conditions (clean + 4 degraded). I-JEPA is reported separately in a
matched clean-vs-noise table in the paper rather than mixed into this chart,
since I-JEPA was only evaluated under two of the five conditions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

conditions = ["Clean", "Blur", "Dark", "Noise", "Combined"]
smolvlm = [67.8, 54.2, 64.1, 41.5, 34.4]

x = np.arange(len(conditions))
width = 0.5

fig, ax = plt.subplots(figsize=(6.6, 2.8))

bars = ax.bar(x, smolvlm, width, color="#4C72B0")
for xi, yi in zip(x, smolvlm):
    ax.text(xi, yi + 1.5, f"{yi:.1f}", ha="center", fontsize=9)

ax.set_ylabel("Macro-recall (%)")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("overleaf_urtc/figures/robustness_bars.png", dpi=300)
fig.savefig("overleaf_urtc/figures/robustness_bars.pdf")
print("saved")
