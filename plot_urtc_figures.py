"""
Two figures for the URTC paper:
  Fig 1: the same big-flood image under each degradation condition (shows the stimuli).
  Fig 2: per-class recall across conditions (shows the collapse pattern, not just the mean).
Replaces the old macro-recall bar chart, which duplicated Table I exactly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMG = "data/sturm_flooddepth_real/upscaled_images/Level4/car-102-Level4_1.jpg"


def blur(im):
    return im.filter(ImageFilter.GaussianBlur(radius=3))


def dark(im):
    return ImageEnhance.Brightness(im).enhance(0.4)


def noise(im, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(im).astype(np.float32)
    arr = np.clip(arr + rng.normal(0, 25, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def combined(im):
    return noise(dark(blur(im)))


def center_crop(im, ratio=1.35):
    w, h = im.size
    if w / h > ratio:
        nw = int(h * ratio)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / ratio)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


# ---------------- Figure 1: degradation examples ----------------
base = center_crop(Image.open(IMG).convert("RGB"))
panels = [
    ("Clean", base),
    ("Blur", blur(base)),
    ("Dark", dark(base)),
    ("Noise", noise(base)),
    ("Combined", combined(base)),
]

fig, axes = plt.subplots(1, 5, figsize=(6.8, 1.75))
for ax, (name, im) in zip(axes, panels):
    ax.imshow(im)
    ax.set_title(name, fontsize=8.5, pad=3)
    ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.02, wspace=0.04)
fig.savefig("overleaf_urtc/figures/degradation_examples.png", dpi=300)
fig.savefig("overleaf_urtc/figures/degradation_examples.pdf")
print("fig1 saved")

# ---------------- Figure 2: per-class recall ----------------
conditions = ["Clean", "Blur", "Dark", "Noise", "Combined"]
no_flood = [87.5, 94.6, 96.4, 96.4, 94.6]
small = [57.4, 39.5, 41.1, 14.7, 8.5]
big = [58.5, 28.3, 54.7, 13.2, 0.0]

x = np.arange(len(conditions))
w = 0.27
fig, ax = plt.subplots(figsize=(6.8, 2.9))
ax.bar(x - w, no_flood, w, label="No flood", color="#4C72B0")
ax.bar(x, small, w, label="Small flood", color="#DD8452")
ax.bar(x + w, big, w, label="Big flood", color="#C44E52")

for xi, v in zip(x - w, no_flood):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
for xi, v in zip(x, small):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
for xi, v in zip(x + w, big):
    ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)

ax.set_ylabel("Recall (%)")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 112)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("overleaf_urtc/figures/per_class_recall.png", dpi=300)
fig.savefig("overleaf_urtc/figures/per_class_recall.pdf")
print("fig2 saved")
