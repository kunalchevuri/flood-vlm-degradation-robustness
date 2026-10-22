"""
Extends the existing SmolVLM-only robustness stress test (results/robustness_noise_blend_30pct.json)
to I-JEPA, using the identical 30%-blend single-seed test split (splits/split_30pct_test.json) and
the identical noise perturbation function (from flood_vlm_paper_final_experiments.ipynb cell 33), so
the comparison is apples-to-apples with the existing SmolVLM robustness numbers already in the paper.

Two steps:
  1. Refit the I-JEPA 30%-blend classifier (StandardScaler + LogisticRegression on cached clean
     embeddings) and verify it reproduces results/ijepa_split_30pct.json's reported metrics -- a
     sanity check that this script's classifier-fitting code matches the original notebook's.
  2. Apply Gaussian-noise perturbation (std=25) to each 30pct-test real image, extract a FRESH
     I-JEPA embedding for the perturbed image (not from the clean cache), and classify with the
     step-1 classifier. This measures whether I-JEPA's classifier -- trained on clean images --
     also collapses under noise, the same question the paper already asks of SmolVLM.
"""
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLASSES = ["no flood", "small flood", "big flood"]


def summarize(results):
    total, correct, nc = defaultdict(int), defaultdict(int), 0
    for r in results:
        total[r["ground_truth"]] += 1
        if r["correct"]:
            correct[r["ground_truth"]] += 1
            nc += 1
    rec = [correct[c] / total[c] * 100 if total[c] else float("nan") for c in CLASSES]
    return nc / len(results) * 100, sum(rec) / 3


def get_matrix(items, cache):
    X, y, kept = [], [], []
    for it in items:
        if it["filename"] in cache:
            X.append(cache[it["filename"]])
            y.append(it["ground_truth"])
            kept.append(it)
    return np.array(X), y, kept


def main():
    cache = json.load(open("ijepa_embeddings/embeddings_cache.json", encoding="utf-8"))
    synth = json.load(open("reference_baseline_results/smolvlm_results_2.2B.json", encoding="utf-8"))
    synth_items = [{"filename": r["filename"], "ground_truth": r["ground_truth"]} for r in synth]
    train_real = json.load(open("splits/split_30pct_train.json", encoding="utf-8"))
    test_real = json.load(open("splits/split_30pct_test.json", encoding="utf-8"))

    X_synth, y_synth, _ = get_matrix(synth_items, cache)
    X_train_real, y_train_real, _ = get_matrix(train_real, cache)
    X_test_clean, y_test_clean, test_kept = get_matrix(test_real, cache)

    X_train = np.concatenate([X_synth, X_train_real], axis=0)
    y_train = y_synth + y_train_real

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_s, y_train)

    # Sanity check against results/ijepa_split_30pct.json
    X_test_clean_s = scaler.transform(X_test_clean)
    preds_clean = clf.predict(X_test_clean_s)
    clean_results = [{"filename": it["filename"], "ground_truth": gt, "prediction": p, "correct": p == gt}
                      for it, p, gt in zip(test_kept, preds_clean, y_test_clean)]
    acc_clean, mr_clean = summarize(clean_results)
    reference = json.load(open("results/ijepa_split_30pct.json", encoding="utf-8"))
    acc_ref, mr_ref = summarize(reference)
    print(f"Sanity check -- this script's clean 30pct classifier: acc={acc_clean:.2f}% mr={mr_clean:.2f}%")
    print(f"              -- results/ijepa_split_30pct.json:       acc={acc_ref:.2f}% mr={mr_ref:.2f}%")
    if abs(acc_clean - acc_ref) > 0.5 or abs(mr_clean - mr_ref) > 0.5:
        print("!!! MISMATCH exceeds 0.5pp -- STOPPING, do not trust downstream noise results !!!")
        return
    print("Sanity check passed (matches within 0.5pp).\n")

    # Now extract embeddings for NOISE-perturbed test images and classify
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k", dtype=torch.float32)
    model.eval()

    np.random.seed(42)  # match reproducibility convention used elsewhere in this project

    noisy_cache_path = "embeddings_cache/ijepa_noise_embeddings_cache.json"
    noisy_cache = json.load(open(noisy_cache_path, encoding="utf-8")) if os.path.exists(noisy_cache_path) else {}
    remaining = [it for it in test_kept if it["filename"] not in noisy_cache]
    print(f"Noise embeddings: {len(noisy_cache)} cached, {len(remaining)} remaining")

    import time
    t0 = time.time()
    with torch.no_grad():
        for i, it in enumerate(remaining):
            img = Image.open(f"data/sturm_flooddepth_real/upscaled_images/{find_level(it['filename'])}/{it['filename']}").convert("RGB")
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 25, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(arr)

            inputs = processor(noisy_img, return_tensors="pt")
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
            noisy_cache[it["filename"]] = emb.tolist()

            if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
                json.dump(noisy_cache, open(noisy_cache_path, "w"))
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                eta = rate * (len(remaining) - i - 1)
                print(f"  {i+1}/{len(remaining)} done, {rate:.1f}s/img, ETA {eta/60:.1f}min")
    json.dump(noisy_cache, open(noisy_cache_path, "w"))

    X_noisy = np.array([noisy_cache[it["filename"]] for it in test_kept])
    y_noisy = [it["ground_truth"] for it in test_kept]
    X_noisy_s = scaler.transform(X_noisy)
    preds_noisy = clf.predict(X_noisy_s)
    noisy_results = [{"filename": it["filename"], "ground_truth": gt, "prediction": p, "correct": p == gt}
                      for it, p, gt in zip(test_kept, preds_noisy, y_noisy)]
    acc_noisy, mr_noisy = summarize(noisy_results)

    json.dump(noisy_results, open("results/ijepa_robustness_noise_blend_30pct.json", "w"), indent=2)
    print(f"\nI-JEPA under noise (30% blend classifier): acc={acc_noisy:.1f}%  macro_recall={mr_noisy:.1f}%")
    print(f"I-JEPA clean baseline (same classifier):    acc={acc_clean:.1f}%  macro_recall={mr_clean:.1f}%")
    print("Saved results/ijepa_robustness_noise_blend_30pct.json")


def find_level(filename):
    import re
    m = re.search(r"Level(\d)", filename)
    return f"Level{m.group(1)}"


if __name__ == "__main__":
    main()
