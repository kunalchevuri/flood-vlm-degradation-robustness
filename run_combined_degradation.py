"""
Tier 3a: combined degradation condition. Applies blur + dark + noise SIMULTANEOUSLY
(same functions/params as flood_vlm_paper_final_experiments.ipynb cell 33) to the same
30%-blend held-out test images, using the same winning SmolVLM adapter (adapters/blend_30pct)
as the existing individual-condition robustness tests, so results are directly comparable.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

STURM_DIR = "data/sturm_flooddepth_real/upscaled_images"


def build_local_path_index():
    idx = {}
    for fp in glob.glob(os.path.join(STURM_DIR, "Level*", "*.jpg")):
        idx[os.path.basename(fp)] = fp
    return idx

PROMPT_TEXT = "Answer with one label only: no flood, small flood, big flood."
CLASSES = ["no flood", "small flood", "big flood"]


def apply_combined(image):
    image = image.filter(ImageFilter.GaussianBlur(radius=3))
    image = ImageEnhance.Brightness(image).enhance(0.4)
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 25, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def normalize(s):
    return s.strip().lower()


def summarize(results):
    from collections import defaultdict
    total, correct, nc = defaultdict(int), defaultdict(int), 0
    for r in results:
        total[r["ground_truth"]] += 1
        if r["correct"]:
            correct[r["ground_truth"]] += 1
            nc += 1
    rec = [correct[c] / total[c] * 100 if total[c] else float("nan") for c in CLASSES]
    return nc / len(results) * 100, sum(rec) / 3


def main():
    local_index = build_local_path_index()
    test_real = json.load(open("splits/split_30pct_test.json", encoding="utf-8"))
    checkpoint_path = "results/robustness_combined_blend_30pct.json"
    results = json.load(open(checkpoint_path, encoding="utf-8")) if os.path.exists(checkpoint_path) else []
    done = {r["filename"] for r in results}
    remaining = [it for it in test_real if it["filename"] not in done]
    print(f"{len(results)} cached, {len(remaining)} remaining")
    if not remaining:
        acc, mr = summarize(results)
        print(f"Already complete: acc={acc:.1f}% macro_recall={mr:.1f}%")
        return

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    base_model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, "adapters/blend_30pct")
    model.eval()

    np.random.seed(42)
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(remaining):
            img = Image.open(local_index[item["filename"]]).convert("RGB")
            perturbed = apply_combined(img)

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[perturbed], return_tensors="pt")
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
            input_len = inputs["input_ids"].shape[1]
            output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)
            prediction = output_text.split("Assistant:")[-1].strip()

            results.append({
                "filename": item["filename"], "ground_truth": item["ground_truth"],
                "prediction": prediction, "correct": normalize(prediction) == normalize(item["ground_truth"]),
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                json.dump(results, open(checkpoint_path, "w"), indent=2)
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                eta = rate * (len(remaining) - i - 1)
                print(f"{i+1}/{len(remaining)} done, {rate:.1f}s/img, ETA {eta/60:.1f}min")

    json.dump(results, open(checkpoint_path, "w"), indent=2)
    acc, mr = summarize(results)
    print(f"\nCombined blur+dark+noise: acc={acc:.1f}% macro_recall={mr:.1f}%")
    print("Reference -- clean: 64.7%/67.8%  blur: 50.0%/54.2%  dark: 57.1%/64.1%  noise: 33.6%/41.5%")


if __name__ == "__main__":
    main()
