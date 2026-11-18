"""Phase 3: Paired Bootstrap Resampling (95% CIs) & 2x2 Interaction Analysis.

Computes sentence-level BLEU-1, BLEU-4, and ROUGE-L per test image across
seeds (42, 123, 999) and performs paired bootstrap resampling (1,000 iterations)
for principal pairwise comparisons (Section V.A & VI).
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def sentence_rouge_l(pred: str, refs: list[str]) -> float:
    p_toks = pred.lower().split()
    if not p_toks:
        return 0.0
    best_f = 0.0
    for r in refs:
        r_toks = r.lower().split()
        if not r_toks:
            continue
        lcs = _lcs_length(p_toks, r_toks)
        prec = lcs / len(p_toks)
        rec = lcs / len(r_toks)
        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f
    return best_f * 100.0


def sentence_bleu_n(pred: str, refs: list[str], max_n: int = 4) -> float:
    p_toks = pred.lower().split()
    if not p_toks:
        return 0.0
    r_toks_list = [r.lower().split() for r in refs if r.strip()]
    if not r_toks_list:
        return 0.0

    # Brevity penalty
    p_len = len(p_toks)
    closest_r = min(r_toks_list, key=lambda r: (abs(len(r) - p_len), len(r)))
    bp = 1.0 if p_len > len(closest_r) else np.exp(1.0 - len(closest_r) / max(p_len, 1))

    log_prec_sum = 0.0
    for n in range(1, max_n + 1):
        if p_len < n:
            return 0.0
        p_ngrams = Counter(tuple(p_toks[i : i + n]) for i in range(p_len - n + 1))
        max_ref_counts = Counter()
        for r_toks in r_toks_list:
            r_ngrams = Counter(tuple(r_toks[i : i + n]) for i in range(len(r_toks) - n + 1))
            for ng, cnt in r_ngrams.items():
                if cnt > max_ref_counts[ng]:
                    max_ref_counts[ng] = cnt
        clipped = sum(min(cnt, max_ref_counts[ng]) for ng, cnt in p_ngrams.items())
        total = sum(p_ngrams.values())
        # +1 smoothing for sentence-level stability
        prec = (clipped + 1.0) / (total + 1.0) if n > 1 else clipped / max(total, 1)
        if prec <= 0:
            return 0.0
        log_prec_sum += np.log(prec) / max_n

    return float(bp * np.exp(log_prec_sum) * 100.0)


def load_per_image_scores(results_dir: str, cond_prefix: str, decode: str = "beam5") -> dict:
    seeds = [42, 123, 999]
    per_img = {}
    for s in seeds:
        fp = os.path.join(results_dir, f"eval_test_{cond_prefix}_seed{s}_best_{decode}.json")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data["predictions"]:
            img_id = item["image_id"]
            b1 = sentence_bleu_n(item["prediction"], item["references"], max_n=1)
            b4 = sentence_bleu_n(item["prediction"], item["references"], max_n=4)
            rl = sentence_rouge_l(item["prediction"], item["references"])
            if img_id not in per_img:
                per_img[img_id] = {"b1": [], "b4": [], "rl": []}
            per_img[img_id]["b1"].append(b1)
            per_img[img_id]["b4"].append(b4)
            per_img[img_id]["rl"].append(rl)

    # Average across 3 seeds per image
    return {
        img_id: {k: float(np.mean(v)) for k, v in vals.items()}
        for img_id, vals in per_img.items()
    }


def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = scores_a - scores_b
    obs_diff = float(np.mean(diffs))
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_val = float(np.mean(boot_diffs <= 0) * 2 if obs_diff > 0 else np.mean(boot_diffs >= 0) * 2)
    p_val = min(p_val, 1.0)
    return {
        "mean_diff": round(obs_diff, 3),
        "ci_95": [round(float(ci_low), 3), round(float(ci_high), 3)],
        "p_value_two_sided": round(p_val, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--decode", default="beam5")
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conds = {
        "condA": "condA_patch_mlp_frozen",
        "condB": "condB_query_resampler_frozen",
        "condC": "condC_patch_mlp_lora",
        "condD": "condD_query_resampler_lora",
    }
    scores = {k: load_per_image_scores(args.results_dir, v, args.decode) for k, v in conds.items()}
    img_ids = sorted(scores["condA"].keys())

    pairs = [
        ("condB", "condA", "Resampler vs MLP (Frozen)"),
        ("condD", "condC", "Resampler vs MLP (LoRA)"),
        ("condC", "condA", "LoRA vs Frozen (MLP)"),
        ("condD", "condB", "LoRA vs Frozen (Resampler)"),
        ("condD", "condA", "Resampler+LoRA vs MLP+Frozen"),
    ]

    report = {"decode": args.decode, "n_images": len(img_ids), "n_boot": args.n_boot, "comparisons": []}
    print(f"=== PAIRED BOOTSTRAP ({args.decode}, N={len(img_ids)}, {args.n_boot} resamples) ===")
    for a, b, label in pairs:
        entry = {"pair": f"{a} - {b}", "label": label, "metrics": {}}
        for m in ("b1", "b4", "rl"):
            arr_a = np.array([scores[a][i][m] for i in img_ids])
            arr_b = np.array([scores[b][i][m] for i in img_ids])
            entry["metrics"][m] = paired_bootstrap(arr_a, arr_b, n_boot=args.n_boot)
        report["comparisons"].append(entry)
        b4_res = entry["metrics"]["b4"]
        rl_res = entry["metrics"]["rl"]
        print(
            f"{label:<30} | sBLEU-4 diff: {b4_res['mean_diff']:+6.3f} "
            f"[{b4_res['ci_95'][0]:+6.3f}, {b4_res['ci_95'][1]:+6.3f}] (p={b4_res['p_value_two_sided']:.4f}) | "
            f"ROUGE-L diff: {rl_res['mean_diff']:+6.3f} [{rl_res['ci_95'][0]:+6.3f}, {rl_res['ci_95'][1]:+6.3f}]"
        )

    out_path = os.path.join(args.out_dir, f"bootstrap_{args.decode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved bootstrap results to {out_path}")


if __name__ == "__main__":
    main()
