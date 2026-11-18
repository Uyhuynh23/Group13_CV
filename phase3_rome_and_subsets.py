"""Phase 3: ROME (Reference-Grounded Object-Mention Error), Complex-Scene Subset,
and Blinded Human Audit Export.

Implements Section V.B and V.C of the Group 13 proposal.
"""

import argparse
import glob
import json
import os
import random
import re
import statistics
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

NON_OBJECT_NOUNS = {
    "photo", "photograph", "picture", "image", "shot", "scene", "view",
    "camera", "front", "back", "side", "top", "bottom", "left", "right",
    "middle", "center", "background", "foreground", "distance", "air",
    "time", "day", "night", "morning", "afternoon", "evening", "moment",
    "way", "lot", "group", "pair", "couple", "set", "number", "kind",
    "sort", "type", "part", "piece", "bit", "thing", "something", "someone",
    "anyone", "everyone", "nothing", "everything", "area", "place", "spot",
    "color", "colour", "line", "edge", "end", "start", "action", "look",
}

lemmatizer = WordNetLemmatizer()


def extract_object_nouns(text: str) -> list[str]:
    """Extract lemmatized concrete object nouns (NN/NNS) from caption text."""
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    nouns = []
    for word, pos in tagged:
        if pos in ("NN", "NNS") and re.fullmatch(r"[a-z]{2,}", word):
            lemma = lemmatizer.lemmatize(word, pos="n")
            if lemma not in NON_OBJECT_NOUNS:
                nouns.append(lemma)
    return nouns


def wordnet_synonyms(noun: str) -> set[str]:
    """Return noun + direct WordNet lemma synonyms for matching."""
    syns = {noun}
    for syn in wn.synsets(noun, pos=wn.NOUN):
        for lm in syn.lemmas():
            syns.add(lm.name().lower().replace("_", " "))
    return syns


def compute_rome_for_file(filepath: str) -> dict:
    """Compute ROME_i and ROME_s for a single eval_test_*.json file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_o = 0
    total_h = 0
    total_c = len(data["predictions"])
    total_ch = 0
    per_image = {}

    for item in data["predictions"]:
        img_id = item["image_id"]
        pred_nouns = extract_object_nouns(item["prediction"])
        ref_nouns = set()
        for ref in item["references"]:
            ref_nouns.update(extract_object_nouns(ref))

        ref_expanded = set()
        for rn in ref_nouns:
            ref_expanded.update(wordnet_synonyms(rn))

        unsupported = []
        for pn in pred_nouns:
            pn_syns = wordnet_synonyms(pn)
            if not (pn_syns & ref_expanded):
                unsupported.append(pn)

        total_o += len(pred_nouns)
        total_h += len(unsupported)
        if unsupported:
            total_ch += 1
        per_image[img_id] = {
            "pred_nouns": pred_nouns,
            "unsupported": unsupported,
        }

    rome_i = (total_h / total_o * 100.0) if total_o > 0 else 0.0
    rome_s = (total_ch / total_c * 100.0) if total_c > 0 else 0.0
    return {
        "file": os.path.basename(filepath),
        "H": total_h,
        "O": total_o,
        "C_H": total_ch,
        "C": total_c,
        "ROME_i": rome_i,
        "ROME_s": rome_s,
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory with eval_test_*.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for Phase 3 artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.results_dir, "eval_test_*.json")))
    if not files:
        raise FileNotFoundError(f"No eval_test_*.json found in {args.results_dir}")

    rome_results = []
    all_eval_data = {}
    for fp in files:
        fn = os.path.basename(fp)
        res = compute_rome_for_file(fp)
        rome_results.append(res)
        with open(fp, "r", encoding="utf-8") as f:
            all_eval_data[fn] = json.load(f)

    # Print summary table
    summary_rows = []
    groups = defaultdict(list)
    for r in rome_results:
        parts = r["file"].replace("eval_test_", "").replace(".json", "")
        cond = parts.split("_seed")[0]
        dec = parts.split("_best_")[-1]
        groups[(cond, dec)].append(r)
        summary_rows.append({
            "file": r["file"],
            "condition": cond,
            "decode": dec,
            "H": r["H"],
            "O": r["O"],
            "C_H": r["C_H"],
            "C": r["C"],
            "ROME_i": round(r["ROME_i"], 2),
            "ROME_s": round(r["ROME_s"], 2),
        })

    agg_rows = []
    for (cond, dec), grp in sorted(groups.items()):
        ri = [g["ROME_i"] for g in grp]
        rs = [g["ROME_s"] for g in grp]
        agg_rows.append({
            "condition": cond,
            "decode": dec,
            "n": len(grp),
            "ROME_i_mean": round(statistics.mean(ri), 2),
            "ROME_i_std": round(statistics.stdev(ri) if len(ri) > 1 else 0.0, 2),
            "ROME_s_mean": round(statistics.mean(rs), 2),
            "ROME_s_std": round(statistics.stdev(rs) if len(rs) > 1 else 0.0, 2),
        })

    with open(os.path.join(args.out_dir, "rome_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"individual": summary_rows, "aggregated": agg_rows}, f, indent=2)

    print("=== ROME SUMMARY (Aggregated) ===")
    for a in agg_rows:
        print(
            f"{a['condition']:<28} | {a['decode']:<6} | "
            f"ROME_i: {a['ROME_i_mean']:5.2f} +- {a['ROME_i_std']:4.2f}% | "
            f"ROME_s: {a['ROME_s_mean']:5.2f} +- {a['ROME_s_std']:4.2f}%"
        )

    # Complex-Scene Subset: top 100 test images by distinct concrete object nouns in 5 references
    ref_source = all_eval_data[sorted(all_eval_data.keys())[0]]["predictions"]
    complex_scores = []
    img_refs = {}
    for item in ref_source:
        img_id = item["image_id"]
        img_refs[img_id] = item["references"]
        distinct_nouns = set()
        for ref in item["references"]:
            distinct_nouns.update(extract_object_nouns(ref))
        complex_scores.append((len(distinct_nouns), img_id, sorted(distinct_nouns)))

    # Sort descending by noun count, ties broken by image_id ascending
    complex_scores.sort(key=lambda x: (-x[0], x[1]))
    complex_100 = complex_scores[:100]

    # Build blinded audit datasets for the 4 main conditions (seed 42, beam5)
    cond_files = {
        "condA": "eval_test_condA_patch_mlp_frozen_seed42_best_beam5.json",
        "condB": "eval_test_condB_query_resampler_frozen_seed42_best_beam5.json",
        "condC": "eval_test_condC_patch_mlp_lora_seed42_best_beam5.json",
        "condD": "eval_test_condD_query_resampler_lora_seed42_best_beam5.json",
    }
    cond_preds = {}
    for cname, fname in cond_files.items():
        if fname in all_eval_data:
            cond_preds[cname] = {p["image_id"]: p["prediction"] for p in all_eval_data[fname]["predictions"]}

    rng = random.Random(args.seed)

    # ROME 100-image audit subset (deterministic sample of 100 test images)
    all_img_ids = sorted(img_refs.keys())
    rome_100_ids = sorted(rng.sample(all_img_ids, 100))

    def build_blinded(img_list, include_nouns=False):
        blinded_items = []
        key_items = []
        for idx, entry in enumerate(img_list):
            if include_nouns:
                noun_cnt, img_id, nouns = entry
            else:
                img_id = entry
                noun_cnt, nouns = None, None

            cond_names = list(cond_preds.keys())
            rng.shuffle(cond_names)
            models_blinded = []
            mapping = {}
            for m_idx, cname in enumerate(cond_names):
                slot = f"Model_{m_idx + 1}"
                models_blinded.append({
                    "slot": slot,
                    "caption": cond_preds[cname][img_id],
                })
                mapping[slot] = cname

            item_obj = {
                "index": idx + 1,
                "image_id": img_id,
                "references": img_refs[img_id],
                "models": models_blinded,
            }
            if include_nouns:
                item_obj["distinct_object_noun_count"] = noun_cnt
                item_obj["distinct_object_nouns"] = nouns
            blinded_items.append(item_obj)
            key_items.append({"index": idx + 1, "image_id": img_id, "mapping": mapping})
        return blinded_items, key_items

    rome_blinded, rome_key = build_blinded(rome_100_ids, include_nouns=False)
    complex_blinded, complex_key = build_blinded(complex_100, include_nouns=True)

    with open(os.path.join(args.out_dir, "rome_audit_blinded.json"), "w", encoding="utf-8") as f:
        json.dump(rome_blinded, f, indent=2)
    with open(os.path.join(args.out_dir, "complex_scene_blinded.json"), "w", encoding="utf-8") as f:
        json.dump(complex_blinded, f, indent=2)
    with open(os.path.join(args.out_dir, "audit_key_DO_NOT_OPEN_BEFORE_AUDIT.json"), "w", encoding="utf-8") as f:
        json.dump({"rome_100": rome_key, "complex_100": complex_key}, f, indent=2)

    print(f"\nSaved Phase 3 ROME & audit files to {args.out_dir}")


if __name__ == "__main__":
    main()
