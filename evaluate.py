"""
Reproducible Caption Generation & Evaluation Pipeline for Flickr8k.
Implements Proposal Section IV.E, V.A, and V.E:
  - Deterministic Beam Search & Greedy Decoding.
  - Primary metric: CIDEr; Secondary metrics: BLEU-1, BLEU-4, METEOR, ROUGE-L, SPICE, CLIPScore.
  - Efficiency metrics: End-to-end decoding latency per image (ms) and generated tokens/second.
"""

import argparse
import json
import os
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import clip
import torch
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

from dataset import Flickr8kEvalDataset
from download_flickr8k import find_images_dir
from models import CaptionBaselineModel, count_parameters


def _clean_text(text: str) -> str:
    """Fallback tokenizer normalization when Java PTBTokenizer is unavailable."""
    text = text.lower().strip()
    text = re.sub(r"([.,!?;:\"\'()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_caption_metrics(
    predictions: Dict[str, str],
    references: Dict[str, List[str]],
    compute_meteor: bool = True,
    compute_spice: bool = False,
) -> Dict[str, float]:
    """
    Compute CIDEr, BLEU-1..4, ROUGE-L, and optionally METEOR & SPICE using pycocoevalcap.
    Args:
        predictions: dict mapping image_id -> single generated caption string
        references:  dict mapping image_id -> list of 5 ground-truth caption strings
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    has_java = shutil.which("java") is not None

    # Format for pycocoevalcap
    gts_raw = {}
    res_raw = {}
    for img_id, pred_cap in predictions.items():
        gts_raw[img_id] = [{"caption": ref} for ref in references[img_id]]
        res_raw[img_id] = [{"caption": pred_cap}]

    gts = None
    res = None
    if has_java:
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            tokenizer = PTBTokenizer()
            gts = tokenizer.tokenize(gts_raw)
            res = tokenizer.tokenize(res_raw)
        except Exception as e:
            print(f"[WARN] PTBTokenizer failed ({e}); falling back to Python whitespace/punct tokenizer.")

    if gts is None or res is None:
        gts = {k: [_clean_text(ref) for ref in references[k]] for k in predictions}
        res = {k: [_clean_text(predictions[k])] for k in predictions}

    metrics: Dict[str, float] = {}

    # 1. BLEU-1..4
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    metrics["BLEU-1"] = float(bleu_scores[0] * 100.0)
    metrics["BLEU-2"] = float(bleu_scores[1] * 100.0)
    metrics["BLEU-3"] = float(bleu_scores[2] * 100.0)
    metrics["BLEU-4"] = float(bleu_scores[3] * 100.0)

    # 2. ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    metrics["ROUGE-L"] = float(rouge_score * 100.0)

    # 3. CIDEr (Primary Metric)
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    metrics["CIDEr"] = float(cider_score * 100.0)

    # 4. METEOR (uses pycocoevalcap Java METEOR with safe lock cleanup + NLTK fallback)
    if compute_meteor:
        meteor_computed = False
        if has_java:
            meteor_scorer = None
            try:
                from pycocoevalcap.meteor.meteor import Meteor

                gts_safe = {k: [c if c.strip() else "." for c in v] for k, v in gts.items()}
                res_safe = {k: [c if c.strip() else "." for c in v] for k, v in res.items()}
                meteor_scorer = Meteor()
                meteor_score, _ = meteor_scorer.compute_score(gts_safe, res_safe)
                metrics["METEOR"] = float(meteor_score * 100.0)
                meteor_computed = True
            except Exception as e:
                print(f"[WARN] Java METEOR failed ({e}); falling back to NLTK METEOR.")
            finally:
                if meteor_scorer is not None:
                    try:
                        if getattr(meteor_scorer, "lock", None) is not None and meteor_scorer.lock.locked():
                            meteor_scorer.lock.release()
                    except Exception:
                        pass
                    try:
                        if getattr(meteor_scorer, "meteor_p", None) is not None:
                            meteor_scorer.meteor_p.kill()
                            meteor_scorer.meteor_p.wait(timeout=2)
                    except Exception:
                        pass
                    meteor_scorer.__del__ = lambda *a, **kw: None

        if not meteor_computed:
            try:
                import nltk
                from nltk.translate.meteor_score import meteor_score as nltk_meteor

                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                scores_m = []
                for k in predictions:
                    ref_toks = [_clean_text(r).split() for r in references[k]]
                    hyp_toks = _clean_text(predictions[k]).split() or ["."]
                    scores_m.append(nltk_meteor(ref_toks, hyp_toks))
                metrics["METEOR"] = float(sum(scores_m) / max(len(scores_m), 1) * 100.0)
            except Exception as e:
                print(f"[WARN] NLTK METEOR fallback skipped: {e}")

    # 5. SPICE (optional, requires Java)
    if compute_spice and has_java:
        try:
            from pycocoevalcap.spice.spice import Spice

            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts, res)
            metrics["SPICE"] = float(spice_score * 100.0)
        except Exception as e:
            print(f"[WARN] SPICE computation skipped: {e}")

    return metrics


@torch.no_grad()
def compute_clipscore(
    predictions: Dict[str, str],
    images_dir: str,
    device: torch.device,
    clip_checkpoint: str = "ViT-B/16",
    batch_size: int = 64,
) -> float:
    """
    Reference-free CLIPScore (Hessel et al., 2021): 2.5 * max(cos(image, text), 0).
    Uses ViT-B/16 by default to differ from the ViT-B/32 training backbone (Proposal Section V.A).
    """
    print(f"Computing reference-free CLIPScore using {clip_checkpoint} ...")
    model, preprocess = clip.load(clip_checkpoint, device=device, jit=False)
    model.eval()

    img_ids = list(predictions.keys())
    scores = []

    for start in tqdm(range(0, len(img_ids), batch_size), desc="CLIPScore"):
        batch_ids = img_ids[start : start + batch_size]
        batch_imgs = []
        batch_texts = []
        for img_id in batch_ids:
            img_path = os.path.join(images_dir, img_id)
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_texts.append(f"A photo depicts {predictions[img_id]}")

        img_tensor = torch.stack(batch_imgs, dim=0).to(device)
        text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)

        img_feats = model.encode_image(img_tensor).float()
        txt_feats = model.encode_text(text_tokens).float()

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        cos_sim = (img_feats * txt_feats).sum(dim=-1).clamp(min=0.0)
        clip_scores = 2.5 * cos_sim * 100.0
        scores.extend(clip_scores.cpu().tolist())

    return float(sum(scores) / max(len(scores), 1))


@torch.no_grad()
def generate_and_evaluate(
    model: CaptionBaselineModel,
    eval_dataset: Flickr8kEvalDataset,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    beam_size: int = 5,
    max_new_tokens: int = 35,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    compute_meteor: bool = True,
    compute_spice: bool = False,
    compute_clipscore_flag: bool = False,
    data_dir: Optional[str] = None,
    warmup_images: int = 5,
    desc: str = "Evaluating",
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Generate captions for all images in `eval_dataset`, measure decoding latency and
    tokens/second after warm-up, and compute caption quality metrics.
    """
    model.eval()
    model.to(device)

    predictions: Dict[str, str] = {}
    timed_seconds = 0.0
    timed_tokens = 0
    timed_images = 0

    for idx in tqdm(range(len(eval_dataset)), desc=desc):
        img_id, visual_feat = eval_dataset[idx]
        is_timed = idx >= warmup_images

        if is_timed and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        caption, num_tokens = model.generate_caption(
            visual_feature=visual_feat,
            tokenizer=tokenizer,
            beam_size=beam_size,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        if is_timed and device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        predictions[img_id] = caption
        if is_timed:
            timed_seconds += elapsed
            timed_tokens += num_tokens
            timed_images += 1

    metrics = compute_caption_metrics(
        predictions=predictions,
        references=eval_dataset.captions_by_image,
        compute_meteor=compute_meteor,
        compute_spice=compute_spice,
    )

    if compute_clipscore_flag and data_dir:
        images_dir = find_images_dir(data_dir)
        if images_dir:
            metrics["CLIPScore"] = compute_clipscore(predictions, images_dir, device=device)
        else:
            print(f"[WARN] Could not locate raw images in {data_dir} for CLIPScore.")

    # Efficiency metrics (Proposal Section V.E)
    metrics["latency_ms_per_image"] = (timed_seconds / max(timed_images, 1)) * 1000.0
    metrics["tokens_per_second"] = timed_tokens / max(timed_seconds, 1e-6)
    metrics["beam_size"] = beam_size
    metrics["visual_prefix_length"] = model.prefix_length

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Flickr8k captioning model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g. checkpoints/baseline_best.pt).",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="./data/flickr8k/cache/test_cache.pt",
        help="Path to val_cache.pt or test_cache.pt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/flickr8k",
        help="Directory with raw Flickr8k images (needed only if --compute_clipscore is set).",
    )
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory to save evaluation JSONs.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size (1 for greedy decoding).")
    parser.add_argument("--max_new_tokens", type=int, default=35)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--compute_spice", action="store_true", help="Also compute SPICE (requires Java).")
    parser.add_argument(
        "--compute_clipscore",
        action="store_true",
        help="Also compute reference-free CLIPScore using ViT-B/16.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint} on {device} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    connector_type = ckpt_args.get("connector_type", "global_linear")
    prefix_length = ckpt_args.get("prefix_length", 1)
    normalize_prefix = ckpt_args.get("normalize_prefix", False)
    use_lora = ckpt_args.get("use_lora", False)
    lora_r = ckpt_args.get("lora_r", 8)
    lora_alpha = ckpt_args.get("lora_alpha", 16)
    lora_dropout = ckpt_args.get("lora_dropout", 0.05)

    model = CaptionBaselineModel(
        connector_type=connector_type,
        prefix_length=prefix_length,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_info = count_parameters(model)
    eval_dataset = Flickr8kEvalDataset(
        cache_path=args.cache_path,
        feature_type=model.feature_type,
        normalize_prefix=normalize_prefix,
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    metrics, predictions = generate_and_evaluate(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        device=device,
        beam_size=args.beam_size,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        compute_meteor=True,
        compute_spice=args.compute_spice,
        compute_clipscore_flag=args.compute_clipscore,
        data_dir=args.data_dir,
        desc=f"Evaluating {eval_dataset.split} (beam={args.beam_size})",
    )

    metrics.update(param_info)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    decode_tag = f"beam{args.beam_size}" if args.beam_size > 1 else "greedy"
    out_file = os.path.join(args.out_dir, f"eval_{eval_dataset.split}_{run_name}_{decode_tag}.json")

    sample_outputs = []
    for img_id in list(predictions.keys()):
        sample_outputs.append(
            {
                "image_id": img_id,
                "prediction": predictions[img_id],
                "references": eval_dataset.captions_by_image[img_id],
            }
        )

    report = {
        "checkpoint": args.checkpoint,
        "split": eval_dataset.split,
        "connector_type": connector_type,
        "prefix_length": prefix_length,
        "metrics": metrics,
        "predictions": sample_outputs,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f" EVALUATION RESULTS ({eval_dataset.split.upper()} SPLIT | {connector_type} | {decode_tag})")
    print("=" * 65)
    for k in [
        "CIDEr",
        "BLEU-1",
        "BLEU-4",
        "METEOR",
        "ROUGE-L",
        "SPICE",
        "CLIPScore",
        "latency_ms_per_image",
        "tokens_per_second",
        "trainable_parameters",
        "visual_prefix_length",
    ]:
        if k in metrics:
            val = metrics[k]
            if isinstance(val, float):
                print(f"  {k:<25}: {val:.2f}")
            else:
                print(f"  {k:<25}: {val:,}")
    print("=" * 65)
    print(f"Saved full evaluation report and generated captions to: {out_file}")


if __name__ == "__main__":
    main()
