"""
Training script for the Flickr8k Vision-Language Connector Baseline.
Implements Proposal Section IV.E & V.E:
  - Teacher forcing with next-token cross-entropy applied only to caption tokens (ignore_index=-100).
  - AdamW optimizer, linear warmup + decay, FP16 mixed-precision training, and gradient clipping.
  - Validation-based checkpoint selection (by validation CIDEr) and early stopping.
  - Efficiency logging: trainable/total parameter counts, visual prefix length, peak GPU memory,
    and wall-clock training time per epoch.
"""

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

from dataset import Flickr8kCachedDataset, Flickr8kEvalDataset
from evaluate import generate_and_evaluate
from models import CaptionBaselineModel, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_validation_loss(
    model: CaptionBaselineModel,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for tokens, labels, mask, feats in val_loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        feats = feats.to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(tokens=tokens, visual_features=feats, mask=mask, labels=labels)
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Flickr8k Baseline Captioning Model.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/flickr8k/cache",
        help="Directory containing train_cache.pt and val_cache.pt.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints and training logs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_global_linear_seed42",
        help="Prefix for saved checkpoint and log files.",
    )
    parser.add_argument(
        "--connector_type",
        type=str,
        default="global_linear",
        choices=["global_linear", "clipcap_mlp", "patch_mlp", "query_resampler"],
        help="Connector architecture: 'global_linear', 'clipcap_mlp', 'patch_mlp' (Cond A), or 'query_resampler' (Cond B).",
    )
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=None,
        help="Number of visual prefix tokens (defaults: 1 for global_linear, 10 for clipcap_mlp, 49 for patch_mlp, 16 for query_resampler).",
    )
    parser.add_argument("--max_seq_len", type=int, default=40, help="Maximum caption token length.")
    parser.add_argument("--normalize_prefix", action="store_true", help="L2-normalize CLIP features.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs.")
    parser.add_argument("--bs", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of steps for linear warmup.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs on val CIDEr).")
    parser.add_argument(
        "--val_beam_size",
        type=int,
        default=1,
        help="Beam size during per-epoch validation CIDEr check (1=greedy for fast epoch checks, 5=beam).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no_amp", action="store_true", help="Disable FP16 mixed-precision training.")
    args = parser.parse_args()

    default_prefix_lengths = {
        "global_linear": 1,
        "clipcap_mlp": 10,
        "patch_mlp": 49,
        "query_resampler": 16,
    }
    if args.prefix_length is None:
        args.prefix_length = default_prefix_lengths[args.connector_type]

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 65)
    print(f" TRAINING FLICKR8K MODEL: {args.run_name}")
    print(f" Device: {device} | Mixed Precision (AMP): {use_amp} | Seed: {args.seed}")
    print(f" Connector: {args.connector_type} | Visual Prefix Length: {args.prefix_length}")
    print("=" * 65)

    model = CaptionBaselineModel(
        connector_type=args.connector_type,
        prefix_length=args.prefix_length,
    ).to(device)
    args.prefix_length = model.prefix_length

    param_info = count_parameters(model)
    print(
        f" Trainable Parameters : {param_info['trainable_parameters']:,}\n"
        f" Frozen GPT-2 Params  : {param_info['gpt2_frozen_parameters']:,}\n"
        f" Total Parameters     : {param_info['total_parameters']:,}"
    )

    train_cache_path = os.path.join(args.cache_dir, "train_cache.pt")
    val_cache_path = os.path.join(args.cache_dir, "val_cache.pt")

    train_dataset = Flickr8kCachedDataset(
        cache_path=train_cache_path,
        prefix_length=args.prefix_length,
        max_seq_len=args.max_seq_len,
        feature_type=model.feature_type,
        normalize_prefix=args.normalize_prefix,
    )
    val_loss_dataset = Flickr8kCachedDataset(
        cache_path=val_cache_path,
        prefix_length=args.prefix_length,
        max_seq_len=args.max_seq_len,
        feature_type=model.feature_type,
        normalize_prefix=args.normalize_prefix,
    )
    val_eval_dataset = Flickr8kEvalDataset(
        cache_path=val_cache_path,
        feature_type=model.feature_type,
        normalize_prefix=args.normalize_prefix,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loss_loader = DataLoader(
        val_loss_dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    best_val_cider = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for tokens, labels, mask, feats in pbar:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            feats = feats.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(tokens=tokens, visual_features=feats, mask=mask, labels=labels)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - epoch_start
        train_loss = running_loss / max(len(train_loader), 1)
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )

        # Validation Loss & Validation CIDEr (Proposal Section IV.E)
        val_loss = evaluate_validation_loss(model, val_loss_loader, device, use_amp)
        val_metrics, _ = generate_and_evaluate(
            model=model,
            eval_dataset=val_eval_dataset,
            tokenizer=tokenizer,
            device=device,
            beam_size=args.val_beam_size,
            max_new_tokens=35,
            compute_meteor=False,
            compute_spice=False,
            compute_clipscore_flag=False,
            desc=f"Val Epoch {epoch}",
        )
        val_cider = val_metrics["CIDEr"]

        epoch_record: Dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_CIDEr": val_cider,
            "val_BLEU-1": val_metrics["BLEU-1"],
            "val_BLEU-4": val_metrics["BLEU-4"],
            "val_ROUGE-L": val_metrics["ROUGE-L"],
            "epoch_train_time_sec": epoch_train_time,
            "peak_gpu_memory_mb": peak_mem_mb,
        }
        history.append(epoch_record)

        print(
            f"  [Epoch {epoch:02d}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"val_CIDEr: {val_cider:.2f} | val_BLEU-4: {val_metrics['BLEU-4']:.2f} | "
            f"time: {epoch_train_time:.1f}s | peak_VRAM: {peak_mem_mb:.1f} MB"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
            "param_info": param_info,
        }

        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, f"{args.run_name}_latest.pt")
        torch.save(ckpt_payload, latest_path)

        # Save best checkpoint by Validation CIDEr
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
            torch.save(ckpt_payload, best_path)
            print(f"  --> New best Validation CIDEr ({best_val_cider:.2f})! Saved to {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"  [Early Stopping] No validation CIDEr improvement for {args.patience} epochs. "
                    f"Best epoch was {best_epoch} (CIDEr={best_val_cider:.2f})."
                )
                break

        # Save updated history JSON after each epoch
        summary_path = os.path.join(args.out_dir, f"{args.run_name}_history.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": args.run_name,
                    "args": vars(args),
                    "param_info": param_info,
                    "best_epoch": best_epoch,
                    "best_val_CIDEr": best_val_cider,
                    "history": history,
                },
                f,
                indent=2,
            )

    print("\n" + "=" * 65)
    print(f" TRAINING COMPLETE | Best Epoch: {best_epoch} | Best Val CIDEr: {best_val_cider:.2f}")
    print(f" Best Checkpoint : {os.path.join(args.out_dir, f'{args.run_name}_best.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
