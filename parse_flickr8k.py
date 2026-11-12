"""
Extract and cache frozen CLIP ViT-B/32 features (both 1x512 global features for the
pipeline sanity baseline and 49x768 patch tokens in FP16 for the main 2x2 study)
across the standard Flickr8k train (6,000), val (1,000), and test (1,000) splits.
"""

import argparse
import os
from typing import Dict, List, Tuple

import clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from download_flickr8k import find_images_dir


class Flickr8kImageDataset(Dataset):
    def __init__(self, image_ids: List[str], images_dir: str, preprocess):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        tensor = self.preprocess(image)
        return idx, img_id, tensor


def load_captions(data_dir: str) -> Dict[str, List[str]]:
    """
    Parse Flickr8k captions from Flickr8k.token.txt (standard format)
    or captions.txt (CSV format fallback).
    Returns a dict mapping image filename -> list of 5 captions.
    """
    token_path = os.path.join(data_dir, "Flickr8k.token.txt")
    csv_path = os.path.join(data_dir, "captions.txt")

    captions_by_image: Dict[str, List[str]] = {}

    if os.path.isfile(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_tag, caption = line.split("\t", 1)
                img_id = img_tag.split("#")[0].strip()
                captions_by_image.setdefault(img_id, []).append(caption.strip())
    elif os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or (i == 0 and line.lower().startswith("image,caption")):
                    continue
                img_id, caption = line.split(",", 1)
                captions_by_image.setdefault(img_id.strip(), []).append(caption.strip().strip('"'))
    else:
        raise FileNotFoundError(
            f"Could not find Flickr8k.token.txt or captions.txt in {data_dir}. "
            "Run `python download_flickr8k.py --data_dir <path>` first."
        )

    return captions_by_image


def load_split_ids(split_file: str) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def extract_clip_vit_features(
    clip_model, images: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract both:
      1. global_features: [B, 512] post-projection global image embedding.
      2. patch_features:  [B, 49, 768] final visual Transformer tokens before global pooling,
                          with the class token removed and fixed ln_post normalization applied
                          (Proposal Section IV.B).
    """
    visual = clip_model.visual
    x = images.type(visual.conv1.weight.dtype)
    x = visual.conv1(x)  # [B, 768, 7, 7]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, 49, 768]

    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)  # [B, 50, 768]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND [50, B, 768]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD [B, 50, 768]

    # Apply fixed final LayerNorm to all tokens
    x_norm = visual.ln_post(x)  # [B, 50, 768]

    # 1. Patch tokens (CLS token at index 0 removed -> 49 x 768)
    patch_features = x_norm[:, 1:, :].to(dtype=torch.float16).cpu()

    # 2. Global pooled feature (CLS token projected to 512-d, identical to encode_image)
    cls_feat = x_norm[:, 0, :]
    if visual.proj is not None:
        global_features = (cls_feat @ visual.proj).to(dtype=torch.float16).cpu()
    else:
        global_features = cls_feat.to(dtype=torch.float16).cpu()

    return global_features, patch_features


def process_split(
    split_name: str,
    image_ids: List[str],
    images_dir: str,
    captions_by_image: Dict[str, List[str]],
    clip_model,
    preprocess,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    out_path: str,
):
    dataset = Flickr8kImageDataset(image_ids, images_dir, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_global = []
    all_patches = []

    for _, _, batch_imgs in tqdm(loader, desc=f"Caching {split_name} ({len(image_ids)} imgs)"):
        batch_imgs = batch_imgs.to(device)
        global_feat, patch_feat = extract_clip_vit_features(clip_model, batch_imgs)
        all_global.append(global_feat)
        all_patches.append(patch_feat)

    global_tensor = torch.cat(all_global, dim=0)  # [N, 512] FP16
    patch_tensor = torch.cat(all_patches, dim=0)  # [N, 49, 768] FP16

    split_captions_by_image = {}
    samples = []
    for img_idx, img_id in enumerate(image_ids):
        if img_id not in captions_by_image:
            raise KeyError(f"Image {img_id} in split {split_name} has no captions in token file.")
        caps = captions_by_image[img_id][:5]
        split_captions_by_image[img_id] = caps
        for cap_idx, cap_text in enumerate(caps):
            samples.append(
                {
                    "image_id": img_id,
                    "image_idx": img_idx,
                    "caption_idx": cap_idx,
                    "caption": cap_text,
                }
            )

    payload = {
        "split": split_name,
        "clip_model": "ViT-B/32",
        "image_ids": image_ids,
        "global_features": global_tensor,
        "patch_features": patch_tensor,
        "captions_by_image": split_captions_by_image,
        "samples": samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(
        f"  -> Saved {out_path} ({size_mb:.1f} MB) | "
        f"global: {tuple(global_tensor.shape)} | patches: {tuple(patch_tensor.shape)} | "
        f"captions: {len(samples)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Cache CLIP ViT-B/32 features for Flickr8k.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/flickr8k",
        help="Directory containing Flickr8k images and Flickr8k.token.txt.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="./data/flickr8k/splits",
        help="Directory containing Flickr_8k.{train,dev,test}Images.txt.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/flickr8k/cache",
        help="Output directory to store train_cache.pt, val_cache.pt, test_cache.pt.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cache exists.")
    args = parser.parse_args()

    split_files = {
        "train": "Flickr_8k.trainImages.txt",
        "val": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }

    # Check if all caches already exist
    expected_caches = [os.path.join(args.cache_dir, f"{sp}_cache.pt") for sp in split_files]
    if not args.overwrite and all(os.path.isfile(p) for p in expected_caches):
        print(f"[OK] All CLIP feature caches already exist in {args.cache_dir}:")
        for p in expected_caches:
            size_mb = os.path.getsize(p) / (1024 ** 2)
            print(f"     - {os.path.basename(p)} ({size_mb:.1f} MB)")
        return

    images_dir = find_images_dir(args.data_dir)
    if not images_dir:
        raise FileNotFoundError(
            f"Could not find Flickr8k images in {args.data_dir}. "
            "Run `python download_flickr8k.py --data_dir <path>` first."
        )

    captions_by_image = load_captions(args.data_dir)
    print(f"Loaded captions for {len(captions_by_image)} images from {args.data_dir}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP ViT-B/32 on {device} ...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    for split_name, filename in split_files.items():
        out_path = os.path.join(args.cache_dir, f"{split_name}_cache.pt")
        if not args.overwrite and os.path.isfile(out_path):
            print(f"[SKIP] {out_path} already exists.")
            continue

        split_path = os.path.join(args.splits_dir, filename)
        if not os.path.isfile(split_path):
            split_path = os.path.join(args.data_dir, filename)
        image_ids = load_split_ids(split_path)

        process_split(
            split_name=split_name,
            image_ids=image_ids,
            images_dir=images_dir,
            captions_by_image=captions_by_image,
            clip_model=clip_model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            out_path=out_path,
        )

    print("[DONE] All Flickr8k CLIP ViT-B/32 feature caches generated successfully.")


if __name__ == "__main__":
    main()
