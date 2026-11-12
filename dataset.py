"""
PyTorch Dataset classes for training and evaluating on cached Flickr8k CLIP features.
Follows Proposal Section IV.A & IV.B:
  - Uses all 5 captions per image as separate training targets (6,000 * 5 = 30,000 targets).
  - Retains all 5 captions per image as references during validation and test evaluation.
  - Tokenizes captions with GPT-2 tokenizer, appends EOS (<|endoftext|>), and masks padding
    with ignore_index = -100 so cross-entropy loss applies strictly to caption tokens.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class Flickr8kCachedDataset(Dataset):
    """
    Training/validation dataset over (image_feature, single_caption) pairs.
    For 6,000 training images with 5 captions each, __len__ is 30,000.
    """

    def __init__(
        self,
        cache_path: str,
        prefix_length: int,
        max_seq_len: int = 40,
        feature_type: str = "global",
        gpt2_type: str = "gpt2",
        normalize_prefix: bool = False,
    ):
        super().__init__()
        assert feature_type in ("global", "patch"), f"Unsupported feature_type: {feature_type}"
        self.prefix_length = prefix_length
        self.max_seq_len = max_seq_len
        self.feature_type = feature_type
        self.normalize_prefix = normalize_prefix

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.eos_token_id = self.tokenizer.eos_token_id  # 50256 (<|endoftext|>)

        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.split = data["split"]
        self.image_ids: List[str] = data["image_ids"]
        self.captions_by_image: Dict[str, List[str]] = data["captions_by_image"]
        self.samples: List[Dict] = data["samples"]

        if feature_type == "global":
            self.features = data["global_features"]  # [N, 512] FP16
        else:
            self.features = data["patch_features"]  # [N, 49, 768] FP16

        # Pre-tokenize all captions with EOS appended
        self.encoded_tokens: List[torch.Tensor] = []
        self.encoded_labels: List[torch.Tensor] = []
        self.encoded_masks: List[torch.Tensor] = []
        self.sample_image_indices: List[int] = []

        for sample in self.samples:
            cap_text = sample["caption"].strip()
            # Encode caption and append EOS token
            token_ids = self.tokenizer.encode(cap_text)
            if len(token_ids) >= self.max_seq_len:
                token_ids = token_ids[: self.max_seq_len - 1] + [self.eos_token_id]
            else:
                token_ids = token_ids + [self.eos_token_id]

            valid_len = len(token_ids)
            pad_len = self.max_seq_len - valid_len

            # Input tokens for GPT-2 embedding lookup (padded with 0)
            tokens = torch.tensor(token_ids + [0] * pad_len, dtype=torch.long)

            # Target labels for cross-entropy loss (-100 is ignored by F.cross_entropy)
            labels = torch.tensor(token_ids + [-100] * pad_len, dtype=torch.long)

            # Attention mask: 1 for visual prefix + valid caption tokens, 0 for padding
            caption_mask = torch.cat(
                [torch.ones(valid_len, dtype=torch.float32), torch.zeros(pad_len, dtype=torch.float32)],
                dim=0,
            )
            full_mask = torch.cat(
                [torch.ones(self.prefix_length, dtype=torch.float32), caption_mask],
                dim=0,
            )

            self.encoded_tokens.append(tokens)
            self.encoded_labels.append(labels)
            self.encoded_masks.append(full_mask)
            self.sample_image_indices.append(sample["image_idx"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encoded_tokens[item]
        labels = self.encoded_labels[item]
        mask = self.encoded_masks[item]
        img_idx = self.sample_image_indices[item]

        feat = self.features[img_idx].float()
        if self.normalize_prefix:
            feat = feat / feat.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)

        return tokens, labels, mask, feat


class Flickr8kEvalDataset(Dataset):
    """
    Evaluation dataset over unique images (1,000 for val, 1,000 for test).
    Returns (image_id, visual_feature) while storing captions_by_image for metric scoring.
    """

    def __init__(
        self,
        cache_path: str,
        feature_type: str = "global",
        normalize_prefix: bool = False,
    ):
        super().__init__()
        assert feature_type in ("global", "patch"), f"Unsupported feature_type: {feature_type}"
        self.feature_type = feature_type
        self.normalize_prefix = normalize_prefix

        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.split = data["split"]
        self.image_ids: List[str] = data["image_ids"]
        self.captions_by_image: Dict[str, List[str]] = data["captions_by_image"]

        if feature_type == "global":
            self.features = data["global_features"]  # [N, 512]
        else:
            self.features = data["patch_features"]  # [N, 49, 768]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        img_id = self.image_ids[idx]
        feat = self.features[idx].float()
        if self.normalize_prefix:
            feat = feat / feat.norm(2, dim=-1, keepdim=True).clamp(min=1e-6)
        return img_id, feat
