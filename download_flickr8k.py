"""
Automatic downloader and verifier for the Flickr8k dataset (images + captions + splits).
Designed to run seamlessly on Google Colab (saving directly to Google Drive or local disk).
"""

import argparse
import os
import urllib.request
import zipfile
from tqdm import tqdm

DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def find_images_dir(base_dir: str) -> str:
    """Locate the directory containing Flickr8k .jpg images."""
    candidates = [
        os.path.join(base_dir, "Flicker8k_Dataset"),
        os.path.join(base_dir, "Flickr8k_Dataset"),
        os.path.join(base_dir, "images"),
        base_dir,
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            jpgs = [f for f in os.listdir(cand) if f.lower().endswith(".jpg")]
            if len(jpgs) >= 8000:
                return cand
    return ""


def verify_dataset(data_dir: str) -> bool:
    images_dir = find_images_dir(data_dir)
    token_file = os.path.join(data_dir, "Flickr8k.token.txt")
    splits = [
        "Flickr_8k.trainImages.txt",
        "Flickr_8k.devImages.txt",
        "Flickr_8k.testImages.txt",
    ]
    if not images_dir or not os.path.isfile(token_file):
        return False
    for sp in splits:
        if not os.path.isfile(os.path.join(data_dir, sp)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and verify Flickr8k dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/flickr8k",
        help="Target directory to extract Flickr8k dataset.",
    )
    parser.add_argument(
        "--zip_cache_dir",
        type=str,
        default="",
        help="Optional directory (e.g. on Google Drive) to persist the raw .zip files so they never need to be re-downloaded.",
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep .zip archives after extraction.",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    zip_dir = args.zip_cache_dir if args.zip_cache_dir else args.data_dir
    os.makedirs(zip_dir, exist_ok=True)
    keep_zip = args.keep_zip or bool(args.zip_cache_dir)

    if verify_dataset(args.data_dir):
        img_dir = find_images_dir(args.data_dir)
        print(f"[OK] Flickr8k is already downloaded and verified at: {args.data_dir}")
        print(f"     Images directory: {img_dir}")
        return

    # 1. Download & extract Flickr8k_text.zip
    text_zip = os.path.join(zip_dir, "Flickr8k_text.zip")
    if not os.path.isfile(os.path.join(args.data_dir, "Flickr8k.token.txt")):
        if not os.path.isfile(text_zip):
            print(f"Downloading Flickr8k_text.zip to {text_zip} ...")
            download_url(TEXT_URL, text_zip)
        print(f"Extracting {text_zip} to {args.data_dir} ...")
        with zipfile.ZipFile(text_zip, "r") as zf:
            zf.extractall(args.data_dir)
        if not keep_zip and os.path.isfile(text_zip):
            os.remove(text_zip)

    # 2. Download & extract Flickr8k_Dataset.zip
    dataset_zip = os.path.join(zip_dir, "Flickr8k_Dataset.zip")
    if not find_images_dir(args.data_dir):
        if not os.path.isfile(dataset_zip):
            print(f"Downloading Flickr8k_Dataset.zip (~1.11 GB) to {dataset_zip} ...")
            download_url(DATASET_URL, dataset_zip)
        print(f"Extracting {dataset_zip} to {args.data_dir} ...")
        with zipfile.ZipFile(dataset_zip, "r") as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            for member in tqdm(members, desc="Extracting images"):
                zf.extract(member, args.data_dir)
        if not keep_zip and os.path.isfile(dataset_zip):
            os.remove(dataset_zip)

    # 3. Verify everything is in place
    if verify_dataset(args.data_dir):
        img_dir = find_images_dir(args.data_dir)
        num_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
        print(f"[SUCCESS] Flickr8k dataset ready at {args.data_dir} ({num_imgs} images found).")
    else:
        raise RuntimeError(f"Dataset verification failed in {args.data_dir}. Please check the contents.")


if __name__ == "__main__":
    main()
