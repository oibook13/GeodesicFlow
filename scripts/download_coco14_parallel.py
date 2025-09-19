import os
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_dataset
from tqdm import tqdm

# Optional: enable to delete non-RGB images (very rare)
enforce_rgb_non_rgb_delete = False
try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    Image = None
    enforce_rgb_non_rgb_delete = False  # can't enforce without Pillow

# --------- Config ----------
dataset_name = "UCSC-VLAA/Recap-COCO-30K"
local_dir = "/opt/dlami/nvme/datasets/coco14"
max_workers = 64          # tune for your bandwidth / host
connect_timeout = 10
read_timeout = 60
retries = 3               # automatic retry on transient errors
# ---------------------------

# Make dirs
Path(local_dir).mkdir(parents=True, exist_ok=True)

# Load dataset (splits usually: 'train', maybe 'validation' or others for this repo)
dataset = load_dataset(dataset_name)

# Session with retries (shared across threads)
def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(['GET', 'HEAD']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

SESSION = make_session()

def safe_download(url: str, dest: Path) -> bool:
    """Download to dest via temp file + atomic rename. Returns True if file exists at end."""
    if dest.exists():
        return True
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with SESSION.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as r:
            r.raise_for_status()
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        tmp.replace(dest)  # atomic on same filesystem
        return True
    except requests.exceptions.TooManyRedirects:
        tqdm.write(f"[skip] Too many redirects: {url}")
    except Exception as e:
        tqdm.write(f"[error] {url} -> {e}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    return dest.exists()

def is_rgb_image(path: Path) -> bool:
    """Return True if image opens successfully and is exactly mode 'RGB'."""
    if not enforce_rgb_non_rgb_delete or Image is None:
        return True  # treat as OK if not enforcing
    try:
        with Image.open(path) as im:
            im.load()
            return im.mode == "RGB"
    except Exception as e:
        tqdm.write(f"[img open error] {path}: {e}")
        return False

def extract_base_from_url(img_url: str) -> str:
    """
    From a COCO URL like .../train2014/COCO_train2014_000000522418.jpg
    or .../train2017/000000522418.jpg → return '000000522418'
    """
    fname = os.path.basename(img_url)
    stem, _ = os.path.splitext(fname)
    # Handle prefixes like 'COCO_train2014_000000522418' → keep the last 12 digits if present
    # Otherwise just return the stem.
    tail_digits = ''.join(ch for ch in stem if ch.isdigit())
    if len(tail_digits) >= 12:
        return tail_digits[-12:]
    return stem

def process_one(split_dir: Path, item: dict) -> str:
    """
    Download image and write caption .txt (and delete non-RGB if enabled).
    Returns one of: 'downloaded', 'skipped', 'non_rgb_deleted', or 'error'.
    """
    img_url = item.get("coco_url") or item.get("image_url")
    if not img_url:
        return "error"

    base = extract_base_from_url(img_url)
    img_path = split_dir / f"{base}.jpg"
    txt_path = split_dir / f"{base}.txt"

    # If image exists, ensure caption exists; still verify RGB if enabled
    if img_path.exists():
        if enforce_rgb_non_rgb_delete and not is_rgb_image(img_path):
            img_path.unlink(missing_ok=True)
            txt_path.unlink(missing_ok=True)
            return "non_rgb_deleted"
        if not txt_path.exists():
            try:
                with open(txt_path, "w") as tf:
                    tf.write(str(item.get("caption", "")))
            except Exception as e:
                tqdm.write(f"[txt error] {txt_path}: {e}")
                return "error"
        return "skipped"

    # Download image
    ok = safe_download(img_url, img_path)
    if not ok:
        return "error"

    # Optionally delete non-RGB
    if enforce_rgb_non_rgb_delete and not is_rgb_image(img_path):
        img_path.unlink(missing_ok=True)
        txt_path.unlink(missing_ok=True)
        return "non_rgb_deleted"

    # Write caption .txt (best-effort; only if image exists)
    try:
        with open(txt_path, "w") as tf:
            tf.write(str(item.get("caption", "")))
    except Exception as e:
        tqdm.write(f"[txt error] {txt_path}: {e}")
        return "error"

    return "downloaded"

def run_split(split_name: str):
    split_dir = Path(local_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    ds_split = dataset[split_name]
    total = len(ds_split)

    results = {"downloaded": 0, "skipped": 0, "non_rgb_deleted": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=total, desc=f"{split_name}", unit="img") as pbar:
        futures = (ex.submit(process_one, split_dir, ds_split[i]) for i in range(total))
        for fut in as_completed(futures):
            status = fut.result()
            results[status] = results.get(status, 0) + 1
            pbar.update(1)
            if (pbar.n % 1000) == 0:
                pbar.set_postfix(results)
    tqdm.write(f"[{split_name}] {results}")

def main():
    # Ensure subdirs exist (only for splits we actually have)
    for split in dataset.keys():
        (Path(local_dir) / split).mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        run_split(split)

    print("Dataset downloaded and organized successfully.")

if __name__ == "__main__":
    main()
