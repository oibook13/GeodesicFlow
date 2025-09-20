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

# --- NEW: image mode check ---
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly truncated files

# --------- Config for Modal ----------
dataset_name = "phiyodr/coco2017"
local_dir = "/datasets"  # Modal volume mount point
max_workers = 64  # tune for your bandwidth / host
connect_timeout = 10
read_timeout = 60
retries = 3  # automatic retry on transient errors
# ---------------------------

# Make dirs
Path(local_dir).mkdir(parents=True, exist_ok=True)

# Load dataset
dataset = load_dataset(
    dataset_name
)  # keys usually: 'train', 'validation' (and maybe 'test')


# Session with retries (shared across threads)
def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


SESSION = make_session()


def safe_download(url: str, dest: Path) -> None:
    """Download to dest via temp file + atomic rename. Skips if exists."""
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with SESSION.get(
            url, stream=True, timeout=(connect_timeout, read_timeout)
        ) as r:
            r.raise_for_status()
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        tmp.replace(dest)  # atomic on same filesystem
    except requests.exceptions.TooManyRedirects:
        tqdm.write(f"[skip] Too many redirects: {url}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    except Exception as e:
        tqdm.write(f"[error] {url} -> {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        # Don't re-raise; we just log and continue


def is_rgb_image(path: Path) -> bool:
    """Return True if image opens successfully and is exactly mode 'RGB'."""
    try:
        with Image.open(path) as im:
            im.load()  # force decode
            return im.mode == "RGB"
    except Exception as e:
        tqdm.write(f"[img open error] {path}: {e}")
        return False  # treat unreadable as not-RGB/bad


def process_one(split_dir: Path, item: dict) -> str:
    """
    Download image and write JSON, deleting any non-RGB images.
    Returns 'downloaded', 'skipped', 'non_rgb_deleted', or 'error' (logged).
    """
    # file_name looks like "train2017/000000522418.jpg" → want "000000522418"
    base = os.path.splitext(os.path.basename(item["file_name"]))[0]
    img_url = item.get("coco_url")
    if not img_url:
        return "error"

    img_path = split_dir / f"{base}.jpg"
    json_path = split_dir / f"{base}.json"

    # If image already exists, ensure JSON exists; we'll still verify RGB below
    if img_path.exists() and not json_path.exists():
        try:
            with open(json_path, "w") as jf:
                json.dump(item, jf)
        except Exception as e:
            tqdm.write(f"[json error] {json_path}: {e}")
            return "error"

    # Download if missing
    if not img_path.exists():
        safe_download(img_url, img_path)
        if not img_path.exists():
            return "error"

    # Check color mode; delete non-RGB (and its JSON to keep dataset clean)
    if not is_rgb_image(img_path):
        img_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return "non_rgb_deleted"

    # Save/overwrite JSON (best-effort)
    try:
        with open(json_path, "w") as jf:
            json.dump(item, jf)
    except Exception as e:
        tqdm.write(f"[json error] {json_path}: {e}")
        return "error"

    # If we reached here and the file existed before, call it 'skipped'
    return (
        "skipped"
        if "downloaded" not in locals() and not (json_path.stat().st_size == 0)
        else "downloaded"
    )


def run_split(split_name: str):
    split_dir = Path(local_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    ds_split = dataset[split_name]
    total = len(ds_split)

    # Use a thread pool and submit per item
    results = {"downloaded": 0, "skipped": 0, "non_rgb_deleted": 0, "error": 0}
    with (
        ThreadPoolExecutor(max_workers=max_workers) as ex,
        tqdm(total=total, desc=f"{split_name}", unit="img") as pbar,
    ):
        futures = (ex.submit(process_one, split_dir, ds_split[i]) for i in range(total))
        for fut in as_completed(futures):
            status = fut.result()
            results[status] = results.get(status, 0) + 1
            pbar.update(1)
            # Optional: show occasional stats
            if (pbar.n % 1000) == 0:
                pbar.set_postfix(results)
    tqdm.write(f"[{split_name}] {results}")


def main():
    print(f"Downloading COCO17 dataset to Modal volume: {local_dir}")

    # Ensure subdirs exist (only for splits we actually have)
    for split in dataset.keys():
        (Path(local_dir) / split).mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        print(f"Processing split: {split}")
        run_split(split)

    print("Dataset downloaded and organized successfully.")

    # Print summary
    for split in dataset.keys():
        split_dir = Path(local_dir) / split
        if split_dir.exists():
            image_count = len(list(split_dir.glob("*.jpg")))
            json_count = len(list(split_dir.glob("*.json")))
            print(f"Split {split}: {image_count} images, {json_count} JSON files")


if __name__ == "__main__":
    main()