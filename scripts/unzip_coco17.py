import zipfile
from pathlib import Path
from tqdm import tqdm

# --- paths ---
zip_path = Path("/opt/dlami/nvme/datasets/coco17_only_txt.zip")
root_dir = Path("/opt/dlami/nvme/datasets/coco17")

# make sure target dir exists
root_dir.mkdir(parents=True, exist_ok=True)

# 1) Extract ONLY .txt files from the zip, flattened into root_dir
extracted_txt = 0
with zipfile.ZipFile(zip_path, "r") as zf:
    members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
    for m in tqdm(members, desc="extract txt", unit="file"):
        # flatten: drop internal zip folders, keep just the filename
        target = root_dir / Path(m).name
        # if the file already exists, skip overwrite (idempotent)
        if target.exists():
            continue
        with zf.open(m) as src, open(target, "wb") as dst:
            dst.write(src.read())
        extracted_txt += 1

# 2) Delete image files that don't share a prefix with ANY existing .jpg
#    i.e., if basename (without ext) is not present among JPG files, remove it.
IMAGE_EXTS = {".png", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"}
jpg_stems = set()

# collect all .jpg basenames recursively
for p in root_dir.rglob("*.jpg"):
    jpg_stems.add(p.stem)

deleted_images = 0
kept_images = 0

# scan for non-jpg images and remove those whose stem is not in jpg_stems
for ext in IMAGE_EXTS:
    for p in root_dir.rglob(f"*{ext}"):
        if p.stem not in jpg_stems:
            try:
                p.unlink()
                deleted_images += 1
            except Exception as e:
                print(f"[warn] failed to delete {p}: {e}")
        else:
            kept_images += 1

print(f"Done.\n  Extracted txt files: {extracted_txt}\n  Deleted non-matching images: {deleted_images}\n  Kept non-jpg images (matched a .jpg): {kept_images}")
