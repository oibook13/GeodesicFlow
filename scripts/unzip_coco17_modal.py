import zipfile
from pathlib import Path, PurePosixPath
from tqdm import tqdm
import shutil

# Modal paths - using the datasets volume mount point
ROOT = Path("/datasets")
ZIP_PATH = ROOT / "coco17_only_txt.zip"

# Normalize any of these to canonical split names
ALIASES = {
    "train": "train",
    "train2014": "train",
    "train2017": "train",
    "validation": "validation",
    "val": "validation",
    "val2014": "validation",
    "val2017": "validation",
    "test": "test",
    "test2014": "test",
    "test2017": "test",
}

# Prepare dirs
ROOT.mkdir(parents=True, exist_ok=True)
for s in ("train", "validation", "test", "_staging_txt"):
    (ROOT / s).mkdir(parents=True, exist_ok=True)


def detect_split_from_parts(parts):
    """
    Look through *all* path parts inside the zip for a known split alias.
    Return canonical split ('train'/'validation'/'test') or None.
    """
    for part in parts:
        key = part.lower()
        if key in ALIASES:
            return ALIASES[key]
    return None


def safe_write_bytes(target: Path, src_fileobj):
    tmp = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as dst:
        shutil.copyfileobj(src_fileobj, dst)
    tmp.replace(target)


def extract_txts():
    extracted = {"train": 0, "validation": 0, "test": 0, "staged": 0, "skipped": 0}

    if not ZIP_PATH.exists():
        print(f"Warning: {ZIP_PATH} not found, skipping text extraction")
        return extracted

    print(f"Opening zip file: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_members = zf.namelist()
        members = [m for m in all_members if m.lower().endswith(".txt")]
        print(f"Found {len(members)} text files in zip (out of {len(all_members)} total files)")

        # Process in batches for better progress reporting
        batch_size = 1000
        total_batches = (len(members) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(members))
            batch_members = members[start_idx:end_idx]

            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_members)} files)")

            for m in batch_members:
                p = PurePosixPath(m)  # zip paths are POSIX-like
                if m.endswith("/"):
                    continue  # skip directories just in case

                split = detect_split_from_parts(p.parts)
                # Flatten (drop internal subfolders), keep only filename
                target_dir = ROOT / (split if split else "_staging_txt")
                target = target_dir / p.name

                if target.exists():
                    extracted["skipped"] += 1
                    continue

                with zf.open(m) as src:
                    safe_write_bytes(target, src)
                if split:
                    extracted[split] += 1
                else:
                    extracted["staged"] += 1

            # Progress update every batch
            total_processed = sum(extracted.values())
            print(f"  Batch {batch_idx + 1} complete. Total processed: {total_processed}")

    return extracted


def build_jpg_indexes():
    """Return dict: split -> {stem set} and also a global stem->split map for resolving staged files."""
    jpg_index = {}
    stem_to_split = {}
    for split in ("train", "validation", "test"):
        split_dir = ROOT / split
        stems = {p.stem for p in split_dir.rglob("*.jpg")}
        jpg_index[split] = stems
        for s in stems:
            # If a stem appears in multiple splits (rare), prefer train > validation > test
            if (
                s not in stem_to_split
                or split == "train"
                or (stem_to_split[s] == "test" and split == "validation")
            ):
                stem_to_split[s] = split
    return jpg_index, stem_to_split


def place_staged_txt(jpg_index, stem_to_split):
    """Move _staging_txt captions next to their JPG split; delete if no matching JPG anywhere."""
    staged_dir = ROOT / "_staging_txt"
    moved, deleted = 0, 0
    for txt in staged_dir.glob("*.txt"):
        stem = txt.stem
        split = stem_to_split.get(stem)
        if split:
            dest = ROOT / split / f"{stem}.txt"
            if dest.exists():
                # already have one; treat staged as duplicate
                txt.unlink(missing_ok=True)
                deleted += 1
            else:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(txt), str(dest))
                    moved += 1
                except Exception:
                    try:
                        shutil.copy2(str(txt), str(dest))
                        txt.unlink(missing_ok=True)
                        moved += 1
                    except Exception as e:
                        print(f"[warn] failed to place {txt} -> {dest}: {e}")
        else:
            # no JPG anywhere -> remove
            txt.unlink(missing_ok=True)
            deleted += 1
    return moved, deleted


def enforce_per_split(split, jpg_stems):
    """
    In split dir:
      - keep at most one .txt per JPG basename
      - delete any .txt with no matching .jpg
    """
    split_dir = ROOT / split
    kept = 0
    del_unmatched = 0
    del_dups = 0
    seen = set()

    # Iterate deterministically to keep the same file each run
    for txt in sorted(split_dir.rglob("*.txt"), key=lambda p: p.name):
        stem = txt.stem
        if stem not in jpg_stems:
            txt.unlink(missing_ok=True)
            del_unmatched += 1
            continue
        if stem in seen:
            txt.unlink(missing_ok=True)
            del_dups += 1
            continue
        seen.add(stem)
        kept += 1
    return kept, del_unmatched, del_dups


def main():
    print(f"Starting text file extraction from {ZIP_PATH}")
    print(f"Dataset root: {ROOT}")

    # Check if ZIP file exists
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found!")
        print("Available files in /datasets:")
        if ROOT.exists():
            for f in ROOT.iterdir():
                print(f"  {f}")
        return

    stats = extract_txts()
    print(f"Extracted: {stats}")

    jpg_index, stem_to_split = build_jpg_indexes()
    moved, deleted = place_staged_txt(jpg_index, stem_to_split)
    print(f"Staged moved: {moved}, staged deleted: {deleted}")

    for split in ("train", "validation", "test"):
        kept, del_unmatched, del_dups = enforce_per_split(split, jpg_index[split])
        print(
            f"[{split}] kept={kept} del_unmatched={del_unmatched} del_dups={del_dups}"
        )

    # Optional: clean up empty staging dir
    try:
        (ROOT / "_staging_txt").rmdir()
    except Exception:
        pass

    print("Text file extraction and organization completed!")


if __name__ == "__main__":
    main()