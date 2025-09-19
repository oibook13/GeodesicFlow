from pathlib import Path

# Root dataset directory
root_dir = Path("/opt/dlami/nvme/datasets/coco17")

# Collect all jpg basenames
jpg_stems = {p.stem for p in root_dir.rglob("*.jpg")}

deleted_unmatched = 0
deleted_duplicates = 0
kept = 0

seen = set()  # track which txt stems we've already kept

for txt_file in root_dir.rglob("*.txt"):
    stem = txt_file.stem

    # Case 1: no matching jpg -> delete
    if stem not in jpg_stems:
        try:
            txt_file.unlink()
            deleted_unmatched += 1
        except Exception as e:
            print(f"[warn] failed to delete {txt_file}: {e}")
        continue

    # Case 2: duplicate txt -> delete
    if stem in seen:
        try:
            txt_file.unlink()
            deleted_duplicates += 1
        except Exception as e:
            print(f"[warn] failed to delete {txt_file}: {e}")
        continue

    # Case 3: keep the first valid txt
    seen.add(stem)
    kept += 1

print(f"Done.")
print(f"  Kept: {kept} unique txt files with matching jpg")
print(f"  Deleted unmatched txt files: {deleted_unmatched}")
print(f"  Deleted duplicate txt files: {deleted_duplicates}")
