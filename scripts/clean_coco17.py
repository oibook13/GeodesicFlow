from pathlib import Path

train_dir = Path("/opt/dlami/nvme/datasets/coco17/train")

jpg_stems = {p.stem for p in train_dir.glob("*.jpg")}
json_stems = {p.stem for p in train_dir.glob("*.json")}
txt_stems = {p.stem for p in train_dir.glob("*.txt")}

# Rogue JSONs (no matching JPG)
rogue_jsons = [train_dir / f"{stem}.json" for stem in (json_stems - jpg_stems)]

# Rogue TXTs (no matching JPG)
rogue_txts = [train_dir / f"{stem}.txt" for stem in (txt_stems - jpg_stems)]

# JPGs missing JSON
jpg_missing_json = [train_dir / f"{stem}.jpg" for stem in (jpg_stems - json_stems)]

# JPGs missing TXT
jpg_missing_txt = [train_dir / f"{stem}.jpg" for stem in (jpg_stems - txt_stems)]

print("=== Rogue JSONs (delete these) ===")
for f in rogue_jsons:
    print(f)

print("\n=== Rogue TXTs (should not exist) ===")
for f in rogue_txts:
    print(f)

print("\n=== JPGs missing JSON ===")
for f in jpg_missing_json[:20]:  # only show first 20
    print(f)
if len(jpg_missing_json) > 20:
    print(f"... and {len(jpg_missing_json)-20} more")

print("\n=== JPGs missing TXT ===")
for f in jpg_missing_txt[:20]:
    print(f)
if len(jpg_missing_txt) > 20:
    print(f"... and {len(jpg_missing_txt)-20} more")

# Uncomment this block to auto-delete the rogues
# for f in rogue_jsons + rogue_txts:
#     print(f"Deleting {f}")
#     f.unlink()
