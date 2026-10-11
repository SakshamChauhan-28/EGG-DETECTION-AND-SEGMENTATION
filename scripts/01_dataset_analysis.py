import os
from pathlib import Path
from PIL import Image
import csv
import json
import hashlib
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent / "data" / "egg.v1i.folder")))

OUTPUT_DIR = Path("dataset_report")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# FIND IMAGES
# ============================================================

images = []

for path in DATASET_ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        images.append(path)

images = sorted(images)

print("=" * 70)
print("Nosh Robotics - Egg Dataset Analysis")
print("=" * 70)

print(f"\nDataset root:")
print(DATASET_ROOT)

print(f"\nTotal images found: {len(images)}")

# ============================================================
# DIRECTORY / SPLIT ANALYSIS
# ============================================================

split_counts = Counter()

for image_path in images:
    relative = image_path.relative_to(DATASET_ROOT)
    parts = relative.parts

    if parts:
        first_folder = parts[0].lower()

        if first_folder in {"train", "valid", "val", "test"}:
            split_counts[first_folder] += 1

print("\nDataset splits:")
for split, count in sorted(split_counts.items()):
    print(f"  {split}: {count}")

# ============================================================
# IMAGE ANALYSIS
# ============================================================

records = []
dimension_counts = Counter()
format_counts = Counter()

print("\nAnalyzing images...")

for i, image_path in enumerate(images, start=1):

    try:
        with Image.open(image_path) as img:

            width, height = img.size
            mode = img.mode
            fmt = img.format

            dimension_counts[(width, height)] += 1
            format_counts[fmt] += 1

            # File size
            file_size_kb = image_path.stat().st_size / 1024

            records.append({
                "path": str(image_path.relative_to(DATASET_ROOT)),
                "width": width,
                "height": height,
                "mode": mode,
                "format": fmt,
                "file_size_kb": round(file_size_kb, 2)
            })

    except Exception as e:
        print(f"ERROR: {image_path}")
        print(e)

    if i % 20 == 0:
        print(f"  Processed {i}/{len(images)}")

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nImages: {len(records)}")

print("\nImage dimensions:")
for dimensions, count in dimension_counts.most_common():
    print(f"  {dimensions[0]} x {dimensions[1]} : {count}")

print("\nImage formats:")
for fmt, count in format_counts.items():
    print(f"  {fmt}: {count}")

# ============================================================
# SAVE IMAGE METADATA
# ============================================================

csv_path = OUTPUT_DIR / "image_metadata.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path",
            "width",
            "height",
            "mode",
            "format",
            "file_size_kb"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved metadata:")
print(csv_path)

# ============================================================
# DUPLICATE DETECTION
# ============================================================

print("\nChecking for exact duplicate files...")

hash_to_files = {}

for image_path in images:

    try:
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        hash_to_files.setdefault(file_hash, []).append(
            str(image_path.relative_to(DATASET_ROOT))
        )

    except Exception:
        pass

duplicate_groups = {
    h: files
    for h, files in hash_to_files.items()
    if len(files) > 1
}

print(f"Exact duplicate groups: {len(duplicate_groups)}")

if duplicate_groups:

    for group_id, files in enumerate(duplicate_groups.values(), start=1):

        print(f"\nDuplicate group {group_id}:")
        for file in files:
            print(f"  {file}")

# ============================================================
# SAVE DUPLICATE REPORT
# ============================================================

duplicate_report = OUTPUT_DIR / "duplicate_report.json"

with open(duplicate_report, "w", encoding="utf-8") as f:
    json.dump(duplicate_groups, f, indent=4)

print(f"\nSaved duplicate report:")
print(duplicate_report)

# ============================================================
# FINAL JSON SUMMARY
# ============================================================

summary = {
    "dataset_root": str(DATASET_ROOT),
    "total_images": len(records),
    "splits": dict(split_counts),
    "dimensions": {
        f"{w}x{h}": count
        for (w, h), count in dimension_counts.items()
    },
    "formats": dict(format_counts),
    "exact_duplicate_groups": len(duplicate_groups)
}

summary_path = OUTPUT_DIR / "dataset_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"\nSaved summary:")
print(summary_path)

print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETE")
print("=" * 70)