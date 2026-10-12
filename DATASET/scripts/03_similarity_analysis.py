import os
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data" / "egg.v1i.folder"))
)

OUTPUT_DIR = Path("dataset_report")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Extract a compact visual representation using:
      - resized grayscale image
      - HSV color histogram

    This is intended for dataset similarity analysis,
    not for object recognition.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # 1. Grayscale appearance
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    gray_features = gray.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # 2. HSV color histogram
    # --------------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    ).flatten()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    features = np.concatenate([
        gray_features.flatten(),
        hist
    ])

    return features


# ============================================================
# EXTRACT FEATURES
# ============================================================

print("=" * 70)
print("Nosh Robotics - Near Duplicate / Leakage Analysis")
print("=" * 70)

print(f"\nImages found: {len(images)}")
print("\nExtracting visual features...")

features = []
records = []

for i, image_path in enumerate(images, start=1):

    feature = extract_features(image_path)

    features.append(feature)

    relative = image_path.relative_to(DATASET_ROOT)

    parts = relative.parts

    split = parts[0] if parts else "unknown"

    records.append({
        "path": str(relative),
        "split": split
    })

    if i % 20 == 0:
        print(f"  Processed {i}/{len(images)}")


features = np.asarray(features, dtype=np.float32)


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(features_scaled)


# ============================================================
# FIND HIGH-SIMILARITY PAIRS
# ============================================================

THRESHOLD = 0.90

pairs = []

for i in range(len(images)):

    for j in range(i + 1, len(images)):

        similarity = similarity_matrix[i, j]

        if similarity >= THRESHOLD:

            pairs.append({
                "image_1": records[i]["path"],
                "split_1": records[i]["split"],
                "image_2": records[j]["path"],
                "split_2": records[j]["split"],
                "similarity": float(similarity)
            })


pairs.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY RESULTS")
print("=" * 70)

print(f"\nSimilarity threshold: {THRESHOLD}")
print(f"High-similarity pairs: {len(pairs)}")

print("\nTop 20 most similar pairs:\n")

for pair in pairs[:20]:

    print(
        f"{pair['similarity']:.4f} | "
        f"{pair['split_1']:5s} | "
        f"{pair['image_1']}  <-->  "
        f"{pair['split_2']:5s} | "
        f"{pair['image_2']}"
    )


# ============================================================
# SAVE SIMILARITY PAIRS
# ============================================================

csv_path = OUTPUT_DIR / "similarity_pairs.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_1",
            "split_1",
            "image_2",
            "split_2",
            "similarity"
        ]
    )

    writer.writeheader()
    writer.writerows(pairs)


print(f"\nSaved:")
print(csv_path)


# ============================================================
# CROSS-SPLIT SUMMARY
# ============================================================

cross_split = defaultdict(int)

for pair in pairs:

    if pair["split_1"] != pair["split_2"]:

        key = tuple(
            sorted([
                pair["split_1"],
                pair["split_2"]
            ])
        )

        cross_split[key] += 1


print("\n" + "=" * 70)
print("CROSS-SPLIT SIMILARITY")
print("=" * 70)

if cross_split:

    for pair_type, count in sorted(cross_split.items()):

        print(
            f"{pair_type[0]} <-> {pair_type[1]} : "
            f"{count} pairs"
        )

else:

    print("No high-similarity cross-split pairs found.")


# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nCreating PCA visualization...")

pca = PCA(n_components=2)

features_2d = pca.fit_transform(features_scaled)

split_colors = {
    "train": "tab:blue",
    "valid": "tab:orange",
    "test": "tab:green"
}

plt.figure(figsize=(10, 7))

for split in ["train", "valid", "test"]:

    indices = [
        i
        for i, record in enumerate(records)
        if record["split"] == split
    ]

    if not indices:
        continue

    plt.scatter(
        features_2d[indices, 0],
        features_2d[indices, 1],
        label=split,
        alpha=0.75,
        s=50
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Dataset Visual Similarity - PCA")
plt.legend()
plt.grid(alpha=0.2)

pca_path = OUTPUT_DIR / "dataset_similarity_pca.png"

plt.tight_layout()
plt.savefig(pca_path, dpi=200)
plt.close()

print(f"Saved:")
print(pca_path)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY ANALYSIS COMPLETE")
print("=" * 70)