import os
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data" / "egg.v1i.folder"))
)

OUTPUT_DIR = Path("dataset_report/scene_grouping")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ------------------------------------------------------------
# Similarity configuration
# ------------------------------------------------------------

# This is intentionally a candidate threshold.
# It should NOT be treated as ground truth.
SIMILARITY_THRESHOLD = 0.82

# Minimum number of images that makes a candidate group
MIN_GROUP_SIZE = 2

# Number of most similar images stored for each image
TOP_K = 5

# Contact sheet size
THUMBNAIL_SIZE = 180
COLUMNS = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("Nosh Robotics - Scene Grouping Analysis")
print("=" * 75)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
)

print(f"\nImages found: {len(images)}")


# ============================================================
# LOAD PRETRAINED RESNET
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Remove final classification layer.
# The output becomes a visual feature representation.
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("ResNet18 loaded.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_resnet_feature(image_path):
    """
    Extract a global visual embedding from the image.
    """

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        feature = model(tensor)

    feature = feature.squeeze(0).cpu().numpy()

    # Normalize
    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature


# ============================================================
# COLOR FEATURE
# ============================================================

def extract_color_feature(image_path):
    """
    Extract a compact HSV color histogram.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

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

    return hist.astype(np.float32)


# ============================================================
# EXTRACT FEATURES
# ============================================================

print("\nExtracting visual features...")

resnet_features = []
color_features = []

records = []

for index, image_path in enumerate(images):

    print(
        f"\rProcessing {index + 1}/{len(images)}",
        end=""
    )

    resnet_feature = extract_resnet_feature(
        image_path
    )

    color_feature = extract_color_feature(
        image_path
    )

    resnet_features.append(resnet_feature)
    color_features.append(color_feature)

    relative = image_path.relative_to(
        DATASET_ROOT
    )

    split = relative.parts[0]

    records.append({
        "index": index,
        "path": str(relative),
        "split": split
    })

print("\nFeature extraction complete.")


resnet_features = np.asarray(
    resnet_features,
    dtype=np.float32
)

color_features = np.asarray(
    color_features,
    dtype=np.float32
)


# ============================================================
# NORMALIZE COLOR FEATURES
# ============================================================

color_norm = np.linalg.norm(
    color_features,
    axis=1,
    keepdims=True
)

color_norm[color_norm == 0] = 1

color_features = (
    color_features / color_norm
)


# ============================================================
# SIMILARITY MATRICES
# ============================================================

print("\nComputing similarity matrices...")

resnet_similarity = cosine_similarity(
    resnet_features
)

color_similarity = cosine_similarity(
    color_features
)


# ============================================================
# COMBINED SIMILARITY
# ============================================================

# Scene appearance is primarily captured by ResNet.
# Color is used as an additional signal.

combined_similarity = (
    0.80 * resnet_similarity
    +
    0.20 * color_similarity
)


# ============================================================
# REMOVE SELF-SIMILARITY
# ============================================================

np.fill_diagonal(
    combined_similarity,
    -1
)


# ============================================================
# TOP-K SIMILAR IMAGES FOR EVERY IMAGE
# ============================================================

top_k_records = []

for i in range(len(images)):

    similarity_row = combined_similarity[i]

    top_indices = np.argsort(
        similarity_row
    )[::-1][:TOP_K]

    for rank, j in enumerate(top_indices, start=1):

        top_k_records.append({

            "image": records[i]["path"],

            "image_split": records[i]["split"],

            "rank": rank,

            "similar_image": records[j]["path"],

            "similar_split": records[j]["split"],

            "similarity": round(
                float(similarity_row[j]),
                5
            )
        })


top_k_df = pd.DataFrame(
    top_k_records
)

top_k_path = (
    OUTPUT_DIR /
    "top_similar_images.csv"
)

top_k_df.to_csv(
    top_k_path,
    index=False
)

print(f"\nSaved:")
print(top_k_path)


# ============================================================
# CREATE CANDIDATE PAIRS
# ============================================================

candidate_pairs = []

for i in range(len(images)):

    for j in range(i + 1, len(images)):

        similarity = combined_similarity[i, j]

        if similarity >= SIMILARITY_THRESHOLD:

            candidate_pairs.append({

                "index_1": i,

                "image_1": records[i]["path"],

                "split_1": records[i]["split"],

                "index_2": j,

                "image_2": records[j]["path"],

                "split_2": records[j]["split"],

                "similarity": round(
                    float(similarity),
                    5
                )
            })


candidate_pairs.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# ============================================================
# SAVE CANDIDATE PAIRS
# ============================================================

candidate_path = (
    OUTPUT_DIR /
    "candidate_scene_pairs.csv"
)

candidate_df = pd.DataFrame(
    candidate_pairs
)

candidate_df.to_csv(
    candidate_path,
    index=False
)

print(f"Saved:")
print(candidate_path)


# ============================================================
# UNION-FIND FOR CANDIDATE GROUPS
# ============================================================

parent = list(range(len(images)))


def find(x):

    while parent[x] != x:

        parent[x] = parent[parent[x]]

        x = parent[x]

    return x


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:

        parent[root_b] = root_a


# Connect candidate similar images

for pair in candidate_pairs:

    union(
        pair["index_1"],
        pair["index_2"]
    )


# ============================================================
# BUILD GROUPS
# ============================================================

groups = defaultdict(list)

for index in range(len(images)):

    root = find(index)

    groups[root].append(index)


candidate_groups = [

    members
    for members in groups.values()
    if len(members) >= MIN_GROUP_SIZE
]


# Sort largest groups first

candidate_groups.sort(
    key=len,
    reverse=True
)


print("\n" + "=" * 75)
print("CANDIDATE SCENE GROUPS")
print("=" * 75)

print(
    f"\nSimilarity threshold: "
    f"{SIMILARITY_THRESHOLD}"
)

print(
    f"Candidate groups found: "
    f"{len(candidate_groups)}"
)


for group_id, members in enumerate(
    candidate_groups,
    start=1
):

    print(
        f"\nCandidate Group {group_id} "
        f"({len(members)} images)"
    )

    for index in members:

        print(
            f"  [{records[index]['split']}] "
            f"{records[index]['path']}"
        )


# ============================================================
# SAVE GROUP ASSIGNMENTS
# ============================================================

group_records = []

for group_id, members in enumerate(
    candidate_groups,
    start=1
):

    for index in members:

        group_records.append({

            "candidate_group": (
                f"candidate_{group_id:03d}"
            ),

            "index": index,

            "image": records[index]["path"],

            "original_split": records[index]["split"],

            # YOU will change this after
            # visual verification.
            "verified_scene_group": ""

        })


# Images that aren't in a candidate group
# are initially marked as singleton.

grouped_indices = set()

for members in candidate_groups:

    grouped_indices.update(members)


singleton_counter = 1

for index in range(len(images)):

    if index not in grouped_indices:

        group_records.append({

            "candidate_group": (
                f"singleton_{singleton_counter:03d}"
            ),

            "index": index,

            "image": records[index]["path"],

            "original_split": records[index]["split"],

            "verified_scene_group": ""

        })

        singleton_counter += 1


group_df = pd.DataFrame(
    group_records
)

group_path = (
    OUTPUT_DIR /
    "scene_groups_review.csv"
)

group_df.to_csv(
    group_path,
    index=False
)

print(f"\nSaved:")
print(group_path)


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

print("\nCreating candidate-group contact sheets...")


def create_contact_sheet(
    group_id,
    member_indices
):

    rows = (
        len(member_indices)
        + COLUMNS
        - 1
    ) // COLUMNS

    width = (
        COLUMNS *
        THUMBNAIL_SIZE
    )

    height = (
        rows *
        (THUMBNAIL_SIZE + 35)
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for position, index in enumerate(
        member_indices
    ):

        image_path = images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image.thumbnail(
            (
                THUMBNAIL_SIZE - 10,
                THUMBNAIL_SIZE - 10
            )
        )

        x = (
            position % COLUMNS
        ) * THUMBNAIL_SIZE

        y = (
            position // COLUMNS
        ) * (
            THUMBNAIL_SIZE + 35
        )

        paste_x = (
            x
            +
            (THUMBNAIL_SIZE - image.width)
            // 2
        )

        paste_y = (
            y
            +
            (THUMBNAIL_SIZE - image.height)
            // 2
        )

        canvas.paste(
            image,
            (paste_x, paste_y)
        )

        label = (
            f"{index}: "
            f"{records[index]['split']}"
        )

        draw.text(
            (
                x + 5,
                y + THUMBNAIL_SIZE
            ),
            label,
            fill="black"
        )

    output = (
        OUTPUT_DIR
        /
        "candidate_groups"
        /
        f"{group_id}.jpg"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.save(
        output,
        quality=95
    )

    return output


for group_id, members in enumerate(
    candidate_groups,
    start=1
):

    create_contact_sheet(
        f"candidate_{group_id:03d}",
        members
    )


print(
    "\nContact sheets saved to:"
)

print(
    OUTPUT_DIR /
    "candidate_groups"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SCENE GROUPING ANALYSIS COMPLETE")
print("=" * 75)

print("\nImportant files:")

print(
    f"\n1. {top_k_path}"
)

print(
    f"2. {candidate_path}"
)

print(
    f"3. {group_path}"
)

print(
    f"4. {OUTPUT_DIR / 'candidate_groups'}"
)

print(
    "\nThe candidate groups are NOT final."
)

print(
    "Visually verify them before creating "
    "the development/evaluation split."
)