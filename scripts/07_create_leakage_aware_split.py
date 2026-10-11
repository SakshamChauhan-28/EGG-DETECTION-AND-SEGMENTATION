import os
from pathlib import Path
import shutil
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent.parent / "data" / "egg.v1i.folder"))
)

GROUP_FILE = Path(
    "dataset_report/scene_grouping/scene_groups_review.csv"
)

OUTPUT_ROOT = Path(
    "split_dataset"
)

DEVELOPMENT_DIR = OUTPUT_ROOT / "development"
EVALUATION_DIR = OUTPUT_ROOT / "evaluation"

REPORT_DIR = Path(
    "dataset_report"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORTANT:
# These candidate groups are kept completely intact.
#
# Candidate Groups 1 and 2 are intentionally kept together
# in DEVELOPMENT because they contain broad/mixed visual
# environments and we do not want to split them.
#
# The following complete groups give exactly 22 images:
#
# Group 4  -> 4
# Group 5  -> 4
# Group 6  -> 4
# Group 10 -> 3
# Group 11 -> 3
# Group 16 -> 2
# Group 17 -> 2
#
# Total evaluation = 22
# Total development = 88
# ============================================================

EVALUATION_GROUPS = {
    "candidate_004",
    "candidate_005",
    "candidate_006",
    "candidate_010",
    "candidate_011",
    "candidate_016",
    "candidate_017",
}


# ============================================================
# SAFETY CHECKS
# ============================================================

print("=" * 75)
print("Nosh Robotics - Leakage-Aware Development/Evaluation Split")
print("=" * 75)

print("\nDataset:")
print(DATASET_ROOT)

print("\nGroup file:")
print(GROUP_FILE)


if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_ROOT}"
    )


if not GROUP_FILE.exists():

    raise FileNotFoundError(
        f"Scene grouping file not found:\n{GROUP_FILE}"
    )


# ============================================================
# LOAD GROUP INFORMATION
# ============================================================

df = pd.read_csv(
    GROUP_FILE
)


required_columns = {
    "candidate_group",
    "index",
    "image",
    "original_split"
}


missing = required_columns - set(df.columns)


if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )


print(
    f"\nImages listed in grouping file: {len(df)}"
)


# ============================================================
# CHECK FOR DUPLICATE IMAGE ENTRIES
# ============================================================

duplicate_images = df[
    df["image"].duplicated(keep=False)
]


if len(duplicate_images) > 0:

    print(
        "\nWARNING: Duplicate image entries detected:"
    )

    print(
        duplicate_images[
            ["candidate_group", "image"]
        ].to_string(index=False)
    )

    raise ValueError(
        "The same image appears more than once "
        "in the grouping file."
    )


# ============================================================
# CHECK GROUP COUNTS
# ============================================================

group_counts = (
    df.groupby("candidate_group")
    .size()
    .sort_index()
)


print("\nCandidate groups:")
print(group_counts.to_string())


# ============================================================
# VERIFY SELECTED GROUPS EXIST
# ============================================================

available_groups = set(
    df["candidate_group"].astype(str)
)


missing_eval_groups = (
    EVALUATION_GROUPS
    - available_groups
)


if missing_eval_groups:

    raise ValueError(
        "The following evaluation groups "
        "do not exist:\n"
        + "\n".join(
            sorted(missing_eval_groups)
        )
    )


# ============================================================
# ASSIGN SPLITS
# ============================================================

df["final_split"] = df[
    "candidate_group"
].apply(
    lambda group:
        "evaluation"
        if group in EVALUATION_GROUPS
        else "development"
)


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SPLIT")
print("=" * 75)


split_counts = (
    df["final_split"]
    .value_counts()
)


development_count = int(
    split_counts.get(
        "development",
        0
    )
)

evaluation_count = int(
    split_counts.get(
        "evaluation",
        0
    )
)

total_count = (
    development_count
    +
    evaluation_count
)


print(
    f"\nTotal images       : {total_count}"
)

print(
    f"Development images : {development_count}"
)

print(
    f"Evaluation images  : {evaluation_count}"
)


development_percentage = (
    development_count
    / total_count
    * 100
)

evaluation_percentage = (
    evaluation_count
    / total_count
    * 100
)


print(
    f"\nDevelopment: "
    f"{development_percentage:.2f}%"
)

print(
    f"Evaluation : "
    f"{evaluation_percentage:.2f}%"
)


# ============================================================
# VERIFY 4:1 RATIO
# ============================================================

print("\nTarget ratio: approximately 4:1")

ratio = (
    development_count
    / evaluation_count
)


print(
    f"Actual ratio: "
    f"{ratio:.2f}:1"
)


# ============================================================
# SHOW GROUP ASSIGNMENTS
# ============================================================

print("\n" + "=" * 75)
print("GROUP ASSIGNMENTS")
print("=" * 75)


for group_name, group_df in (
    df.groupby("candidate_group")
):

    split = group_df[
        "final_split"
    ].iloc[0]

    count = len(group_df)

    print(
        f"{group_name:18s} "
        f"{count:2d} images -> "
        f"{split.upper()}"
    )


# ============================================================
# IMPORTANT VERIFICATION:
# EACH GROUP MUST BELONG TO ONLY ONE SPLIT
# ============================================================

group_split_counts = (
    df.groupby("candidate_group")[
        "final_split"
    ]
    .nunique()
)


violating_groups = (
    group_split_counts[
        group_split_counts > 1
    ]
)


if len(violating_groups) > 0:

    print(
        "\nERROR: Group leakage detected!"
    )

    print(
        violating_groups
    )

    raise RuntimeError(
        "A candidate group appears "
        "in both development and evaluation."
    )


print(
    "\n✓ No candidate group is split "
    "between development and evaluation."
)


# ============================================================
# CLEAN OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_ROOT.exists():

    print(
        "\nRemoving previous split_dataset..."
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )


DEVELOPMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY IMAGES
# ============================================================

print("\n" + "=" * 75)
print("COPYING IMAGES")
print("=" * 75)


copy_errors = []

for _, row in df.iterrows():

    relative_path = Path(
        row["image"]
    )

    source = (
        DATASET_ROOT
        /
        relative_path
    )

    if not source.exists():

        copy_errors.append({
            "image": str(
                relative_path
            ),
            "error": "Source file not found"
        })

        continue


    split = row[
        "final_split"
    ]

    if split == "development":

        destination_root = (
            DEVELOPMENT_DIR
        )

    else:

        destination_root = (
            EVALUATION_DIR
        )


    # Preserve a clean, unique filename.
    #
    # We use the original filename because
    # Roboflow-generated filenames are already unique.

    destination = (
        destination_root
        /
        source.name
    )


    shutil.copy2(
        source,
        destination
    )


    print(
        f"[{split:11s}] "
        f"{source.name}"
    )


# ============================================================
# CHECK COPY ERRORS
# ============================================================

if copy_errors:

    error_path = (
        REPORT_DIR
        /
        "split_copy_errors.json"
    )

    with open(
        error_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            copy_errors,
            f,
            indent=2
        )

    raise RuntimeError(
        f"{len(copy_errors)} files "
        f"could not be copied. "
        f"See {error_path}"
    )


# ============================================================
# VERIFY FILE COUNTS ON DISK
# ============================================================

development_files = [
    p for p in DEVELOPMENT_DIR.iterdir()
    if p.is_file()
]

evaluation_files = [
    p for p in EVALUATION_DIR.iterdir()
    if p.is_file()
]


print("\n" + "=" * 75)
print("DISK VERIFICATION")
print("=" * 75)


print(
    f"\nDevelopment files: "
    f"{len(development_files)}"
)

print(
    f"Evaluation files : "
    f"{len(evaluation_files)}"
)


if len(development_files) != development_count:

    raise RuntimeError(
        "Development file count mismatch."
    )


if len(evaluation_files) != evaluation_count:

    raise RuntimeError(
        "Evaluation file count mismatch."
    )


print(
    "\n✓ File counts match the split plan."
)


# ============================================================
# CREATE SPLIT CSV
# ============================================================

split_csv = (
    REPORT_DIR
    /
    "final_split_assignments.csv"
)


df.to_csv(
    split_csv,
    index=False
)


print(
    f"\nSaved split assignments:"
)

print(
    split_csv
)


# ============================================================
# CREATE GROUP SUMMARY
# ============================================================

group_summary = (
    df.groupby(
        [
            "candidate_group",
            "final_split"
        ]
    )
    .size()
    .reset_index(
        name="image_count"
    )
)


group_summary_csv = (
    REPORT_DIR
    /
    "final_group_split_summary.csv"
)


group_summary.to_csv(
    group_summary_csv,
    index=False
)


print(
    f"Saved group summary:"
)

print(
    group_summary_csv
)


# ============================================================
# CREATE JSON REPORT
# ============================================================

report = {

    "dataset_root": str(
        DATASET_ROOT
    ),

    "total_images": total_count,

    "development_images":
        development_count,

    "evaluation_images":
        evaluation_count,

    "development_percentage":
        round(
            development_percentage,
            2
        ),

    "evaluation_percentage":
        round(
            evaluation_percentage,
            2
        ),

    "development_evaluation_ratio":
        round(
            ratio,
            3
        ),

    "evaluation_groups":
        sorted(
            EVALUATION_GROUPS
        ),

    "development_groups":
        sorted(
            set(
                df[
                    df["final_split"]
                    == "development"
                ]["candidate_group"]
            )
        ),

    "group_leakage_check":
        "PASS",

    "original_dataset_modified":
        False
}


report_path = (
    REPORT_DIR
    /
    "final_split_report.json"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print(
    f"Saved final report:"
)

print(
    report_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SPLIT COMPLETE")
print("=" * 75)

print(
    "\nOriginal dataset:"
)

print(
    f"  {DATASET_ROOT}"
)

print(
    "\nNew assignment dataset:"
)

print(
    f"  {OUTPUT_ROOT.resolve()}"
)

print("\nStructure:")

print(
    "  split_dataset/"
)

print(
    "    development/"
)

print(
    f"      {development_count} images"
)

print(
    "    evaluation/"
)

print(
    f"      {evaluation_count} images"
)


print("\nLeakage protection:")

print(
    "  ✓ Candidate groups kept intact"
)

print(
    "  ✓ Candidate Groups 1 + 2 "
    "kept together in development"
)

print(
    "  ✓ No group appears in both splits"
)

print(
    "  ✓ Original dataset untouched"
)

print(
    "\n" + "=" * 75
)
print("READY FOR TASK 1")
print("=" * 75)