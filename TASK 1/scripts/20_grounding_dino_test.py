from pathlib import Path
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

IMAGE_PATH = (Path(__file__).resolve().parent / r"split_dataset\development\not_damaged_155_jpg.rf.a65395f65d93bcd6cbf6bdab6d94e551.jpg")

OUTPUT_PATH = "gdino_test_output.jpg"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("GROUNDING DINO TEST")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(DEVICE)

model.eval()

print("Model loaded.")


# ============================================================
# LOAD IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"\nImage: {IMAGE_PATH}")
print(f"Size: {image.size}")


# ============================================================
# TEXT PROMPT
# ============================================================

# IMPORTANT:
# Grounding DINO expects lowercase text ending with a period.
text = "egg."


inputs = processor(
    images=image,
    text=text,
    return_tensors="pt"
).to(DEVICE)


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning Grounding DINO...")

with torch.no_grad():
    outputs = model(**inputs)


# ============================================================
# POST PROCESS
# ============================================================

results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[(image.height, image.width)]
)

result = results[0]

boxes = result["boxes"].detach().cpu()
scores = result["scores"].detach().cpu()
labels = result["text_labels"]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DETECTIONS")
print("=" * 60)

print(f"Number of detections: {len(boxes)}")

for i, (box, score, label) in enumerate(
    zip(boxes, scores, labels)
):

    print(
        f"{i + 1}. "
        f"label={label} "
        f"score={score.item():.3f} "
        f"box={[round(x, 1) for x in box.tolist()]}"
    )


# ============================================================
# VISUALIZATION
# ============================================================

draw = ImageDraw.Draw(image)

for i, (box, score, label) in enumerate(
    zip(boxes, scores, labels)
):

    x1, y1, x2, y2 = box.tolist()

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )

    text_label = f"egg {score.item():.2f}"

    draw.text(
        (x1, max(0, y1 - 20)),
        text_label,
        fill="red"
    )


image.save(OUTPUT_PATH)

print("\nOutput saved to:")
print(os.path.abspath(OUTPUT_PATH))

print("\nDONE")