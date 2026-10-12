import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path(Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent / "data" / "egg.v1i.folder")))
OUTPUT_PATH = Path("dataset_report/contact_sheet.jpg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

images = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

# Use up to 110 images
images = images[:110]

THUMBNAIL_SIZE = 160
LABEL_HEIGHT = 25
COLUMNS = 10

ROWS = (len(images) + COLUMNS - 1) // COLUMNS

sheet_width = COLUMNS * THUMBNAIL_SIZE
sheet_height = ROWS * (THUMBNAIL_SIZE + LABEL_HEIGHT)

sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, image_path in enumerate(images):

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))

        x = (i % COLUMNS) * THUMBNAIL_SIZE
        y = (i // COLUMNS) * (THUMBNAIL_SIZE + LABEL_HEIGHT)

        # Center image
        paste_x = x + (THUMBNAIL_SIZE - img.width) // 2
        paste_y = y + (THUMBNAIL_SIZE - img.height) // 2

        sheet.paste(img, (paste_x, paste_y))

        # Dataset split
        relative = image_path.relative_to(DATASET_ROOT)
        split = relative.parts[0] if len(relative.parts) > 1 else ""

        label = f"{i+1}: {split}"

        draw.text(
            (x + 3, y + THUMBNAIL_SIZE + 3),
            label,
            fill="black"
        )

    except Exception as e:
        print(f"Could not process {image_path}: {e}")

OUTPUT_PATH.parent.mkdir(exist_ok=True)

sheet.save(OUTPUT_PATH, quality=95)

print(f"Saved contact sheet to:")
print(OUTPUT_PATH)