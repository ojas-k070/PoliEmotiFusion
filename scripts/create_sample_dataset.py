from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "image_emotion"

EMOTIONS = [
    "Anger",
    "Joy",
    "Sadness",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]

COLORS = {
    "Anger": (220, 50, 50),
    "Joy": (255, 200, 50),
    "Sadness": (60, 100, 200),
    "Fear": (140, 60, 180),
    "Surprise": (255, 120, 40),
    "Disgust": (50, 160, 80),
    "Neutral": (150, 150, 150),
}

SPLITS = {
    "train": 10,
    "val": 3,
    "test": 3,
}

def create_sample_image(emotion: str, index: int, size=(260, 260)) -> Image.Image:
    base_color = COLORS.get(emotion, (128, 128, 128))
    img = Image.new("RGB", size, color=base_color)
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    offset = (index * 7) % 20
    draw.ellipse([cx - 50 + offset, cy - 50, cx + 50 + offset, cy + 50], outline=(255, 255, 255), width=3)
    draw.line([(cx - 30, cy + 20), (cx + 30, cy + 20)], fill=(255, 255, 255), width=3)
    return img

def main():
    print(f"Generating sample dataset in {DATA_DIR}...")
    total_created = 0
    for split, count in SPLITS.items():
        for emotion in EMOTIONS:
            split_dir = DATA_DIR / split / emotion
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = create_sample_image(emotion, i)
                filename = f"{emotion.lower()}_{split}_{i:03d}.jpg"
                file_path = split_dir / filename
                img.save(file_path, "JPEG", quality=90)
                total_created += 1
    print(f"Successfully generated {total_created} sample images across {len(EMOTIONS)} classes and {len(SPLITS)} splits.")

if __name__ == "__main__":
    main()
