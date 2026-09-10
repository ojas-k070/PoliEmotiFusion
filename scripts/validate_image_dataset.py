import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "image_emotion"

EXPECTED_CLASSES = [
    "Anger",
    "Joy",
    "Sadness",
    "Fear",
    "Surprise",
    "Disgust",
    "Neutral",
]
EXPECTED_SPLITS = ["train", "val", "test"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_dataset(data_dir: Path = DATA_DIR) -> bool:
    if not data_dir.exists():
        print(f"ERROR: Dataset directory {data_dir} does not exist.")
        return False

    all_passed = True
    print(f"=== Validating Image Emotion Dataset at: {data_dir} ===")

    for split in EXPECTED_SPLITS:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"[WARNING] Missing split directory: {split_path}")
            all_passed = False
            continue

        print(f"\nChecking split '{split}':")
        split_total = 0

        for cls_name in EXPECTED_CLASSES:
            cls_dir = split_path / cls_name
            if not cls_dir.exists():
                print(f"  [MISSING] Class directory missing: {cls_dir}")
                all_passed = False
                continue

            images = [p for p in cls_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
            split_total += len(images)
            corrupted = 0

            for img_p in images:
                try:
                    with Image.open(img_p) as img:
                        img.verify()
                except Exception:
                    corrupted += 1
                    all_passed = False

            status = f"OK ( {len(images)} images)" if corrupted == 0 else f"FAILED ({corrupted} corrupted)"
            print(f"  - {cls_name:10s}: {status}")

        print(f"Split ' {split}' total valid images: {split_total}")

    if all_passed:
        print("\nDataset validation completed successfully!")
    else:
        print("\nDataset validation completed with issues.")
    return all_passed

if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
