from pathlib import Path
import random
import shutil
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/classification")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.data)
    train_dir = root / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)
        val_count = int(total * args.val_ratio)
        test_count = int(total * args.test_ratio)

        val_images = images[:val_count]
        test_images = images[val_count:val_count + test_count]

        for split_name, split_images in [
            ("val", val_images),
            ("test", test_images),
        ]:
            target_dir = root / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                target_path = target_dir / image_path.name
                shutil.move(str(image_path), str(target_path))

        remaining_train = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(
            f"{class_dir.name}: total={total}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}, "
            f"train_remaining={len(remaining_train)}"
        )

    print("Done splitting dataset.")

if __name__ == "__main__":
    main()