"""
Create a YOLOv8 classification dataset from YOLO detection annotations.

Input expected Roboflow YOLOv8 format:
datasets/raw_roboflow/<dataset_name>/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
  test/images/*.jpg
  test/labels/*.txt

Output classification format:
datasets/classification/
  train/seatbelt/*.jpg
  train/no_seatbelt/*.jpg
  train/phone_usage/*.jpg
  train/no_phone_usage/*.jpg
  val/...
  test/...

Why this exists:
Your project is using YOLO-labelled object detection datasets from Roboflow,
but your current prototype needs a classification dataset. This script crops
labelled bounding boxes and places each crop into the correct classification folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def load_aliases(alias_file: Path) -> Dict[str, List[str]]:
    with alias_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    target_classes = cfg.get("target_classes", {})
    aliases: Dict[str, List[str]] = {}
    for target, names in target_classes.items():
        normalized_target = normalize_name(target)
        aliases[normalized_target] = sorted({normalize_name(n) for n in names + [target]})
    return aliases


def read_data_yaml(dataset_dir: Path) -> List[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml in {dataset_dir}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, list):
        return names

    raise ValueError(f"Could not read class names from {data_yaml}")


def map_source_classes_to_targets(source_names: List[str], aliases: Dict[str, List[str]]) -> Dict[int, str]:
    class_map: Dict[int, str] = {}
    normalized_source = [normalize_name(name) for name in source_names]

    for idx, source_name in enumerate(normalized_source):
        for target, target_aliases in aliases.items():
            if source_name in target_aliases:
                class_map[idx] = target
                break

    return class_map


def find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    padding: float = 0.08,
) -> Tuple[int, int, int, int]:
    box_w = width * image_width
    box_h = height * image_height
    cx = x_center * image_width
    cy = y_center * image_height

    pad_w = box_w * padding
    pad_h = box_h * padding

    x1 = max(0, int(cx - box_w / 2 - pad_w))
    y1 = max(0, int(cy - box_h / 2 - pad_h))
    x2 = min(image_width, int(cx + box_w / 2 + pad_w))
    y2 = min(image_height, int(cy + box_h / 2 + pad_h))

    return x1, y1, x2, y2


def convert_split(
    dataset_dir: Path,
    output_dir: Path,
    split_name: str,
    class_map: Dict[int, str],
    min_crop_size: int,
    padding: float,
) -> Dict[str, int]:
    # Roboflow usually uses "valid"; Ultralytics classification expects "val".
    output_split = "val" if split_name == "valid" else split_name

    images_dir = dataset_dir / split_name / "images"
    labels_dir = dataset_dir / split_name / "labels"

    counts: Dict[str, int] = {target: 0 for target in set(class_map.values())}

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Skipping missing split: {dataset_dir / split_name}")
        return counts

    for label_file in labels_dir.glob("*.txt"):
        image_path = find_image_for_label(images_dir, label_file.stem)
        if image_path is None:
            print(f"Warning: no image found for {label_file}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: cannot read image {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        with label_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for obj_index, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            if class_id not in class_map:
                continue

            target_class = class_map[class_id]
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy(
                x_center, y_center, box_width, box_height, image_width, image_height, padding
            )

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_h < min_crop_size or crop_w < min_crop_size:
                continue

            class_dir = output_dir / output_split / target_class
            class_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{dataset_dir.name}_{label_file.stem}_{obj_index}_{target_class}.jpg"
            output_path = class_dir / output_name
            cv2.imwrite(str(output_path), crop)

            counts[target_class] = counts.get(target_class, 0) + 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="datasets/raw_roboflow", help="Folder containing Roboflow YOLOv8 datasets")
    parser.add_argument("--out", default="datasets/classification", help="Output classification dataset folder")
    parser.add_argument("--aliases", default="config/class_aliases.yaml", help="Class aliases YAML")
    parser.add_argument("--reset", action="store_true", help="Delete output folder before creating it")
    parser.add_argument("--min-crop-size", type=int, default=24, help="Minimum crop width/height in pixels")
    parser.add_argument("--padding", type=float, default=0.08, help="Padding around bounding boxes")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.out)
    alias_file = Path(args.aliases)

    if args.reset and output_dir.exists():
        shutil.rmtree(output_dir)

    aliases = load_aliases(alias_file)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"{raw_root} does not exist. Put your extracted Roboflow YOLOv8 datasets there."
        )

    dataset_dirs = [p for p in raw_root.iterdir() if p.is_dir() and (p / "data.yaml").exists()]
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No Roboflow YOLOv8 datasets found inside {raw_root}. "
            "Expected each dataset folder to contain data.yaml."
        )

    total_summary = {}

    for dataset_dir in dataset_dirs:
        source_names = read_data_yaml(dataset_dir)
        class_map = map_source_classes_to_targets(source_names, aliases)

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_dir}")
        print(f"Source classes: {source_names}")
        print(f"Matched class IDs: {class_map}")

        if not class_map:
            print("No target classes matched. Edit config/class_aliases.yaml if needed.")
            continue

        dataset_summary = {}
        for split_name in ["train", "valid", "test"]:
            counts = convert_split(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                split_name=split_name,
                class_map=class_map,
                min_crop_size=args.min_crop_size,
                padding=args.padding,
            )
            dataset_summary[split_name] = counts
            print(f"{split_name}: {counts}")

        total_summary[dataset_dir.name] = dataset_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(total_summary, f, indent=2)

    print("\nDone.")
    print(f"Classification dataset created at: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
