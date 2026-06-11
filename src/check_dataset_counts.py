from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/classification")
    args = parser.parse_args()

    root = Path(args.data)
    rows = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            count = len([p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
            rows.append({"split": split, "class": cls_dir.name, "image_count": count})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = root / "dataset_counts.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
