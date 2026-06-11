"""
Train YOLOv8 classification model.

Example:
python src/train_classifier.py --data datasets/classification --model yolov8n-cls.pt --epochs 30 --imgsz 224
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/classification")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="traffic_violation_cls")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    print("\nTraining complete.")
    print(f"Best model usually saved under: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
