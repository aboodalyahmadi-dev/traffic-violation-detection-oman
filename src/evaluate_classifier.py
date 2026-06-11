"""
Evaluate classification model with accuracy, precision, recall, F1-score,
classification report, and confusion matrix.

Example:
python src/evaluate_classifier.py --model runs/classify/traffic_violation_cls/weights/best.pt --data datasets/classification/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_dir: Path):
    image_paths = []
    y_true = []

    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        for image_path in (data_dir / class_name).rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)
                y_true.append(class_to_idx[class_name])

    return image_paths, y_true, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained best.pt")
    parser.add_argument("--data", default="datasets/classification/test", help="Test split folder")
    parser.add_argument("--out", default="runs/evaluation", help="Output folder for metrics")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, y_true, class_names = collect_images(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    model = YOLO(args.model)

    y_pred = []
    confidences = []

    for image_path in image_paths:
        result = model.predict(str(image_path), verbose=False)[0]
        pred_id = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        y_pred.append(pred_id)
        confidences.append(conf)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    metrics_df = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    summary_df = pd.DataFrame([
        {"metric": "accuracy", "value": accuracy},
        {"metric": "macro_precision", "value": macro_precision},
        {"metric": "macro_recall", "value": macro_recall},
        {"metric": "macro_f1", "value": macro_f1},
        {"metric": "weighted_precision", "value": weighted_precision},
        {"metric": "weighted_recall", "value": weighted_recall},
        {"metric": "weighted_f1", "value": weighted_f1},
    ])

    metrics_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("\nSummary metrics")
    print(summary_df.to_string(index=False))

    print("\nPer-class metrics")
    print(metrics_df.to_string(index=False))

    print("\nClassification report")
    print(report)

    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
