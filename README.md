# Traffic Violation Classification Prototype

This prototype supports the project:

**Intelligent Image Processing System for Enhanced Traffic Violation Detection in Oman**

Current scope:
- `seatbelt`
- `no_seatbelt`
- `phone_usage`
- `no_phone_usage`

The idea is to use Roboflow YOLOv8 object-detection datasets, crop labelled objects from YOLO annotations, and automatically create a YOLOv8 classification dataset.

---

## 1. Project structure

```text
traffic_violation_classification_prototype/
  app.py
  requirements.txt
  config/
    class_aliases.yaml
  src/
    prepare_classification_dataset_from_yolo.py
    check_dataset_counts.py
    train_classifier.py
    evaluate_classifier.py
  datasets/
    raw_roboflow/
      put_your_extracted_roboflow_datasets_here/
    classification/
      train/
      val/
      test/
  runs/
  outputs/
```

---

## 2. Install requirements

```bash
pip install -r requirements.txt
```

---

## 3. Add Roboflow datasets

Download/export the Roboflow datasets in **YOLOv8** format and extract them here:

```text
datasets/raw_roboflow/
```

Example:

```text
datasets/raw_roboflow/seatbelt_dataset/
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels

datasets/raw_roboflow/phone_usage_dataset/
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

The script reads every dataset folder that contains `data.yaml`.

---

## 4. Convert YOLO annotations into classification folders

```bash
python src/prepare_classification_dataset_from_yolo.py --reset
```

This creates:

```text
datasets/classification/train/seatbelt
datasets/classification/train/no_seatbelt
datasets/classification/train/phone_usage
datasets/classification/train/no_phone_usage
datasets/classification/val/...
datasets/classification/test/...
```

Check counts:

```bash
python src/check_dataset_counts.py
```

If one class has zero images, edit:

```text
config/class_aliases.yaml
```

and add the exact class names from your Roboflow `data.yaml`.

---

## 5. Train YOLOv8 classification model

```bash
python src/train_classifier.py --epochs 30 --imgsz 224 --batch 16
```

The best model should be saved at:

```text
runs/classify/traffic_violation_cls/weights/best.pt
```

---

## 6. Evaluate with recall and F1-score

```bash
python src/evaluate_classifier.py --model runs/classify/traffic_violation_cls/weights/best.pt --data datasets/classification/test
```

Outputs:

```text
runs/evaluation/summary_metrics.csv
runs/evaluation/per_class_metrics.csv
runs/evaluation/classification_report.txt
runs/evaluation/confusion_matrix.csv
runs/evaluation/confusion_matrix.png
```

Metrics include:
- accuracy
- precision
- recall
- F1-score
- per-class precision/recall/F1
- confusion matrix

---

## 7. Run the web app

```bash
streamlit run app.py
```

Upload cropped driver/vehicle-region images and the app will classify them into the four target classes.

---

## Notes for Chapter 5

Use screenshots of:
1. Dataset folder structure
2. Dataset counts
3. Training command/output
4. Training graphs from `runs/classify/...`
5. Web app upload page
6. Prediction result page
7. Evaluation metrics CSV/report
8. Confusion matrix image

Important limitation:
This version is a classification prototype. It classifies cropped object/driver-region images. A full roadside system would need an object-detection stage first to crop the relevant region automatically from full traffic images.
