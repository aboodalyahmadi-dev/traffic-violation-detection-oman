"""
Streamlit prototype for the bachelor project:
Intelligent Image Processing System for Enhanced Traffic Violation Detection in Oman

This app classifies uploaded cropped driver/vehicle-region images into:
seatbelt, no_seatbelt, phone_usage, no_phone_usage.

Run:
streamlit run app.py
"""
from pathlib import Path
from datetime import datetime
import csv

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


RESULTS_CSV = Path("outputs/prediction_history.csv")
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="Traffic Violation Classification Prototype", layout="wide")

st.title("Traffic Violation Classification Prototype")
st.caption("YOLOv8 classification prototype for seatbelt and phone-usage detection.")

model_path = st.sidebar.text_input(
    "Model path",
    value="runs/classify/runs/classify/traffic_violation_cls/weights/best.pt",
)

confidence_warning_threshold = st.sidebar.slider(
    "Low-confidence warning threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload a cropped driver/vehicle-region image",
    type=["jpg", "jpeg", "png", "webp"],
)

def save_prediction(filename: str, predicted_class: str, confidence: float):
    file_exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "predicted_class", "confidence"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), filename, predicted_class, confidence])


if uploaded_file:
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"Model not found: {model_file}")
            st.info("Train the model first, then update the model path in the sidebar.")
        else:
            model = YOLO(str(model_file))
            result = model.predict(image, verbose=False)[0]

            predicted_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            predicted_class = result.names[predicted_id]

            st.metric("Predicted class", predicted_class)
            st.metric("Confidence", f"{confidence:.2%}")

            if confidence < confidence_warning_threshold:
                st.warning("Low confidence. This should be sent for manual review.")

            save_prediction(uploaded_file.name, predicted_class, confidence)

            probs = result.probs.data.cpu().numpy()
            labels = [result.names[i] for i in range(len(probs))]
            st.bar_chart({label: float(prob) for label, prob in zip(labels, probs)})

st.divider()
st.subheader("Prediction history")

if RESULTS_CSV.exists():
    history_df = pd.read_csv(RESULTS_CSV)
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No predictions yet.")