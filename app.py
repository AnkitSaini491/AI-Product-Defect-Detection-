

import streamlit as st
import cv2
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Product Quality Analytics",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Product Defect Detection")
st.write("Detect product defects and analyze quality using Computer Vision.")

MODEL_PATH = Path("models/best.pt")

# Load model
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))

# Sidebar
st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Product Image")
    st.image(image, width=500)

    if st.button("Detect Defects"):

        if not MODEL_PATH.exists():
            st.error(
                "Model not found! Please add your trained "
                "YOLO model to models/best.pt."
            )
            st.stop()

        try:
            model = load_model()

            with st.spinner("Analyzing product..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

            result = results[0]

            # Display detection result
            annotated = cv2.cvtColor(
                result.plot(),
                cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Results")
            st.image(annotated, width=700)

            # Extract detection data
            records = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                score = float(box.conf[0].item())

                records.append({
                    "Defect Type": class_name,
                    "Confidence (%)": round(score * 100, 2)
                })

            df = pd.DataFrame(
                records,
                columns=["Defect Type", "Confidence (%)"]
            )

            total_defects = len(df)
            status = "Defective" if total_defects > 0 else "No Defects Detected"

            # Quality metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("Defects Detected", total_defects)
            col2.metric(
                "Defect Categories",
                df["Defect Type"].nunique()
            )
            col3.metric("Inspection Status", status)

            st.subheader("Defect Analysis")

            if not df.empty:
                st.dataframe(df, use_container_width=True)

                counts = (
                    df["Defect Type"]
                    .value_counts()
                    .rename_axis("Defect Type")
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    counts,
                    x="Defect Type",
                    y="Count",
                    title="Defects by Category",
                    text="Count"
                )

                st.plotly_chart(fig, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Inspection Report",
                    data=csv,
                    file_name="inspection_report.csv",
                    mime="text/csv"
                )

            else:
                st.success(
                    "No defects detected at the selected confidence level."
                )

        except Exception as e:
            st.error(f"Detection failed: {e}")

else:
    st.info("Upload a product image to start inspection.")
