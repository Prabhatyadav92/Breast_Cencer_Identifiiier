import streamlit as st
import torch
import numpy as np
from model import BreastCancerModel

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    INPUT_FEATURES = 30
    model = BreastCancerModel(INPUT_FEATURES)
    model.load_state_dict(
        torch.load("model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()

# ---------------- UI ----------------
st.title("🩺 Breast Cancer Identification App")
st.markdown(
    "This app uses a **PyTorch deep learning model** to predict whether a tumor is **Benign or Malignant**."
)

st.divider()

# Feature names (Wisconsin dataset)
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

features = []

st.subheader("🔢 Enter Patient Medical Features")

for name in feature_names:
    value = st.number_input(
        name,
        min_value=0.0,
        format="%.4f"
    )
    features.append(value)

# ---------------- PREDICTION ----------------
st.divider()

if st.button("🔍 Predict Cancer Type", use_container_width=True):
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)
    input_tensor = torch.tensor(input_data)

    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0

    st.subheader("🧾 Result")

    if prediction == 1:
        st.success(f"✅ **Benign (No Cancer Detected)**\n\nProbability: **{probability:.2%}**")
    else:
        st.error(f"⚠️ **Malignant (Cancer Detected)**\n\nProbability: **{1 - probability:.2%}**")
