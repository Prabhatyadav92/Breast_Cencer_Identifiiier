import streamlit as st
import torch
import numpy as np
from model import BreastCancerModel

# Page config
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

# Load model
INPUT_FEATURES = 30   # Breast cancer dataset has 30 features

model = BreastCancerModel(INPUT_FEATURES)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

st.title("🩺 Breast Cancer Detection App")
st.write("Enter patient medical details to predict cancer type")

# Collect inputs
features = []
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

for name in feature_names:
    value = st.number_input(name, min_value=0.0)
    features.append(value)

# Predict
if st.button("Predict"):
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)
    input_tensor = torch.tensor(input_data)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).item()

    if prediction == 1:
        st.success("✅ Benign (No Cancer Detected)")
    else:
        st.error("⚠️ Malignant (Cancer Detected)")
