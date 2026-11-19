import streamlit as st
import numpy as np
import torch
import sys
import os
import json
import pandas as pd


# =====================================================
# PROJECT PATH SETUP
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hospital.local_model import get_model


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Live Patient Prediction",
    layout="wide"
)

st.title("🩺 Live Patient Severity Prediction")

LABEL_MAP = {
    0: "Stable",
    1: "Medium Risk",
    2: "High Risk",
    3: "Critical"
}

LABEL_NAMES = [
    "Stable",
    "Medium Risk",
    "High Risk",
    "Critical"
]

SEQUENCE_LENGTH = int(
    os.getenv("SEQUENCE_LENGTH", 10)
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize_input(hr, spo2, bp, rr, temp, ecg):
    """
    Fixed-range normalization for live dashboard input.

    Training data was MinMax-scaled, so live values also need
    to be scaled approximately to 0-1.
    """

    hr_n = (hr - 30.0) / (200.0 - 30.0)
    spo2_n = (spo2 - 50.0) / (100.0 - 50.0)
    bp_n = (bp - 50.0) / (220.0 - 50.0)
    rr_n = (rr - 5.0) / (50.0 - 5.0)
    temp_n = (temp - 30.0) / (45.0 - 30.0)
    ecg_n = float(ecg)

    values = np.array(
        [
            hr_n,
            spo2_n,
            bp_n,
            rr_n,
            temp_n,
            ecg_n
        ],
        dtype=np.float32
    )

    values = np.clip(
        values,
        0.0,
        1.0
    )

    return values


def get_model_path(hospital_id, model_choice):
    if model_choice == "Best Selected FL":
        return (
            f"hospital/logs/"
            f"hospital_{hospital_id}_best_selected_fl.pth"
        )

    if model_choice == "Local Baseline":
        return (
            f"hospital/logs/"
            f"hospital_{hospital_id}_local_baseline.pth"
        )

    return None


def load_model(model_path):
    model = get_model()

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


def predict_severity(model, sequence):
    with torch.no_grad():

        outputs = model(sequence)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        pred_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    probs = probabilities.cpu().numpy()[0]

    confidence = float(
        probs[pred_class]
    )

    return pred_class, confidence, probs


def severity_badge(pred_class):
    if pred_class == 0:
        return "🟢 Stable"
    elif pred_class == 1:
        return "🟡 Medium Risk"
    elif pred_class == 2:
        return "🟠 High Risk"
    else:
        return "🔴 Critical"


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Model Settings")

hospital_id = st.sidebar.selectbox(
    "Select Hospital",
    [0, 1, 2]
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Best Selected FL",
        "Local Baseline"
    ]
)

model_path = get_model_path(
    hospital_id,
    model_choice
)

st.sidebar.write("Model path:")
st.sidebar.code(model_path)


# =====================================================
# INPUTS
# =====================================================

st.subheader("Patient Vital Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    hr = st.number_input(
        "Heart Rate (HR)",
        min_value=30.0,
        max_value=200.0,
        value=80.0
    )

    spo2 = st.number_input(
        "SpO2",
        min_value=50.0,
        max_value=100.0,
        value=98.0
    )

with c2:
    bp = st.number_input(
        "Blood Pressure",
        min_value=50.0,
        max_value=220.0,
        value=120.0
    )

    rr = st.number_input(
        "Respiratory Rate",
        min_value=5.0,
        max_value=50.0,
        value=16.0
    )

with c3:
    temp = st.number_input(
        "Temperature",
        min_value=30.0,
        max_value=45.0,
        value=37.0
    )

    ecg = st.selectbox(
        "ECG Abnormality",
        [0, 1]
    )


# =====================================================
# PREDICT
# =====================================================

if st.button("Predict Severity"):

    if model_path is None or not os.path.exists(model_path):

        st.error(
            "Model file not found. "
            "Run training first and make sure the selected model exists."
        )

        st.stop()

    model = load_model(
        model_path
    )

    normalized_values = normalize_input(
        hr,
        spo2,
        bp,
        rr,
        temp,
        ecg
    )

    x = np.array(
        [
            [
                normalized_values
            ]
        ],
        dtype=np.float32
    )

    x = np.repeat(
        x,
        SEQUENCE_LENGTH,
        axis=1
    )

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32
    )

    pred_class, confidence, probs = predict_severity(
        model,
        x_tensor
    )

    label = LABEL_MAP[pred_class]

    st.subheader("Prediction Result")

    if pred_class == 0:
        st.success(
            f"{severity_badge(pred_class)} "
            f"({confidence * 100:.2f}% confidence)"
        )

    elif pred_class == 1:
        st.warning(
            f"{severity_badge(pred_class)} "
            f"({confidence * 100:.2f}% confidence)"
        )

    elif pred_class == 2:
        st.warning(
            f"{severity_badge(pred_class)} "
            f"({confidence * 100:.2f}% confidence)"
        )

    else:
        st.error(
            f"{severity_badge(pred_class)} "
            f"({confidence * 100:.2f}% confidence)"
        )

    st.write(
        f"Predicted class: `{pred_class}` — **{label}**"
    )

    prob_df = pd.DataFrame({
        "Class ID": [0, 1, 2, 3],
        "Severity": LABEL_NAMES,
        "Probability (%)": probs * 100
    })

    st.subheader("Class Probabilities")

    st.bar_chart(
        prob_df.set_index("Severity")["Probability (%)"]
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )


# =====================================================
# INFORMATION
# =====================================================

st.info(
    """
Prediction labels:
- 0 = Stable
- 1 = Medium Risk
- 2 = High Risk
- 3 = Critical

The model uses softmax probabilities and selects the class with the highest probability.
"""
)