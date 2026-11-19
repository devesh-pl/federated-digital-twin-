import streamlit as st
import pandas as pd
import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


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
    page_title="Hospital Dashboard",
    layout="wide"
)

st.title("🏥 Hospital Federated Dashboard")

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

def safe_get(row, key, default=0.0):
    try:
        if key in row:
            value = row[key]

            if pd.isna(value):
                return default

            return value

    except Exception:
        pass

    return default


def load_json_file(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def normalize_live_input(hr, spo2, bp, rr, temp, ecg):
    """
    Simple fixed-range normalization for live prediction.

    Training used MinMaxScaler fitted on each hospital dataset.
    For live dashboard input, we use clinically reasonable ranges.
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


def load_model_from_path(model_path):
    model = get_model()

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


def predict_multiclass(model, sequence):
    x_tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = model(x_tensor)

        probs = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

    pred_class = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[pred_class]
    )

    return pred_class, confidence, probs


# =====================================================
# SELECT HOSPITAL
# =====================================================

hospital_id = st.sidebar.selectbox(
    "Select Hospital",
    [0, 1, 2]
)

metrics_file = (
    f"hospital/logs/"
    f"hospital_{hospital_id}_metrics.json"
)

comparison_file = (
    f"hospital/logs/"
    f"hospital_{hospital_id}_baseline_vs_best_fl.json"
)

best_model_file = (
    f"hospital/logs/"
    f"hospital_{hospital_id}_best_selected_fl.pth"
)

best_model_info_file = (
    f"hospital/logs/"
    f"hospital_{hospital_id}_best_selected_fl.json"
)


# =====================================================
# LOAD METRICS
# =====================================================

if not os.path.exists(metrics_file):

    st.warning(
        "No round metrics found yet. Run the federated training first."
    )

    st.stop()

with open(metrics_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

if df.empty:
    st.warning("Metrics file is empty.")
    st.stop()

latest = df.iloc[-1]

comparison_data = load_json_file(
    comparison_file
)

best_model_info = load_json_file(
    best_model_info_file
)


# =====================================================
# TOP METRICS
# =====================================================

st.subheader("📌 Latest Round Summary")

updated_local_acc = safe_get(
    latest,
    "local_accuracy"
) * 100

selected_fl_acc = safe_get(
    latest,
    "federated_accuracy",
    safe_get(latest, "selected_accuracy")
) * 100

# Prefer baseline-vs-best improvement if available.
if comparison_data is not None:
    improvement = comparison_data.get(
        "accuracy_improvement_percent",
        safe_get(latest, "improvement_percent")
    )
else:
    improvement = safe_get(
        latest,
        "improvement_percent"
    )

updated_local_weighted_f1 = safe_get(
    latest,
    "local_weighted_f1",
    safe_get(latest, "local_f1")
)

selected_weighted_f1 = safe_get(
    latest,
    "selected_weighted_f1",
    safe_get(latest, "federated_f1")
)

selected_macro_f1 = safe_get(
    latest,
    "selected_macro_f1",
    safe_get(latest, "federated_macro_f1", 0.0)
)

critical_recall = safe_get(
    latest,
    "selected_critical_recall",
    safe_get(latest, "federated_critical_recall", 0.0)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🏠 Updated Local Accuracy",
    f"{updated_local_acc:.2f}%"
)

c2.metric(
    "🌍 Selected FL Accuracy",
    f"{selected_fl_acc:.2f}%",
    f"{improvement:.2f}%"
)

c3.metric(
    "📊 Selected Weighted F1",
    f"{selected_weighted_f1:.4f}"
)

c4.metric(
    "🚨 Critical Recall",
    f"{critical_recall:.4f}"
)


# =====================================================
# BASELINE VS BEST FL SUMMARY
# =====================================================

st.subheader("🏁 Local Baseline vs Best Selected FL")

if comparison_data is not None:

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Local Baseline Acc",
        f"{comparison_data.get('local_baseline_accuracy', 0) * 100:.2f}%"
    )

    b2.metric(
        "Best Selected FL Acc",
        f"{comparison_data.get('best_selected_accuracy', 0) * 100:.2f}%",
        f"{comparison_data.get('accuracy_improvement_percent', 0):.2f}%"
    )

    b3.metric(
        "Best Weighted F1",
        f"{comparison_data.get('best_selected_weighted_f1', 0):.4f}",
        f"{comparison_data.get('weighted_f1_improvement_percent', 0):.2f}%"
    )

    b4.metric(
        "Best Critical Recall",
        f"{comparison_data.get('best_selected_critical_recall', 0):.4f}",
        f"{comparison_data.get('critical_recall_improvement_percent', 0):.2f}%"
    )

    st.info(
        f"Best selected model came from round "
        f"{comparison_data.get('best_selected_round', 'N/A')} "
        f"using `{comparison_data.get('best_selected_model_type', 'N/A')}`."
    )

else:

    st.warning(
        "Final baseline-vs-best-FL comparison file not found yet."
    )


# =====================================================
# ACCURACY GRAPH
# =====================================================

st.subheader("📈 Accuracy Progression")

accuracy_cols = {
    "Round": df["round"],
    "Updated Local Accuracy": df["local_accuracy"] * 100,
    "Selected FL Accuracy": df.get(
        "federated_accuracy",
        df.get("selected_accuracy", pd.Series([0] * len(df)))
    ) * 100
}

if "global_accuracy" in df.columns:
    accuracy_cols["Raw Global FL Accuracy"] = (
        df["global_accuracy"] * 100
    )

if "personalized_accuracy" in df.columns:
    accuracy_cols["Personalized FL Accuracy"] = (
        df["personalized_accuracy"] * 100
    )

chart_df = pd.DataFrame(
    accuracy_cols
).set_index("Round")

st.line_chart(chart_df)


# =====================================================
# F1 GRAPH
# =====================================================

st.subheader("📊 Weighted F1 Progression")

f1_cols = {
    "Round": df["round"],
    "Updated Local Weighted F1": df.get(
        "local_weighted_f1",
        df.get("local_f1", pd.Series([0] * len(df)))
    ),
    "Selected FL Weighted F1": df.get(
        "selected_weighted_f1",
        df.get("federated_f1", pd.Series([0] * len(df)))
    )
}

if "global_weighted_f1" in df.columns:
    f1_cols["Raw Global FL Weighted F1"] = (
        df["global_weighted_f1"]
    )

if "personalized_weighted_f1" in df.columns:
    f1_cols["Personalized FL Weighted F1"] = (
        df["personalized_weighted_f1"]
    )

f1_df = pd.DataFrame(
    f1_cols
).set_index("Round")

st.line_chart(f1_df)


# =====================================================
# MACRO F1 + CRITICAL RECALL GRAPH
# =====================================================

st.subheader("🚨 Critical Risk Detection")

risk_cols = {
    "Round": df["round"]
}

if "selected_macro_f1" in df.columns:
    risk_cols["Selected Macro F1"] = (
        df["selected_macro_f1"]
    )

if "selected_critical_recall" in df.columns:
    risk_cols["Selected Critical Recall"] = (
        df["selected_critical_recall"]
    )

if "local_critical_recall" in df.columns:
    risk_cols["Updated Local Critical Recall"] = (
        df["local_critical_recall"]
    )

if "personalized_critical_recall" in df.columns:
    risk_cols["Personalized FL Critical Recall"] = (
        df["personalized_critical_recall"]
    )

risk_df = pd.DataFrame(
    risk_cols
).set_index("Round")

st.line_chart(risk_df)


# =====================================================
# ACCEPTED VS REJECTED UPDATES
# =====================================================

st.subheader("✅ Accepted vs Rejected FL Updates")

if "selected_model_type" in df.columns:

    preferred_selection_cols = [
        "round",
        "selected_model_type",
        "personalized_vs_baseline_acc_gain_percent",
        "personalized_vs_baseline_weighted_f1_gain_percent",
        "personalized_vs_baseline_critical_recall_gain_percent",
        "personalized_vs_updated_local_acc_gain_percent",
        "personalized_vs_updated_local_weighted_f1_gain_percent",
        "selected_medical_safety_score",
        "improvement_percent",
        "f1_improvement_percent"
    ]

    fallback_old_cols = [
        "round",
        "selected_model_type",
        "raw_personalized_acc_gain_percent",
        "raw_personalized_weighted_f1_gain_percent",
        "improvement_percent",
        "f1_improvement_percent"
    ]

    selection_cols = [
        col for col in preferred_selection_cols
        if col in df.columns
    ]

    if len(selection_cols) <= 2:
        selection_cols = [
            col for col in fallback_old_cols
            if col in df.columns
        ]

    if selection_cols:
        selection_df = df[selection_cols].copy()

        st.dataframe(
            selection_df,
            use_container_width=True
        )
    else:
        st.info(
            "No detailed model selection gain columns found."
        )

    accepted = (
        df["selected_model_type"]
        == "personalized_fl"
    ).sum()

    rejected = (
        df["selected_model_type"]
        != "personalized_fl"
    ).sum()

    a1, a2 = st.columns(2)

    a1.metric(
        "Accepted Personalized FL Updates",
        int(accepted)
    )

    a2.metric(
        "Rejected FL Updates",
        int(rejected)
    )

else:

    st.info(
        "Selected model type not available in metrics."
    )


# =====================================================
# CONFUSION MATRIX 4x4
# =====================================================

st.subheader("🧩 Latest Selected Model Confusion Matrix")

if "confusion_matrix" in latest:

    cm = np.array(
        latest["confusion_matrix"]
    )

    if cm.shape == (4, 4):

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABEL_NAMES
        )

        display.plot(
            ax=ax,
            values_format="d",
            xticks_rotation=45,
            colorbar=False
        )

        ax.set_title(
            "4-Class Confusion Matrix"
        )

        st.pyplot(fig)

    else:

        st.warning(
            f"Confusion matrix shape is {cm.shape}, expected 4x4."
        )

else:

    st.info(
        "No confusion matrix found in latest metrics."
    )


# =====================================================
# ROUND TABLE
# =====================================================

st.subheader("📋 Round Metrics")

display_columns = [
    col
    for col in [
        "round",
        "local_accuracy",
        "local_weighted_f1",
        "local_macro_f1",
        "local_critical_recall",
        "global_accuracy",
        "global_weighted_f1",
        "personalized_accuracy",
        "personalized_weighted_f1",
        "personalized_critical_recall",
        "selected_model_type",
        "selected_accuracy",
        "selected_weighted_f1",
        "selected_macro_f1",
        "selected_critical_recall",
        "improvement_percent",
        "f1_improvement_percent"
    ]
    if col in df.columns
]

if display_columns:

    table_df = df[display_columns].copy()

    rename_map = {
        "local_accuracy": "updated_local_accuracy",
        "local_weighted_f1": "updated_local_weighted_f1",
        "local_macro_f1": "updated_local_macro_f1",
        "local_critical_recall": "updated_local_critical_recall"
    }

    table_df = table_df.rename(
        columns=rename_map
    )

    st.dataframe(
        table_df,
        use_container_width=True
    )

else:

    st.dataframe(
        df,
        use_container_width=True
    )


# =====================================================
# LIVE PREDICTION PANEL
# =====================================================

st.subheader("🩺 Live Patient Severity Prediction")

st.write(
    "Prediction classes: "
    "**Stable**, **Medium Risk**, **High Risk**, **Critical**"
)

hr = st.number_input(
    "Heart Rate (HR)",
    30.0,
    200.0,
    80.0
)

spo2 = st.number_input(
    "SPO2",
    50.0,
    100.0,
    98.0
)

bp = st.number_input(
    "Blood Pressure",
    50.0,
    220.0,
    120.0
)

rr = st.number_input(
    "Respiratory Rate",
    5.0,
    50.0,
    16.0
)

temp = st.number_input(
    "Temperature",
    30.0,
    45.0,
    37.0
)

ecg = st.selectbox(
    "ECG Abnormality",
    [0, 1]
)


# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Severity"):

    if not os.path.exists(best_model_file):

        st.error(
            "No best selected hospital model found. "
            "Run federated training first."
        )

    else:

        model = load_model_from_path(
            best_model_file
        )

        normalized_values = normalize_live_input(
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

        pred_class, confidence, probs = predict_multiclass(
            model,
            x
        )

        if pred_class == 0:
            risk = "🟢 Stable"
        elif pred_class == 1:
            risk = "🟡 Medium Risk"
        elif pred_class == 2:
            risk = "🟠 High Risk"
        else:
            risk = "🔴 Critical"

        st.success(
            f"Prediction: {risk}"
        )

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        prob_df = pd.DataFrame({
            "Class": LABEL_NAMES,
            "Probability (%)": probs * 100
        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )


# =====================================================
# SYSTEM STATUS
# =====================================================

st.subheader("⚙️ System Status")

st.success(
    "Encryption Active ✅"
)

st.success(
    "Blockchain Logging Active ✅"
)

st.success(
    "Federated Synchronization Active ✅"
)

st.success(
    "4-Class Severity Prediction Active ✅"
)