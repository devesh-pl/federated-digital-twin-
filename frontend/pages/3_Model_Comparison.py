import sys
import os
import json
import pandas as pd
import streamlit as st


# =====================================================
# PROJECT PATH SETUP
# =====================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

st.title("📈 Local vs Federated Model Comparison")


# =====================================================
# HELPERS
# =====================================================

def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def safe_get(data, key, default=0.0):
    if data is None:
        return default

    return data.get(key, default)


# =====================================================
# LOAD COMPARISON RESULTS
# =====================================================

results = []
round_results = []

for hospital_id in [0, 1, 2]:

    comparison_path = (
        f"hospital/logs/"
        f"hospital_{hospital_id}_baseline_vs_best_fl.json"
    )

    metrics_path = (
        f"hospital/logs/"
        f"hospital_{hospital_id}_metrics.json"
    )

    comparison = load_json(comparison_path)

    # -------------------------------------------------
    # Preferred: use final baseline_vs_best_fl.json
    # -------------------------------------------------

    if comparison is not None:

        local_acc = safe_get(
            comparison,
            "local_baseline_accuracy"
        ) * 100

        best_acc = safe_get(
            comparison,
            "best_selected_accuracy"
        ) * 100

        acc_improvement = safe_get(
            comparison,
            "accuracy_improvement_percent"
        )

        local_weighted_f1 = safe_get(
            comparison,
            "local_baseline_weighted_f1"
        )

        best_weighted_f1 = safe_get(
            comparison,
            "best_selected_weighted_f1"
        )

        weighted_f1_improvement = safe_get(
            comparison,
            "weighted_f1_improvement_percent"
        )

        local_macro_f1 = safe_get(
            comparison,
            "local_baseline_macro_f1"
        )

        best_macro_f1 = safe_get(
            comparison,
            "best_selected_macro_f1"
        )

        macro_f1_improvement = safe_get(
            comparison,
            "macro_f1_improvement_percent"
        )

        local_critical_recall = safe_get(
            comparison,
            "local_baseline_critical_recall"
        )

        best_critical_recall = safe_get(
            comparison,
            "best_selected_critical_recall"
        )

        critical_recall_improvement = safe_get(
            comparison,
            "critical_recall_improvement_percent"
        )

        best_round = safe_get(
            comparison,
            "best_selected_round",
            "N/A"
        )

        best_model_type = safe_get(
            comparison,
            "best_selected_model_type",
            "N/A"
        )

        results.append({
            "Hospital": hospital_id,

            "Local Baseline Accuracy": local_acc,
            "Best Selected FL Accuracy": best_acc,
            "Accuracy Improvement %": acc_improvement,

            "Local Weighted F1": local_weighted_f1,
            "Best Selected Weighted F1": best_weighted_f1,
            "Weighted F1 Improvement %": weighted_f1_improvement,

            "Local Macro F1": local_macro_f1,
            "Best Selected Macro F1": best_macro_f1,
            "Macro F1 Improvement %": macro_f1_improvement,

            "Local Critical Recall": local_critical_recall,
            "Best Selected Critical Recall": best_critical_recall,
            "Critical Recall Improvement %": critical_recall_improvement,

            "Best Round": best_round,
            "Best Model Type": best_model_type
        })

    # -------------------------------------------------
    # Also load round-wise metrics if available
    # -------------------------------------------------

    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as f:
            data = json.load(f)

        df_rounds = pd.DataFrame(data)

        if not df_rounds.empty:

            for _, row in df_rounds.iterrows():

                round_results.append({
                    "Hospital": hospital_id,
                    "Round": row.get("round", None),

                    "Local Accuracy":
                        row.get("local_accuracy", 0.0) * 100,

                    "Global FL Accuracy":
                        row.get("global_accuracy", 0.0) * 100,

                    "Personalized FL Accuracy":
                        row.get("personalized_accuracy", 0.0) * 100,

                    "Selected FL Accuracy":
                        row.get("selected_accuracy", row.get("federated_accuracy", 0.0)) * 100,

                    "Local Weighted F1":
                        row.get("local_weighted_f1", row.get("local_f1", 0.0)),

                    "Global Weighted F1":
                        row.get("global_weighted_f1", row.get("global_f1", 0.0)),

                    "Personalized Weighted F1":
                        row.get("personalized_weighted_f1", row.get("personalized_f1", 0.0)),

                    "Selected Weighted F1":
                        row.get("selected_weighted_f1", row.get("federated_f1", 0.0)),

                    "Selected Macro F1":
                        row.get("selected_macro_f1", 0.0),

                    "Selected Critical Recall":
                        row.get("selected_critical_recall", 0.0),

                    "Selected Model Type":
                        row.get("selected_model_type", "N/A")
                })


# =====================================================
# EMPTY CHECK
# =====================================================

if not results:

    st.warning(
        "No final comparison files found yet. "
        "Run hospital training first so baseline_vs_best_fl.json files are created."
    )

    st.stop()


df_results = pd.DataFrame(results)


# =====================================================
# SUMMARY CARDS
# =====================================================

st.subheader("📌 Overall Research Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Avg Local Baseline Acc",
    f"{df_results['Local Baseline Accuracy'].mean():.2f}%"
)

c2.metric(
    "Avg Best Selected FL Acc",
    f"{df_results['Best Selected FL Accuracy'].mean():.2f}%",
    f"{df_results['Accuracy Improvement %'].mean():.2f}%"
)

c3.metric(
    "Avg Best Weighted F1",
    f"{df_results['Best Selected Weighted F1'].mean():.4f}",
    f"{df_results['Weighted F1 Improvement %'].mean():.2f}%"
)

c4.metric(
    "Avg Critical Recall",
    f"{df_results['Best Selected Critical Recall'].mean():.4f}",
    f"{df_results['Critical Recall Improvement %'].mean():.2f}%"
)


# =====================================================
# RESEARCH COMPARISON TABLE
# =====================================================

st.subheader("📋 Local Baseline vs Best Selected FL Table")

display_df = df_results.copy()

percentage_columns = [
    "Local Baseline Accuracy",
    "Best Selected FL Accuracy",
    "Accuracy Improvement %"
]

for col in percentage_columns:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.2f}%"
        )

metric_columns = [
    "Local Weighted F1",
    "Best Selected Weighted F1",
    "Local Macro F1",
    "Best Selected Macro F1",
    "Local Critical Recall",
    "Best Selected Critical Recall"
]

for col in metric_columns:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}"
        )

improvement_metric_columns = [
    "Weighted F1 Improvement %",
    "Macro F1 Improvement %",
    "Critical Recall Improvement %"
]

for col in improvement_metric_columns:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.2f}%"
        )

st.dataframe(
    display_df,
    use_container_width=True
)


# =====================================================
# ACCURACY COMPARISON
# =====================================================

st.subheader("📊 Accuracy: Local Baseline vs Best Selected FL")

accuracy_chart = df_results.set_index("Hospital")[
    [
        "Local Baseline Accuracy",
        "Best Selected FL Accuracy"
    ]
]

st.bar_chart(accuracy_chart)


# =====================================================
# WEIGHTED F1 COMPARISON
# =====================================================

st.subheader("📊 Weighted F1: Local Baseline vs Best Selected FL")

weighted_f1_chart = df_results.set_index("Hospital")[
    [
        "Local Weighted F1",
        "Best Selected Weighted F1"
    ]
]

st.bar_chart(weighted_f1_chart)


# =====================================================
# MACRO F1 COMPARISON
# =====================================================

st.subheader("📊 Macro F1: Local Baseline vs Best Selected FL")

macro_f1_chart = df_results.set_index("Hospital")[
    [
        "Local Macro F1",
        "Best Selected Macro F1"
    ]
]

st.bar_chart(macro_f1_chart)


# =====================================================
# CRITICAL RECALL COMPARISON
# =====================================================

st.subheader("🚨 Critical Recall: Local Baseline vs Best Selected FL")

critical_recall_chart = df_results.set_index("Hospital")[
    [
        "Local Critical Recall",
        "Best Selected Critical Recall"
    ]
]

st.bar_chart(critical_recall_chart)


# =====================================================
# IMPROVEMENT CHARTS
# =====================================================

st.subheader("📈 Improvement by Hospital")

improvement_chart = df_results.set_index("Hospital")[
    [
        "Accuracy Improvement %",
        "Weighted F1 Improvement %",
        "Macro F1 Improvement %",
        "Critical Recall Improvement %"
    ]
]

st.bar_chart(improvement_chart)


# =====================================================
# ROUND-WISE COMPARISON
# =====================================================

if round_results:

    st.subheader("🔁 Round-wise Model Comparison")

    df_rounds_all = pd.DataFrame(round_results)

    selected_hospital = st.selectbox(
        "Select hospital for round-wise view",
        sorted(df_rounds_all["Hospital"].unique())
    )

    hospital_rounds = df_rounds_all[
        df_rounds_all["Hospital"] == selected_hospital
    ].copy()

    hospital_rounds = hospital_rounds.sort_values(
        by="Round"
    )

    st.write("### Accuracy by Round")

    accuracy_round_chart = hospital_rounds.set_index("Round")[
        [
            "Local Accuracy",
            "Global FL Accuracy",
            "Personalized FL Accuracy",
            "Selected FL Accuracy"
        ]
    ]

    st.line_chart(accuracy_round_chart)

    st.write("### Weighted F1 by Round")

    f1_round_chart = hospital_rounds.set_index("Round")[
        [
            "Local Weighted F1",
            "Global Weighted F1",
            "Personalized Weighted F1",
            "Selected Weighted F1"
        ]
    ]

    st.line_chart(f1_round_chart)

    st.write("### Selected Model Details")

    st.dataframe(
        hospital_rounds[
            [
                "Hospital",
                "Round",
                "Selected Model Type",
                "Selected FL Accuracy",
                "Selected Weighted F1",
                "Selected Macro F1",
                "Selected Critical Recall"
            ]
        ],
        use_container_width=True
    )


# =====================================================
# INTERPRETATION
# =====================================================

st.info(
    """
This comparison uses the local baseline model trained only on each hospital's own data,
then compares it against the best validation-gated selected federated model.

For the 4-class severity prediction task, the dashboard reports:
- Accuracy
- Weighted F1
- Macro F1
- Weighted precision and recall through saved metrics
- Critical recall for detecting the most severe patient state

The field `federated_f1` is kept for compatibility, but it now represents
the selected model's weighted F1 score.
"""
)