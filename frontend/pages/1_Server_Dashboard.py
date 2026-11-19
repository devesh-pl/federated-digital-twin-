import streamlit as st
import pandas as pd
import json
import os
import requests


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Federated Server Dashboard",
    layout="wide"
)

st.title("📊 Federated Server Dashboard")

SERVER_URL = "http://127.0.0.1:8000"


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def safe_series(df, preferred_col, fallback_col=None, default=0.0):
    if preferred_col in df.columns:
        return df[preferred_col]

    if fallback_col and fallback_col in df.columns:
        return df[fallback_col]

    return pd.Series(
        [default] * len(df),
        index=df.index
    )


def load_best_selected_fl_results(num_hospitals=3):
    results = []

    for hospital_id in range(num_hospitals):

        path = (
            f"hospital/logs/"
            f"hospital_{hospital_id}_"
            f"baseline_vs_best_fl.json"
        )

        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            data = json.load(f)

        results.append(data)

    return results


# =====================================================
# SERVER STATUS
# =====================================================

st.subheader("🖥️ Server Status")

try:
    response = requests.get(
        f"{SERVER_URL}/",
        timeout=5
    )

    status_data = response.json()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Round",
        status_data.get("current_round", "N/A")
    )

    c2.metric(
        "Registered Hospitals",
        status_data.get("registered_hospitals", "N/A")
    )

    c3.metric(
        "Total Hospitals",
        status_data.get("total_hospitals", "N/A")
    )

    c4.metric(
        "Max Rounds",
        status_data.get("max_rounds", "N/A")
    )

    st.success("Server Running ✅")

    st.write(
        f"Aggregation Mode: "
        f"`{status_data.get('aggregation_mode', 'N/A')}`"
    )

except Exception as e:
    st.error("Server Offline ❌")
    st.caption(str(e))


# =====================================================
# LOAD SERVER METRICS
# =====================================================

global_metrics_path = "server/logs/global_metrics.json"
federated_metrics_path = "server/logs/federated_metrics.json"
hospital_metrics_path = "server/logs/hospital_metrics.json"

global_data = load_json(global_metrics_path)
federated_data = load_json(federated_metrics_path)
hospital_data = load_json(hospital_metrics_path)


# =====================================================
# BEST SELECTED FL METRICS
# =====================================================

st.subheader("🌍 Best Selected FL Metrics")

best_selected_results = load_best_selected_fl_results(
    num_hospitals=3
)

if best_selected_results:

    selected_df = pd.DataFrame(best_selected_results)

    required_cols = [
        "best_selected_accuracy",
        "accuracy_improvement_percent",
        "best_selected_weighted_f1",
        "weighted_f1_improvement_percent",
        "best_selected_macro_f1",
        "macro_f1_improvement_percent",
        "best_selected_critical_recall"
    ]

    for col in required_cols:
        if col not in selected_df.columns:
            selected_df[col] = 0.0

    if "hospital_id" not in selected_df.columns:
        selected_df["hospital_id"] = range(len(selected_df))

    if "best_selected_round" not in selected_df.columns:
        selected_df["best_selected_round"] = "N/A"

    if "best_selected_model_type" not in selected_df.columns:
        selected_df["best_selected_model_type"] = "N/A"

    # -------------------------------------------------
    # BEST ROW BY ACCURACY
    # -------------------------------------------------

    best_row = selected_df.loc[
        selected_df["best_selected_accuracy"].idxmax()
    ]

    best_selected_acc = (
        best_row["best_selected_accuracy"] * 100
    )

    accuracy_improvement = (
        best_row["accuracy_improvement_percent"]
    )

    best_selected_weighted_f1 = (
        best_row["best_selected_weighted_f1"]
    )

    weighted_f1_improvement = (
        best_row["weighted_f1_improvement_percent"]
    )

    best_selected_macro_f1 = (
        best_row["best_selected_macro_f1"]
    )

    macro_f1_improvement = (
        best_row["macro_f1_improvement_percent"]
    )

    best_selected_critical_recall = (
        best_row["best_selected_critical_recall"]
    )

    # -------------------------------------------------
    # TOP METRICS
    # -------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best Selected FL Acc",
        f"{best_selected_acc:.2f}%",
        f"{accuracy_improvement:.2f}%"
    )

    c2.metric(
        "Best Weighted F1",
        f"{best_selected_weighted_f1:.4f}",
        f"{weighted_f1_improvement:.2f}%"
    )

    c3.metric(
        "Best Macro F1",
        f"{best_selected_macro_f1:.4f}",
        f"{macro_f1_improvement:.2f}%"
    )

    c4.metric(
        "Best Critical Recall",
        f"{best_selected_critical_recall:.4f}"
    )

    st.info(
        f"""
Best selected FL checkpoint came from Hospital {int(best_row['hospital_id'])}.

Best Selected FL Acc: {best_selected_acc:.2f}%  
Accuracy Improvement: {accuracy_improvement:.2f}%

This dashboard reports the **best selected personalized FL model**, not only the raw global model,
because raw global FL can be lower when hospital data is non-IID.
"""
    )

    # -------------------------------------------------
    # TABLE
    # -------------------------------------------------

    st.subheader("📋 Best Selected FL by Hospital")

    display_cols = [
        "hospital_id",
        "local_baseline_accuracy",
        "best_selected_accuracy",
        "accuracy_improvement_percent",
        "local_baseline_weighted_f1",
        "best_selected_weighted_f1",
        "weighted_f1_improvement_percent",
        "local_baseline_macro_f1",
        "best_selected_macro_f1",
        "macro_f1_improvement_percent",
        "local_baseline_critical_recall",
        "best_selected_critical_recall",
        "critical_recall_improvement_percent",
        "best_selected_round",
        "best_selected_model_type"
    ]

    display_cols = [
        col for col in display_cols
        if col in selected_df.columns
    ]

    display_df = selected_df[display_cols].copy()

    percent_cols = [
        "local_baseline_accuracy",
        "best_selected_accuracy"
    ]

    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col] * 100

    display_df = display_df.rename(
        columns={
            "hospital_id": "Hospital",
            "local_baseline_accuracy": "Local Baseline Acc %",
            "best_selected_accuracy": "Best Selected FL Acc %",
            "accuracy_improvement_percent": "Accuracy Improvement %",
            "local_baseline_weighted_f1": "Local Baseline Weighted F1",
            "best_selected_weighted_f1": "Best Selected Weighted F1",
            "weighted_f1_improvement_percent": "Weighted F1 Improvement %",
            "local_baseline_macro_f1": "Local Baseline Macro F1",
            "best_selected_macro_f1": "Best Selected Macro F1",
            "macro_f1_improvement_percent": "Macro F1 Improvement %",
            "local_baseline_critical_recall": "Local Baseline Critical Recall",
            "best_selected_critical_recall": "Best Selected Critical Recall",
            "critical_recall_improvement_percent": "Critical Recall Improvement %",
            "best_selected_round": "Best Round",
            "best_selected_model_type": "Best Model Type"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # -------------------------------------------------
    # BAR CHART
    # -------------------------------------------------

    st.subheader("📈 Local Baseline vs Best Selected FL Accuracy")

    chart_df = selected_df.copy()

    chart_df["Local Baseline Acc"] = (
        chart_df.get("local_baseline_accuracy", 0.0) * 100
    )

    chart_df["Best Selected FL Acc"] = (
        chart_df.get("best_selected_accuracy", 0.0) * 100
    )

    chart_df = chart_df.set_index("hospital_id")[
        [
            "Local Baseline Acc",
            "Best Selected FL Acc"
        ]
    ]

    st.bar_chart(chart_df)

else:
    st.warning(
        "No best selected FL comparison files found yet. "
        "Run all hospital clients first."
    )


# =====================================================
# RAW GLOBAL METRICS
# =====================================================

st.subheader("🌐 Raw Global Aggregation Metrics")

if global_data:

    df = pd.DataFrame(global_data)

    if df.empty:
        st.warning("Global metrics file is empty.")

    else:
        if "weighted_f1" not in df.columns and "f1" in df.columns:
            df["weighted_f1"] = df["f1"]

        if "weighted_precision" not in df.columns and "precision" in df.columns:
            df["weighted_precision"] = df["precision"]

        if "weighted_recall" not in df.columns and "recall" in df.columns:
            df["weighted_recall"] = df["recall"]

        if "macro_f1" not in df.columns:
            df["macro_f1"] = 0.0

        if "critical_recall" not in df.columns:
            df["critical_recall"] = 0.0

        if "accuracy" not in df.columns:
            df["accuracy"] = 0.0

        if "round" not in df.columns:
            df["round"] = range(1, len(df) + 1)

        st.caption(
            "These are raw server-side global model metrics. "
            "They may be lower than selected personalized FL metrics."
        )

        best_acc_idx = df["accuracy"].idxmax()
        best_acc_row = df.loc[best_acc_idx]

        best_f1_idx = df["weighted_f1"].idxmax()
        best_f1_row = df.loc[best_f1_idx]

        best_macro_idx = df["macro_f1"].idxmax()
        best_macro_row = df.loc[best_macro_idx]

        best_critical_idx = df["critical_recall"].idxmax()
        best_critical_row = df.loc[best_critical_idx]

        best_acc = best_acc_row.get("accuracy", 0.0) * 100
        best_acc_round = int(best_acc_row.get("round", 0))

        best_weighted_f1 = best_f1_row.get("weighted_f1", 0.0)
        best_weighted_f1_round = int(best_f1_row.get("round", 0))

        best_macro_f1 = best_macro_row.get("macro_f1", 0.0)
        best_macro_f1_round = int(best_macro_row.get("round", 0))

        best_critical_recall = best_critical_row.get("critical_recall", 0.0)
        best_critical_round = int(best_critical_row.get("round", 0))

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Best Raw Global Acc",
            f"{best_acc:.2f}%"
        )

        c2.metric(
            "Best Raw Weighted F1",
            f"{best_weighted_f1:.4f}"
        )

        c3.metric(
            "Best Raw Macro F1",
            f"{best_macro_f1:.4f}"
        )

        c4.metric(
            "Best Raw Critical Recall",
            f"{best_critical_recall:.4f}"
        )

        with st.expander("Show Raw Global Metric Graphs"):

            st.write(
                f"""
Best raw global checkpoints:

- Accuracy: Round {best_acc_round}
- Weighted F1: Round {best_weighted_f1_round}
- Macro F1: Round {best_macro_f1_round}
- Critical Recall: Round {best_critical_round}
"""
            )

            st.subheader("📈 Raw Global Accuracy Progression")

            accuracy_values = df["accuracy"] * 100
            best_accuracy_values = accuracy_values.cummax()

            acc_df = pd.DataFrame({
                "Round": df["round"],
                "Raw Global Accuracy": accuracy_values,
                "Best Raw Global Accuracy So Far": best_accuracy_values
            }).set_index("Round")

            st.line_chart(acc_df)

            st.subheader("📊 Raw Global Weighted F1 Progression")

            weighted_f1_series = safe_series(
                df,
                "weighted_f1",
                "f1",
                0.0
            )

            best_weighted_f1_series = weighted_f1_series.cummax()

            f1_df = pd.DataFrame({
                "Round": df["round"],
                "Raw Global Weighted F1": weighted_f1_series,
                "Best Raw Weighted F1 So Far": best_weighted_f1_series
            }).set_index("Round")

            st.line_chart(f1_df)

            st.subheader("🧠 Raw Global Macro F1 Progression")

            macro_f1_series = safe_series(
                df,
                "macro_f1",
                None,
                0.0
            )

            best_macro_f1_series = macro_f1_series.cummax()

            macro_df = pd.DataFrame({
                "Round": df["round"],
                "Raw Global Macro F1": macro_f1_series,
                "Best Raw Macro F1 So Far": best_macro_f1_series
            }).set_index("Round")

            st.line_chart(macro_df)

            st.subheader("🚨 Raw Global Critical Recall Progression")

            critical_recall_series = safe_series(
                df,
                "critical_recall",
                None,
                0.0
            )

            best_critical_recall_series = critical_recall_series.cummax()

            critical_df = pd.DataFrame({
                "Round": df["round"],
                "Raw Global Critical Recall": critical_recall_series,
                "Best Raw Critical Recall So Far": best_critical_recall_series
            }).set_index("Round")

            st.line_chart(critical_df)

            st.subheader("📋 Raw Global Metrics Table")

            display_cols = [
                col
                for col in [
                    "round",
                    "accuracy",
                    "weighted_precision",
                    "weighted_recall",
                    "weighted_f1",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "critical_recall",
                    "critical_f1",
                    "high_recall",
                    "high_f1",
                    "roc_auc"
                ]
                if col in df.columns
            ]

            if display_cols:
                display_df = df[display_cols].copy()

                if "accuracy" in display_df.columns:
                    display_df["accuracy"] = display_df["accuracy"] * 100
                    display_df = display_df.rename(
                        columns={
                            "accuracy": "accuracy_percent"
                        }
                    )

                st.dataframe(
                    display_df,
                    use_container_width=True
                )

            else:
                st.dataframe(
                    df,
                    use_container_width=True
                )

else:
    st.info(
        "No raw global metrics found yet. "
        "Expected file: server/logs/global_metrics.json"
    )


# =====================================================
# FEDERATED ROUND DETAILS
# =====================================================

st.subheader("⚖️ Aggregation Details")

if federated_data:

    fed_df = pd.DataFrame(federated_data)

    if not fed_df.empty:

        latest_fed = fed_df.iloc[-1]

        st.write(
            f"Latest Aggregation Mode: "
            f"`{latest_fed.get('aggregation_mode', 'N/A')}`"
        )

        aggregation_weights = latest_fed.get(
            "aggregation_weights",
            {}
        )

        if aggregation_weights:

            weights_df = pd.DataFrame({
                "Hospital": list(aggregation_weights.keys()),
                "Aggregation Weight": list(aggregation_weights.values())
            })

            st.dataframe(
                weights_df,
                use_container_width=True
            )

            st.bar_chart(
                weights_df.set_index("Hospital")
            )

        st.write("### Federated Round History")

        compact_rows = []

        for item in federated_data:
            gm = item.get("global_metrics", {})

            compact_rows.append({
                "Round": item.get("round"),
                "Aggregation Mode": item.get("aggregation_mode"),
                "Accuracy %": gm.get("accuracy", 0.0) * 100,
                "Weighted F1": gm.get(
                    "weighted_f1",
                    gm.get("f1", 0.0)
                ),
                "Macro F1": gm.get("macro_f1", 0.0),
                "Critical Recall": gm.get("critical_recall", 0.0),
                "ROC-AUC": gm.get("roc_auc", 0.0)
            })

        st.dataframe(
            pd.DataFrame(compact_rows),
            use_container_width=True
        )

else:
    st.info(
        "No federated aggregation details found yet. "
        "Expected file: server/logs/federated_metrics.json"
    )


# =====================================================
# HOSPITAL UPLOAD METRICS
# =====================================================

st.subheader("🏥 Hospital Upload Metrics")

if hospital_data:

    hospital_df = pd.DataFrame(hospital_data)

    if not hospital_df.empty:

        display_cols = [
            col
            for col in [
                "round",
                "hospital_id",
                "local_accuracy",
                "local_weighted_f1",
                "local_weighted_precision",
                "local_weighted_recall",
                "local_roc_auc"
            ]
            if col in hospital_df.columns
        ]

        st.dataframe(
            hospital_df[display_cols]
            if display_cols
            else hospital_df,
            use_container_width=True
        )

        if {
            "round",
            "hospital_id",
            "local_accuracy"
        }.issubset(hospital_df.columns):

            st.write("### Hospital Updated Local Accuracy by Round")

            acc_pivot = hospital_df.pivot_table(
                index="round",
                columns="hospital_id",
                values="local_accuracy",
                aggfunc="mean"
            ) * 100

            st.line_chart(acc_pivot)

        if {
            "round",
            "hospital_id",
            "local_weighted_f1"
        }.issubset(hospital_df.columns):

            st.write("### Hospital Weighted F1 by Round")

            f1_pivot = hospital_df.pivot_table(
                index="round",
                columns="hospital_id",
                values="local_weighted_f1",
                aggfunc="mean"
            )

            st.line_chart(f1_pivot)

else:
    st.info(
        "No hospital upload metrics found yet. "
        "Expected file: server/logs/hospital_metrics.json"
    )


# =====================================================
# BLOCKCHAIN EVENTS
# =====================================================

st.subheader("🔗 Blockchain Aggregation Events")

blockchain_path = "server/logs/blockchain.json"

chain = load_json(blockchain_path)

if chain:

    latest_blocks = chain[-5:]

    for i, block in enumerate(
        reversed(latest_blocks)
    ):

        block_number = len(chain) - i

        block_data = block.get("data", {})

        st.info(
            f"""
Block #{block_number}

Data:
{json.dumps(block_data, indent=2)}

Hash:
{block.get('hash', 'N/A')}
"""
        )

else:
    st.warning("No blockchain events yet.")


# =====================================================
# SYSTEM STATUS
# =====================================================

st.subheader("⚙️ System Components")

s1, s2, s3, s4 = st.columns(4)

s1.success("Federated Aggregation Active ✅")
s2.success("Encrypted Upload Pipeline Active ✅")
s3.success("Blockchain Audit Trail Active ✅")
s4.success("4-Class Severity Model Active ✅")


# =====================================================
# RESEARCH DESCRIPTION
# =====================================================

st.subheader("🧠 System Overview")

st.info(
    """
This federated learning framework enables multiple hospitals to collaboratively train
a healthcare severity prediction model without sharing raw patient data.

The current model predicts four patient severity classes:

0 = Stable  
1 = Medium Risk  
2 = High Risk  
3 = Critical  

This server dashboard reports the **best selected FL model** from hospital-side
personalized federated learning results, because selected FL is the final model used
by each hospital.

Raw global aggregation metrics are still shown separately because global-only FL can
fluctuate and may be lower when hospital datasets are non-IID.

For the multiclass setup, the dashboard reports:

- Best Selected FL Accuracy
- Accuracy Improvement
- Weighted F1
- Macro F1
- Critical Recall
- Hospital-wise baseline vs selected FL comparison
- Raw global metrics separately
"""
)