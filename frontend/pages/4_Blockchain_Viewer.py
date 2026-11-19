import streamlit as st
import json
import os
import pandas as pd


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Blockchain Audit Trail",
    layout="wide"
)

st.title("🔗 Blockchain Audit Trail")

path = "server/logs/blockchain.json"


# =====================================================
# LOAD BLOCKCHAIN
# =====================================================

if not os.path.exists(path):

    st.warning("No blockchain data found yet.")
    st.stop()

with open(path, "r") as f:
    chain = json.load(f)

if not chain:

    st.warning("Blockchain file is empty.")
    st.stop()


# =====================================================
# SUMMARY
# =====================================================

st.subheader("📌 Blockchain Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Blocks",
    len(chain)
)

latest_block = chain[-1]

latest_data = latest_block.get(
    "data",
    {}
)

c2.metric(
    "Latest Round",
    latest_data.get("round", "N/A")
)

c3.metric(
    "Latest Event",
    latest_data.get("event", "N/A")
)


# =====================================================
# LATEST AGGREGATION DETAILS
# =====================================================

st.subheader("⚖️ Latest Aggregation Details")

aggregation_mode = latest_data.get(
    "aggregation_mode",
    "N/A"
)

st.write(
    f"Aggregation Mode: `{aggregation_mode}`"
)

aggregation_weights = latest_data.get(
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

else:

    st.info("No aggregation weights found in latest block.")


# =====================================================
# LATEST GLOBAL METRICS
# =====================================================

st.subheader("🌍 Latest Global Model Metrics")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{latest_data.get('accuracy', 0.0) * 100:.2f}%"
)

m2.metric(
    "Weighted F1",
    f"{latest_data.get('weighted_f1', latest_data.get('f1', 0.0)):.4f}"
)

m3.metric(
    "Macro F1",
    f"{latest_data.get('macro_f1', 0.0):.4f}"
)

m4.metric(
    "Critical Recall",
    f"{latest_data.get('critical_recall', 0.0):.4f}"
)


# =====================================================
# BLOCKCHAIN TABLE
# =====================================================

st.subheader("📋 Blockchain Round Table")

rows = []

for index, block in enumerate(chain):

    data = block.get("data", {})

    rows.append({
        "Block": index,
        "Round": data.get("round", "N/A"),
        "Event": data.get("event", "N/A"),
        "Aggregation Mode": data.get("aggregation_mode", "N/A"),
        "Hospitals": data.get("hospitals", []),
        "Accuracy": data.get("accuracy", 0.0),
        "Weighted F1": data.get("weighted_f1", data.get("f1", 0.0)),
        "Macro F1": data.get("macro_f1", 0.0),
        "Critical Recall": data.get("critical_recall", 0.0),
        "Hash": block.get("hash", "N/A")
    })

df = pd.DataFrame(rows)

st.dataframe(
    df,
    use_container_width=True
)


# =====================================================
# FULL BLOCK VIEWER
# =====================================================

st.subheader("🔍 Full Block Details")

selected_block = st.selectbox(
    "Select Block",
    list(range(len(chain))),
    index=len(chain) - 1
)

st.json(
    chain[selected_block]
)


# =====================================================
# RAW BLOCKCHAIN VIEW
# =====================================================

with st.expander("Show complete blockchain JSON"):

    st.json(chain)


# =====================================================
# INFO
# =====================================================

st.info(
    """
The blockchain audit trail records every federated aggregation event.

For the 4-class severity model, each block can store:
- Round number
- Participating hospitals
- Aggregation mode
- Aggregation weights
- Global accuracy
- Weighted F1
- Macro F1
- Critical recall
- Model audit hash

This provides a tamper-evident record of federated learning progress.
"""
)