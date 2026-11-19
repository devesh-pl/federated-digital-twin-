import streamlit as st
import requests
import pandas as pd
import json


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Federated Control Panel",
    layout="wide"
)

st.title("⚙️ Federated Control Panel")


# =====================================================
# SERVER INPUT
# =====================================================

server_ip = st.text_input(
    "Server IP",
    "127.0.0.1"
)

server_url = f"http://{server_ip}:8000"


# =====================================================
# CHECK SERVER STATUS
# =====================================================

st.subheader("🖥️ Server Status")

if st.button("Check Server Status"):

    try:

        r = requests.get(
            f"{server_url}/status",
            timeout=5
        )

        status = r.json()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Round",
            status.get("current_round", "N/A")
        )

        c2.metric(
            "Max Rounds",
            status.get("max_rounds", "N/A")
        )

        c3.metric(
            "Registered Hospitals",
            status.get("registered_hospitals", "N/A")
        )

        c4.metric(
            "Uploaded Weights",
            status.get("uploaded_weights", "N/A")
        )

        st.success("Server reachable ✅")

        st.write(
            f"Aggregation Mode: "
            f"`{status.get('aggregation_mode', 'N/A')}`"
        )

        # -------------------------
        # HOSPITAL STATUS
        # -------------------------

        hospital_status = status.get(
            "hospital_status",
            {}
        )

        if hospital_status:

            st.subheader("🏥 Hospital Sync Status")

            rows = []

            for hospital_id, info in hospital_status.items():

                rows.append({
                    "Hospital": hospital_id,
                    "Uploaded": info.get("uploaded", False),
                    "Round": info.get("round", 0)
                })

            hospital_df = pd.DataFrame(rows)

            st.dataframe(
                hospital_df,
                use_container_width=True
            )

        # -------------------------
        # AGGREGATION WEIGHTS
        # -------------------------

        aggregation_weights = status.get(
            "aggregation_weights",
            {}
        )

        if aggregation_weights:

            st.subheader("⚖️ Current Aggregation Weights")

            weights_df = pd.DataFrame({
                "Hospital": list(aggregation_weights.keys()),
                "Weight": list(aggregation_weights.values())
            })

            st.dataframe(
                weights_df,
                use_container_width=True
            )

            st.bar_chart(
                weights_df.set_index("Hospital")
            )

        # -------------------------
        # LATEST GLOBAL METRICS
        # -------------------------

        latest_global_metrics = status.get(
            "latest_global_metrics",
            {}
        )

        if latest_global_metrics:

            st.subheader("🌍 Latest Global Metrics")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Accuracy",
                f"{latest_global_metrics.get('accuracy', 0.0) * 100:.2f}%"
            )

            m2.metric(
                "Weighted F1",
                f"{latest_global_metrics.get('weighted_f1', latest_global_metrics.get('f1', 0.0)):.4f}"
            )

            m3.metric(
                "Macro F1",
                f"{latest_global_metrics.get('macro_f1', 0.0):.4f}"
            )

            m4.metric(
                "Critical Recall",
                f"{latest_global_metrics.get('critical_recall', 0.0):.4f}"
            )

        with st.expander("Show raw server status JSON"):

            st.json(status)

    except Exception as e:

        st.error("Server unreachable ❌")
        st.caption(str(e))


# =====================================================
# HOME ENDPOINT
# =====================================================

st.subheader("🏠 Server Home Endpoint")

if st.button("Check Home Endpoint"):

    try:

        r = requests.get(
            server_url,
            timeout=5
        )

        st.json(
            r.json()
        )

    except Exception as e:

        st.error("Could not connect to server.")
        st.caption(str(e))


# =====================================================
# GLOBAL MODEL ENDPOINT
# =====================================================

st.subheader("🌍 Global Model Availability")

if st.button("Check Global Model"):

    try:

        r = requests.get(
            f"{server_url}/global_model",
            timeout=5
        )

        data = r.json()

        status = data.get(
            "status",
            "unknown"
        )

        if status == "ready":

            st.success(
                f"Global model ready for round "
                f"{data.get('round', 'N/A')} ✅"
            )

            st.write(
                f"Aggregation Mode: "
                f"`{data.get('aggregation_mode', 'N/A')}`"
            )

            global_metrics = data.get(
                "global_metrics",
                {}
            )

            if global_metrics:

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Accuracy",
                    f"{global_metrics.get('accuracy', 0.0) * 100:.2f}%"
                )

                c2.metric(
                    "Weighted F1",
                    f"{global_metrics.get('weighted_f1', global_metrics.get('f1', 0.0)):.4f}"
                )

                c3.metric(
                    "Macro F1",
                    f"{global_metrics.get('macro_f1', 0.0):.4f}"
                )

                c4.metric(
                    "Critical Recall",
                    f"{global_metrics.get('critical_recall', 0.0):.4f}"
                )

        elif status == "not_ready":

            st.warning("Global model is not ready yet.")

        elif status == "finished":

            st.success("Federated learning is finished 🎉")

        else:

            st.info(f"Global model status: {status}")

        with st.expander("Show raw global model response"):

            # Avoid rendering very large weights by hiding them
            if "weights" in data:
                data_preview = data.copy()
                data_preview["weights"] = "[hidden large weight array]"
                st.json(data_preview)
            else:
                st.json(data)

    except Exception as e:

        st.error("Could not fetch global model.")
        st.caption(str(e))


# =====================================================
# RUN COMMANDS
# =====================================================

st.subheader("▶️ How to Run Federated Training")

st.code(
    """
# Terminal 1: Run server
python -m server.federated_server

# Terminal 2: Run Hospital 0
python -m hospital.hospital_client --id 0

# Terminal 3: Run Hospital 1
python -m hospital.hospital_client --id 1

# Terminal 4: Run Hospital 2
python -m hospital.hospital_client --id 2

# Terminal 5: Run dashboard
streamlit run frontend/app.py
""",
    language="bash"
)


# =====================================================
# INFO
# =====================================================

st.info(
    """
This control panel checks the federated server status, hospital synchronization,
aggregation weights, and latest global model metrics.

For the 4-class severity model:
- local_f1 means weighted F1
- selected/federated F1 means weighted F1
- macro F1 and critical recall are used to judge severity prediction quality
"""
)