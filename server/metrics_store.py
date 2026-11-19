import json
import os
import time


# =====================================================
# PATHS
# =====================================================

HOSPITAL_METRICS_FILE = "server/logs/hospital_metrics.json"


# =====================================================
# SAVE HOSPITAL METRICS
# =====================================================

def save_hospital_metrics(metrics):
    """
    Save hospital-side metrics received by the federated server.

    For the 4-class severity model:
    - local_f1 means weighted F1
    - local_precision means weighted precision
    - local_recall means weighted recall
    - local_auc means multiclass ROC-AUC if available, else 0.0
    """

    os.makedirs(
        "server/logs",
        exist_ok=True
    )

    # -------------------------
    # LOAD EXISTING METRICS
    # -------------------------

    if os.path.exists(HOSPITAL_METRICS_FILE):

        with open(HOSPITAL_METRICS_FILE, "r") as f:
            data = json.load(f)

    else:

        data = []

    # -------------------------
    # NORMALIZE METRIC ENTRY
    # -------------------------

    clean_metrics = {
        "timestamp": time.time(),

        "hospital_id":
            metrics.get("hospital_id"),

        "round":
            metrics.get("round"),

        "local_accuracy":
            float(metrics.get("local_accuracy", 0.0)),

        # Backward-compatible field.
        # In multiclass mode, this is weighted F1.
        "local_f1":
            float(metrics.get("local_f1", 0.0)),

        "local_weighted_f1":
            float(
                metrics.get(
                    "local_weighted_f1",
                    metrics.get("local_f1", 0.0)
                )
            ),

        # Backward-compatible field.
        # In multiclass mode, this is weighted precision.
        "local_precision":
            float(metrics.get("local_precision", 0.0)),

        "local_weighted_precision":
            float(
                metrics.get(
                    "local_weighted_precision",
                    metrics.get("local_precision", 0.0)
                )
            ),

        # Backward-compatible field.
        # In multiclass mode, this is weighted recall.
        "local_recall":
            float(metrics.get("local_recall", 0.0)),

        "local_weighted_recall":
            float(
                metrics.get(
                    "local_weighted_recall",
                    metrics.get("local_recall", 0.0)
                )
            ),

        "local_auc":
            float(metrics.get("local_auc", 0.0)),

        "local_roc_auc":
            float(
                metrics.get(
                    "local_roc_auc",
                    metrics.get("local_auc", 0.0)
                )
            )
    }

    # -------------------------
    # APPEND + SAVE
    # -------------------------

    data.append(clean_metrics)

    with open(HOSPITAL_METRICS_FILE, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )

    print(
        f"[METRICS STORE] Saved metrics "
        f"for Hospital {clean_metrics.get('hospital_id')} | "
        f"Round {clean_metrics.get('round')} | "
        f"Acc: {clean_metrics['local_accuracy']:.4f} | "
        f"Weighted F1: {clean_metrics['local_weighted_f1']:.4f}"
    )