import torch
import os
import json
import time


# -------------------------
# LOG DIRECTORY
# -------------------------

LOG_DIR = "server/logs"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# -------------------------
# FILES
# -------------------------

LATEST_MODEL_FILE = (
    f"{LOG_DIR}/latest_model.json"
)

GLOBAL_HISTORY_FILE = (
    f"{LOG_DIR}/global_metrics.json"
)

MODEL_REGISTRY_FILE = (
    f"{LOG_DIR}/model_registry.json"
)


# -------------------------
# SAVE GLOBAL MODEL
# -------------------------

def save_global_model(
    weights,
    round_num,
    metrics=None
):

    # -------------------------
    # MODEL PATH
    # -------------------------

    model_path = (
        f"{LOG_DIR}/"
        f"global_model_round_"
        f"{round_num}.pth"
    )

    # -------------------------
    # SAVE MODEL
    # -------------------------

    torch.save(
        weights,
        model_path
    )

    print(
        f"[SERVER] "
        f"💾 Saved Global Model → "
        f"{model_path}"
    )

    # -------------------------
    # SAVE LATEST POINTER
    # -------------------------

    latest_payload = {
        "round": round_num,
        "timestamp": time.time(),
        "model_path": model_path,
        "metrics": metrics
    }

    with open(
        LATEST_MODEL_FILE,
        "w"
    ) as f:

        json.dump(
            latest_payload,
            f,
            indent=4
        )

    # -------------------------
    # SAVE MODEL REGISTRY
    # -------------------------

    if os.path.exists(
        MODEL_REGISTRY_FILE
    ):

        with open(
            MODEL_REGISTRY_FILE,
            "r"
        ) as f:

            registry = json.load(f)

    else:

        registry = []

    registry.append({
        "round": round_num,
        "timestamp": time.time(),
        "model_path": model_path,

        # Helpful summary fields
        "accuracy": get_metric(metrics, "accuracy"),
        "weighted_f1": get_metric(metrics, "weighted_f1"),
        "macro_f1": get_metric(metrics, "macro_f1"),
        "critical_recall": get_metric(metrics, "critical_recall")
    })

    with open(
        MODEL_REGISTRY_FILE,
        "w"
    ) as f:

        json.dump(
            registry,
            f,
            indent=4
        )

    # -------------------------
    # SAVE METRICS HISTORY
    # -------------------------

    if metrics:

        save_global_metrics(
            round_num,
            metrics
        )


# -------------------------
# SAFE METRIC GETTER
# -------------------------

def get_metric(
    metrics,
    key,
    default=0.0
):

    if metrics is None:

        return default

    return metrics.get(
        key,
        default
    )


# -------------------------
# SAVE GLOBAL METRICS
# -------------------------

def save_global_metrics(
    round_num,
    metrics
):

    # -------------------------
    # LOAD OLD HISTORY
    # -------------------------

    if os.path.exists(
        GLOBAL_HISTORY_FILE
    ):

        with open(
            GLOBAL_HISTORY_FILE,
            "r"
        ) as f:

            history = json.load(f)

    else:

        history = []

    # -------------------------
    # CREATE MULTICLASS ENTRY
    # -------------------------

    entry = {
        "round": round_num,
        "timestamp": time.time(),

        # -------------------------
        # Backward-compatible metrics
        # -------------------------

        "accuracy": get_metric(
            metrics,
            "accuracy"
        ),

        "precision": get_metric(
            metrics,
            "precision"
        ),

        "recall": get_metric(
            metrics,
            "recall"
        ),

        "f1": get_metric(
            metrics,
            "f1"
        ),

        "roc_auc": get_metric(
            metrics,
            "roc_auc"
        ),

        # -------------------------
        # Multiclass metrics
        # -------------------------

        "weighted_precision": get_metric(
            metrics,
            "weighted_precision"
        ),

        "weighted_recall": get_metric(
            metrics,
            "weighted_recall"
        ),

        "weighted_f1": get_metric(
            metrics,
            "weighted_f1"
        ),

        "macro_precision": get_metric(
            metrics,
            "macro_precision"
        ),

        "macro_recall": get_metric(
            metrics,
            "macro_recall"
        ),

        "macro_f1": get_metric(
            metrics,
            "macro_f1"
        ),

        # -------------------------
        # Severity-specific metrics
        # -------------------------

        "stable_precision": get_metric(
            metrics,
            "stable_precision"
        ),

        "stable_recall": get_metric(
            metrics,
            "stable_recall"
        ),

        "stable_f1": get_metric(
            metrics,
            "stable_f1"
        ),

        "medium_precision": get_metric(
            metrics,
            "medium_precision"
        ),

        "medium_recall": get_metric(
            metrics,
            "medium_recall"
        ),

        "medium_f1": get_metric(
            metrics,
            "medium_f1"
        ),

        "high_precision": get_metric(
            metrics,
            "high_precision"
        ),

        "high_recall": get_metric(
            metrics,
            "high_recall"
        ),

        "high_f1": get_metric(
            metrics,
            "high_f1"
        ),

        "critical_precision": get_metric(
            metrics,
            "critical_precision"
        ),

        "critical_recall": get_metric(
            metrics,
            "critical_recall"
        ),

        "critical_f1": get_metric(
            metrics,
            "critical_f1"
        ),

        # -------------------------
        # Confusion matrix
        # -------------------------

        "confusion_matrix": get_metric(
            metrics,
            "confusion_matrix",
            []
        )
    }

    history.append(entry)

    # -------------------------
    # SAVE FILE
    # -------------------------

    with open(
        GLOBAL_HISTORY_FILE,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    print(
        f"[SERVER] "
        f"📊 Saved Global Metrics "
        f"for Round {round_num}"
    )


# -------------------------
# LOAD LATEST MODEL INFO
# -------------------------

def load_latest_model_info():

    if not os.path.exists(
        LATEST_MODEL_FILE
    ):

        return None

    with open(
        LATEST_MODEL_FILE,
        "r"
    ) as f:

        return json.load(f)


# -------------------------
# LOAD METRICS HISTORY
# -------------------------

def load_global_metrics():

    if not os.path.exists(
        GLOBAL_HISTORY_FILE
    ):

        return []

    with open(
        GLOBAL_HISTORY_FILE,
        "r"
    ) as f:

        return json.load(f)


# -------------------------
# LOAD MODEL REGISTRY
# -------------------------

def load_model_registry():

    if not os.path.exists(
        MODEL_REGISTRY_FILE
    ):

        return []

    with open(
        MODEL_REGISTRY_FILE,
        "r"
    ) as f:

        return json.load(f)