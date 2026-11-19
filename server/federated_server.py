import uvicorn
import asyncio
import os
import json
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form

from server.aggregator import (
    federated_average,
    weighted_federated_average
)

from server.crypto_utils import decrypt
from server.global_model import save_global_model
from server.blockchain import add_block
from server.metrics_store import save_hospital_metrics
from server.global_evaluator import evaluate_global_model


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

app = FastAPI()


# =====================================================
# CONFIG
# =====================================================

MAX_ROUNDS = int(
    os.getenv("ROUNDS", 20)
)

TOTAL_HOSPITALS = int(
    os.getenv("NUM_HOSPITALS", 3)
)

SERVER_HOST = os.getenv(
    "SERVER_HOST",
    "0.0.0.0"
)

SERVER_PORT = int(
    os.getenv("SERVER_PORT", 8000)
)

AGGREGATION_MODE = os.getenv(
    "AGGREGATION_MODE",
    "f1_weighted"
)

MIN_AGG_WEIGHT = float(
    os.getenv("MIN_AGG_WEIGHT", 0.05)
)


print("\n========== SERVER CONFIG ==========")
print(f"SERVER_HOST={SERVER_HOST}")
print(f"SERVER_PORT={SERVER_PORT}")
print(f"NUM_HOSPITALS={TOTAL_HOSPITALS}")
print(f"ROUNDS={MAX_ROUNDS}")
print(f"AGGREGATION_MODE={AGGREGATION_MODE}")
print(f"MIN_AGG_WEIGHT={MIN_AGG_WEIGHT}")
print("===================================\n")


# =====================================================
# STATE
# =====================================================

registered_hospitals = set()

received_weights = {}

received_metrics = {}

hospital_status = {}

latest_global_weights = None

latest_global_metrics = None

latest_aggregation_weights = {}

current_round = 1

lock = asyncio.Lock()


# =====================================================
# LOG DIRECTORIES
# =====================================================

os.makedirs("server/logs", exist_ok=True)
os.makedirs("hospital/logs", exist_ok=True)


# =====================================================
# REGISTER HOSPITAL
# =====================================================

@app.post("/register")
def register_hospital(hospital_id: int):

    registered_hospitals.add(hospital_id)

    hospital_status[hospital_id] = {
        "uploaded": False,
        "round": 0
    }

    print(f"[SERVER] Hospital {hospital_id} registered")

    return {
        "status": "registered",
        "hospital_id": hospital_id,
        "registered": len(registered_hospitals),
        "required": TOTAL_HOSPITALS,
        "current_round": current_round,
        "max_rounds": MAX_ROUNDS,
        "aggregation_mode": AGGREGATION_MODE
    }


# =====================================================
# BUILD F1-WEIGHTED AGGREGATION WEIGHTS
# =====================================================

def build_f1_aggregation_weights(hospital_ids):
    """
    For multiclass classification:
    local_f1 is expected to represent weighted F1.
    Kept as local_f1 for backward compatibility.
    """

    weights = []

    for hospital_id in hospital_ids:

        metrics = received_metrics.get(
            hospital_id,
            {}
        )

        local_f1 = float(
            metrics.get("local_f1", 0.0)
        )

        weight = max(
            local_f1,
            MIN_AGG_WEIGHT
        )

        weights.append(weight)

    total = sum(weights)

    if total == 0:

        weights = [
            1.0 / len(hospital_ids)
            for _ in hospital_ids
        ]

    else:

        weights = [
            w / total
            for w in weights
        ]

    return weights


# =====================================================
# SAVE FEDERATED ROUND METRICS
# =====================================================

def save_federated_metrics(
    round_num,
    global_metrics,
    aggregation_weights
):

    save_path = "server/logs/federated_metrics.json"

    if os.path.exists(save_path):

        with open(save_path, "r") as f:
            history = json.load(f)

    else:

        history = []

    round_data = {
        "round": round_num,
        "aggregation_mode": AGGREGATION_MODE,
        "aggregation_weights": aggregation_weights,
        "global_metrics": global_metrics,
        "hospitals": received_metrics
    }

    history.append(round_data)

    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)


# =====================================================
# SAFE DEFAULT GLOBAL METRICS
# =====================================================

def default_multiclass_metrics():
    return {
        "accuracy": 0.0,

        # Backward-compatible names
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,

        # Multiclass metric names
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "weighted_f1": 0.0,

        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,

        "critical_recall": 0.0,
        "critical_f1": 0.0,

        "high_recall": 0.0,
        "high_f1": 0.0,

        "roc_auc": 0.0,

        "confusion_matrix": [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
    }


# =====================================================
# AGGREGATION
# =====================================================

def run_aggregation():

    global latest_global_weights
    global latest_global_metrics
    global latest_aggregation_weights
    global received_weights
    global received_metrics
    global current_round
    global hospital_status

    print(f"\n[SERVER] 🚀 Aggregating Round {current_round}")

    if len(received_weights) == 0:

        print("[SERVER] No weights received")

        return

    hospital_ids = sorted(
        received_weights.keys()
    )

    all_weights = [
        received_weights[hospital_id]
        for hospital_id in hospital_ids
    ]

    # -------------------------------------------------
    # F1-WEIGHTED FEDAVG
    # -------------------------------------------------

    if AGGREGATION_MODE == "f1_weighted":

        aggregation_weights = build_f1_aggregation_weights(
            hospital_ids
        )

        print("[SERVER] Aggregation weights:")

        for hospital_id, weight in zip(
            hospital_ids,
            aggregation_weights
        ):

            local_f1 = received_metrics[hospital_id].get(
                "local_f1",
                0.0
            )

            print(
                f"  Hospital {hospital_id} | "
                f"local_weighted_f1={local_f1:.4f} | "
                f"weight={weight:.4f}"
            )

        global_weights = weighted_federated_average(
            all_weights,
            aggregation_weights
        )

    else:

        aggregation_weights = [
            1.0 / len(hospital_ids)
            for _ in hospital_ids
        ]

        global_weights = federated_average(
            all_weights
        )

    latest_global_weights = global_weights

    latest_aggregation_weights = {
        str(hospital_id): float(weight)
        for hospital_id, weight in zip(
            hospital_ids,
            aggregation_weights
        )
    }

    # -------------------------------------------------
    # SAFE GLOBAL EVALUATION
    # -------------------------------------------------

    try:

        metrics = evaluate_global_model(
            global_weights
        )

    except Exception as e:

        print("[GLOBAL EVALUATION ERROR]", e)

        metrics = default_multiclass_metrics()

    latest_global_metrics = metrics

    accuracy = metrics.get("accuracy", 0.0)

    weighted_f1 = metrics.get(
        "weighted_f1",
        metrics.get("f1", 0.0)
    )

    macro_f1 = metrics.get(
        "macro_f1",
        0.0
    )

    precision = metrics.get(
        "weighted_precision",
        metrics.get("precision", 0.0)
    )

    recall = metrics.get(
        "weighted_recall",
        metrics.get("recall", 0.0)
    )

    critical_recall = metrics.get(
        "critical_recall",
        0.0
    )

    roc_auc = metrics.get(
        "roc_auc",
        0.0
    )

    print(
        f"[SERVER] 📊 Round {current_round} | "
        f"Acc: {accuracy:.4f} | "
        f"Weighted F1: {weighted_f1:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Weighted Precision: {precision:.4f} | "
        f"Weighted Recall: {recall:.4f} | "
        f"Critical Recall: {critical_recall:.4f} | "
        f"ROC-AUC: {roc_auc:.4f}"
    )

    # -------------------------------------------------
    # SAVE GLOBAL MODEL
    # -------------------------------------------------

    save_global_model(
        global_weights,
        current_round,
        metrics
    )

    # -------------------------------------------------
    # SAVE FEDERATED METRICS
    # -------------------------------------------------

    save_federated_metrics(
        current_round,
        metrics,
        latest_aggregation_weights
    )

    # -------------------------------------------------
    # BLOCKCHAIN ENTRY
    # -------------------------------------------------

    add_block({
        "round": current_round,
        "event": "Federated Aggregation Complete",
        "aggregation_mode": AGGREGATION_MODE,
        "aggregation_weights": latest_aggregation_weights,
        "hospitals": hospital_ids,

        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "critical_recall": critical_recall,
        "roc_auc": roc_auc
    })

    print(
        f"[SERVER] ✅ Round {current_round} complete"
    )

    # -------------------------------------------------
    # NEXT ROUND
    # -------------------------------------------------

    current_round += 1

    received_weights = {}

    received_metrics = {}

    for h in hospital_status:

        hospital_status[h]["uploaded"] = False


# =====================================================
# UPLOAD LOCAL MODEL
# =====================================================

@app.post("/upload")
async def upload(
    file: UploadFile,
    hospital_id: int = Form(...),
    local_accuracy: float = Form(...),
    local_f1: float = Form(...),
    local_precision: float = Form(...),
    local_recall: float = Form(...),
    local_auc: float = Form(...)
):

    global received_weights
    global received_metrics
    global current_round
    global hospital_status

    try:

        print(
            f"\n[SERVER] Receiving from "
            f"Hospital {hospital_id}"
        )

        encrypted_data = await file.read()

        weights = decrypt(
            encrypted_data
        )

        received_weights[hospital_id] = weights

        hospital_status[hospital_id] = {
            "uploaded": True,
            "round": current_round
        }

        # For multiclass:
        # local_f1 is weighted_f1.
        # local_precision is weighted_precision.
        # local_recall is weighted_recall.
        # local_auc is multiclass weighted ROC-AUC if available,
        # otherwise 0.0 from client.
        received_metrics[hospital_id] = {
            "hospital_id": hospital_id,
            "round": current_round,

            "local_accuracy": float(local_accuracy),

            "local_f1": float(local_f1),
            "local_weighted_f1": float(local_f1),

            "local_precision": float(local_precision),
            "local_weighted_precision": float(local_precision),

            "local_recall": float(local_recall),
            "local_weighted_recall": float(local_recall),

            "local_auc": float(local_auc),
            "local_roc_auc": float(local_auc)
        }

        print(
            f"[SERVER] Local metrics | "
            f"Hospital {hospital_id} | "
            f"Acc: {local_accuracy:.4f} | "
            f"Weighted F1: {local_f1:.4f} | "
            f"Weighted Precision: {local_precision:.4f} | "
            f"Weighted Recall: {local_recall:.4f} | "
            f"ROC-AUC: {local_auc:.4f}"
        )

        print(
            f"[SERVER] Received "
            f"{len(received_weights)}/"
            f"{TOTAL_HOSPITALS}"
        )

        async with lock:

            if len(registered_hospitals) < TOTAL_HOSPITALS:

                return {
                    "status": "waiting_for_hospitals",
                    "registered": len(registered_hospitals),
                    "required": TOTAL_HOSPITALS,
                    "current_round": current_round
                }

            if current_round > MAX_ROUNDS:

                return {
                    "status": "finished"
                }

            if len(received_weights) == TOTAL_HOSPITALS:

                for h_id in received_metrics:

                    save_hospital_metrics(
                        received_metrics[h_id]
                    )

                run_aggregation()

                return {
                    "status": "aggregated",
                    "round": current_round - 1,
                    "aggregation_mode": AGGREGATION_MODE,
                    "aggregation_weights": latest_aggregation_weights
                }

        return {
            "status": "waiting",
            "received": len(received_weights),
            "required": TOTAL_HOSPITALS,
            "current_round": current_round
        }

    except Exception as e:

        print("[SERVER UPLOAD ERROR]", e)

        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# GET GLOBAL MODEL
# =====================================================

@app.get("/global_model")
def global_model():

    global latest_global_weights
    global current_round

    if latest_global_weights is None:

        return {
            "status": "not_ready"
        }

    if current_round > MAX_ROUNDS:

        return {
            "status": "finished"
        }

    serializable_weights = []

    for layer in latest_global_weights:

        serializable_weights.append(
            np.array(layer).tolist()
        )

    return {
        "status": "ready",
        "round": current_round - 1,
        "weights": serializable_weights,
        "global_metrics": latest_global_metrics,
        "aggregation_mode": AGGREGATION_MODE,
        "aggregation_weights": latest_aggregation_weights
    }


# =====================================================
# STATUS
# =====================================================

@app.get("/status")
def status():

    return {
        "current_round": current_round,
        "max_rounds": MAX_ROUNDS,
        "total_hospitals": TOTAL_HOSPITALS,
        "registered_hospitals": len(registered_hospitals),
        "uploaded_weights": len(received_weights),
        "hospital_status": hospital_status,
        "aggregation_mode": AGGREGATION_MODE,
        "aggregation_weights": latest_aggregation_weights,
        "latest_global_metrics": latest_global_metrics
    }


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "current_round": current_round,
        "registered_hospitals": len(registered_hospitals),
        "total_hospitals": TOTAL_HOSPITALS,
        "max_rounds": MAX_ROUNDS,
        "aggregation_mode": AGGREGATION_MODE
    }


# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT
    )