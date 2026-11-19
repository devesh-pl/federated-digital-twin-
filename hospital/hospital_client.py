import requests
import torch
import argparse
import time
import os
import json
import copy

from dotenv import load_dotenv
from datetime import datetime

from hospital.data_loader import load_data
from hospital.local_model import get_model
from hospital.masking import add_mask
from hospital.crypto_utils import encrypt

from hospital.trainer import (
    train_model,
    evaluate_model,
    clone_model_parameters
)


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()


# =====================================================
# ENV CONFIG
# =====================================================

EPOCHS = int(os.getenv("EPOCHS", 5))
BASELINE_EPOCHS = int(os.getenv("BASELINE_EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

LR = float(os.getenv("LEARNING_RATE", 0.001))
TOTAL_ROUNDS = int(os.getenv("ROUNDS", 10))

FEDPROX_MU = float(os.getenv("FEDPROX_MU", 0.005))
PERSONALIZED_ALPHA = float(os.getenv("PERSONALIZED_ALPHA", 0.4))

LOCAL_BASELINE_FRACTION = float(
    os.getenv("LOCAL_BASELINE_FRACTION", 0.25)
)

BEST_MODEL_METRIC = os.getenv(
    "BEST_MODEL_METRIC",
    "medical_safety_score"
)

MIN_ACC_GAIN = float(os.getenv("MIN_ACCEPT_ACC_GAIN", 0.0))
MIN_F1_GAIN = float(os.getenv("MIN_ACCEPT_F1_GAIN", 0.0))

CRITICAL_RECALL_TOLERANCE = float(
    os.getenv("CRITICAL_RECALL_TOLERANCE", 0.03)
)


# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--id",
    type=int,
    required=True
)

parser.add_argument(
    "--server_ip",
    type=str,
    default="127.0.0.1"
)

args = parser.parse_args()


# =====================================================
# CONFIG
# =====================================================

HOSPITAL_ID = args.id
SERVER_URL = f"http://{args.server_ip}:8000"
DATA_PATH = f"datasets/hospital_{HOSPITAL_ID}.csv"

os.makedirs("hospital/logs", exist_ok=True)


# =====================================================
# VERIFY CONFIG
# =====================================================

print("\n========== CONFIG ==========")
print(f"EPOCHS={EPOCHS}")
print(f"BASELINE_EPOCHS={BASELINE_EPOCHS}")
print(f"BATCH_SIZE={BATCH_SIZE}")
print(f"LEARNING_RATE={LR}")
print(f"ROUNDS={TOTAL_ROUNDS}")
print(f"FEDPROX_MU={FEDPROX_MU}")
print(f"PERSONALIZED_ALPHA={PERSONALIZED_ALPHA}")
print(f"LOCAL_BASELINE_FRACTION={LOCAL_BASELINE_FRACTION}")
print(f"MIN_ACCEPT_ACC_GAIN={MIN_ACC_GAIN}")
print(f"MIN_ACCEPT_F1_GAIN={MIN_F1_GAIN}")
print(f"BEST_MODEL_METRIC={BEST_MODEL_METRIC}")
print(f"CRITICAL_RECALL_TOLERANCE={CRITICAL_RECALL_TOLERANCE}")
print("============================\n")


# =====================================================
# LOAD DATA
# =====================================================

print(f"[Hospital {HOSPITAL_ID}] Loading data...")

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
) = load_data(DATA_PATH)


# =====================================================
# MODEL
# =====================================================

model = get_model()

global_reference_params = None


# =====================================================
# REGISTER
# =====================================================

def register():

    try:

        r = requests.post(
            f"{SERVER_URL}/register",
            params={
                "hospital_id": HOSPITAL_ID
            },
            timeout=10
        )

        print("[REGISTER]", r.json())

    except Exception as e:

        print("[REGISTER ERROR]", e)


# =====================================================
# GET GLOBAL MODEL
# =====================================================

def get_global_model():

    try:

        r = requests.get(
            f"{SERVER_URL}/global_model",
            timeout=5
        )

        if r.status_code != 200:

            return {
                "status": "error",
                "message": f"HTTP {r.status_code}"
            }

        return r.json()

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# SET MODEL WEIGHTS
# =====================================================

def set_weights(target_model, weights):

    with torch.no_grad():

        for p, w in zip(
            target_model.parameters(),
            weights
        ):

            p.data = torch.tensor(
                w,
                dtype=torch.float32,
                device=p.data.device
            )


# =====================================================
# GET MODEL WEIGHTS
# =====================================================

def get_weights(source_model):

    return [
        p.detach().cpu().numpy()
        for p in source_model.parameters()
    ]


# =====================================================
# PERSONALIZED MODEL
# =====================================================

def build_personalized_model(
    local_state_dict,
    global_weights
):
    """
    personalized = alpha * global + (1 - alpha) * local
    """

    personalized_model = get_model()

    personalized_model.load_state_dict(
        local_state_dict
    )

    with torch.no_grad():

        for p, global_w in zip(
            personalized_model.parameters(),
            global_weights
        ):

            local_w = p.data.clone()

            global_tensor = torch.tensor(
                global_w,
                dtype=torch.float32,
                device=p.data.device
            )

            mixed = (
                PERSONALIZED_ALPHA * global_tensor
                + (1.0 - PERSONALIZED_ALPHA) * local_w
            )

            p.data = mixed

    return personalized_model


# =====================================================
# SAVE HELPERS
# =====================================================

def save_json_append(file_path, payload):

    if os.path.exists(file_path):

        with open(file_path, "r") as f:
            data = json.load(f)

    else:

        data = []

    payload["timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    data.append(payload)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_round_metrics(metrics):

    file_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_metrics.json"
    )

    save_json_append(file_path, metrics)


# =====================================================
# TRAIN LOCAL BASELINE
# =====================================================

def train_local_baseline():
    """
    Train local-only baseline model before federation starts.

    This baseline intentionally uses only a fraction of local data.
    This supports the comparison:
    Limited Local Baseline vs Federated Learning.
    """

    print(
        f"\n[Hospital {HOSPITAL_ID}] "
        f"🏠 Training Local Baseline"
    )

    baseline_fraction = LOCAL_BASELINE_FRACTION

    baseline_size = int(
        len(X_train) * baseline_fraction
    )

    if baseline_size < BATCH_SIZE:
        baseline_size = min(
            len(X_train),
            BATCH_SIZE
        )

    X_baseline = X_train[:baseline_size]
    y_baseline = y_train[:baseline_size]

    print(
        f"[LOCAL BASELINE] Using only "
        f"{baseline_fraction * 100:.0f}% local data "
        f"({baseline_size} samples)"
    )

    baseline_model = get_model()

    baseline_model, baseline_history = train_model(
        baseline_model,
        X_baseline,
        y_baseline,
        epochs=BASELINE_EPOCHS,
        lr=LR,
        fedprox_mu=0.0,
        global_params=None,
        batch_size=BATCH_SIZE
    )

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test
    )

    baseline_metrics["medical_safety_score"] = (
        baseline_metrics["macro_f1"]
        + baseline_metrics["weighted_f1"]
        + baseline_metrics["critical_recall"]
    )

    print(
        f"[LOCAL BASELINE] "
        f"Acc: {baseline_metrics['accuracy']:.4f} | "
        f"Weighted F1: {baseline_metrics['weighted_f1']:.4f} | "
        f"Macro F1: {baseline_metrics['macro_f1']:.4f} | "
        f"Critical Recall: {baseline_metrics['critical_recall']:.4f} | "
        f"Medical Safety Score: "
        f"{baseline_metrics['medical_safety_score']:.4f}"
    )

    baseline_model_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_"
        f"local_baseline.pth"
    )

    torch.save(
        baseline_model.state_dict(),
        baseline_model_path
    )

    baseline_payload = {
        "hospital_id": HOSPITAL_ID,
        "model_type": "limited_local_baseline",

        "local_baseline_fraction": baseline_fraction,
        "local_baseline_samples": baseline_size,

        "accuracy": baseline_metrics["accuracy"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "f1": baseline_metrics["f1"],

        "weighted_precision": baseline_metrics["weighted_precision"],
        "weighted_recall": baseline_metrics["weighted_recall"],
        "weighted_f1": baseline_metrics["weighted_f1"],

        "macro_precision": baseline_metrics["macro_precision"],
        "macro_recall": baseline_metrics["macro_recall"],
        "macro_f1": baseline_metrics["macro_f1"],

        "critical_recall": baseline_metrics["critical_recall"],
        "critical_f1": baseline_metrics["critical_f1"],
        "high_recall": baseline_metrics["high_recall"],
        "high_f1": baseline_metrics["high_f1"],

        "medical_safety_score":
            baseline_metrics["medical_safety_score"],

        "roc_auc": baseline_metrics["roc_auc"],
        "confusion_matrix": baseline_metrics["confusion_matrix"],

        "epochs": BASELINE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_path": baseline_model_path,
        "loss_history": baseline_history["loss"],
        "train_accuracy_history": baseline_history["accuracy"],

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    baseline_metrics_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_"
        f"local_baseline.json"
    )

    with open(baseline_metrics_path, "w") as f:
        json.dump(
            baseline_payload,
            f,
            indent=4
        )

    return (
        baseline_metrics,
        baseline_model_path
    )


# =====================================================
# BEST SELECTED MODEL TRACKER
# =====================================================

def is_better_model(
    current_metrics,
    best_metrics
):

    if best_metrics is None:
        return True

    current_score = current_metrics.get(
        BEST_MODEL_METRIC,
        0.0
    )

    best_score = best_metrics.get(
        BEST_MODEL_METRIC,
        0.0
    )

    return current_score > best_score


def save_best_selected_model(
    selected_model,
    selected_metrics,
    round_num,
    selected_model_type
):

    best_model_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_"
        f"best_selected_fl.pth"
    )

    torch.save(
        selected_model.state_dict(),
        best_model_path
    )

    payload = {
        "hospital_id": HOSPITAL_ID,
        "round": round_num,
        "model_type": "best_selected_federated_model",
        "selected_model_type": selected_model_type,
        "selection_metric": BEST_MODEL_METRIC,

        "accuracy": selected_metrics["accuracy"],
        "precision": selected_metrics["precision"],
        "recall": selected_metrics["recall"],
        "f1": selected_metrics["f1"],

        "weighted_precision": selected_metrics["weighted_precision"],
        "weighted_recall": selected_metrics["weighted_recall"],
        "weighted_f1": selected_metrics["weighted_f1"],

        "macro_precision": selected_metrics["macro_precision"],
        "macro_recall": selected_metrics["macro_recall"],
        "macro_f1": selected_metrics["macro_f1"],

        "stable_precision": selected_metrics["stable_precision"],
        "stable_recall": selected_metrics["stable_recall"],
        "stable_f1": selected_metrics["stable_f1"],

        "medium_precision": selected_metrics["medium_precision"],
        "medium_recall": selected_metrics["medium_recall"],
        "medium_f1": selected_metrics["medium_f1"],

        "high_precision": selected_metrics["high_precision"],
        "high_recall": selected_metrics["high_recall"],
        "high_f1": selected_metrics["high_f1"],

        "critical_precision": selected_metrics["critical_precision"],
        "critical_recall": selected_metrics["critical_recall"],
        "critical_f1": selected_metrics["critical_f1"],

        "medical_safety_score": selected_metrics.get(
            "medical_safety_score",
            (
                selected_metrics["macro_f1"]
                + selected_metrics["weighted_f1"]
                + selected_metrics["critical_recall"]
            )
        ),

        "roc_auc": selected_metrics["roc_auc"],
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "model_path": best_model_path,

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    best_metrics_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_"
        f"best_selected_fl.json"
    )

    with open(best_metrics_path, "w") as f:
        json.dump(
            payload,
            f,
            indent=4
        )

    print(
        f"[BEST MODEL UPDATED] "
        f"Round {round_num} | "
        f"Acc: {selected_metrics['accuracy']:.4f} | "
        f"Weighted F1: {selected_metrics['weighted_f1']:.4f} | "
        f"Macro F1: {selected_metrics['macro_f1']:.4f} | "
        f"Critical Recall: {selected_metrics['critical_recall']:.4f} | "
        f"Safety Score: "
        f"{payload['medical_safety_score']:.4f} | "
        f"Saved: {best_model_path}"
    )


# =====================================================
# FINAL BASELINE VS BEST FL COMPARISON
# =====================================================

def save_final_baseline_vs_best_comparison(
    baseline_metrics,
    best_metrics
):

    if best_metrics is None:

        print(
            "[FINAL COMPARISON] "
            "No best selected FL model found."
        )

        return

    accuracy_improvement = round(
        (
            best_metrics["accuracy"]
            - baseline_metrics["accuracy"]
        ) * 100,
        2
    )

    weighted_f1_improvement = round(
        (
            best_metrics["weighted_f1"]
            - baseline_metrics["weighted_f1"]
        ) * 100,
        2
    )

    macro_f1_improvement = round(
        (
            best_metrics["macro_f1"]
            - baseline_metrics["macro_f1"]
        ) * 100,
        2
    )

    critical_recall_improvement = round(
        (
            best_metrics["critical_recall"]
            - baseline_metrics["critical_recall"]
        ) * 100,
        2
    )

    medical_safety_score_improvement = round(
        (
            best_metrics.get(
                "medical_safety_score",
                best_metrics["macro_f1"]
                + best_metrics["weighted_f1"]
                + best_metrics["critical_recall"]
            )
            - baseline_metrics.get(
                "medical_safety_score",
                baseline_metrics["macro_f1"]
                + baseline_metrics["weighted_f1"]
                + baseline_metrics["critical_recall"]
            )
        ) * 100,
        2
    )

    comparison = {
        "hospital_id": HOSPITAL_ID,

        "baseline_type": "limited_local_baseline",
        "local_baseline_fraction": LOCAL_BASELINE_FRACTION,

        "local_baseline_accuracy": baseline_metrics["accuracy"],
        "local_baseline_weighted_f1": baseline_metrics["weighted_f1"],
        "local_baseline_macro_f1": baseline_metrics["macro_f1"],
        "local_baseline_critical_recall":
            baseline_metrics["critical_recall"],
        "local_baseline_medical_safety_score":
            baseline_metrics.get(
                "medical_safety_score",
                baseline_metrics["macro_f1"]
                + baseline_metrics["weighted_f1"]
                + baseline_metrics["critical_recall"]
            ),

        "best_selected_accuracy": best_metrics["accuracy"],
        "best_selected_weighted_f1": best_metrics["weighted_f1"],
        "best_selected_macro_f1": best_metrics["macro_f1"],
        "best_selected_critical_recall": best_metrics["critical_recall"],
        "best_selected_medical_safety_score":
            best_metrics.get(
                "medical_safety_score",
                best_metrics["macro_f1"]
                + best_metrics["weighted_f1"]
                + best_metrics["critical_recall"]
            ),

        "best_selected_round": best_metrics["round"],
        "best_selected_model_type": best_metrics["selected_model_type"],

        "accuracy_improvement_percent": accuracy_improvement,
        "weighted_f1_improvement_percent": weighted_f1_improvement,
        "macro_f1_improvement_percent": macro_f1_improvement,
        "critical_recall_improvement_percent":
            critical_recall_improvement,
        "medical_safety_score_improvement_percent":
            medical_safety_score_improvement,

        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    comparison_path = (
        f"hospital/logs/"
        f"hospital_{HOSPITAL_ID}_"
        f"baseline_vs_best_fl.json"
    )

    with open(comparison_path, "w") as f:
        json.dump(
            comparison,
            f,
            indent=4
        )

    print("\n========== FINAL COMPARISON ==========")
    print(
        f"Local Baseline Acc: "
        f"{baseline_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"Best Selected FL Acc: "
        f"{best_metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"Accuracy Improvement: "
        f"{accuracy_improvement:.2f}%"
    )
    print(
        f"Local Baseline Weighted F1: "
        f"{baseline_metrics['weighted_f1']:.4f}"
    )
    print(
        f"Best Selected FL Weighted F1: "
        f"{best_metrics['weighted_f1']:.4f}"
    )
    print(
        f"Weighted F1 Improvement: "
        f"{weighted_f1_improvement:.2f}%"
    )
    print(
        f"Local Baseline Macro F1: "
        f"{baseline_metrics['macro_f1']:.4f}"
    )
    print(
        f"Best Selected FL Macro F1: "
        f"{best_metrics['macro_f1']:.4f}"
    )
    print(
        f"Macro F1 Improvement: "
        f"{macro_f1_improvement:.2f}%"
    )
    print(
        f"Critical Recall Improvement: "
        f"{critical_recall_improvement:.2f}%"
    )
    print(
        f"Medical Safety Score Improvement: "
        f"{medical_safety_score_improvement:.2f}%"
    )
    print("======================================\n")


# =====================================================
# MODEL SELECTION
# =====================================================

def select_best_model(
    local_state_dict,
    local_metrics,
    personalized_model,
    personalized_metrics,
    baseline_metrics
):
    """
    Medical-safety-aware model selection.

    Select personalized FL only if:
    1. it improves over the limited local baseline
    2. it preserves critical recall within tolerance
    """

    # -------------------------------------------------
    # GAINS VS LIMITED LOCAL BASELINE
    # -------------------------------------------------

    baseline_acc_gain = (
        personalized_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    baseline_weighted_f1_gain = (
        personalized_metrics["weighted_f1"]
        - baseline_metrics["weighted_f1"]
    )

    baseline_macro_f1_gain = (
        personalized_metrics["macro_f1"]
        - baseline_metrics["macro_f1"]
    )

    baseline_critical_recall_gain = (
        personalized_metrics["critical_recall"]
        - baseline_metrics["critical_recall"]
    )

    # -------------------------------------------------
    # GAINS VS UPDATED LOCAL MODEL
    # -------------------------------------------------

    updated_local_acc_gain = (
        personalized_metrics["accuracy"]
        - local_metrics["accuracy"]
    )

    updated_local_weighted_f1_gain = (
        personalized_metrics["weighted_f1"]
        - local_metrics["weighted_f1"]
    )

    updated_local_macro_f1_gain = (
        personalized_metrics["macro_f1"]
        - local_metrics["macro_f1"]
    )

    # -------------------------------------------------
    # CRITICAL RECALL SAFETY CHECK
    # -------------------------------------------------

    baseline_critical_recall = baseline_metrics[
        "critical_recall"
    ]

    personalized_critical_recall = personalized_metrics[
        "critical_recall"
    ]

    min_allowed_critical_recall = (
        baseline_critical_recall
        - CRITICAL_RECALL_TOLERANCE
    )

    personalized_critical_safe = (
        personalized_critical_recall
        >= min_allowed_critical_recall
    )

    # -------------------------------------------------
    # MEDICAL SAFETY SCORE
    # -------------------------------------------------

    personalized_medical_safety_score = (
        personalized_metrics["macro_f1"]
        + personalized_metrics["weighted_f1"]
        + personalized_metrics["critical_recall"]
    )

    local_medical_safety_score = (
        local_metrics["macro_f1"]
        + local_metrics["weighted_f1"]
        + local_metrics["critical_recall"]
    )

    baseline_medical_safety_score = (
        baseline_metrics["macro_f1"]
        + baseline_metrics["weighted_f1"]
        + baseline_metrics["critical_recall"]
    )

    personalized_metrics["medical_safety_score"] = (
        personalized_medical_safety_score
    )

    local_metrics["medical_safety_score"] = (
        local_medical_safety_score
    )

    baseline_metrics["medical_safety_score"] = (
        baseline_medical_safety_score
    )

    # -------------------------------------------------
    # PRINT CHECKS
    # -------------------------------------------------

    print(
        f"[CRITICAL SAFETY] "
        f"Baseline Critical Recall: "
        f"{baseline_critical_recall:.4f} | "
        f"Personalized Critical Recall: "
        f"{personalized_critical_recall:.4f} | "
        f"Minimum Allowed: "
        f"{min_allowed_critical_recall:.4f} | "
        f"Safe: {personalized_critical_safe}"
    )

    print(
        f"[MEDICAL SAFETY SCORE] "
        f"Baseline: {baseline_medical_safety_score:.4f} | "
        f"Updated Local: {local_medical_safety_score:.4f} | "
        f"Personalized FL: "
        f"{personalized_medical_safety_score:.4f}"
    )

    # -------------------------------------------------
    # ACCEPTANCE LOGIC
    # -------------------------------------------------

    improves_over_baseline = (
        baseline_acc_gain >= MIN_ACC_GAIN
        or baseline_weighted_f1_gain >= MIN_F1_GAIN
        or baseline_macro_f1_gain >= MIN_F1_GAIN
    )

    accept_personalized = (
        improves_over_baseline
        and personalized_critical_safe
    )

    if accept_personalized:

        selected_model = personalized_model
        selected_model_type = "personalized_fl"
        selected_metrics = personalized_metrics

        print(
            f"[MODEL SELECTED] Personalized FL accepted vs local baseline | "
            f"Baseline Acc gain: "
            f"{baseline_acc_gain * 100:.2f}% | "
            f"Baseline Weighted F1 gain: "
            f"{baseline_weighted_f1_gain * 100:.2f}% | "
            f"Baseline Macro F1 gain: "
            f"{baseline_macro_f1_gain * 100:.2f}% | "
            f"Critical Recall gain: "
            f"{baseline_critical_recall_gain * 100:.2f}%"
        )

        if updated_local_acc_gain < 0:

            print(
                f"[NOTE] Personalized FL is below updated local model by "
                f"{abs(updated_local_acc_gain) * 100:.2f}% accuracy, "
                f"but it improves over the limited local baseline "
                f"and passes critical safety."
            )

    else:

        selected_model = get_model()
        selected_model.load_state_dict(
            local_state_dict
        )

        selected_model_type = "updated_local_model"
        selected_metrics = local_metrics

        print(
            f"[MODEL SELECTED] Updated local model kept | "
            f"Personalized FL rejected."
        )

        if not improves_over_baseline:

            print(
                "[REJECT REASON] Personalized FL did not improve "
                "over local baseline."
            )

        if not personalized_critical_safe:

            print(
                "[REJECT REASON] Personalized FL reduced critical recall "
                "beyond allowed tolerance."
            )

    # -------------------------------------------------
    # SELECTED MODEL IMPROVEMENT VS BASELINE
    # -------------------------------------------------

    selected_accuracy_improvement = round(
        (
            selected_metrics["accuracy"]
            - baseline_metrics["accuracy"]
        ) * 100,
        2
    )

    selected_f1_improvement = round(
        (
            selected_metrics["weighted_f1"]
            - baseline_metrics["weighted_f1"]
        ) * 100,
        2
    )

    return (
        selected_model,
        selected_model_type,
        selected_metrics,
        selected_accuracy_improvement,
        selected_f1_improvement,
        baseline_acc_gain,
        baseline_weighted_f1_gain,
        baseline_critical_recall_gain,
        updated_local_acc_gain,
        updated_local_weighted_f1_gain,
        updated_local_macro_f1_gain
    )


# =====================================================
# TRAIN + UPLOAD
# =====================================================

def train_and_upload(round_num):

    global model
    global global_reference_params

    print(f"\n[Hospital {HOSPITAL_ID}] 🔥 Training")

    print(
        f"[CONFIG] Epochs: {EPOCHS} | "
        f"Batch Size: {BATCH_SIZE} | "
        f"Learning Rate: {LR} | "
        f"FedProx Mu: {FEDPROX_MU}"
    )

    train_start = time.time()

    model, history = train_model(
        model,
        X_train,
        y_train,
        epochs=EPOCHS,
        lr=LR,
        fedprox_mu=FEDPROX_MU,
        global_params=global_reference_params,
        batch_size=BATCH_SIZE
    )

    train_time = time.time() - train_start

    local_state_dict = copy.deepcopy(
        model.state_dict()
    )

    local_metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    local_metrics["medical_safety_score"] = (
        local_metrics["macro_f1"]
        + local_metrics["weighted_f1"]
        + local_metrics["critical_recall"]
    )

    print(
        f"[UPDATED LOCAL] "
        f"Acc: {local_metrics['accuracy']:.4f} | "
        f"Weighted F1: {local_metrics['weighted_f1']:.4f} | "
        f"Macro F1: {local_metrics['macro_f1']:.4f} | "
        f"Critical Recall: {local_metrics['critical_recall']:.4f} | "
        f"Safety Score: "
        f"{local_metrics['medical_safety_score']:.4f}"
    )

    weights = get_weights(model)

    masked, _ = add_mask(weights)

    encrypted = encrypt(masked)

    file_path = f"hospital_{HOSPITAL_ID}.bin"

    with open(file_path, "wb") as f:
        f.write(encrypted)

    success = False

    while not success:

        try:

            with open(file_path, "rb") as f:

                files = {
                    "file": f
                }

                data = {
                    "hospital_id": HOSPITAL_ID,
                    "local_accuracy": local_metrics["accuracy"],
                    "local_f1": local_metrics["weighted_f1"],
                    "local_precision":
                        local_metrics["weighted_precision"],
                    "local_recall":
                        local_metrics["weighted_recall"],
                    "local_auc":
                        local_metrics.get("roc_auc", 0.0)
                }

                r = requests.post(
                    f"{SERVER_URL}/upload",
                    files=files,
                    data=data,
                    timeout=30
                )

                try:

                    response = r.json()
                    print("[UPLOAD]", response)

                    if (
                        response.get("status") == "aggregated"
                        and response.get("round") != round_num
                    ):

                        print(
                            f"[UPLOAD ROUND MISMATCH] "
                            f"Expected Round {round_num}, "
                            f"but server aggregated Round "
                            f"{response.get('round')}."
                        )

                        print(
                            "Stop all clients, clear logs, "
                            "restart server."
                        )

                        return (
                            local_metrics,
                            local_state_dict,
                            history,
                            round(train_time, 2)
                        )

                    if response.get("status") != "error":

                        success = True

                except Exception as e:

                    print("[UPLOAD RESPONSE ERROR]", e)
                    print("[UPLOAD STATUS]", r.status_code)
                    print("[UPLOAD TEXT]", r.text[:500])
                    time.sleep(3)

        except Exception as e:

            print("[UPLOAD ERROR]", e)
            time.sleep(3)

    if os.path.exists(file_path):

        os.remove(file_path)

    return (
        local_metrics,
        local_state_dict,
        history,
        round(train_time, 2)
    )


# =====================================================
# MAIN LOOP
# =====================================================

def run():

    global model
    global global_reference_params

    (
        local_baseline_metrics,
        local_baseline_model_path
    ) = train_local_baseline()

    model = get_model()

    best_selected_metrics = None

    register()

    for round_num in range(TOTAL_ROUNDS):

        print(
            f"\n========== "
            f"FL ROUND {round_num + 1}/"
            f"{TOTAL_ROUNDS} "
            f"=========="
        )

        (
            local_metrics,
            local_state_dict,
            history,
            train_time
        ) = train_and_upload(round_num + 1)

        print(
            f"[Hospital {HOSPITAL_ID}] "
            f"⏳ Waiting global model..."
        )

        start_time = time.time()

        while True:

            if time.time() - start_time > 300:

                print(
                    f"[Hospital {HOSPITAL_ID}] "
                    f"❌ Timeout waiting for global model"
                )

                save_final_baseline_vs_best_comparison(
                    local_baseline_metrics,
                    best_selected_metrics
                )

                return

            data = get_global_model()

            if data.get("status") == "finished":

                print(
                    f"[Hospital {HOSPITAL_ID}] "
                    f"🎉 Federated Learning Finished"
                )

                save_final_baseline_vs_best_comparison(
                    local_baseline_metrics,
                    best_selected_metrics
                )

                return

            if data.get("status") == "ready":

                server_round = data.get("round")
                expected_round = round_num + 1

                if server_round > expected_round:

                    print(
                        f"[ROUND MISMATCH] "
                        f"Expected Round {expected_round}, "
                        f"but server is already at "
                        f"Round {server_round}."
                    )

                    print(
                        "Stop all clients, clear logs, "
                        "restart server."
                    )

                    save_final_baseline_vs_best_comparison(
                        local_baseline_metrics,
                        best_selected_metrics
                    )

                    return

                if server_round == expected_round:

                    print(
                        f"[Hospital {HOSPITAL_ID}] "
                        f"✅ Received Global Model "
                        f"Round {server_round}"
                    )

                    global_weights = data["weights"]

                    global_model = get_model()

                    set_weights(
                        global_model,
                        global_weights
                    )

                    global_metrics = evaluate_model(
                        global_model,
                        X_test,
                        y_test
                    )

                    global_metrics["medical_safety_score"] = (
                        global_metrics["macro_f1"]
                        + global_metrics["weighted_f1"]
                        + global_metrics["critical_recall"]
                    )

                    personalized_model = build_personalized_model(
                        local_state_dict,
                        global_weights
                    )

                    personalized_metrics = evaluate_model(
                        personalized_model,
                        X_test,
                        y_test
                    )

                    personalized_metrics["medical_safety_score"] = (
                        personalized_metrics["macro_f1"]
                        + personalized_metrics["weighted_f1"]
                        + personalized_metrics["critical_recall"]
                    )

                    (
                        model,
                        selected_model_type,
                        selected_metrics,
                        selected_accuracy_improvement,
                        selected_f1_improvement,
                        baseline_acc_gain,
                        baseline_weighted_f1_gain,
                        baseline_critical_recall_gain,
                        updated_local_acc_gain,
                        updated_local_weighted_f1_gain,
                        updated_local_macro_f1_gain
                    ) = select_best_model(
                        local_state_dict,
                        local_metrics,
                        personalized_model,
                        personalized_metrics,
                        local_baseline_metrics
                    )

                    global_reference_params = clone_model_parameters(
                        global_model
                    )

                    torch.save(
                        global_model.state_dict(),
                        (
                            f"hospital/logs/"
                            f"hospital_{HOSPITAL_ID}_"
                            f"global_round_"
                            f"{server_round}.pth"
                        )
                    )

                    torch.save(
                        personalized_model.state_dict(),
                        (
                            f"hospital/logs/"
                            f"hospital_{HOSPITAL_ID}_"
                            f"personalized_round_"
                            f"{server_round}.pth"
                        )
                    )

                    torch.save(
                        model.state_dict(),
                        (
                            f"hospital/logs/"
                            f"hospital_{HOSPITAL_ID}_"
                            f"selected_round_"
                            f"{server_round}.pth"
                        )
                    )

                    print(
                        f"[GLOBAL FL] "
                        f"Acc: {global_metrics['accuracy']:.4f} | "
                        f"Weighted F1: "
                        f"{global_metrics['weighted_f1']:.4f} | "
                        f"Macro F1: {global_metrics['macro_f1']:.4f} | "
                        f"Critical Recall: "
                        f"{global_metrics['critical_recall']:.4f} | "
                        f"Safety Score: "
                        f"{global_metrics['medical_safety_score']:.4f}"
                    )

                    print(
                        f"[PERSONALIZED FL] "
                        f"Acc: "
                        f"{personalized_metrics['accuracy']:.4f} | "
                        f"Weighted F1: "
                        f"{personalized_metrics['weighted_f1']:.4f} | "
                        f"Macro F1: "
                        f"{personalized_metrics['macro_f1']:.4f} | "
                        f"Critical Recall: "
                        f"{personalized_metrics['critical_recall']:.4f} | "
                        f"Safety Score: "
                        f"{personalized_metrics['medical_safety_score']:.4f}"
                    )

                    print(
                        f"[SELECTED MODEL IMPROVEMENT VS BASELINE] "
                        f"Acc: {selected_accuracy_improvement:.2f}% | "
                        f"Weighted F1: {selected_f1_improvement:.2f}% | "
                        f"Critical Recall Gain: "
                        f"{baseline_critical_recall_gain * 100:.2f}% | "
                        f"Selected: {selected_model_type}"
                    )

                    candidate_best_metrics = {
                        "round": round_num + 1,

                        "accuracy": selected_metrics["accuracy"],

                        "precision": selected_metrics["precision"],
                        "recall": selected_metrics["recall"],
                        "f1": selected_metrics["f1"],

                        "weighted_precision":
                            selected_metrics["weighted_precision"],
                        "weighted_recall":
                            selected_metrics["weighted_recall"],
                        "weighted_f1":
                            selected_metrics["weighted_f1"],

                        "macro_precision":
                            selected_metrics["macro_precision"],
                        "macro_recall":
                            selected_metrics["macro_recall"],
                        "macro_f1":
                            selected_metrics["macro_f1"],

                        "stable_precision":
                            selected_metrics["stable_precision"],
                        "stable_recall":
                            selected_metrics["stable_recall"],
                        "stable_f1":
                            selected_metrics["stable_f1"],

                        "medium_precision":
                            selected_metrics["medium_precision"],
                        "medium_recall":
                            selected_metrics["medium_recall"],
                        "medium_f1":
                            selected_metrics["medium_f1"],

                        "high_precision":
                            selected_metrics["high_precision"],
                        "high_recall":
                            selected_metrics["high_recall"],
                        "high_f1":
                            selected_metrics["high_f1"],

                        "critical_precision":
                            selected_metrics["critical_precision"],
                        "critical_recall":
                            selected_metrics["critical_recall"],
                        "critical_f1":
                            selected_metrics["critical_f1"],

                        "medical_safety_score":
                            selected_metrics.get(
                                "medical_safety_score",
                                (
                                    selected_metrics["macro_f1"]
                                    + selected_metrics["weighted_f1"]
                                    + selected_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "roc_auc": selected_metrics["roc_auc"],
                        "confusion_matrix":
                            selected_metrics["confusion_matrix"],

                        "selected_model_type": selected_model_type
                    }

                    if is_better_model(
                        candidate_best_metrics,
                        best_selected_metrics
                    ):

                        best_selected_metrics = candidate_best_metrics

                        save_best_selected_model(
                            model,
                            best_selected_metrics,
                            round_num + 1,
                            selected_model_type
                        )

                    save_round_metrics({

                        "round": round_num + 1,
                        "hospital_id": HOSPITAL_ID,

                        "local_baseline_accuracy":
                            local_baseline_metrics["accuracy"],

                        "local_baseline_weighted_f1":
                            local_baseline_metrics["weighted_f1"],

                        "local_baseline_macro_f1":
                            local_baseline_metrics["macro_f1"],

                        "local_baseline_critical_recall":
                            local_baseline_metrics["critical_recall"],

                        "local_baseline_medical_safety_score":
                            local_baseline_metrics.get(
                                "medical_safety_score",
                                (
                                    local_baseline_metrics["macro_f1"]
                                    + local_baseline_metrics[
                                        "weighted_f1"
                                    ]
                                    + local_baseline_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "local_accuracy":
                            local_metrics["accuracy"],

                        "updated_local_accuracy":
                            local_metrics["accuracy"],

                        "local_precision":
                            local_metrics["precision"],

                        "local_recall":
                            local_metrics["recall"],

                        "local_f1":
                            local_metrics["weighted_f1"],

                        "local_weighted_f1":
                            local_metrics["weighted_f1"],

                        "local_macro_f1":
                            local_metrics["macro_f1"],

                        "local_critical_recall":
                            local_metrics["critical_recall"],

                        "local_medical_safety_score":
                            local_metrics.get(
                                "medical_safety_score",
                                (
                                    local_metrics["macro_f1"]
                                    + local_metrics["weighted_f1"]
                                    + local_metrics["critical_recall"]
                                )
                            ),

                        "local_roc_auc":
                            local_metrics["roc_auc"],

                        "global_accuracy":
                            global_metrics["accuracy"],

                        "global_precision":
                            global_metrics["precision"],

                        "global_recall":
                            global_metrics["recall"],

                        "global_f1":
                            global_metrics["weighted_f1"],

                        "global_weighted_f1":
                            global_metrics["weighted_f1"],

                        "global_macro_f1":
                            global_metrics["macro_f1"],

                        "global_critical_recall":
                            global_metrics["critical_recall"],

                        "global_medical_safety_score":
                            global_metrics.get(
                                "medical_safety_score",
                                (
                                    global_metrics["macro_f1"]
                                    + global_metrics["weighted_f1"]
                                    + global_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "global_roc_auc":
                            global_metrics["roc_auc"],

                        "personalized_accuracy":
                            personalized_metrics["accuracy"],

                        "personalized_precision":
                            personalized_metrics["precision"],

                        "personalized_recall":
                            personalized_metrics["recall"],

                        "personalized_f1":
                            personalized_metrics["weighted_f1"],

                        "personalized_weighted_f1":
                            personalized_metrics["weighted_f1"],

                        "personalized_macro_f1":
                            personalized_metrics["macro_f1"],

                        "personalized_critical_recall":
                            personalized_metrics["critical_recall"],

                        "personalized_medical_safety_score":
                            personalized_metrics.get(
                                "medical_safety_score",
                                (
                                    personalized_metrics["macro_f1"]
                                    + personalized_metrics[
                                        "weighted_f1"
                                    ]
                                    + personalized_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "personalized_roc_auc":
                            personalized_metrics["roc_auc"],

                        "personalized_vs_baseline_acc_gain_percent":
                            round(baseline_acc_gain * 100, 2),

                        "personalized_vs_baseline_weighted_f1_gain_percent":
                            round(baseline_weighted_f1_gain * 100, 2),

                        "personalized_vs_baseline_critical_recall_gain_percent":
                            round(
                                baseline_critical_recall_gain * 100,
                                2
                            ),

                        "personalized_vs_updated_local_acc_gain_percent":
                            round(updated_local_acc_gain * 100, 2),

                        "personalized_vs_updated_local_weighted_f1_gain_percent":
                            round(
                                updated_local_weighted_f1_gain * 100,
                                2
                            ),

                        "personalized_vs_updated_local_macro_f1_gain_percent":
                            round(
                                updated_local_macro_f1_gain * 100,
                                2
                            ),

                        "selected_model_type":
                            selected_model_type,

                        "selected_accuracy":
                            selected_metrics["accuracy"],

                        "selected_precision":
                            selected_metrics["precision"],

                        "selected_recall":
                            selected_metrics["recall"],

                        "selected_f1":
                            selected_metrics["weighted_f1"],

                        "selected_weighted_f1":
                            selected_metrics["weighted_f1"],

                        "selected_macro_f1":
                            selected_metrics["macro_f1"],

                        "selected_critical_recall":
                            selected_metrics["critical_recall"],

                        "selected_medical_safety_score":
                            selected_metrics.get(
                                "medical_safety_score",
                                (
                                    selected_metrics["macro_f1"]
                                    + selected_metrics[
                                        "weighted_f1"
                                    ]
                                    + selected_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "selected_roc_auc":
                            selected_metrics["roc_auc"],

                        "federated_accuracy":
                            selected_metrics["accuracy"],

                        "federated_precision":
                            selected_metrics["precision"],

                        "federated_recall":
                            selected_metrics["recall"],

                        "federated_f1":
                            selected_metrics["weighted_f1"],

                        "federated_roc_auc":
                            selected_metrics["roc_auc"],

                        "improvement_percent":
                            selected_accuracy_improvement,

                        "f1_improvement_percent":
                            selected_f1_improvement,

                        "critical_recall_improvement_percent":
                            round(
                                (
                                    selected_metrics["critical_recall"]
                                    - local_baseline_metrics[
                                        "critical_recall"
                                    ]
                                ) * 100,
                                2
                            ),

                        "confusion_matrix":
                            selected_metrics["confusion_matrix"],

                        "best_so_far_accuracy":
                            best_selected_metrics["accuracy"],

                        "best_so_far_weighted_f1":
                            best_selected_metrics["weighted_f1"],

                        "best_so_far_macro_f1":
                            best_selected_metrics["macro_f1"],

                        "best_so_far_critical_recall":
                            best_selected_metrics["critical_recall"],

                        "best_so_far_medical_safety_score":
                            best_selected_metrics.get(
                                "medical_safety_score",
                                (
                                    best_selected_metrics["macro_f1"]
                                    + best_selected_metrics[
                                        "weighted_f1"
                                    ]
                                    + best_selected_metrics[
                                        "critical_recall"
                                    ]
                                )
                            ),

                        "best_so_far_round":
                            best_selected_metrics["round"],

                        "best_vs_local_baseline_accuracy_improvement_percent":
                            round(
                                (
                                    best_selected_metrics["accuracy"]
                                    - local_baseline_metrics["accuracy"]
                                ) * 100,
                                2
                            ),

                        "best_vs_local_baseline_weighted_f1_improvement_percent":
                            round(
                                (
                                    best_selected_metrics["weighted_f1"]
                                    - local_baseline_metrics[
                                        "weighted_f1"
                                    ]
                                ) * 100,
                                2
                            ),

                        "best_vs_local_baseline_critical_recall_improvement_percent":
                            round(
                                (
                                    best_selected_metrics[
                                        "critical_recall"
                                    ]
                                    - local_baseline_metrics[
                                        "critical_recall"
                                    ]
                                ) * 100,
                                2
                            ),

                        "loss_history":
                            history["loss"],

                        "train_accuracy_history":
                            history["accuracy"],

                        "training_time_seconds":
                            train_time,

                        "batch_size":
                            BATCH_SIZE,

                        "fedprox_mu":
                            FEDPROX_MU,

                        "personalized_alpha":
                            PERSONALIZED_ALPHA,

                        "min_accept_acc_gain":
                            MIN_ACC_GAIN,

                        "min_accept_f1_gain":
                            MIN_F1_GAIN,

                        "critical_recall_tolerance":
                            CRITICAL_RECALL_TOLERANCE,

                        "best_model_metric":
                            BEST_MODEL_METRIC
                    })

                    break

            time.sleep(2)

    save_final_baseline_vs_best_comparison(
        local_baseline_metrics,
        best_selected_metrics
    )


# =====================================================
# START
# =====================================================

if __name__ == "__main__":

    run()