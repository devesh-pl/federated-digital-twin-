import torch
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# =====================================================
# LABELS
# =====================================================

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


# =====================================================
# EVALUATE MULTICLASS MODEL
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    hospital_id,
    mode="local"
):
    """
    Evaluate 4-class severity prediction model.

    Classes:
    0 = Stable
    1 = Medium Risk
    2 = High Risk
    3 = Critical
    """

    os.makedirs(
        "hospital/logs",
        exist_ok=True
    )

    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = np.array(
        y_test
    ).astype(int).flatten()

    # -------------------------------------------------
    # PREDICTIONS
    # -------------------------------------------------

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        ).cpu().numpy()

        predictions = np.argmax(
            probabilities,
            axis=1
        )

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    weighted_precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    per_class_precision = precision_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0
    )

    per_class_recall = recall_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0
    )

    # -------------------------------------------------
    # MULTICLASS ROC AUC
    # -------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        roc_auc = 0.0

    # -------------------------------------------------
    # CONFUSION MATRIX 4x4
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3]
    )

    # -------------------------------------------------
    # SAVE CONFUSION MATRIX IMAGE
    # -------------------------------------------------

    try:

        fig, ax = plt.subplots(
            figsize=(7, 6)
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
            f"{mode.upper()} 4-Class Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            f"hospital/logs/"
            f"h{hospital_id}_{mode}_cm.png"
        )

        plt.close()

    except Exception as e:

        print(
            "[EVALUATOR] Could not save confusion matrix:",
            e
        )

    # -------------------------------------------------
    # SAVE CLASS PROBABILITY DISTRIBUTION IMAGE
    # -------------------------------------------------

    try:

        avg_probs = probabilities.mean(
            axis=0
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.bar(
            LABEL_NAMES,
            avg_probs
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.ylabel(
            "Average Predicted Probability"
        )

        plt.title(
            f"{mode.upper()} Average Class Probability"
        )

        plt.tight_layout()

        plt.savefig(
            f"hospital/logs/"
            f"h{hospital_id}_{mode}_class_probs.png"
        )

        plt.close()

    except Exception as e:

        print(
            "[EVALUATOR] Could not save probability plot:",
            e
        )

    # -------------------------------------------------
    # RETURN METRICS
    # -------------------------------------------------

    return {
        "accuracy": float(accuracy),

        # Backward-compatible names.
        # These now mean weighted multiclass metrics.
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),

        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),

        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),

        "roc_auc": float(roc_auc),

        "confusion_matrix": cm.tolist(),

        "stable_precision": float(per_class_precision[0]),
        "stable_recall": float(per_class_recall[0]),
        "stable_f1": float(per_class_f1[0]),

        "medium_precision": float(per_class_precision[1]),
        "medium_recall": float(per_class_recall[1]),
        "medium_f1": float(per_class_f1[1]),

        "high_precision": float(per_class_precision[2]),
        "high_recall": float(per_class_recall[2]),
        "high_f1": float(per_class_f1[2]),

        "critical_precision": float(per_class_precision[3]),
        "critical_recall": float(per_class_recall[3]),
        "critical_f1": float(per_class_f1[3]),

        "label_map": LABEL_MAP
    }