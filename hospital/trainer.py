import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# =====================================================
# VERBOSE CONFIG
# =====================================================

VERBOSE_TRAINING = os.getenv(
    "VERBOSE_TRAINING",
    "false"
).lower() == "true"


# =====================================================
# LABEL MAP
# =====================================================

LABEL_MAP = {
    0: "Stable",
    1: "Medium Risk",
    2: "High Risk",
    3: "Critical"
}


# =====================================================
# COPY GLOBAL PARAMETERS FOR FEDPROX
# =====================================================

def clone_model_parameters(model):
    """
    Creates a frozen copy of current model parameters.
    Used as the FedProx reference model.
    """

    return [
        p.detach().clone()
        for p in model.parameters()
    ]


# =====================================================
# COMPUTE STABLE MULTICLASS CLASS WEIGHTS
# =====================================================

def compute_class_weights(y_train, num_classes=4):
    """
    Compute stable multiclass weights using square-root inverse frequency.
    """

    y_np = np.array(y_train).astype(int).flatten()

    counts = np.bincount(
        y_np,
        minlength=num_classes
    )

    total = counts.sum()

    weights = []

    for class_count in counts:

        if class_count == 0:
            weights.append(1.0)

        else:
            freq = class_count / total
            weights.append(
                1.0 / np.sqrt(freq)
            )

    weights = np.array(
        weights,
        dtype=np.float32
    )

    weights = weights / weights.mean()

    weights = np.clip(
        weights,
        0.5,
        3.0
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    ), counts


# =====================================================
# TRAIN MODEL WITH MINI-BATCH + FEDPROX
# =====================================================

def train_model(
    model,
    X_train,
    y_train,
    epochs=5,
    lr=0.001,
    fedprox_mu=0.0,
    global_params=None,
    num_classes=4,
    batch_size=64
):
    device = torch.device("cpu")

    model.to(device)

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    ).view(-1)

    # -------------------------
    # CLASS WEIGHTS
    # -------------------------

    class_weights, class_counts = compute_class_weights(
        y_train.numpy(),
        num_classes=num_classes
    )

    class_weights = class_weights.to(device)

    if VERBOSE_TRAINING:

        print("[TRAIN] Class distribution:")

        for class_id in range(num_classes):

            print(
                f"  Class {class_id} "
                f"({LABEL_MAP.get(class_id, 'Unknown')}): "
                f"{int(class_counts[class_id])} samples | "
                f"weight={class_weights[class_id].item():.4f}"
            )

    # -------------------------
    # FEDPROX SETUP
    # -------------------------

    if fedprox_mu > 0 and global_params is not None:

        if VERBOSE_TRAINING:

            print(
                f"[TRAIN] FedProx enabled | "
                f"mu={fedprox_mu}"
            )

        global_params = [
            gp.to(device)
            for gp in global_params
        ]

    # -------------------------
    # DATALOADER
    # -------------------------

    dataset = torch.utils.data.TensorDataset(
        X_train,
        y_train
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # -------------------------
    # LOSS + OPTIMIZER
    # -------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    history = {
        "loss": [],
        "accuracy": []
    }

    # -------------------------
    # TRAINING LOOP
    # -------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        class_pred_counts = np.zeros(
            num_classes,
            dtype=int
        )

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            classification_loss = criterion(
                outputs,
                batch_y
            )

            prox_loss = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device
            )

            if fedprox_mu > 0 and global_params is not None:

                for local_param, global_param in zip(
                    model.parameters(),
                    global_params
                ):

                    prox_loss += torch.sum(
                        (
                            local_param
                            - global_param
                        ) ** 2
                    )

                loss = (
                    classification_loss
                    + (fedprox_mu / 2.0) * prox_loss
                )

            else:

                loss = classification_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            for class_id in range(num_classes):

                class_pred_counts[class_id] += int(
                    torch.sum(
                        preds == class_id
                    ).item()
                )

            correct += int(
                torch.sum(
                    preds == batch_y
                ).item()
            )

            total += batch_y.size(0)

            total_loss += (
                loss.item()
                * batch_y.size(0)
            )

        epoch_loss = total_loss / total
        epoch_acc = correct / total

        history["loss"].append(
            float(epoch_loss)
        )

        history["accuracy"].append(
            float(epoch_acc)
        )

        if VERBOSE_TRAINING:

            print(
                f"[TRAIN] Epoch {epoch + 1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} | "
                f"Pred Dist: {class_pred_counts.tolist()}"
            )

    print(
        f"[TRAIN COMPLETE] "
        f"Final Loss: {history['loss'][-1]:.4f} | "
        f"Final Train Acc: {history['accuracy'][-1]:.4f}"
    )

    return model, history


# =====================================================
# RAW CLASS PROBABILITIES
# =====================================================

def predict_probabilities(
    model,
    X
):
    """
    Returns multiclass probabilities for each sample.

    Output shape:
    [num_samples, 4]
    """

    device = torch.device("cpu")

    model.to(device)

    X = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probs = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()

    return probs


# =====================================================
# EVALUATE MULTICLASS MODEL
# =====================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Multiclass evaluator.

    Classes:
    0 = Stable
    1 = Medium Risk
    2 = High Risk
    3 = Critical
    """

    y_test = np.array(
        y_test
    ).astype(int).flatten()

    probs = predict_probabilities(
        model,
        X_test
    )

    preds = np.argmax(
        probs,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        preds
    )

    weighted_precision = precision_score(
        y_test,
        preds,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        preds,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        preds,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_test,
        preds,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        preds,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        preds,
        average="macro",
        zero_division=0
    )

    per_class_precision = precision_score(
        y_test,
        preds,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    per_class_recall = recall_score(
        y_test,
        preds,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_test,
        preds,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        preds,
        labels=[0, 1, 2, 3]
    ).tolist()

    try:

        roc_auc = roc_auc_score(
            y_test,
            probs,
            multi_class="ovr",
            average="weighted"
        )

    except Exception:

        roc_auc = 0.0

    high_risk_recall = float(
        per_class_recall[2]
    )

    high_risk_f1 = float(
        per_class_f1[2]
    )

    critical_recall = float(
        per_class_recall[3]
    )

    critical_f1 = float(
        per_class_f1[3]
    )

    metrics = {
        "accuracy": float(accuracy),

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
        "confusion_matrix": cm,

        "stable_precision": float(per_class_precision[0]),
        "stable_recall": float(per_class_recall[0]),
        "stable_f1": float(per_class_f1[0]),

        "medium_precision": float(per_class_precision[1]),
        "medium_recall": float(per_class_recall[1]),
        "medium_f1": float(per_class_f1[1]),

        "high_precision": float(per_class_precision[2]),
        "high_recall": float(per_class_recall[2]),
        "high_f1": float(per_class_f1[2]),

        "high_risk_recall": high_risk_recall,
        "high_risk_f1": high_risk_f1,

        "critical_precision": float(per_class_precision[3]),
        "critical_recall": critical_recall,
        "critical_f1": critical_f1,

        "label_map": LABEL_MAP
    }

    return metrics


# =====================================================
# FIND BEST "THRESHOLD" COMPATIBILITY FUNCTION
# =====================================================

def find_best_threshold(
    model,
    X_test,
    y_test,
    thresholds=None
):
    """
    Compatibility function for old hospital_client.py.

    In multiclass classification, there is no threshold.
    We simply evaluate using argmax.
    """

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    metrics["threshold"] = None
    metrics["best_threshold"] = None

    return None, metrics


# =====================================================
# PREDICT SINGLE PATIENT
# =====================================================

def predict_patient(
    model,
    sequence,
    threshold=None
):
    """
    Predict one patient sequence.

    Returns severity class:
    0 = Stable
    1 = Medium Risk
    2 = High Risk
    3 = Critical
    """

    device = torch.device("cpu")

    model.to(device)

    model.eval()

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

    prediction = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[prediction]
    )

    return {
        "prediction": prediction,
        "label": LABEL_MAP[prediction],
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": {
            "stable": round(float(probs[0]) * 100, 2),
            "medium_risk": round(float(probs[1]) * 100, 2),
            "high_risk": round(float(probs[2]) * 100, 2),
            "critical": round(float(probs[3]) * 100, 2)
        }
    }