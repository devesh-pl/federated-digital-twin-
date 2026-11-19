import torch
import numpy as np

from hospital.local_model import get_model

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

def load_model(model_path):

    model = get_model()

    weights = torch.load(
        model_path,
        map_location="cpu"
    )

    with torch.no_grad():

        for p, w in zip(
            model.parameters(),
            weights
        ):

            p.data = torch.tensor(
                w,
                dtype=p.data.dtype
            )

    model.eval()

    return model

# =====================================================
# PREPARE INPUT
# =====================================================

def prepare_input(
    hr,
    spo2,
    bp,
    rr,
    temp,
    ecg
):

    # -------------------------------------------------
    # VERY IMPORTANT
    # SAME ORDER AS TRAINING
    # -------------------------------------------------

    values = np.array([
        hr,
        spo2,
        bp,
        rr,
        temp,
        ecg
    ], dtype=np.float32)

    # -------------------------------------------------
    # SIMPLE NORMALIZATION
    # -------------------------------------------------

    normalized = np.array([

        values[0] / 200.0,   # hr
        values[1] / 100.0,   # spo2
        values[2] / 200.0,   # bp
        values[3] / 40.0,    # rr
        values[4] / 45.0,    # temp
        values[5] / 1.0      # ecg

    ])

    # -------------------------------------------------
    # CREATE SEQUENCE
    # -------------------------------------------------

    sequence = np.tile(
        normalized,
        (10, 1)
    )

    return torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

# =====================================================
# PREDICTION
# =====================================================

def predict_patient(
    model,
    hr,
    spo2,
    bp,
    rr,
    temp,
    ecg
):

    input_tensor = prepare_input(
        hr,
        spo2,
        bp,
        rr,
        temp,
        ecg
    )

    with torch.no_grad():

        output = model(input_tensor)

        probability = (
            output.item()
        )

    # -------------------------------------------------
    # LABEL
    # -------------------------------------------------

    if probability > 0.5:

        label = "Critical"

    else:

        label = "Stable"

    confidence = max(
        probability,
        1 - probability
    )

    return {

        "prediction": label,

        "probability": float(probability),

        "confidence": float(confidence)
    }