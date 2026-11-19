import torch

from hospital.local_model import get_model
from hospital.trainer import evaluate_model
from hospital.data_loader import load_data


def evaluate_global_model(weights):
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_data("server/global_validation.csv")

    model = get_model()

    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.data = torch.tensor(
                w,
                dtype=torch.float32,
                device=p.data.device
            )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    return metrics