from shared.model_architecture import LSTMModel


def get_model():
    """
    Returns the hospital model.

    Multiclass severity prediction:
    0 = Stable
    1 = Medium Risk
    2 = High Risk
    3 = Critical
    """

    return LSTMModel(
        input_size=6,
        hidden_size=64,
        num_layers=1,
        num_classes=4
    )