import numpy as np


# =====================================================
# VALIDATION HELPERS
# =====================================================

def validate_weights(all_weights):
    """
    Validate that all hospitals uploaded compatible model weights.
    """

    if not all_weights:
        raise ValueError("No model weights received for aggregation.")

    reference_shapes = [
        np.array(layer).shape
        for layer in all_weights[0]
    ]

    for hospital_index, hospital_weights in enumerate(all_weights):

        if len(hospital_weights) != len(reference_shapes):

            raise ValueError(
                f"Hospital {hospital_index} has "
                f"{len(hospital_weights)} layers, expected "
                f"{len(reference_shapes)}."
            )

        for layer_index, layer in enumerate(hospital_weights):

            current_shape = np.array(layer).shape
            expected_shape = reference_shapes[layer_index]

            if current_shape != expected_shape:

                raise ValueError(
                    f"Shape mismatch at hospital {hospital_index}, "
                    f"layer {layer_index}: got {current_shape}, "
                    f"expected {expected_shape}."
                )


# =====================================================
# NORMAL FEDAVG
# =====================================================

def federated_average(all_weights):
    """
    Standard FedAvg.
    Works for both binary and multiclass models.
    """

    validate_weights(all_weights)

    avg_weights = []

    for layer_weights in zip(*all_weights):

        layer_stack = np.array(
            layer_weights,
            dtype=np.float32
        )

        avg_layer = np.mean(
            layer_stack,
            axis=0
        )

        avg_weights.append(
            avg_layer.astype(np.float32)
        )

    return avg_weights


# =====================================================
# F1-WEIGHTED FEDAVG
# =====================================================

def weighted_federated_average(
    all_weights,
    aggregation_weights
):
    """
    F1-weighted Federated Averaging.

    For the 4-class severity model, local_f1 means weighted F1.

    all_weights:
        [
            hospital_0_weights,
            hospital_1_weights,
            hospital_2_weights
        ]

    aggregation_weights:
        [
            hospital_0_weight,
            hospital_1_weight,
            hospital_2_weight
        ]
    """

    validate_weights(all_weights)

    aggregation_weights = np.array(
        aggregation_weights,
        dtype=np.float32
    )

    if len(aggregation_weights) != len(all_weights):

        raise ValueError(
            f"Aggregation weight count "
            f"{len(aggregation_weights)} does not match "
            f"hospital weight count {len(all_weights)}."
        )

    if aggregation_weights.sum() == 0:

        aggregation_weights = np.ones_like(
            aggregation_weights
        )

    aggregation_weights = (
        aggregation_weights
        / aggregation_weights.sum()
    )

    avg_weights = []

    for layer_group in zip(*all_weights):

        weighted_layer = None

        for hospital_weight, layer in zip(
            aggregation_weights,
            layer_group
        ):

            layer = np.array(
                layer,
                dtype=np.float32
            )

            if weighted_layer is None:

                weighted_layer = (
                    hospital_weight * layer
                )

            else:

                weighted_layer += (
                    hospital_weight * layer
                )

        avg_weights.append(
            weighted_layer.astype(np.float32)
        )

    return avg_weights