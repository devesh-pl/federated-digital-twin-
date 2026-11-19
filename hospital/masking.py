import numpy as np

def add_mask(weights):
    masked_weights = []
    masks = []

    for w in weights:
        noise = np.random.normal(0, 0.01, w.shape)
        masked_weights.append(w + noise)
        masks.append(noise)

    return masked_weights, masks


def remove_mask(masked_weights, masks):
    original_weights = []

    for w, m in zip(masked_weights, masks):
        original_weights.append(w - m)

    return original_weights