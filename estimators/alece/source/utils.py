import numpy as np

def cards_to_labels(cards: np.ndarray, scaling_ratio: float):
    card_min = np.min(cards)
    assert card_min >= 0

    cards += 1
    labels = np.log(cards) / scaling_ratio

    return labels


def label_preds_to_card_preds(preds: np.ndarray, scaling_ratio: float):
    preds = np.reshape(preds, [-1])
    preds *= scaling_ratio
    preds = np.clip(preds, a_max=25, a_min=0)
    preds = np.exp(preds) - 1
    return preds


def get_weights_from_labels(labels: np.ndarray):
    label_mean = np.mean(labels)
    weights = labels / label_mean
    weights = np.reshape(weights, [-1])
    weights = np.clip(weights, a_min=1e-3, a_max=np.max(weights))
    return weights
