"""Metrics use float64 and top-label ECE with 15 equal-width bins."""
import numpy as np


def calibration_bins(probs, labels, n_bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    bin_id = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = bin_id == i
        bins.append({"bin": i, "lower": i / n_bins, "upper": (i + 1) / n_bins,
                     "count": int(mask.sum()),
                     "confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return bins


def compute_metrics(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert probs.shape == (len(labels), 10)
    assert np.isfinite(probs).all() and (probs >= 0).all()
    assert np.max(np.abs(probs.sum(axis=1) - 1)) < 2e-6
    one_hot = np.eye(10)[labels]
    bins = calibration_bins(probs, labels)
    predictions = probs.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {
        "n": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "ece15": float(sum(b["count"] / len(labels) * abs(b["confidence"] - b["accuracy"])
                           for b in bins if b["count"])),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "calibration_bins": bins,
        "confusion_matrix": confusion.tolist(),
    }
