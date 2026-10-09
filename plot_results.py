"""Regenerate publication-quality figures from saved probabilities and logs."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from metrics import calibration_bins

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
COLORS = {"baseline": "#537489", "vbll": "#D47632"}
NAMES = {"baseline": "Deterministic CNN", "vbll": "CNN + D-VBLL"}
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def setup():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#BBC4CA", "text.color": "#203442",
                         "axes.labelcolor": "#203442", "xtick.color": "#4B5C67",
                         "ytick.color": "#4B5C67", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "grid.color": "#E6EBEE"})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def learning_curves():
    rows = list(csv.DictReader((RESULTS / "training_log.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), constrained_layout=True)
    for ax, key, ylabel in zip(axes, ("val_nll", "val_accuracy"), ("Validation NLL (lower is better)", "Validation accuracy")):
        for kind in NAMES:
            values = np.array([[float(r[key]) for r in rows if r["model"] == kind and int(r["seed"]) == seed] for seed in (42, 43, 44)])
            x = np.arange(1, values.shape[1] + 1)
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(x, mean, color=COLORS[kind], lw=2.1, label=NAMES[kind])
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[kind], alpha=0.14)
        ax.set(xlabel="Epoch", ylabel=ylabel, xticks=[1, 2, 4, 6, 8, 10])
        ax.grid(axis="y", alpha=0.7)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.suptitle("Learning curves · mean ± sample SD across three paired seeds", fontsize=12)
    save(fig, "learning_curves")


def metric_comparison(summary):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.0), constrained_layout=True)
    for ax, metric, title in zip(axes, ("accuracy", "nll", "brier", "ece15"), ("Accuracy ↑", "NLL ↓", "Brier score ↓", "ECE (15 bins) ↓")):
        for j, kind in enumerate(NAMES):
            entries = [next(v for v in summary["aggregate"] if v["model"] == kind and v["condition"] == condition)
                       for condition in ("clean", "gaussian_noise_0.2")]
            means = [e[metric]["mean"] for e in entries]
            stds = [e[metric]["std"] for e in entries]
            x = np.arange(2) + (-0.16 if j == 0 else 0.16)
            ax.bar(x, means, width=0.28, color=COLORS[kind], label=NAMES[kind], zorder=2)
            ax.errorbar(x, means, yerr=stds, color="#203442", fmt="none", capsize=3, lw=0.9, zorder=3)
        ax.set_title(title)
        ax.set_xticks([0, 1], ["Clean", "Noise σ=.2"])
        ax.grid(axis="y", zorder=0)
        if metric in ("accuracy", "ece15"):
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if metric == "accuracy":
            ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Test metrics · mean ± sample SD across three paired seeds", fontsize=12)
    save(fig, "metric_comparison")


def pooled(kind, condition):
    arrays = [np.load(RESULTS / "predictions" / f"{kind}_seed{seed}_{condition}.npz") for seed in (42, 43, 44)]
    return np.concatenate([a["probabilities"] for a in arrays]), np.concatenate([a["labels"] for a in arrays])


def reliability():
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.25), constrained_layout=True)
    for ax, condition, title in zip(axes, ("clean", "gaussian_noise_0.2"), ("Clean test images", "Gaussian noise σ = 0.2")):
        ax.plot([0, 1], [0, 1], color="#9AA8B2", ls="--", lw=1)
        for kind in NAMES:
            probs, labels = pooled(kind, condition)
            bins = [b for b in calibration_bins(probs, labels) if b["count"]]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                    "o-", color=COLORS[kind], lw=1.7, ms=4, label=NAMES[kind])
        ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy",
               title=title, xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Reliability diagrams · pooled predictions, 15 equal-width bins", fontsize=12)
    save(fig, "reliability")


def confusion():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for ax, kind in zip(axes, NAMES):
        probs, labels = pooled(kind, "clean")
        cm = np.zeros((10, 10))
        np.add.at(cm, (labels, probs.argmax(1)), 1)
        cm = cm / cm.sum(1, keepdims=True)
        image = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.04:
                    ax.text(j, i, f"{cm[i,j]*100:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > .55 else "#203442")
        ax.set_xticks(range(10), CLASS_NAMES, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
        ax.set(xlabel="Predicted class", ylabel="True class", title=NAMES[kind])
    fig.colorbar(image, ax=axes, shrink=.78, label="Fraction within true class")
    fig.suptitle("Clean-test confusion · pooled predictions, rows normalized", fontsize=12)
    save(fig, "confusion_matrices")


if __name__ == "__main__":
    setup()
    summary = json.loads((RESULTS / "summary.json").read_text())
    learning_curves()
    metric_comparison(summary)
    reliability()
    confusion()
    print("Saved four figures as PNG and SVG in figures/.")
