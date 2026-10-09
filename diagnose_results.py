"""Post hoc descriptive diagnostics. No retraining or hyperparameter selection."""
from pathlib import Path
import json
import numpy as np
import torch
from model import matched_pair

ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    report = {
        "scope": "Descriptive post hoc inspection; no causal intervention was performed.",
        "interpretation": "Large initial and retained noise variance in seed 44 is consistent with its slower convergence; it does not prove the cause.",
        "variance_definition": "exp(2 * noise_logdiag), logdiag is a log standard deviation",
        "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        "seeds": [],
    }
    for seed in (42, 43, 44):
        initial = matched_pair(seed)["vbll"]
        checkpoint = torch.load(results / "checkpoints" / f"vbll_seed{seed}.pt", map_location="cpu", weights_only=True)
        initial_var = torch.exp(2 * initial.head.noise_logdiag).detach().numpy()
        selected_var = torch.exp(2 * checkpoint["state_dict"]["head.noise_logdiag"]).numpy()
        recalls = {}
        for kind in ("baseline", "vbll"):
            saved = np.load(results / "predictions" / f"{kind}_seed{seed}_clean.npz")
            predicted = saved["probabilities"].argmax(1)
            recalls[kind] = [float((predicted[saved["labels"] == c] == c).mean()) for c in range(10)]
        report["seeds"].append({"seed": seed, "selected_vbll_epoch": checkpoint["epoch"],
                                 "initial_noise_variance": initial_var.tolist(),
                                 "selected_noise_variance": selected_var.tolist(),
                                 "clean_per_class_recall": recalls})
    (results / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved results/diagnostics.json")


if __name__ == "__main__":
    main()
