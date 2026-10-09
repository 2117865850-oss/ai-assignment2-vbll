"""Run the pre-specified compact FashionMNIST adaptation of D-VBLL."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from torchvision.datasets import FashionMNIST
import vbll
from model import matched_pair, trainable_parameters
from metrics import compute_metrics

ROOT = Path(__file__).resolve().parent
METRICS = ("accuracy", "nll", "brier", "ece15", "mean_confidence")


def array_hash(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def split_indices(labels, seed=2026):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        train.extend(indices[:1200])
        val.extend(indices[1200:1400])
    return rng.permutation(train).astype(np.int64), rng.permutation(val).astype(np.int64)


@torch.inference_mode()
def predict(model, images, device, mc_seed, n_samples=128, batch_size=256):
    """CPU MC uses a forked RNG; it cannot change training permutations or RNG."""
    model.eval()
    head = deepcopy(model.head).cpu().eval()
    outputs = []
    with torch.random.fork_rng(devices=[]):
        # Set only CPU RNG, unlike torch.manual_seed which also seeds accelerators.
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(mc_seed).get_state())
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(device)
            features = model.backbone(x).cpu()
            if model.kind == "baseline":
                probs = torch.softmax(head(features), dim=-1)
            else:
                probs = head.predictive(features, n_samples=n_samples)
            outputs.append(probs.numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true", help="Download official FashionMNIST files if absent")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; specify --device cpu")
    out = args.output.resolve()
    for folder in (out, out / "checkpoints", out / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    start_wall = time.time()
    config = {
        "algorithm": "Discriminative variational Bayesian last layers (D-VBLL), diagonal covariance",
        "package": "vbll==0.4.9", "dataset": "FashionMNIST",
        "data_root": "FashionMNIST dataset root supplied via --data-root", "device": args.device,
        "seeds": args.seeds, "split_seed": 2026,
        "train_size": 12000, "validation_size": 2000, "test_size": 10000,
        "epochs": args.epochs, "batch_size": 128, "eval_batch_size": 256,
        "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
        "input": "float32 x/255; normalize as (x-0.5)/0.5; no augmentation",
        "feature_dimension": 64, "last_layer_bias": False,
        "vbll": {"parameterization": "diagonal", "regularization_weight": 1 / 12000,
                 "prior_scale_argument": 1.0, "effective_weight_prior_variance": 2 / 64,
                 "wishart_scale": 1.0, "dof_argument": 1.0, "internal_wishart_coefficient": 6.0},
        "mc_samples": 128, "validation_mc_seed": "100000 + training seed (reset each epoch)",
        "test_mc_seed": "200000 + training seed (same clean/noisy draws)",
        "model_selection": "minimum validation predictive NLL; earliest epoch on exact tie",
        "noise": {"distribution": "Gaussian", "sigma": 0.2, "seed": 20261008,
                  "clipping": [0, 1], "applied_before_normalization": True},
        "ece_bins": 15, "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_write(out / "config.json", config)
    environment = {"python": sys.version, "executable": "Python interpreter used to run train.py",
                   "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                   "mps_available": torch.backends.mps.is_available(),
                   "versions": {n: importlib.metadata.version(n) for n in
                                ("torch", "torchvision", "numpy", "vbll", "matplotlib")}}
    class_path = Path(inspect.getfile(vbll.DiscClassification))
    environment["vbll_classification_source_sha256"] = hashlib.sha256(class_path.read_bytes()).hexdigest()
    environment["vbll_source_file"] = "vbll/layers/classification.py"
    json_write(out / "environment.json", environment)
    # The official test split is not accessed until all validation selections finish.
    # Official dataset repository supplies the same MD5-verified torchvision files.
    FashionMNIST.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    raw_train = FashionMNIST(str(args.data_root), train=True, download=args.download)
    train_idx, val_idx = split_indices(raw_train.targets.numpy())
    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.unique(np.concatenate([train_idx, val_idx]))) == 14000
    np.savez_compressed(out / "split_indices.npz", train=train_idx, validation=val_idx,
                        unused=np.setdiff1d(np.arange(60000), np.concatenate([train_idx, val_idx])))
    manifest = {"split_seed": 2026, "train_count": len(train_idx), "val_count": len(val_idx),
                "unused_count": 46000, "train_sha256_int64": array_hash(train_idx),
                "validation_sha256_int64": array_hash(val_idx), "overlap": 0,
                "train_per_class": np.bincount(raw_train.targets.numpy()[train_idx]).tolist(),
                "validation_per_class": np.bincount(raw_train.targets.numpy()[val_idx]).tolist(),
                "dataset_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((args.data_root / "FashionMNIST/raw").glob("*")) if p.is_file()}}
    json_write(out / "split_manifest.json", manifest)
    x_train = (raw_train.data[train_idx].float().unsqueeze(1) / 255 - 0.5) / 0.5
    y_train = raw_train.targets[train_idx]
    x_val = (raw_train.data[val_idx].float().unsqueeze(1) / 255 - 0.5) / 0.5
    y_val = raw_train.targets[val_idx].numpy()
    training_rows, init_checks, order_hashes, selections = [], [], [], []
    log_fields = ["seed", "model", "epoch", "train_objective", "val_accuracy", "val_nll",
                  "val_brier", "val_ece15", "epoch_seconds_pair", "selected_so_far"]
    with (out / "training_log.csv").open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()
        for seed in args.seeds:
            models = matched_pair(seed)
            initial_feature_hash = array_hash(torch.cat([p.detach().flatten() for p in models["baseline"].backbone.parameters()]).numpy())
            init_checks.append({"seed": seed, "backbone_equal": True, "mean_head_equal": True,
                                "backbone_sha256": initial_feature_hash,
                                "head_mean_sha256": array_hash(models["baseline"].head.weight.detach().numpy()),
                                "trainable_parameters": {k: trainable_parameters(v) for k, v in models.items()}})
            models = {k: v.to(args.device) for k, v in models.items()}
            optimizers = {k: torch.optim.Adam(v.parameters(), lr=0.001, weight_decay=0.0)
                          for k, v in models.items()}
            best = {k: {"nll": float("inf"), "epoch": None} for k in models}
            for epoch in range(1, args.epochs + 1):
                t_epoch = time.perf_counter()
                for model in models.values():
                    model.train()
                batch_generator = torch.Generator(device="cpu").manual_seed(seed * 10000 + epoch)
                order = torch.randperm(len(train_idx), generator=batch_generator)
                order_hashes.append({"seed": seed, "epoch": epoch, "shared_order_sha256": array_hash(order.numpy())})
                losses = dict.fromkeys(models, 0.0)
                for start in range(0, len(order), 128):
                    batch = order[start:start + 128]
                    images = x_train[batch].to(args.device)
                    labels = y_train[batch].to(args.device)
                    for kind, model in models.items():
                        optimizers[kind].zero_grad(set_to_none=True)
                        loss = model.training_loss(images, labels)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss: {seed}/{kind}/{epoch}")
                        loss.backward()
                        optimizers[kind].step()
                        losses[kind] += float(loss.detach().cpu()) * len(batch)
                val_results = {}
                for kind, model in models.items():
                    probs = predict(model, x_val, args.device, 100000 + seed)
                    metric = compute_metrics(probs, y_val)
                    val_results[kind] = metric
                    if metric["nll"] < best[kind]["nll"]:
                        best[kind] = {"nll": metric["nll"], "epoch": epoch}
                        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                        torch.save({"model": kind, "seed": seed, "epoch": epoch,
                                    "validation_metrics": metric, "state_dict": state},
                                   out / "checkpoints" / f"{kind}_seed{seed}.pt")
                        np.savez_compressed(out / "predictions" / f"{kind}_seed{seed}_validation.npz",
                                            probabilities=probs, labels=y_val, original_indices=val_idx)
                elapsed = time.perf_counter() - t_epoch
                for kind in models:
                    metric = val_results[kind]
                    row = {"seed": seed, "model": kind, "epoch": epoch,
                           "train_objective": losses[kind] / len(train_idx),
                           "val_accuracy": metric["accuracy"], "val_nll": metric["nll"],
                           "val_brier": metric["brier"], "val_ece15": metric["ece15"],
                           "epoch_seconds_pair": elapsed,
                           "selected_so_far": best[kind]["epoch"] == epoch}
                    training_rows.append(row)
                    writer.writerow(row)
                log_file.flush()
                print(f"seed {seed} epoch {epoch:02d}/{args.epochs} | " + " | ".join(
                    f"{kind} val NLL {val_results[kind]['nll']:.4f} acc {val_results[kind]['accuracy']:.4f}"
                    for kind in models) + f" | {elapsed:.1f}s", flush=True)
            selections.extend({"seed": seed, "model": kind, "selected_epoch": value["epoch"],
                               "validation_nll": value["nll"]} for kind, value in best.items())
            del models, optimizers
        json_write(out / "initialization_checks.json", init_checks)
        json_write(out / "batch_order_hashes.json", order_hashes)
        json_write(out / "selected_models.json", selections)
    # All six checkpoint choices are now fixed. Test data never select or tune a model.
    raw_test = FashionMNIST(str(args.data_root), train=False, download=args.download)
    clean_pixels = raw_test.data.float().unsqueeze(1) / 255
    noise_generator = torch.Generator(device="cpu").manual_seed(20261008)
    noisy_pixels = (clean_pixels + 0.2 * torch.randn(clean_pixels.shape, generator=noise_generator)).clamp(0, 1)
    images = {"clean": (clean_pixels - 0.5) / 0.5, "gaussian_noise_0.2": (noisy_pixels - 0.5) / 0.5}
    y_test = raw_test.targets.numpy()
    config["noise"]["noisy_pixels_sha256_float32"] = array_hash(noisy_pixels.numpy())
    config["test_labels_sha256_int64"] = array_hash(y_test)
    rows, details = [], []
    for selection in selections:
        seed, kind = selection["seed"], selection["model"]
        model = matched_pair(seed)[kind]
        checkpoint = torch.load(out / "checkpoints" / f"{kind}_seed{seed}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(args.device)
        for condition, data in images.items():
            probs = predict(model, data, args.device, 200000 + seed)
            metric = compute_metrics(probs, y_test)
            np.savez_compressed(out / "predictions" / f"{kind}_seed{seed}_{condition}.npz",
                                probabilities=probs, labels=y_test, original_indices=np.arange(10000))
            row = {**selection, "condition": condition, **{k: metric[k] for k in METRICS}}
            rows.append(row)
            details.append({**selection, "condition": condition, **metric})
            print(f"TEST {kind} seed{seed} {condition}: acc={metric['accuracy']:.4f} NLL={metric['nll']:.4f} ECE={metric['ece15']:.4f}", flush=True)
        del model
    summary = {"config": config, "runs": rows, "aggregate": [], "paired_differences_vbll_minus_baseline": []}
    for condition in images:
        for kind in ("baseline", "vbll"):
            subset = [r for r in rows if r["condition"] == condition and r["model"] == kind]
            summary["aggregate"].append({"condition": condition, "model": kind, "seeds": len(subset),
                                          **{m: {"mean": float(np.mean([r[m] for r in subset])),
                                                 "std": float(np.std([r[m] for r in subset], ddof=1)) if len(subset) > 1 else None}
                                             for m in METRICS}})
        for metric in METRICS:
            delta = [next(r[metric] for r in rows if r["condition"] == condition and r["seed"] == seed and r["model"] == "vbll") -
                     next(r[metric] for r in rows if r["condition"] == condition and r["seed"] == seed and r["model"] == "baseline") for seed in args.seeds]
            summary["paired_differences_vbll_minus_baseline"].append({"condition": condition, "metric": metric,
                                                                      "per_seed": delta, "mean": float(np.mean(delta)),
                                                                      "std": float(np.std(delta, ddof=1)) if len(delta) > 1 else None})
    config["elapsed_seconds"] = time.time() - start_wall
    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(out / "config.json", config)
    json_write(out / "summary.json", summary)
    json_write(out / "results.json", summary)
    json_write(out / "metrics_detail.json", details)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed in {config['elapsed_seconds']:.1f}s. Results saved to the output directory.", flush=True)


if __name__ == "__main__":
    main()
