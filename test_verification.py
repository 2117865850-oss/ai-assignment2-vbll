"""Independent mathematical checks plus a saved-results audit (no retraining)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import vbll
from model import matched_pair
from metrics import compute_metrics
from train import split_indices, predict


class MathematicalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(2026)

    def test_official_loss_matches_independent_diagonal_formula(self):
        head = vbll.DiscClassification(4, 3, regularization_weight=1 / 12000,
                                       parameterization="diagonal").double()
        x = torch.randn(7, 4, dtype=torch.float64)
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        mu = x @ head.W_mean.T
        var_w = torch.exp(2 * head.W_logdiag)
        noise_var = torch.exp(2 * head.noise_logdiag)
        variance = x.square() @ var_w.T + noise_var
        nll_bound = (torch.logsumexp(mu + 0.5 * variance, -1) - mu[torch.arange(len(x)), y]).mean()
        # The official code intentionally omits the constant -K*D/2 in Gaussian KL.
        q = head.prior_scale
        kl_without_constant = 0.5 * ((head.W_mean.square() + var_w).sum() / q
                                     + head.W_mean.numel() * np.log(q) - torch.log(var_w).sum())
        logdet_precision = -torch.log(noise_var).sum()
        wishart = head.dof * logdet_precision - 0.5 * head.wishart_scale * (1 / noise_var).sum()
        expected = nll_bound + (kl_without_constant - wishart) / 12000
        actual = head._get_train_loss_fn(x)(y)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_jensen_bound_exceeds_monte_carlo_expected_cross_entropy(self):
        # A genuinely independent Monte Carlo estimate, not the package's bound method.
        mu = torch.tensor([[0.8, -0.4, 0.2], [-0.2, 0.5, 0.1]], dtype=torch.float64)
        variance = torch.tensor([[0.6, 0.3, 0.8], [0.4, 0.7, 0.5]], dtype=torch.float64)
        labels = torch.tensor([0, 2])
        analytic = torch.logsumexp(mu + variance / 2, -1) - mu[torch.arange(2), labels]
        z = mu + torch.randn(200000, 2, 3, dtype=torch.float64) * variance.sqrt()
        samples = -F.log_softmax(z, -1)[:, torch.arange(2), labels]
        monte_carlo = samples.mean(0)
        standard_error = samples.std(0) / np.sqrt(len(samples))
        self.assertTrue(torch.all(analytic > monte_carlo + 5 * standard_error))
        self.bound_values = {"analytic": analytic.tolist(), "mc": monte_carlo.tolist()}

    def test_zero_variance_bound_is_cross_entropy(self):
        mu = torch.randn(20, 10, dtype=torch.float64)
        labels = torch.arange(20) % 10
        bound = (torch.logsumexp(mu, -1) - mu[torch.arange(20), labels]).mean()
        torch.testing.assert_close(bound, F.cross_entropy(mu, labels), rtol=1e-12, atol=1e-12)

    def test_matched_initialization(self):
        models = matched_pair(42)
        for a, b in zip(models["baseline"].backbone.parameters(), models["vbll"].backbone.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(models["baseline"].head.weight, models["vbll"].head.W_mean))
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            deterministic = models["baseline"].head(models["baseline"].backbone(x))
            bayesian_mean = models["vbll"].head.logit_predictive(models["vbll"].backbone(x)).mean
        torch.testing.assert_close(deterministic, bayesian_mean)

    def test_predictive_probabilities_rng_reproducibility(self):
        model = matched_pair(42)["vbll"]
        images = torch.randn(13, 1, 28, 28)
        original_rng = torch.get_rng_state().clone()
        p = predict(model, images, "cpu", 999, batch_size=8)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        q = predict(model, images, "cpu", 999, batch_size=8)
        np.testing.assert_array_equal(p, q)
        np.testing.assert_allclose(p.sum(1), 1, atol=2e-7)
        self.assertTrue((p >= 0).all())

    def test_disjoint_stratified_split(self):
        labels = np.repeat(np.arange(10), 6000)
        train, val = split_indices(labels)
        self.assertEqual(len(train), 12000)
        self.assertEqual(len(val), 2000)
        self.assertEqual(len(np.intersect1d(train, val)), 0)
        np.testing.assert_array_equal(np.bincount(labels[train]), np.repeat(1200, 10))
        np.testing.assert_array_equal(np.bincount(labels[val]), np.repeat(200, 10))

    def test_metric_reference_values(self):
        y = np.arange(10)
        perfect = compute_metrics(np.eye(10), y)
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["nll"], 0)
        self.assertEqual(perfect["brier"], 0)
        self.assertEqual(perfect["ece15"], 0)
        uniform = compute_metrics(np.ones((10, 10)) / 10, y)
        self.assertAlmostEqual(uniform["nll"], np.log(10))
        self.assertAlmostEqual(uniform["brier"], 0.9)
        self.assertAlmostEqual(uniform["accuracy"], 0.1)
        self.assertAlmostEqual(uniform["ece15"], 0)


def audit_results(root):
    summary = json.loads((root / "summary.json").read_text())
    checked = []
    for run in summary["runs"]:
        path = root / "predictions" / f"{run['model']}_seed{run['seed']}_{run['condition']}.npz"
        with np.load(path) as saved:
            metric = compute_metrics(saved["probabilities"], saved["labels"])
            assert np.array_equal(saved["original_indices"], np.arange(10000))
        for name in ("accuracy", "nll", "brier", "ece15", "mean_confidence"):
            assert abs(metric[name] - run[name]) < 1e-12
        checked.append(path.name)
    indices = np.load(root / "split_indices.npz")
    assert len(np.unique(np.concatenate([indices[k] for k in ("train", "validation", "unused")]))) == 60000
    assert len(indices["train"]) + len(indices["validation"]) + len(indices["unused"]) == 60000
    # Confirm each checkpoint is the minimum NLL epoch using the archived log.
    import csv
    rows = list(csv.DictReader((root / "training_log.csv").open()))
    selections = json.loads((root / "selected_models.json").read_text())
    for selection in selections:
        candidates = [r for r in rows if int(r["seed"]) == selection["seed"] and r["model"] == selection["model"]]
        chosen = min(candidates, key=lambda r: float(r["val_nll"]))
        assert int(chosen["epoch"]) == selection["selected_epoch"]
        assert abs(float(chosen["val_nll"]) - selection["validation_nll"]) < 1e-12
    return {"prediction_files_recomputed": checked, "all_metrics_match": True,
            "all_checkpoints_selected_by_validation_nll": True, "split_partition_valid": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"mathematical_tests_run": result.testsRun, "passed": result.wasSuccessful(),
              "failures": len(result.failures), "errors": len(result.errors)}
    if args.results and result.wasSuccessful():
        report["results_audit"] = audit_results(args.results)
        (args.results / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)
