# Discriminative VBLL on FashionMNIST

A reproducible, small-data adaptation of **Variational Bayesian Last Layers**
(Harrison, Willes and Snoek, ICLR 2024). This project uses the official
`vbll==0.4.9` implementation and compares a CNN with a diagonal discriminative
VBLL head against a matched deterministic CNN. It is not a reproduction of the
paper's WideResNet/CIFAR benchmark and does not establish state-of-the-art results.

## Run

Python 3.12 was used. Install the pinned requirements in a virtual environment:

```bash
python -m pip install -r requirements.txt
python test_verification.py
python train.py --data-root data --download --device mps
python test_verification.py --results results
python plot_results.py
python diagnose_results.py
```

Use `--device cpu` on machines without Apple MPS. The official FashionMNIST files
are downloaded from the dataset author's GitHub repository and checked against
torchvision's expected MD5 values. For an existing dataset, pass its parent folder
using `--data-root PATH` and omit `--download`. The folder must contain
`FashionMNIST/raw/`. Dataset files are excluded from this submission.

To keep the submitted run intact, write a new run to `--output rerun_results`.
`plot_results.py` and `diagnose_results.py` read the submitted `results/` by default.
The recorded run reused the local dataset with downloads disabled. Subsequent
source edits only improved portability, documentation and diagnostics; they did
not change the recorded model, loss, split, optimizer or evaluation calculations.

## Fixed experimental protocol

| Setting | Value |
|---|---|
| Dataset | FashionMNIST, 28 × 28 grayscale, ten classes |
| Training subset | 12,000; exactly 1,200 per class |
| Validation subset | 2,000; exactly 200 per class |
| Split | Seed 2026, disjoint subsets of the official 60,000 training images |
| Unused training images | 46,000 |
| Test set | All 10,000 official test images |
| Training seeds | 42, 43, 44 |
| Backbone | Conv1→16/ReLU/Pool2; Conv16→32/ReLU/Pool2; flatten1568→64/ReLU |
| Last-layer bias | None in either model |
| Initialization | Exact same backbone parameters and last-layer mean in each pair |
| Batch order | Identical within each model pair in every epoch |
| Training | Adam, learning rate 0.001, batch 128, ten epochs |
| Explicit feature weight prior / weight decay | None; an adaptation from the paper's general MAP-feature formulation |
| Input | `x/255`, then `(x−0.5)/0.5`; no augmentation |
| Selection | Lowest validation predictive NLL independently for each model; earliest exact tie |
| Prediction | 128 Monte Carlo draws for D-VBLL; deterministic softmax for baseline |
| Evaluation random state | CPU RNG fork, fixed per seed; validation seed reset each epoch |
| Additional test condition | Gaussian noise σ=0.2 in [0,1], clip to [0,1], then normalize |
| Noise random state | Fixed CPU generator seed 20261008; same noisy inputs for all models |

All six checkpoint choices were fixed before test-set evaluation. The reported
test metrics were not used for hyperparameter, epoch or seed selection.
There is one fixed validation split; the three seeds vary initialization and
training order, not the data split. MPS training is not guaranteed to reproduce
bit-for-bit across software/hardware versions.

## Bayesian head and objective

The head is `vbll.DiscClassification(64, 10, regularization_weight=1/12000,
parameterization="diagonal", prior_scale=1, wishart_scale=1, dof=1)`.
The package scales the weight-prior covariance to `prior_scale × 2 / 64`, hence
the effective prior variance is **0.03125**. It converts the Wishart coefficient
to `(dof + number_of_classes + 1)/2 = 6`. Both posterior weight and logit-noise
`logdiag` parameters are **log standard deviations**, not log variances.

For feature vector φ, the marginal logits have mean
`μ = W_mean φ` and diagonal variance
`v = φ² @ exp(2 W_logdiag).T + exp(2 noise_logdiag)`.
The analytic classification loss upper bound is
`logsumexp(μ + v/2) − μ_y`. The official objective adds the scaled Gaussian
weight KL and subtracts the scaled Wishart log prior for logit-noise precision.
The official Gaussian KL implementation omits an additive parameter-independent
constant. Consequently the recorded VBLL training objective is not NLL and its
absolute value must not be compared with baseline cross entropy as the same loss.

Training calls the pinned official `_get_train_loss_fn(features)(labels)`.
This avoids the unnecessary predictive draws produced by the public `forward`
method; it does not alter the official objective. Training the analytic bound is
sampling-free. **Prediction is not sampling-free**: `head.predictive(features,
n_samples=128)` samples marginal Gaussian logits, applies softmax and averages the
probabilities. The feature extractor is deterministic; only the last-layer
weights have a variational posterior. Noise scale parameters are point estimates
under their prior. This is not a fully Bayesian convolutional network.

The baseline uses 105,856 trainable parameters; D-VBLL uses 106,506. Both include
the same 105,216-parameter backbone. The baseline uses cross entropy and no
regularizer; the method-specific Bayesian priors are intrinsic to D-VBLL.

## Recorded results

Mean ± sample standard deviation across three seeds. Accuracy and ECE are
percentages; NLL and Brier are unitless. Brier is the mean **sum** over ten classes.

| Condition | Model | Accuracy % ↑ | NLL ↓ | Brier ↓ | ECE % ↓ |
|---|---|---:|---:|---:|---:|
| Clean | Deterministic CNN | 87.763 ± 0.091 | 0.34510 ± 0.00148 | 0.17668 ± 0.00077 | 0.872 ± 0.071 |
| Clean | CNN + D-VBLL | 86.010 ± 3.274 | 0.45064 ± 0.18460 | 0.20678 ± 0.05237 | 2.358 ± 2.615 |
| Gaussian σ=0.2 | Deterministic CNN | 74.653 ± 0.741 | 0.66887 ± 0.01819 | 0.34811 ± 0.00966 | 6.598 ± 1.217 |
| Gaussian σ=0.2 | CNN + D-VBLL | 78.210 ± 3.781 | 0.61936 ± 0.09862 | 0.30535 ± 0.04800 | 3.458 ± 2.821 |

D-VBLL is worse on the clean-test average and better on the chosen noisy-test
average, with substantial seed variability. Seed 44 converges much more slowly
within the fixed ten-epoch budget. A post hoc inspection finds that its initial
Shirt-class logit-noise variance is 46.36 and remains 20.45 at the selected epoch;
the largest initial noise variances for seeds 42 and 43 are about 1.15 and 1.10.
This observation is consistent with initialization sensitivity, but is not a
causal test. All seeds are retained. Three seeds, one small dataset and one noise
severity are insufficient to establish general calibration or robustness gains.
Sample SD is a descriptive measure, not a confidence interval.

Selected epochs: baseline **8 / 9 / 10**; D-VBLL **9 / 9 / 10** for seeds
42 / 43 / 44. The full training and test run took **123.6 seconds** on Apple M4,
24 GiB memory, with MPS training and CPU posterior-predictive sampling. This is
an observed wall-clock time, not a hardware-independent speed benchmark.

## Metrics and verification

- Accuracy: fraction of correct maximum-probability predictions.
- NLL: average `−log(p_true)` with a numerical floor of `1e−12`.
- Brier: mean `sum_k (p_k − one_hot(y)_k)²`, with no division by ten.
- ECE: top-label calibration error using 15 equal-width confidence bins,
  sample-count weighting, and confidence 1 included in the last bin.

Seven independent checks verify the loss against an explicit diagonal formula,
the Jensen upper bound against 200,000 Monte Carlo samples, the zero-variance
cross-entropy limit, matched initialization, normalized/repeatable predictions
with RNG preservation, disjoint stratified indices, and known metric cases.
The saved-results audit recomputes every test metric from all twelve prediction
files and verifies each selected epoch against the validation log. Checks pass.

Reliability and confusion figures pool predictions across all three seeds
(30,000 predictions of the same 10,000 images). This is a descriptive pool, not
an ensemble, not 30,000 independent test images, and not the same calculation as
averaging three per-run ECE values.

## Files

- `model.py`, `train.py`, `metrics.py`: model, experiment and metric implementation.
- `test_verification.py`: mathematical checks and retained-result audit.
- `plot_results.py`, `diagnose_results.py`: figures and post hoc diagnostics.
- `results/config.json`, `environment.json`: configuration and package provenance.
- `results/split_indices.npz`, `split_manifest.json`: indices and data/split hashes.
- `results/batch_order_hashes.json`, `initialization_checks.json`: paired controls.
- `results/training_log.csv`, `selected_models.json`: full validation history and choices.
- `results/metrics.csv`, `summary.json`, `results.json`: results; the last two are aliases.
- `results/predictions/`: per-example probabilities, labels and original indices.
- `results/checkpoints/`: six selected CPU state dictionaries and selection metadata.
- `results/checks.json`, `diagnostics.json`: validation and descriptive inspection.
- `figures/`: four figures, each in PNG and editable SVG.
- `experiment.log`: actual training/evaluation output, with local path removed.

## Sources

- Paper: https://arxiv.org/abs/2404.11599
- Official implementation: https://github.com/VectorInstitute/vbll
- Pinned package: https://pypi.org/project/vbll/0.4.9/
- FashionMNIST: https://github.com/zalandoresearch/fashion-mnist

The experiment runner, paired control, metric code and plots were written for
this assignment. The probabilistic layer is imported from the official package;
its authors retain credit for the algorithm and implementation. The paper's
theory is used directly; this report's measured results come from the saved run.
