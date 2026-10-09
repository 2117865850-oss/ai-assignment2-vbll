# Independent algorithm audit — CISC3024 Assignment 2

Audit date: 2026-10-08. Reviewer: separate AI research/audit agent.

## Status

**Passed implementation review: no blocking algorithm or data-leakage defect found.** Reviewed the installed official package and the local `model.py`, `train.py`, and `metrics.py`, then ran independent numerical checks. Training outcome and report claims remain subject to review after actual results exist. No experiment implementation files were changed by this reviewer.

## Official package and numerical cross-check

- Installed package is `vbll==0.4.9`.
- Reviewed `vbll/layers/classification.py` and `vbll/utils/distributions.py` in `work/experiment-env/lib/python3.12/site-packages`.
- Independently computed the diagonal D-VBLL Jensen bound, Gaussian KL and learned-noise prior in CPU float64 for a random 7-example, 5-feature, 3-class batch (`torch.manual_seed(710)`, regularization 1/100).
- Manual loss: 1.7432258160012406. Official `_get_train_loss_fn` loss: 1.7432258160012406. Absolute difference: 0.0.
- `prior_scale=1` yields actual Gaussian weight prior covariance `2/d`; test with d=5 yielded 0.4.
- Verified that the analytic loss closure does not change the global torch RNG state, but the public classifier `forward()` consumes RNG draws because it also calculates Monte Carlo predictive probabilities.

## Concrete checks for the local experiment

1. `W_logdiag` and `noise_logdiag` are log standard deviations. Variance is `exp(2 * logdiag)`.
2. Diagonal logit mean and variance must be `x @ W_mean.T` and `x.square() @ exp(2*W_logdiag).T + exp(2*noise_logdiag)`.
3. Minimized objective must include `logsumexp(mean + variance/2) - mean_y`, Gaussian KL divided by training N, and the package's noise-prior term. The package omits the constant term in Gaussian KL; absolute training objective must not be called test NLL.
4. Prediction averages softmax probabilities from Gaussian logit samples; it must not apply softmax to the averaged logits. Per-example marginal logit samples are sufficient for the pointwise accuracy/NLL/Brier/ECE metrics.
5. Fair comparison requires matching initial backbone parameters and deterministic head weights to Bayesian head mean. Identical seed alone is insufficient if constructors consume randomness differently.
6. Shuffling must use a separate generator; validation Monte Carlo must not change training data order or model randomness. `torch.random.fork_rng()` does not automatically isolate MPS in all versions; local CPU generators are simple and auditable.
7. Fixed stratified train/validation indices must be disjoint, within the original training split. Test must stay untouched until the declared checkpoint selection and hyperparameters are fixed.
8. Report predictive NLL from actual normalized averaged probabilities; multiclass Brier must use one declared convention (sum over classes is recommended), and ECE bin convention/count must be stated.
9. Record package versions, final hyperparameters, seeds, actual sample counts, selected epoch, and model/initialization equivalence checks. Distinguish this compact Fashion-MNIST adaptation from the original CIFAR/WideResNet experiments.

## Final code review

### Objective and model

- `model.py` calls the official `DiscClassification(64,10,regularization_weight=1/12000,parameterization='diagonal',softmax_bound='jensen',prior_scale=1,wishart_scale=1,dof=1)`.
- The baseline has the same 64-dimensional CNN features and a bias-free 10-class linear head. `matched_pair` copies the feature extractor and initializes deterministic head weights exactly to the Bayesian posterior mean, with equality assertions.
- The official analytic loss closure is used directly, avoiding the public `forward` method's unnecessary MC draws. Pinned package version and source hash are recorded.
- Local model loss was independently recomputed from `W_mean`, `W_logdiag`, `noise_logdiag`, prior variance and the Jensen expression. For a 7-example synthetic image batch (seed803), manual and local losses were both **2.571376323699951**, absolute difference **0.0**.
- No weight decay is applied to either model. This is consistent with the logged config, but is a simplification relative to the paper's feature-weight MAP-prior discussion and must not be described as an exact full experimental reproduction.

### Predictive distribution and random streams

- `predict` computes features once per image batch, moves a copied head and features to CPU, and calls official `head.predictive(..., n_samples=128)` for D-VBLL. Thus logits have the correct Gaussian marginal mean/variance, followed by averaging of softmax probabilities.
- CPU evaluation randomness is isolated with `torch.random.fork_rng(devices=[])` and a CPU generator's explicit state. It neither reseeds nor samples MPS RNG. The architecture contains no stochastic dropout or random augmentation.
- Every epoch uses a dedicated CPU shuffling generator seeded by `seed*10000+epoch`, and both models consume the same minibatch inside the same loop.
- Independently confirmed that `predict` preserves global CPU RNG state and returns byte-identical predictions under a repeated evaluation seed. Maximum probability row-sum error on the audit batch was 5.960464477539063e-08.
- Validation seed is fixed across epochs for each training seed; test seed is distinct from validation seed. Clean and noisy conditions deliberately share predictive standard-normal draws, reducing avoidable sampling variation in the paired condition comparison.

### Split, model selection, and leakage

- Independently loaded original training labels and recomputed the exact stratified split. There are **12,000 train / 2,000 validation**, **zero overlap**, **1,200 train and 200 validation examples per class**, and all indices lie within the original 60,000-example training split.
- Input normalization uses fixed constants (0.5,0.5), so no normalization statistics are estimated from test data.
- Best checkpoint is selected solely by minimum validation predictive NLL, with strict `<` comparison preserving the earlier epoch on an exact tie.
- All six checkpoint selections finish before the test dataset is loaded for prediction. The earlier file-hash manifest reads raw test-file bytes for provenance, but performs no label/model evaluation and creates no statistical leakage.
- Gaussian corruption is defined in the saved configuration before training and generated after selection with a separate fixed seed. Corruption is applied in [0,1] pixel space, clipped there, then normalized.

### Independent metric checks

Metrics use float64 accumulation. NLL clips true-class probabilities to [1e-12,1]; Brier is the **sum over all 10 classes** averaged over examples; ECE is top-label confidence ECE with **15 equal-width bins**, [lower,upper) except the last bin includes1.

| Input probabilities | Accuracy | NLL | Brier | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| Uniform probabilities, one example per class | 0.1 | 2.302585092994046 | 0.9000000000000001 | 0.0 |
| Perfect one-hot predictions | 1.0 | 0.0 | 0.0 | 0.0 |
| Certain incorrect predictions | 0.0 | 27.631021115928547 | 2.0 | 1.0 |

For an additional independent random probability matrix (117 examples), explicit Python-loop calculations agreed with production metrics to absolute errors **0 accuracy, 0 NLL, 3.33e-16 Brier, 2.78e-17 ECE**. Confusion-matrix orientation in code is true label by predicted label.

### Nonblocking reporting notes sent to implementation and parent agents

1. Prefer **pre-specified protocol** over **preregistered** unless an actual external preregistration exists.
2. State that training uses the deterministic Jensen objective, while evaluation uses128 final-layer samples; the whole network is not Bayesian.
3. Treat this as a compact adaptation with no explicit CNN weight decay, diagonal weight covariance, a different dataset/backbone and three seeds. Do not promise or generalize accuracy/calibration gains.
4. Report sample standard deviation across three seeds as run variation, not a confidence interval or a significance result.
