# CISC3024 Assignment 2 — research and implementation notes

Research date: 2026-10-08.

## Recommendation

Select **Variational Bayesian Last Layers (VBLL)**, James Harrison, John Willes, Jasper Snoek, ICLR 2024. Implement the **discriminative** version, D-VBLL, with a small deterministic CNN and a diagonal Gaussian posterior on the final layer. Compare against the same CNN with a deterministic softmax head. This gives a direct Bayesian classification algorithm with a current formal publication and maintained official PyTorch code. All experimental numbers must come from the local run.

## Candidate comparison

| Candidate | Primary evidence | Feasibility and decision |
| --- | --- | --- |
| Variational Bayesian Last Layers, ICLR 2024 | Proceedings https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html ; official code https://github.com/VectorInstitute/vbll | Selected. Direct image-classification evaluation in paper; small Bayesian output layer can be attached to an existing CNN. |
| Improving Neural Additive Models with Bayesian Principles (LA-NAM), ICML 2024 | https://proceedings.mlr.press/v235/bouchiat24a.html ; official code https://github.com/fortuinlab/LA-NAM | Valid Bayesian pattern recognition candidate. Uses a Laplace posterior and empirical-Bayes feature selection. Its official implementation targets tabular regression/binary classification; adapting to ten-class image pixels is less natural and adds scope. |
| Flexible Bayesian Last Layer Models Using Implicit Priors and Diffusion Posterior Sampling, 2024 preprint | https://arxiv.org/abs/2408.03746 | Bayesian last-layer extension with implicit priors and diffusion sampling. Formal publication and official runnable repository not established in this short verification, so not recommended for this assignment. |

## Verified bibliographic sources

1. ICLR 2024 official proceedings page: https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html
2. ICLR paper PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/ee56aa4fe26a189782f507d843fd5272-Paper-Conference.pdf
3. Authors' arXiv full text, version 1, 17 April 2024: https://arxiv.org/html/2404.11599v1
4. Official implementation, MIT license: https://github.com/VectorInstitute/vbll
5. Exact classifier source inspected: https://raw.githubusercontent.com/VectorInstitute/vbll/main/vbll/layers/classification.py
6. Official documentation: https://vbll.readthedocs.io/en/latest/
7. Fashion-MNIST official dataset repository: https://github.com/zalandoresearch/fashion-mnist

Use a pinned package version or exact Git commit in the final code/environment record because GitHub main can change.

## Algorithm essentials and equations

Let the deterministic CNN produce features h = phi_theta(x), with feature dimension d and K classes. Bayesian inference concerns the final weight matrix W, while theta is optimized as a point estimate. Bayes' rule motivates p(W|D) proportional to p(D|W)p(W). VBLL optimizes a tractable Gaussian approximation q(W) rather than integrating the exact posterior.

The discriminative model and variational family (paper equations 3 and 14):

    p(y=k|x,W,epsilon) = softmax(W h + epsilon)_k
    epsilon ~ N(0,Sigma), Sigma = diag(sigma_1^2,...,sigma_K^2)
    p(w_k) = N(0, alpha I)
    q(W) = product_k N(w_k; m_k,S_k)
    mu_k = m_k^T h
    v_k = h^T S_k h + sigma_k^2

With one-hot labels and N training examples, the deterministic per-example log-likelihood lower bound is:

    b(x,y) = mu_y - log sum_k exp(mu_k + v_k/2).

The minimized minibatch objective is (paper equations 14 and 16):

    loss = -mean_batch b(x,y)
           + ( KL[q(W)||p(W)] - log p(Sigma) ) / N
           + feature-weight regularization.

The Gaussian KL term, including the constant often omitted in code, is:

    KL = 1/2 sum_k [ tr(S_k)/alpha + ||m_k||^2/alpha
                    - d + d log(alpha) - log det(S_k) ].

For a diagonal posterior, S_k = diag(s_k1^2,...,s_kd^2), so

    h^T S_k h = sum_j h_j^2 s_kj^2.

The bound is below the ordinary ELBO because Jensen's inequality is applied; it is not an exact marginal likelihood. Diagonal S_k also discards within-class posterior correlations. Prediction (paper equation 19) averages softmax probabilities under the learned posterior. For D-VBLL the paper permits inexpensive sampling only at the last layer:

    p_hat(y=k|x,D) = (1/M) sum_m softmax(z^(m))_k,
    z_k^(m) = mu_k + sqrt(v_k) epsilon_k^(m), epsilon^(m) ~ N(0,I).

**Important wording:** D-VBLL has a sampling-free *training bound* and a single CNN feature pass. Its classification posterior predictive is estimated using last-layer sampling. Do not describe all evaluation as sampling-free or claim that the full neural network is Bayesian.

## Official implementation findings

The official class is `vbll.DiscClassification`. Use `in_features=d`, `out_features=10`, `regularization_weight=1/N`, `parameterization='diagonal'`. The returned object's `train_loss_fn(labels)` computes the training loss, and `predictive.probs` provides class probabilities. The source's `predictive(features, n_samples=20)` defaults to 20 logit samples; explicitly choose and record a larger fixed evaluation count such as 100.

The source scales its weight prior covariance as `prior_scale * (2 / in_features)`. `W_logdiag` and `noise_logdiag` parameterize standard deviations after exponentiation, not variances. The loss includes a Gaussian KL and a learned-noise covariance prior term. Its Gaussian KL omits an additive constant, so its absolute training loss should not be presented as a conventional NLL. For held-out NLL, use the actual averaged predictive probabilities.

The current package also contains newer classifier variants. Use the paper's `DiscClassification`, not `tDiscClassification`, unless intentionally expanding the research scope.

## Proposed local experiment (design choices, not paper claims)

- Dataset: Fashion-MNIST, 28 x 28 grayscale, 10 classes. Use a stratified 12,000 training examples and 2,000 validation examples selected only from the original training split; use the complete original 10,000-image test split. Record chosen indices and split seed.
- Use identical CNN feature architecture, preprocessing and optimizer settings for the deterministic and Bayesian heads. A 64-dimensional feature vector is ample for this compact experiment; a constant feature could represent a Bayesian bias if intentionally implemented and documented.
- Try 10 epochs and three seeds, subject to measured local speed. Report actual settings, not this proposed plan if changed.
- Fix all tuning with validation data; evaluate final selected checkpoints on the test split. Do not choose the better result or checkpoint based on test metrics.
- Report accuracy, NLL, multiclass Brier score, and ECE with a declared bin count. Reliability plots help interpret the probability estimates.
- A matched model comparison tests the adaptation under a small compute budget. It is not a reproduction of the paper's WideResNet-28-10 experiments on CIFAR-10/100 and cannot establish the general superiority of VBLL.
- Calibration or accuracy gains are not guaranteed. Report negative or mixed results faithfully, alongside per-seed values and mean/spread. Changing dataset, architecture, covariance approximation, hyperparameters and compute all limit transfer of the original claims.

## Suggested report structure matching the assignment

1. Finding the algorithm: describe AI-assisted literature search, why ICLR 2024 qualifies as recent, candidate screening, and why VBLL fits Bayesian classification and local compute.
2. Description: Bayes' rule, Gaussian prior/posterior, learned CNN features, Jensen lower bound, KL regularization, probability averaging; distinguish deterministic features and Bayesian head.
3. AI implementation: identify the exact official component used, local training/evaluation code generated by AI, checks performed, dependency versions, and any adaptation.
4. Settings and results: dataset/splits, seed protocol, hardware/software, optimization and posterior sampling count, actual metric table, learning/reliability plots, and a concise failure analysis.
5. Learning: suggested reflection themes are likelihood versus probability calibration, approximate inference, compute/uncertainty tradeoffs, and verification of AI code. The student must review and personalize any first-person reflection.
6. Source webpage: complete paper citation plus direct paper, repository, implementation, and dataset hyperlinks.

## Reference entry

Harrison, J., Willes, J., & Snoek, J. (2024). *Variational Bayesian Last Layers*. International Conference on Learning Representations (ICLR). https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html
