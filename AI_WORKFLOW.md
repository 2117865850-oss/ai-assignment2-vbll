# AI workflow for Assignment 2

Date: 8 October 2026. This file records the actual workflow used for this project.

## User request

The user attached the CISC3024 AI Assignment 2 brief and wrote: “完成AI assignment 2” (complete AI Assignment 2). The brief asks for a recent probability- or Bayes-based classification/recognition algorithm and permits AI to perform the search, implementation, experiments and report writing. It requires six report components, including a source-code webpage.

## Research instructions

The lead AI agent asked a research agent to identify recent 2023–2026 Bayesian or probabilistic classification algorithms with primary papers and official code, and to recommend a method feasible for an image-classification experiment on the available computer. The following is a condensed instruction, not a verbatim user prompt:

> Find a recent probability- or Bayes-based classification algorithm. Verify its publication and official implementation, explain its central inference mechanism, and recommend a reproducible small image-classification experiment.

The search considered Variational Bayesian Last Layers (ICLR 2024), Bayesian neural additive models (ICML 2024), and a 2024 preprint on implicit-prior Bayesian last layers. VBLL was selected because its discriminative classification layer directly fits a small convolutional network, and both its mathematical objective and official implementation are available.

## Implementation instructions

The lead AI agent directed the implementation agent to use the official discriminative VBLL layer with a small CNN, compare it with a deterministic softmax head, retain identical initial shared weights and training batches, run three seeds, select checkpoints on validation NLL, and measure clean and corrupted test predictions. Training uses 12,000 examples, validation uses 2,000, and the held-out official test set uses 10,000. The Bayesian prediction procedure averages 128 last-layer logit samples.

The agent generated the surrounding model, dataset, training, evaluation, checks, plots and documentation. The Bayesian layer is attributed to the official `vbll` package; this project does not claim authorship of that upstream implementation. The student was not asked to write program code.

## Verification and reporting instructions

A separate agent reviewed the probability model, variational bound, official API, data splits, initialization, random-number handling and evaluation design. The lead agent checked the retained outputs against the report, packaged the reproducible source and published it to the user's previously authorized public GitHub repository. Another agent drafted and rendered the report in Word and PDF, covering all six required sections.

Results are measurements from local execution, not values copied from the paper or invented estimates. The report describes a small Fashion-MNIST adaptation; it does not reproduce the paper's original image-classification benchmark. The reflection is AI-written from the work performed and should be reviewed by the student before submission.

## Primary sources

- Paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html
- Paper full text: https://arxiv.org/html/2404.11599v1
- Official library: https://github.com/VectorInstitute/vbll
- Dataset: https://github.com/zalandoresearch/fashion-mnist
