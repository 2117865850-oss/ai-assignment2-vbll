"""Matched CNNs with the official, pinned discriminative VBLL layer."""
from copy import deepcopy
import torch
from torch import nn
import vbll


class Backbone(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
        )


class Classifier(nn.Module):
    def __init__(self, kind, n_train=12000):
        super().__init__()
        self.kind = kind
        self.backbone = Backbone()
        if kind == "vbll":
            self.head = vbll.DiscClassification(
                64, 10, regularization_weight=1.0 / n_train,
                parameterization="diagonal", softmax_bound="jensen",
                prior_scale=1.0, wishart_scale=1.0, dof=1.0,
            )
        elif kind == "baseline":
            self.head = nn.Linear(64, 10, bias=False)
        else:
            raise ValueError(kind)

    def training_loss(self, images, labels):
        features = self.backbone(images)
        if self.kind == "baseline":
            return nn.functional.cross_entropy(self.head(features), labels)
        # Official vbll==0.4.9 deterministic Jensen objective. Its public
        # forward also draws predictions unnecessarily during training.
        return self.head._get_train_loss_fn(features)(labels)


def matched_pair(seed, n_train=12000):
    torch.manual_seed(seed)
    bayesian = Classifier("vbll", n_train)
    baseline = Classifier("baseline", n_train)
    baseline.backbone = deepcopy(bayesian.backbone)
    with torch.no_grad():
        baseline.head.weight.copy_(bayesian.head.W_mean)
    for a, b in zip(baseline.backbone.parameters(), bayesian.backbone.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(baseline.head.weight, bayesian.head.W_mean)
    return {"baseline": baseline, "vbll": bayesian}


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
