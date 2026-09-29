"""
calibration.py
==============
Make the model's confidence numbers MEAN something.

Neural networks are usually over-confident: a model that says "93% sure" may
only be right 80% of the time. That matters here because "unknown" is decided
by a confidence threshold (config.CONFIDENCE_THRESHOLD = 0.60).

TEMPERATURE SCALING (Guo et al., 2017) fixes this with ONE number T fitted on
the validation set after training:

    calibrated probabilities = softmax(logits / T)

T > 1 softens over-confident predictions, T < 1 sharpens under-confident ones.
The predicted class never changes (dividing all logits by the same T keeps
their order), so accuracy is identical -- only the confidence becomes honest.

ECE (expected calibration error) measures the mismatch: group predictions by
confidence, compare average confidence with actual accuracy in each group,
and average the gaps (0 = perfectly calibrated).
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for feats, y in loader:
        logits.append(model(feats.to(device)).float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


def expected_calibration_error(logits, labels, temperature: float = 1.0, n_bins: int = 15) -> float:
    probs = F.softmax(logits / temperature, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    ece = torch.zeros(1)
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(ece)


def fit_temperature(logits, labels) -> float:
    """Temperature T minimising the negative log-likelihood of the validation set."""
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T so T stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def calibrate_model(model, loader, device):
    """Fit T on `loader` (validation data). Returns (T, ece_before, ece_after)."""
    logits, labels = collect_logits(model, loader, device)
    t = fit_temperature(logits, labels)
    before = expected_calibration_error(logits, labels)
    after = expected_calibration_error(logits, labels, t)
    if after > before:
        # Can happen on a small validation set (T is fitted to the likelihood, not ECE).
        # Don't make things worse: keep the raw softmax.
        return 1.0, before, before
    return t, before, after
