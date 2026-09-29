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


# ---------------------------------------------------------------------------
# "Unknown" threshold tuning (on VALIDATION data only -- never the test set)
# ---------------------------------------------------------------------------
# The final decision is: answer = top class, but "unknown" if confidence < threshold.
# The old fixed 0.60 was a guess. Label smoothing deliberately lowers confidence
# (targets 0.9 instead of 1.0), so a fixed cutoff starts rejecting real digits.
# Instead we pick the threshold that maximises BALANCED accuracy on the validation
# set: the average of per-class recall over all 11 classes, so rejecting non-digits
# counts as much as recognising each digit (plain accuracy would favour digits,
# which outnumber 'unknown' clips 14:1).

def apply_threshold(probs: torch.Tensor, threshold: float, unknown_idx: int) -> torch.Tensor:
    conf, pred = probs.max(dim=1)
    pred = pred.clone()
    pred[conf < threshold] = unknown_idx
    return pred


def decision_metrics(pred: torch.Tensor, labels: torch.Tensor, unknown_idx: int) -> dict:
    classes = labels.unique().tolist()
    recalls = [float((pred[labels == c] == c).float().mean()) for c in classes]
    is_unk = labels == unknown_idx
    return {
        "balanced_acc": sum(recalls) / len(recalls),
        "overall_acc": float((pred == labels).float().mean()),
        "digit_acc": float((pred[~is_unk] == labels[~is_unk]).float().mean()) if (~is_unk).any() else float("nan"),
        "digit_rejected": float((pred[~is_unk] == unknown_idx).float().mean()) if (~is_unk).any() else float("nan"),
        "unknown_rejection": float((pred[is_unk] == unknown_idx).float().mean()) if is_unk.any() else float("nan"),
    }


def tune_threshold(probs: torch.Tensor, labels: torch.Tensor, unknown_idx: int,
                   lo: float = 0.20, hi: float = 0.95, step: float = 0.01):
    """Best 'unknown' threshold on validation data. Returns (threshold, metrics, metrics_at_0.60).

    Among thresholds within 0.2 points of the best balanced accuracy, the middle one
    is chosen, so the result doesn't jump around on a flat curve.
    """
    grid = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    scores = [(t, decision_metrics(apply_threshold(probs, t, unknown_idx), labels, unknown_idx)) for t in grid]
    best = max(m["balanced_acc"] for _, m in scores)
    near = [t for t, m in scores if m["balanced_acc"] >= best - 0.002]
    t_best = near[len(near) // 2]
    m_best = dict(scores)[t_best]
    m_default = decision_metrics(apply_threshold(probs, 0.60, unknown_idx), labels, unknown_idx)
    return t_best, m_best, m_default
