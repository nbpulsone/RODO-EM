import torch 
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

def evaluate_with_threshold_search(model, iterator):
    """return best F1 and threshold for one dataset."""
    all_y = []
    all_probs = []

    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    best_th = 0.5
    best_f1 = 0.0

    for th in np.arange(0.0, 1.0, 0.05):
        pred = [1 if p > th else 0 for p in all_probs]
        new_f1 = metrics.f1_score(all_y, pred)
        if new_f1 > best_f1:
            best_f1 = new_f1
            best_th = th

    return best_f1, best_th

def evaluate_binary_metrics(model, iterator, threshold):
    """return thresholded binary metrics for one domain."""
    all_y = []
    all_probs = []

    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_y.extend(y.cpu().numpy().tolist())

    pred = np.asarray(
        [1 if p > threshold else 0 for p in all_probs],
        dtype=np.int64
    )
    labels = np.asarray(all_y, dtype=np.int64)

    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))

    ppv = tp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    f1 = metrics.f1_score(labels, pred, zero_division=0)

    return {
        "f1": float(f1),
        "ppv": float(ppv),
        "tpr": float(tpr),
    }

def evaluate_cc_partition(model, iterator, threshold, cc_only=False):
    """evaluate FPR on an all-negative CC set or F1 on a non-CC set."""
    all_y = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_y.extend(y.cpu().numpy().tolist())

    labels = np.asarray(all_y, dtype=np.int64)
    pred = np.asarray(
        [1 if p > threshold else 0 for p in all_probs],
        dtype=np.int64
    )

    if cc_only:
        negatives = labels == 0
        n_negatives = int(np.sum(negatives))

        if n_negatives == 0:
            raise ValueError(
                "CC FPR cannot be computed because the CC test set "
                "contains no negative examples."
            )

        false_positives = int(np.sum((pred == 1) & negatives))

        return {
            "fpr": false_positives / n_negatives,
            "false_positives": false_positives,
            "negative_examples": n_negatives,
        }

    return {
        "f1": float(metrics.f1_score(labels, pred, zero_division=0)),
        "examples": int(len(labels)),
    }


def normalized_entropy_numpy(values, eps=1e-12):
    """normalized entropy in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)

    if len(values) <= 1:
        return 1.0

    values = np.clip(values, eps, None)
    probabilities = values / values.sum()

    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(values)))


def calculate_parity_statistics(domain_metrics):
    """calculate evaluation-time disparity and parity scores."""
    f1s = np.asarray(
        [metric["f1"] for metric in domain_metrics],
        dtype=np.float64
    )
    ppvs = np.asarray(
        [metric["ppv"] for metric in domain_metrics],
        dtype=np.float64
    )
    tprs = np.asarray(
        [metric["tpr"] for metric in domain_metrics],
        dtype=np.float64
    )

    disparities = {
        "entropy": 1.0 - normalized_entropy_numpy(f1s),
        "variance": float(np.var(f1s)),
        "ppvp": float(np.var(ppvs)),
        "tprp": float(np.var(tprs)),
    }

    # Higher values are better, which matches the existing early-stopping code.
    scores = {
        name: 1.0 / (1.0 + disparity)
        for name, disparity in disparities.items()
    }

    return {
        "domain_ppvs": ppvs.tolist(),
        "domain_tprs": tprs.tolist(),
        "f1_variance": float(np.var(f1s)),
        "f1_entropy": normalized_entropy_numpy(f1s),
        "ppv_variance": float(np.var(ppvs)),
        "tpr_variance": float(np.var(tprs)),
        "parity_disparities": disparities,
        "parity_scores": scores,
    }

def evaluate_loss(model, iterator):
    """evaluate a model on a validation/test dataset, returning the raw loss value""" 
    all_y = []
    all_logits = []
    # calcualte cross entropy loss across the date provided by the iterator
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            #probs = logits.softmax(dim=1)[:, 1]
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

        # compute loss across all samples
        all_logits = torch.cat(all_logits, dim=0).to(model.device)
        all_y = torch.cat(all_y, dim=0).to(model.device)
        loss = criterion(all_logits, all_y)

    return loss.item()

def update_ema_losses(prev_ema, current_losses, alpha=0.8):
    """use exponential moving average to update current loss"""
    current_losses = np.array(current_losses, dtype=np.float64)
    if prev_ema is None:
        return current_losses
    return alpha * np.array(prev_ema, dtype=np.float64) + (1.0 - alpha) * current_losses
