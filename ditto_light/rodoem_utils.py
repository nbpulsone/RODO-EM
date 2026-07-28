import os
import numpy as np
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils import data

""" ~~~ domain robustness optimization helpers ~~~ """
def get_objective_k(opt):
    if opt is None:
        return 1

    parts = opt.split("-")

    if len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1])

    return 1

def median_k_idx(values, k):
    values = np.array(values, dtype=np.float64)
    k = min(k, len(values))
    sorted_idx = np.argsort(values)
    start = max(0, (len(values) - k) // 2)
    return sorted_idx[start:start + k]


def median_proximity_weights_torch(values, eta):
    """Smoothly upweight values close to the median."""
    values = values.float()
    med = torch.median(values)
    scale = torch.std(values).clamp(min=1e-6)
    dist = torch.abs(values - med) / scale
    return torch.softmax(-eta * dist, dim=0)

"""Parity-based helpers"""
def safe_population_variance(values):
    """Population variance that is safe when only one domain is in a batch."""
    if values.numel() <= 1:
        return values.new_tensor(0.0)

    return torch.mean((values - values.mean()) ** 2)


def normalized_entropy(values, eps=1e-8):
    """Return entropy normalized to [0, 1].

    A value of 1 means the mass is distributed uniformly across domains.
    """
    if values.numel() <= 1:
        return values.new_tensor(1.0)

    values = values.clamp(min=eps)
    probs = values / values.sum().clamp(min=eps)

    entropy = -(probs * torch.log(probs.clamp(min=eps))).sum()
    max_entropy = torch.log(
        values.new_tensor(float(values.numel()))
    ).clamp(min=eps)

    return entropy / max_entropy


def soft_binary_domain_metrics(prediction, y, domain_ids, eps=1e-8):
    """Compute differentiable PPV, TPR, and soft F1 for each domain.

    Positive-class probabilities replace hard binary predictions so that the
    resulting parity penalties remain differentiable.
    """
    positive_probs = torch.softmax(prediction, dim=1)[:, 1]
    y_float = y.float()

    domain_ppvs = []
    domain_tprs = []
    domain_f1s = []

    for d in torch.unique(domain_ids):
        mask = domain_ids == d

        probs_d = positive_probs[mask]
        labels_d = y_float[mask]

        soft_tp = torch.sum(probs_d * labels_d)
        soft_fp = torch.sum(probs_d * (1.0 - labels_d))
        soft_fn = torch.sum((1.0 - probs_d) * labels_d)

        ppv = soft_tp / (soft_tp + soft_fp + eps)
        tpr = soft_tp / (soft_tp + soft_fn + eps)
        soft_f1 = (2.0 * ppv * tpr) / (ppv + tpr + eps)

        domain_ppvs.append(ppv)
        domain_tprs.append(tpr)
        domain_f1s.append(soft_f1)

    return (
        torch.stack(domain_ppvs),
        torch.stack(domain_tprs),
        torch.stack(domain_f1s),
    )


def parity_training_loss(
    prediction,
    y,
    domain_ids,
    group_losses,
    parity_metric,
    parity_weight,
):
    """Return classification loss plus a domain-parity penalty."""
    base_loss = group_losses.mean()

    domain_ppvs, domain_tprs, domain_f1s = soft_binary_domain_metrics(
        prediction,
        y,
        domain_ids,
    )

    if parity_metric == "variance":
        # Equalize domain classification losses.
        parity_penalty = safe_population_variance(group_losses)

    elif parity_metric == "entropy":
        # High entropy means soft domain F1 is distributed evenly.
        # Convert this maximization objective into a minimization penalty.
        parity_penalty = 1.0 - normalized_entropy(domain_f1s)

    elif parity_metric == "ppvp":
        parity_penalty = safe_population_variance(domain_ppvs)

    elif parity_metric == "tprp":
        parity_penalty = safe_population_variance(domain_tprs)

    else:
        raise ValueError(
            f"Unknown parity metric: {parity_metric}. "
            "Expected entropy, variance, ppvp, or tprp."
        )

    return base_loss + parity_weight * parity_penalty


def objective_indices(domain_objective, losses, domain_sizes, k):
    if domain_objective == "best":
        return np.argsort(losses)[:k]
    elif domain_objective == "worst":
        return np.argsort(losses)[-k:]
    elif domain_objective == "median":
        return median_k_idx(losses, k)
    elif domain_objective == "biggest":
        return np.argsort(domain_sizes)[-k:]
    elif domain_objective == "smallest":
        return np.argsort(domain_sizes)[:k]
    elif domain_objective == "median_size":
        return median_k_idx(domain_sizes, k)
    else:
        return np.arange(len(losses))


# compute average f1 across domains of those domains with the 3 worst losses
def avg_f1_for_worst_k(losses, f1s, k):
    losses = np.array(losses, dtype=np.float64)
    f1s = np.array(f1s, dtype=np.float64)
    k = min(k, len(losses))
    worst_idx = np.argsort(losses)[-k:]
    return float(np.mean(f1s[worst_idx]))

# compute average f1 across domains of those domains with the worst 3 f1s
def avg_f1_for_lowest_k_f1(f1s, k):
    f1s = np.array(f1s, dtype=np.float64)
    k = min(k, len(f1s))
    worst_idx = np.argsort(f1s)[:k]
    return float(np.mean(f1s[worst_idx]))


def avg_f1_for_highest_k_f1(f1s, k):
    f1s = np.array(f1s, dtype=np.float64)
    k = min(k, len(f1s))
    best_idx = np.argsort(f1s)[-k:]
    return float(np.mean(f1s[best_idx]))


def avg_best_k_loss(values, k):
    values = np.array(values, dtype=np.float64)
    k = min(k, len(values))
    return float(np.mean(np.sort(values)[:k]))


def summarize_domain_f1s(f1s):
    f1s = np.array(f1s, dtype=np.float64)
    sorted_idx = np.argsort(f1s)

    return {
        "per_domain_f1s": f1s.tolist(),
        "worst_domain_idx": int(sorted_idx[0]),
        "worst_domain_f1": float(f1s[sorted_idx[0]]),
        "median_domain_idx": int(sorted_idx[len(sorted_idx) // 2]),
        "median_domain_f1": float(f1s[sorted_idx[len(sorted_idx) // 2]]),
        "best_domain_idx": int(sorted_idx[-1]),
        "best_domain_f1": float(f1s[sorted_idx[-1]]),
    }


def save_line_plot(values, ylabel, title, path, best_epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(values) + 1)
    plt.figure()
    plt.plot(epochs, values, marker="o")
    # highlight early-stopped checkpoint
    if best_epoch is not None:
        plt.scatter(
            best_epoch,
            values[best_epoch - 1],
            color="red",
            marker="*",
            s=250,
            zorder=10,
            label="Selected checkpoint"
        )
        plt.legend()

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_domain_weighted_sampler(trainset, dom_losses, dom_f1s=None, resample_by='loss'):
    """ build a WeightedRandomSampler that over/under-represents certain domains.

    objective is inferred from hp.opt suffix, e.g.
      robust-dro-worst
      robust-dro-best
      robust-size-biggest
      robust-size-smallest
    """
    if trainset.domain_labels is None or dom_losses is None:
        return None

    domain_objective = getattr(trainset.hp, "opt", "robust-dro-worst").split('-')[-1]

    # count training samples per domain
    n_domains = len(dom_losses)
    domain_counts = torch.zeros(n_domains, dtype=torch.float32)
    for dom_id in trainset.domain_labels:
        domain_counts[int(dom_id)] += 1.0

    if resample_by == 'f1' and dom_f1s is not None:
        f1_tensor = torch.tensor(dom_f1s, dtype=torch.float32)

        if domain_objective == "best":
            weights = f1_tensor.clamp(0.0, 1.0)
        else:
            weights = 1.0 - f1_tensor.clamp(0.0, 1.0)

    elif resample_by == 'loss':
        loss_tensor = torch.tensor(dom_losses, dtype=torch.float32)

        if domain_objective == "best":
            weights = 1.0 / loss_tensor.clamp(min=1e-6)
        elif domain_objective == "median":
            weights = median_proximity_weights_torch(
                loss_tensor,
                getattr(trainset.hp, "robust_weight", 5.0)
            )
        else:
            weights = loss_tensor

    elif resample_by == 'size':
        # resample proportional or inversely proportional to domain size
        if domain_objective == "biggest":
            weights = domain_counts
        elif domain_objective == "median_size" or domain_objective == "median":
            weights = median_proximity_weights_torch(
                domain_counts,
                getattr(trainset.hp, "robust_weight", 5.0)
            )
        else:
            weights = 1.0 / domain_counts.clamp(min=1.0)
    else:
        weights = torch.ones(n_domains, dtype=torch.float32)

    mean_w = weights.mean()
    weights = weights / mean_w if mean_w > 0 else torch.ones_like(weights)

    sample_weights = torch.zeros(len(trainset), dtype=torch.float32)
    for i, dom_id in enumerate(trainset.domain_labels):
        sample_weights[i] = weights[int(dom_id)]

    return data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(trainset),
        replacement=True
    )

class BalancedDomainBatchSampler(data.Sampler):
    # create batches with approximately equal # samples from each domain

    def __init__(self, domain_labels, batch_size, drop_last=False):
        self.domain_labels = list(domain_labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.domain_to_indices = {}
        for idx, d in enumerate(self.domain_labels):
            self.domain_to_indices.setdefault(int(d), []).append(idx)

        self.domains = sorted(self.domain_to_indices.keys())
        self.n_domains = len(self.domains)

        if self.n_domains == 0:
            raise ValueError("No domain labels found for balanced batching.")

        self.num_batches = len(self.domain_labels) // batch_size
        if not drop_last and len(self.domain_labels) % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        per_domain = self.batch_size // self.n_domains
        remainder = self.batch_size % self.n_domains

        for _ in range(self.num_batches):
            batch = []
            random.shuffle(self.domains)

            for j, d in enumerate(self.domains):
                n_take = per_domain + (1 if j < remainder else 0)
                candidates = self.domain_to_indices[d]

                # sample with replacement so small domains can still appear every batch
                batch.extend(random.choices(candidates, k=n_take))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
