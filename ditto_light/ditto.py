import os
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

from .robust_eval_utils import *
from .rodoem_utils import *
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None, embed=False):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        if embed:
            return enc
        else:
            return self.fc(enc) # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th

def evaluate_cc_checkpoint(
    checkpoint_path,
    cc_dataset,
    non_cc_dataset,
    hp,
):
    """Load a selected checkpoint and compute CC FPR and non-CC F1."""
    device = "cuda" if hp.use_gpu and torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DittoModel(
        device=device,
        lm=hp.lm,
        alpha_aug=hp.alpha_aug
    ).to(device)

    model.load_state_dict(checkpoint["model"], strict=True)

    cc_iter = data.DataLoader(
        dataset=cc_dataset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=cc_dataset.pad
    )

    non_cc_iter = data.DataLoader(
        dataset=non_cc_dataset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=non_cc_dataset.pad
    )

    threshold = float(checkpoint.get("threshold", 0.5))

    cc_metrics = evaluate_cc_partition(
        model,
        cc_iter,
        threshold=threshold,
        cc_only=True
    )
    non_cc_metrics = evaluate_cc_partition(
        model,
        non_cc_iter,
        threshold=threshold,
        cc_only=False
    )

    return {
        "threshold": threshold,
        "cc_fpr": cc_metrics["fpr"],
        "cc_false_positives": cc_metrics["false_positives"],
        "cc_negative_examples": cc_metrics["negative_examples"],
        "non_cc_f1": non_cc_metrics["f1"],
        "non_cc_examples": non_cc_metrics["examples"],
    }


def train_step(train_iter, model, optimizer, scheduler, hp, dom_losses=None, opt=None):
    """Perform a single training step."""
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        domain_ids = None
        cc_ids = None

        # parse the training batch for samples, domain ids, augmentation, etc.
        if len(batch) == 2:
            # x, label
            x, y = batch
            prediction = model(x)
        elif len(batch) == 3 and batch[1].dim() == 1:
            # x, label, domain_id
            x, y, domain_ids = batch
            prediction = model(x)
        elif len(batch) == 3:
            # x, x_aug, label
            x1, x2, y = batch
            prediction = model(x1, x2)
        elif len(batch) == 4 and batch[1].dim() == 1:
            # x, label, domain_id, cc_label
            x, y, domain_ids, cc_ids = batch
            prediction = model(x)
        elif len(batch) == 4:
            # x, x_aug, label, domain_id
            x1, x2, y, domain_ids = batch
            prediction = model(x1, x2)
        else:
            # x, x_aug, label, domain_id, cc_label
            x1, x2, y, domain_ids, cc_ids = batch
            prediction = model(x1, x2)

        # parse the optimizer and implement robustness objective if specified
        if opt is not None and opt.startswith('robust') and domain_ids is not None:
            y = y.to(model.device)
            domain_ids = domain_ids.to(model.device)
            if cc_ids is not None:
                cc_ids = cc_ids.to(model.device)

            domain_objective = opt.split("-")[-1]
            robust_weight = getattr(hp, "robust_weight", 5.0)

            per_sample_loss = criterion_none(prediction, y)

            # track per-group losses, counts, etc. to inform robust optimizers
            group_losses = []
            group_counts = []
            group_cc_rates = []
            group_ids = []

            for d in torch.unique(domain_ids):
                mask = domain_ids == d

                group_ids.append(d)
                group_losses.append(per_sample_loss[mask].mean())
                group_counts.append(mask.sum().float())

                if cc_ids is not None:
                    group_cc_rates.append(cc_ids[mask].float().mean())

            group_losses = torch.stack(group_losses)
            group_counts = torch.stack(group_counts)

            if cc_ids is not None:
                group_cc_rates = torch.stack(group_cc_rates)

            if opt.startswith("robust-static") and cc_ids is not None:
                # static robust optimization based on provided information
                # e.g. whicn samples are corner-cases
                weights = 1.0 + robust_weight * group_cc_rates
                weights = weights / weights.sum()
                loss = torch.sum(weights * group_losses)
            elif opt.startswith("robust-dro"):
                # group distributionally robust optimizerion
                dro_eta = robust_weight

                if domain_objective == "best":
                    # soft-min over domain losses
                    loss = -torch.logsumexp(-dro_eta * group_losses, dim=0) / dro_eta

                elif domain_objective == "worst":
                    # soft-max over domain losses
                    loss = torch.logsumexp(dro_eta * group_losses, dim=0) / dro_eta

                elif domain_objective == "median":
                    weights = median_proximity_weights_torch(group_losses, dro_eta)
                    loss = torch.sum(weights * group_losses)

                elif domain_objective == "median_size":
                    weights = median_proximity_weights_torch(group_counts, dro_eta)
                    loss = torch.sum(weights * group_losses)

                elif domain_objective == "biggest":
                    # size-adjusted soft-max: large domains get larger DRO prior weight
                    weights = group_counts / group_counts.sum()
                    log_weights = torch.log(weights.clamp(min=1e-12))
                    loss = torch.logsumexp(log_weights + dro_eta * group_losses, dim=0) / dro_eta

                else:  # smallest
                    # inverse-size-adjusted soft-max: small domains get larger DRO prior weight
                    weights = 1.0 / group_counts.clamp(min=1.0)
                    weights = weights / weights.sum()
                    log_weights = torch.log(weights.clamp(min=1e-12))
                    loss = torch.logsumexp(log_weights + dro_eta * group_losses, dim=0) / dro_eta
            # parity based robustness objectives
            elif opt.startswith("robust-parity-"):
                parity_metric = opt.split("-")[-1]
                # delegate parity loss calculation to helper fxn for clarity
                loss = parity_training_loss(
                    prediction=prediction,
                    y=y,
                    domain_ids=domain_ids,
                    group_losses=group_losses,
                    parity_metric=parity_metric,
                    parity_weight=robust_weight,
                )

            elif opt.startswith("robust-uniform"):
                # equal domain weighting
                loss = group_losses.mean()

            elif opt.startswith("robust-size"):
                # sample size-based robustness objectives
                if domain_objective == "biggest":
                    weights = group_counts
                elif domain_objective == "median_size" or domain_objective == "median":
                    weights = median_proximity_weights_torch(group_counts, robust_weight)
                else:
                    weights = 1.0 / group_counts.clamp(min=1.0)

                weights = weights / weights.sum()
                loss = torch.sum(weights * group_losses)

            elif opt.startswith("robust-"):
                # robust-k-objective, e.g. robust-1-worst, robust-2-best
                parts = opt.split("-")
                try:
                    k = int(parts[1])
                except (IndexError, ValueError):
                    raise ValueError(
                        f"Unknown robust option: {opt}. "
                        "Expected robust-k-objective, e.g. robust-1-worst."
                    )

                k = min(k, group_losses.numel())

                if domain_objective == "best":
                    selected_idx = torch.topk(group_losses, k, largest=False).indices

                elif domain_objective == "worst":
                    selected_idx = torch.topk(group_losses, k, largest=True).indices

                elif domain_objective == "biggest":
                    selected_idx = torch.topk(group_counts, k, largest=True).indices

                elif domain_objective == "median":
                    selected_idx = torch.tensor(
                        median_k_idx(group_losses.detach().cpu().numpy(), k),
                        device=group_losses.device
                    )

                elif domain_objective == "median_size":
                    selected_idx = torch.tensor(
                        median_k_idx(group_counts.detach().cpu().numpy(), k),
                        device=group_losses.device
                    )

                else:  # smallest
                    selected_idx = torch.topk(group_counts, k, largest=False).indices

                weights = torch.ones_like(group_losses)
                weights[selected_idx] = robust_weight
                weights = weights / weights.sum()

                loss = torch.sum(weights * group_losses)

            else:
                loss = criterion(prediction, y)
        # no robustness objective specified: use default optimizer
        else:
            loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss.item()}")

        del loss


def train(trainset, validset, testset, run_tag, hp, domain_evals=None,
          opt=None, eval_plots_path=None, decay=0.0, report_opt=None, init_checkpoint=None):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # train_iter is rebuilt each epoch when domain resampling is on
    def make_train_iter(sampler=None, balanced_batches=False):
        if balanced_batches and trainset.domain_labels is not None:
            batch_sampler = BalancedDomainBatchSampler(
                trainset.domain_labels,
                batch_size=hp.batch_size,
                drop_last=False
            )
            return data.DataLoader(
                dataset=trainset,
                batch_sampler=batch_sampler,
                num_workers=0,
                collate_fn=padder
            )

        return data.DataLoader(
            dataset=trainset,
            batch_size=hp.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            collate_fn=padder
        )

    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    dom_iters = [data.DataLoader(dataset=de, batch_size=hp.batch_size*16,
                  shuffle=False, num_workers=0, collate_fn=padder)
                  for de in domain_evals] if domain_evals else None

    # whether to overrepresent poor performing domains in next epoch or not
    use_domain_resample = getattr(hp, 'domain_resample', False) and opt is not None
    resample_by = getattr(hp, 'resample_by', 'loss')
    report_opt = report_opt if report_opt is not None else opt
    domain_objective = report_opt.split("-")[-1] if report_opt else "worst"
    domain_sizes = np.array([len(de) for de in domain_evals], dtype=np.float64) if domain_evals else None

    # size helpers
    def size_k_idx(k, largest=False):
        k = min(k, len(domain_sizes))
        return np.argsort(domain_sizes)[-k:] if largest else np.argsort(domain_sizes)[:k]
    def weighted_f1_score(f1s):
        weights = domain_sizes / domain_sizes.sum()
        return float(np.sum(np.array(f1s) * weights))

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    model = model.to(device)

    # continue training from previous checkpoint
    if init_checkpoint is not None:
        if not os.path.isfile(init_checkpoint):
            raise FileNotFoundError(
                f"Initial checkpoint not found: {init_checkpoint}"
            )

        checkpoint = torch.load(init_checkpoint, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=True)

        print(f"Initialized model weights from: {init_checkpoint}")

    # optimizer (with the option of l2 regularization)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    # initialize metrics 
    model.eval()
    best_dev_f1 = best_test_f1 = 0.0
    best_epoch = None
    dom_losses = None
    if dom_iters:
        raw_dom_losses = [evaluate_loss(model, dom_it) for dom_it in dom_iters]
        dom_losses = update_ema_losses(None, raw_dom_losses, hp.ema_alpha)
        total_dom_losses = [dom_losses]
        _, th = evaluate(model, valid_iter)
        total_dom_f1s = [[evaluate(model, dom_it, threshold=th) for dom_it in dom_iters]]
        tracked_k_losses = []
        tracked_k_f1s = []
        total_parity_stats = []
        tracked_parity_scores = []
        tracked_parity_disparities = []

    # main train loop
    for epoch in range(1, hp.n_epochs+1):
        # rebuild train_iter with updated domain weights if resampling is on
        # incliude parser.add_argument("--balanced_batches", action="store_true")
        # ---> use --balanced_batches mutually exclusive w.r.t. --domain_resample --resample_by f1
        use_balanced_batches = getattr(hp, "balanced_batches", False) and trainset.domain_labels is not None
        if use_balanced_batches and opt is not None:
            train_iter = make_train_iter(balanced_batches=True)
        elif use_domain_resample and dom_losses is not None and opt is not None:
            sampler = make_domain_weighted_sampler(
                trainset, dom_losses,
                dom_f1s=dom_f1s if epoch > 1 else None,
                resample_by=resample_by
            )
            train_iter = make_train_iter(sampler)
        else:
            train_iter = make_train_iter()

        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp, dom_losses, opt)

        # eval (f1)
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)
        if dom_iters:
            # eval (domain loss)
            raw_dom_losses = [evaluate_loss(model, dom_it) for dom_it in dom_iters]
            dom_losses = update_ema_losses(dom_losses, raw_dom_losses, hp.ema_alpha)
            dom_f1s = [evaluate(model, dom_it, threshold=th) for dom_it in dom_iters]
            domain_binary_metrics = [evaluate_binary_metrics(model, dom_it, threshold=th) for dom_it in dom_iters]
            parity_stats = calculate_parity_statistics(domain_binary_metrics)

            # domain-specific threshold oracle on validation domains
            dom_f1s_domain_th = []
            dom_thresholds = []
            for dom_it in dom_iters:
                d_f1, d_th = evaluate_with_threshold_search(model, dom_it)
                dom_f1s_domain_th.append(d_f1)
                dom_thresholds.append(d_th)

            # track the given objective's metric to be reported in the plots
            plot_k = get_objective_k(report_opt)
            if report_opt is not None and report_opt.startswith("robust-parity-"):
                idx = np.arange(len(dom_losses))
            else:
                idx = objective_indices(
                    domain_objective,
                    np.array(dom_losses),
                    domain_sizes,
                    plot_k
                )
            if report_opt is not None and report_opt.startswith("robust-parity-"):
                parity_score = parity_stats["parity_scores"][domain_objective]
                parity_disparity = parity_stats["parity_disparities"][domain_objective]

                tracked_k_losses.append(parity_disparity)
                tracked_k_f1s.append(parity_score)
                tracked_parity_scores.append(parity_score)
                tracked_parity_disparities.append(parity_disparity)
            elif domain_objective == "weighted_f1":
                weights = domain_sizes / domain_sizes.sum()
                tracked_k_losses.append(
                    float(np.sum(np.array(dom_losses) * weights))
                )
                tracked_k_f1s.append(weighted_f1_score(dom_f1s))
            elif domain_objective == "macro_f1":
                tracked_k_losses.append(float(np.mean(dom_losses)))
                tracked_k_f1s.append(float(np.mean(dom_f1s)))
            else:
                tracked_k_losses.append(
                    float(np.mean(np.array(dom_losses)[idx]))
                )
                tracked_k_f1s.append(
                    float(np.mean(np.array(dom_f1s)[idx]))
                )
            
            # track other stats: macro f1, etc
            macro_f1 = np.mean(dom_f1s)
            macro_f1_domain_th = np.mean(dom_f1s_domain_th)
            tracked_f1_global_th = min(dom_f1s) 
            tracked_f1_domain_th = min(dom_f1s_domain_th) 
            total_dom_losses.append(dom_losses)
            total_dom_f1s.append(dom_f1s)
            total_parity_stats.append(parity_stats)

        # determine selection score based on criteria
        if (not dom_iters) or (opt is None) or opt.startswith("robust-static"):
            selection_score = dev_f1
        elif opt.startswith("robust-parity-"):
            selection_score = parity_stats["parity_scores"][domain_objective]
        elif domain_objective in ["best", "worst", "median"]: # performance-based
            f1s = np.array(dom_f1s, dtype=np.float64)
            k = min(get_objective_k(report_opt), len(f1s))

            if domain_objective == "worst":
                idx = np.argsort(f1s)[:k]      # lowest-k validation F1s
            elif domain_objective == "best":
                idx = np.argsort(f1s)[-k:]     # highest-k validation F1s
            else:  # median
                idx = median_k_idx(f1s, k)     # middle-k validation F1s

            selection_score = float(np.mean(f1s[idx]))
        elif domain_objective in ["biggest", "smallest", "median_size"]: # size-based
            idx = objective_indices(
                domain_objective,
                np.array(dom_losses),
                domain_sizes,
                get_objective_k(report_opt)
            )
            selection_score = float(np.mean(np.array(dom_f1s)[idx]))
        else:  # weighted_f1
            selection_score = weighted_f1_score(dom_f1s)
        
        if selection_score > best_dev_f1:
            # best_dev_f1 = dev_f1
            best_dev_f1 = selection_score
            best_test_f1 = test_f1
            best_epoch = epoch
            if hp.save_model:
                # create the directory if not exist
                #directory = os.path.join(hp.logdir, hp.task)
                directory = hp.ckpt_dir
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = hp.ckpt_path
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        # Global validation threshold used to evaluate this selected checkpoint.
                        'threshold': float(th),
                        'selection_score': float(selection_score),
                        'task': hp.task,
                        'lm': hp.lm,
                        'max_len': hp.max_len,
                        'summarize': bool(getattr(hp, "summarize", False)),
                        'opt': opt
                        }
                torch.save(ckpt, ckpt_path)

        # report per-epoch robust optimization progress
        if dom_iters:
            print(
                f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, "
                f"macro_f1={macro_f1}, tracked_f1_global_th={tracked_f1_global_th}, "
                f"macro_f1_domain_th={macro_f1_domain_th}, tracked_f1_domain_th={tracked_f1_domain_th}, "
                f"best_selection_score={best_dev_f1}"
            )
            # (debug) correlation between loss/f1
            corr = np.corrcoef(dom_losses, dom_f1s)[0,1]
            print(f"Loss/F1 correlation = {corr:.3f}")
        else:
            print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    # calculate macro and F1 scores if doing domain-aware evaluation
    results = []
    if dom_iters:
        macro_f1s = []
        weighted_f1s = []
        for i in range(1, len(total_dom_f1s)):
            macro_f1s.append(np.mean(total_dom_f1s[i]))
            weights = np.array([len(dom) for dom in domain_evals], dtype=np.float64)
            weights /= weights.sum()
            weighted_f1s.append(np.sum(np.array(total_dom_f1s[i]) * weights))

        # determine tracked domain set, store average loss and F1 over that set
        tracked_doms = []
        tracked_dom_losses = []
        tracked_dom_f1s = []
        k = get_objective_k(report_opt)

        # update the tracked metric based on the results from this current training iteration
        for ep in range(1, len(total_dom_losses)):
            ep_loss = np.array(total_dom_losses[ep], dtype=np.float64)
            ep_f1s = np.array(total_dom_f1s[ep], dtype=np.float64)

            k_ep = min(k, len(ep_loss))

            if (domain_objective in ["macro_f1", "weighted_f1"] or 
                (report_opt is not None and report_opt.startswith("robust-parity-"))):
                tracked_idxs = np.arange(len(ep_loss))
            else:
                tracked_idxs = objective_indices(
                    domain_objective,
                    ep_loss,
                    domain_sizes,
                    k_ep
                )

            tracked_doms.append(tracked_idxs.tolist())

            if report_opt is not None and report_opt.startswith("robust-parity-"):
                epoch_parity = total_parity_stats[ep - 1]

                tracked_dom_losses.append(
                    epoch_parity["parity_disparities"][domain_objective]
                )
                tracked_dom_f1s.append(
                    epoch_parity["parity_scores"][domain_objective]
                )
            elif domain_objective == "weighted_f1":
                weights = domain_sizes / domain_sizes.sum()
                tracked_dom_losses.append(float(np.sum(ep_loss * weights)))
                tracked_dom_f1s.append(float(np.sum(ep_f1s * weights)))
            else:
                tracked_dom_losses.append(float(np.mean(ep_loss[tracked_idxs])))
                tracked_dom_f1s.append(float(np.mean(ep_f1s[tracked_idxs])))

        # format results list of dictionaries here
        for ep in range(len(macro_f1s)):
            domain_summary = summarize_domain_f1s(total_dom_f1s[ep + 1])
            epoch_parity = total_parity_stats[ep]
            results.append(
                {
                    "macro_f1": macro_f1s[ep],
                    "weighted_f1": weighted_f1s[ep],

                    # objective-specific reporting
                    "tracked_domains": tracked_doms[ep],
                    "tracked_domain_loss": tracked_dom_losses[ep],
                    "tracked_domain_f1": tracked_dom_f1s[ep],

                    # parity reporting
                    "domain_ppvs": epoch_parity["domain_ppvs"],
                    "domain_tprs": epoch_parity["domain_tprs"],
                    "f1_variance": epoch_parity["f1_variance"],
                    "f1_entropy": epoch_parity["f1_entropy"],
                    "ppv_variance": epoch_parity["ppv_variance"],
                    "tpr_variance": epoch_parity["tpr_variance"],
                    "parity_disparities": epoch_parity["parity_disparities"],
                    "parity_scores": epoch_parity["parity_scores"],

                    # objective-independent reporting
                    **domain_summary,
                }
            )

        # early-stopping summary: selected checkpoint vs final epoch
        if best_epoch is not None and len(results) > 0:
            selected = results[best_epoch - 1]
            final = results[-1]

            for r in results:
                r["selected_epoch"] = best_epoch
                r["final_epoch"] = len(results)
                r["is_selected_epoch"] = (r is selected)

            early_stop_summary = {
                "selected_epoch": best_epoch,
                "final_epoch": len(results),

                "selected_macro_f1": selected["macro_f1"],
                "final_macro_f1": final["macro_f1"],
                "delta_macro_f1": selected["macro_f1"] - final["macro_f1"],

                "selected_weighted_f1": selected["weighted_f1"],
                "final_weighted_f1": final["weighted_f1"],
                "delta_weighted_f1": selected["weighted_f1"] - final["weighted_f1"],

                "selected_tracked_domain_f1": selected["tracked_domain_f1"],
                "final_tracked_domain_f1": final["tracked_domain_f1"],
                "delta_tracked_domain_f1": selected["tracked_domain_f1"] - final["tracked_domain_f1"],
            }

            for r in results:
                r["early_stop_summary"] = early_stop_summary

        # plot worst/best domain performance (loss and f1) across training iterations
        if eval_plots_path is not None: 
            plot_name = domain_objective if domain_objective is not None else "worst"
            plot_name = plot_name.replace("_", "-").capitalize()
            safe_run_tag = run_tag.replace("/", "_")
            save_line_plot(
                tracked_k_losses,
                ylabel=f"Average {plot_name}-k Domain Loss",
                title=f"{plot_name}-k Domain Average Loss",
                path=os.path.join(eval_plots_path, f"{safe_run_tag}_tracked_k_loss.png"),
                best_epoch=best_epoch
            )
            save_line_plot(
                tracked_k_f1s,
                ylabel=f"Average {plot_name}-k Domain F1",
                title=f"{plot_name}-k Domain Average F1",
                path=os.path.join(eval_plots_path, f"{safe_run_tag}_tracked_k_f1.png"),
                best_epoch=best_epoch
            )

    writer.close()
    return results if dom_iters else best_test_f1