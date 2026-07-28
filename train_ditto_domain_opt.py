import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

def get_selected_result(results):
    """return the result dictionary for the early-stopped checkpoint."""
    summary = results[0]["early_stop_summary"]
    selected_epoch = summary["selected_epoch"]
    return results[selected_epoch - 1]

def reset_seed(seed):
    """set seed before each run for consitency of results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_domain_f1_report(f, result, domain_names):
    """write report of domain-level performance to file"""
    per_domain = {
        domain_names[i]: result["per_domain_f1s"][i]
        for i in range(len(domain_names))
    }
    f.write(f'Per-Domain F1s: {per_domain}\n')
    f.write(
        f'Worst Domain F1: '
        f'{domain_names[result["worst_domain_idx"]]} = {result["worst_domain_f1"]}\n'
    )
    f.write(
        f'Median Domain F1: '
        f'{domain_names[result["median_domain_idx"]]} = {result["median_domain_f1"]}\n'
    )
    f.write(
        f'Best Domain F1: '
        f'{domain_names[result["best_domain_idx"]]} = {result["best_domain_f1"]}\n'
    )

def percent_reduction(base_value, robust_value):
    """return percentage reduction/positive means disparity improved"""
    if abs(base_value) < 1e-12:
        return 0.0

    return 100.0 * (base_value - robust_value) / base_value

def write_summary(f, base_results, opt_results):
    """write selected-checkpoint performance and parity summaries."""
    sbase = base_results[0]["early_stop_summary"]
    sopt = opt_results[0]["early_stop_summary"]

    base_selected = get_selected_result(base_results)
    robust_selected = get_selected_result(opt_results)

    f.write("\n~~~ SUMMARY ~~~\n")

    # general performance
    f.write(f"Base Selected Epoch: {sbase['selected_epoch']}\n")
    f.write(f"Robust Selected Epoch: {sopt['selected_epoch']}\n")
    f.write(f"Base Macro F1: {sbase['selected_macro_f1']}\n")
    f.write(f"Base Weighted F1: {sbase['selected_weighted_f1']}\n")
    f.write(f"Robust Macro F1: {sopt['selected_macro_f1']}\n")
    f.write(f"Robust Weighted F1: {sopt['selected_weighted_f1']}\n")
    f.write(
        f"Delta Macro F1: "
        f"{sopt['selected_macro_f1'] - sbase['selected_macro_f1']}\n"
    )
    f.write(
        f"Delta Weighted F1: "
        f"{sopt['selected_weighted_f1'] - sbase['selected_weighted_f1']}\n"
    )

    # raw parity metrics at each model's selected checkpoint
    base_entropy = base_selected["f1_entropy"]
    robust_entropy = robust_selected["f1_entropy"]
    base_variance = base_selected["f1_variance"]
    robust_variance = robust_selected["f1_variance"]
    base_ppvp = base_selected["ppv_variance"]
    robust_ppvp = robust_selected["ppv_variance"]
    base_tprp = base_selected["tpr_variance"]
    robust_tprp = robust_selected["tpr_variance"]

    f.write("\n~~~ PARITY SUMMARY ~~~\n")
    # entropy: higher is better
    f.write(f"Base F1 Entropy: {base_entropy}\n")
    f.write(f"Robust F1 Entropy: {robust_entropy}\n")
    f.write(f"Delta F1 Entropy: {robust_entropy - base_entropy}\n")
    # variance-based disparities: lower is better
    f.write(f"Base F1 Variance: {base_variance}\n")
    f.write(f"Robust F1 Variance: {robust_variance}\n")
    f.write(f"Delta F1 Variance: {robust_variance - base_variance}\n")
    f.write(f"Base PPVP Disparity: {base_ppvp}\n")
    f.write(f"Robust PPVP Disparity: {robust_ppvp}\n")
    f.write(f"Delta PPVP Disparity: {robust_ppvp - base_ppvp}\n")
    f.write(f"Base TPRP Disparity: {base_tprp}\n")
    f.write(f"Robust TPRP Disparity: {robust_tprp}\n")
    f.write(f"Delta TPRP Disparity: {robust_tprp - base_tprp}\n")
    f.write(f"F1 Variance Reduction (%): {percent_reduction(base_variance, robust_variance)}\n")
    f.write(f"PPVP Disparity Reduction (%): {percent_reduction(base_ppvp, robust_ppvp)}\n")
    f.write(f"TPRP Disparity Reduction (%): {percent_reduction(base_tprp, robust_tprp)}\n")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--opt", type=str, default=None) # specifies robustness optimizer
    # --> format: robust-<optimizer>-<metric>
    # --> e.g.: robust-dro-worst, robust-1-worst, robust-parity-ppvp, etc.

    # robustness optimization parameters
    parser.add_argument("--robust_weight", type=float, default=5.0) # how much to weight the poor performing domain's samples
    parser.add_argument("--ema_alpha", type=float, default=0.0) # whether to smooth across epochs and by how much
    parser.add_argument("--eval_plots_path", type=str, default="eval_plots") # path to loss/f1 plots
    parser.add_argument("--domain_resample", dest="domain_resample", action="store_true") # whether to overly represent poor performing domain's samples in the next epoch
    parser.add_argument("--resample_by", type=str, default="loss", choices=["loss", "f1", "size"])
    parser.add_argument("--balanced_batches", action="store_true") # whether to represent domains evenly in batches
    parser.add_argument("--weight_decay", type=float, default=0.0) # how much l2 weight decay to provide during training

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task
    opt = hp.opt

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    # load relevant datasets
    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    train_domain_sets = [dom['train'] for dom in config.get('domains', {}).values()]
    val_domain_sets = [dom['valid'] for dom in config.get('domains', {}).values()]
    domain_names = list(config.get('domains', {}).keys())

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)
        for i in range(len(val_domain_sets)):
            val_domain_sets[i] = summarizer.transform_file(val_domain_sets[i], max_len=hp.max_len)

    # domain knowlege injection
    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)
        for i in range(len(val_domain_sets)):
            val_domain_sets[i] = injector.transform_file(val_domain_sets[i])

    # load train/dev/test sets
    domain_train_ids = []
    for i, path in enumerate(train_domain_sets):
        with open(path) as f:
            n_lines = sum(1 for _ in f)
        domain_train_ids.extend([i] * n_lines)
    train_dataset = DittoDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da,
                                   hp=hp,
                                   dom_ids=domain_train_ids)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    # domain (validation) datasets
    domain_evals = [DittoDataset(dom, lm=hp.lm) for dom in val_domain_sets]

    # create checkpoint path for this model/run
    hp.ckpt_dir = os.path.join(hp.logdir, hp.task.replace("/", "_"), hp.opt or "base", f"run_{hp.run_id}")
    hp.base_ckpt_path = os.path.join(hp.ckpt_dir, "base_model.pt")
    weight_decay = getattr(hp, "weight_decay", 0.0)

    # train and evaluate the base general model
    print(f"\n\n~~~TRAINING BASE GENERAL MODEL~~~\n\n")
    hp.opt = None # no domain optimization objective for base general model
    hp.ckpt_path = hp.base_ckpt_path
    base_run_tag = run_tag + "_base"
    reset_seed(hp.run_id)
    base_results = train(
                        train_dataset,
                        valid_dataset,
                        test_dataset,
                        base_run_tag,
                        hp,
                        domain_evals=domain_evals,
                        opt=None,                 # no robust training
                        report_opt=opt,           # but report using robust target
                        eval_plots_path=hp.eval_plots_path,
                        decay=0
                    )

    # train and evaluate the general model optimized for lowest-performing domain
    hp.opt = opt
    robust_run_tag = run_tag + "_robust"
    hp.ckpt_path = os.path.join(hp.ckpt_dir, "robust_model.pt")
    reset_seed(hp.run_id)
    opt_results = train(
                        train_dataset,
                        valid_dataset,
                        test_dataset,
                        robust_run_tag,
                        hp,
                        domain_evals=domain_evals,
                        opt=hp.opt,
                        report_opt=hp.opt,
                        eval_plots_path=hp.eval_plots_path,
                        decay=weight_decay
                    )

    # format the evaluation into result file and plots
    # results structure: list of {"macro_f1": _, "weighted_f1": _, "worst_domain": _, "worst_domain_loss": _, "worst_domain_f1": _}
    #                    dictionary for each epoch
    if hp.outfile is not None:
        with open(hp.outfile, 'w') as f:
            f.write(f"~~~ BASE GENERAL MODEL RESULTS ~~~\n")
            for ep in range(len(base_results)):
                base_tracked_names = [domain_names[i] for i in base_results[ep]["tracked_domains"]]
                f.write(f'== ITER {ep} ==\n')
                f.write(f'Macro F1: {base_results[ep]["macro_f1"]}\n')
                f.write(f'Weighted F1: {base_results[ep]["weighted_f1"]}\n')
                f.write(f'Tracked Domain(s): {base_tracked_names}\n')
                f.write(f'Tracked Domains\' Loss: {base_results[ep]["tracked_domain_loss"]}\n')
                f.write(f'Tracked Domains\' F1: {base_results[ep]["tracked_domain_f1"]}\n')
                if opt is not None and opt.startswith("robust-parity-"):
                    parity_metric = opt.split("-")[-1]
                    f.write(f'Optimized Parity Metric: {parity_metric}\n')
                    f.write(f'Parity Disparity: {base_results[ep]["parity_disparities"][parity_metric]}\n')
                    f.write(f'Parity Score: {base_results[ep]["parity_scores"][parity_metric]}\n')
                    f.write(f'Domain PPVs: {base_results[ep]["domain_ppvs"]}\n')
                    f.write(f'Domain TPRs: {base_results[ep]["domain_tprs"]}\n')
                
            f.write(f"\n~~~ ROBUST GENERAL MODEL RESULTS ~~~\n")
            for ep in range(len(opt_results)):
                opt_tracked_names = [domain_names[i] for i in opt_results[ep]["tracked_domains"]]
                f.write(f'== ITER {ep} ==\n')
                f.write(f'Macro F1: {opt_results[ep]["macro_f1"]}\n')
                f.write(f'Weighted F1: {opt_results[ep]["weighted_f1"]}\n')
                f.write(f'Tracked Domain(s): {opt_tracked_names}\n')
                f.write(f'Tracked Domains\' Loss: {opt_results[ep]["tracked_domain_loss"]}\n')
                f.write(f'Tracked Domains\'  F1: {opt_results[ep]["tracked_domain_f1"]}\n')
                if opt is not None and opt.startswith("robust-parity-"):
                    parity_metric = opt.split("-")[-1]
                    f.write(f'Optimized Parity Metric: {parity_metric}\n')
                    f.write(f'Parity Disparity: {opt_results[ep]["parity_disparities"][parity_metric]}\n')
                    f.write(f'Parity Score: {opt_results[ep]["parity_scores"][parity_metric]}\n')
                    f.write(f'Domain PPVs: {opt_results[ep]["domain_ppvs"]}\n')
                    f.write(f'Domain TPRs: {opt_results[ep]["domain_tprs"]}\n')

            # write summary statistics at the bottom of the result file
            base_selected = get_selected_result(base_results)
            robust_selected = get_selected_result(opt_results)
            f.write(f"\n~~~ DOMAIN SUMMARY (BASE) ~~~\n")
            write_domain_f1_report(f, base_selected, domain_names)
            f.write(f"\n~~~ DOMAIN SUMMARY (ROBUST) ~~~\n")
            write_domain_f1_report(f, robust_selected, domain_names)
            write_summary(f, base_results, opt_results)
