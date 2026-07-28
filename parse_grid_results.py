#!/usr/bin/env python3
"""
parse robust optimization grid-search result files from opt_grid.sh

Expected input files look like:
  task=WDC_products_small_opt=robust-dro-worst_bs=64_lr=1e-5_rw=2.0_ema=0.0_sampler=resample_loss_wd=0.00_seed=0.txt

Expected summary block contains lines like:
  Robust Tracked F1: 0.81
  Delta Tracked F1: -0.015

Usage:
  python parse_grid_results.py --results_dir <DIR> --sort_by <METRIC>
  # e.g. python parse_grid_results.py --result_dir WDCMD_DRO_grid_results --sort_by robust_worst_f1

Outputs:
  <DIR>/grid_summary.csv
  <DIR>/grid_top_by_delta_tracked.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path
import ast
import pandas as pd


FLOAT_PATTERN = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_metric(text, label):
    """extract a floating-point metric line like 'Delta Macro F1: 0.123'."""
    pattern = re.escape(label) + r"\s*:\s*" + FLOAT_PATTERN
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def parse_domain_f1s(text, section_name):
    """parse the per-Domain f1s dictionary from one domain-summary section."""
    pattern = (
        rf"~~~ DOMAIN SUMMARY \({re.escape(section_name)}\) ~~~"
        rf".*?Per-Domain F1s:\s*(\{{.*?\}})"
    )
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    try:
        values = ast.literal_eval(match.group(1))
        return {str(k): float(v) for k, v in values.items()}
    except (ValueError, SyntaxError, TypeError):
        return None


def summarize_domain_metrics(domain_f1s):
    """return raw worst, median, and best F1 across domains."""
    if not domain_f1s:
        return {
            "worst_f1": None,
            "median_f1": None,
            "best_f1": None,
        }

    values = sorted(domain_f1s.values())
    return {
        "worst_f1": values[0],
        "median_f1": values[len(values) // 2],
        "best_f1": values[-1],
    }


def compute_worst_k_f1(domain_f1s, k):
    """return the mean F1 of the k lowest-performing domains."""
    if not domain_f1s:
        return None

    values = sorted(domain_f1s.values())
    k = min(k, len(values))
    return sum(values[:k]) / k


def get_named_domain_f1(domain_f1s, domain_name):
    """return the F1 for a user-specified domain, if present."""
    if not domain_name or not domain_f1s:
        return None
    return domain_f1s.get(domain_name)


def parse_filename(path):
    """extract hyperparameters encoded as key=value pieces in the filename."""
    stem = Path(path).stem
    fields = {}

    # patterns intentionally specific so task/opt values with hyphens or underscores do not confuse the parser.
    patterns = {
        "task": r"task=(.*?)_opt=",
        "opt": r"_opt=(.*?)_bs=",
        "batch_size": r"_bs=([^_]+)",
        "lr": r"_lr=([^_]+)",
        "robust_weight": r"_rw=([^_]+)",
        "ema_alpha": r"_ema=([^_]+)",
        "sampler": r"_sampler=(.*?)_wd=",
        "weight_decay": r"_wd=([^_]+)",
        "seed": r"_seed=([^_]+)$",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, stem)
        fields[key] = match.group(1) if match else None

    # cast numeric fields when possible.
    int_fields = ["batch_size", "seed"]
    float_fields = ["lr", "robust_weight", "ema_alpha", "weight_decay"]

    for key in int_fields:
        if fields.get(key) is not None:
            try:
                fields[key] = int(fields[key])
            except ValueError:
                pass

    for key in float_fields:
        if fields.get(key) is not None:
            try:
                fields[key] = float(fields[key])
            except ValueError:
                pass

    return fields

def get_worst_k_from_opt(opt):
    """extract k from options such as robust-3-worst."""
    if not opt:
        return 1

    match = re.match(r"robust-(\d+)-worst$", opt)
    return int(match.group(1)) if match else 1

def parse_result_file(path, biggest_domain=None, smallest_domain=None):
    text = Path(path).read_text(errors="replace")
    row = parse_filename(path)
    worst_k = get_worst_k_from_opt(row.get("opt"))

    # parse base and robust per-domain metrics
    base_domains = parse_domain_f1s(text, "BASE")
    robust_domains = parse_domain_f1s(text, "ROBUST")
    base_worst_k_f1 = compute_worst_k_f1(base_domains, worst_k)
    robust_worst_k_f1 = compute_worst_k_f1(robust_domains, worst_k)

    # parse base and robust summart metrics
    base_summary = summarize_domain_metrics(base_domains)
    robust_summary = summarize_domain_metrics(robust_domains)
    domain_changes = summarize_domain_changes(base_domains, robust_domains)

    # parse base and robust size metrics (if applicable)
    base_biggest_f1 = get_named_domain_f1(base_domains, biggest_domain)
    robust_biggest_f1 = get_named_domain_f1(robust_domains, biggest_domain)
    base_smallest_f1 = get_named_domain_f1(base_domains, smallest_domain)
    robust_smallest_f1 = get_named_domain_f1(robust_domains, smallest_domain)

    # codense results
    row.update({
        "file": str(path),

        # existing aggregate metrics
        "base_macro_f1": parse_metric(text, "Base Macro F1"),
        "base_weighted_f1": parse_metric(text, "Base Weighted F1"),
        "robust_macro_f1": parse_metric(text, "Robust Macro F1"),
        "robust_weighted_f1": parse_metric(text, "Robust Weighted F1"),
        "delta_macro_f1": parse_metric(text, "Delta Macro F1"),
        "delta_weighted_f1": parse_metric(text, "Delta Weighted F1"),

        # percentage of domains whose F1 improved or stayed the same
        "domains_non_decreased_pct": domain_changes["domains_non_decreased"],
        "num_domains_non_decreased": domain_changes["num_domains_non_decreased"],
        "num_domains_compared": domain_changes["num_domains_compared"],        

        # tracked f1
        "base_tracked_f1": parse_metric(text, "Base Tracked F1"),
        "robust_tracked_f1": parse_metric(text, "Robust Tracked F1"),
        "delta_tracked_f1": parse_metric(text, "Delta Tracked F1"),

        # actual raw domain-performance metrics
        "base_worst_f1": base_summary["worst_f1"],
        "robust_worst_f1": robust_summary["worst_f1"],
        "base_median_f1": base_summary["median_f1"],
        "robust_median_f1": robust_summary["median_f1"],
        "base_best_f1": base_summary["best_f1"],
        "robust_best_f1": robust_summary["best_f1"],
        "worst_k": worst_k,
        "base_worst_k_f1": base_worst_k_f1,
        "robust_worst_k_f1": robust_worst_k_f1,
        "delta_worst_k_f1": (
            robust_worst_k_f1 - base_worst_k_f1
            if base_worst_k_f1 is not None and robust_worst_k_f1 is not None
            else None
        ),

        # user-specified size-domain metrics
        "biggest_domain": biggest_domain,
        "base_biggest_f1": base_biggest_f1,
        "robust_biggest_f1": robust_biggest_f1,

        "smallest_domain": smallest_domain,
        "base_smallest_f1": base_smallest_f1,
        "robust_smallest_f1": robust_smallest_f1,

        # parity metrics from selected checkpoints
        "base_f1_entropy": parse_metric(text, "Base F1 Entropy"),
        "robust_f1_entropy": parse_metric(text, "Robust F1 Entropy"),
        "delta_f1_entropy": parse_metric(text, "Delta F1 Entropy"),

        "base_f1_variance": parse_metric(text, "Base F1 Variance"),
        "robust_f1_variance": parse_metric(text, "Robust F1 Variance"),
        "delta_f1_variance": parse_metric(text, "Delta F1 Variance"),

        "base_ppvp_disparity": parse_metric(text, "Base PPVP Disparity"),
        "robust_ppvp_disparity": parse_metric(text, "Robust PPVP Disparity"),
        "delta_ppvp_disparity": parse_metric(text, "Delta PPVP Disparity"),

        "base_tprp_disparity": parse_metric(text, "Base TPRP Disparity"),
        "robust_tprp_disparity": parse_metric(text, "Robust TPRP Disparity"),
        "delta_tprp_disparity": parse_metric(text, "Delta TPRP Disparity"),
    })

    # compute absolute deltas
    for metric in ["worst", "median", "best", "biggest", "smallest"]:
        base_value = row.get(f"base_{metric}_f1")
        robust_value = row.get(f"robust_{metric}_f1")

        row[f"delta_{metric}_f1"] = (
            robust_value - base_value
            if base_value is not None and robust_value is not None
            else None
        )

    row["parse_ok"] = row["robust_worst_f1"] is not None
    return row

def summarize_domain_changes(base_domain_f1s, robust_domain_f1s):
    """return the percentage of domains whose robust f1 is >= base f1"""
    if not base_domain_f1s or not robust_domain_f1s:
        return {
            "domains_non_decreased": None,
            "num_domains_non_decreased": None,
            "num_domains_compared": None,
        }

    common_domains = sorted(
        set(base_domain_f1s) & set(robust_domain_f1s)
    )

    if not common_domains:
        return {
            "domains_non_decreased": None,
            "num_domains_non_decreased": None,
            "num_domains_compared": None,
        }

    num_non_decreased = sum(
        robust_domain_f1s[domain] >= base_domain_f1s[domain]
        for domain in common_domains
    )

    return {
        "domains_non_decreased": (
            100.0 * num_non_decreased / len(common_domains)
        ),
        "num_domains_non_decreased": num_non_decreased,
        "num_domains_compared": len(common_domains),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="dro_grid_results",
        help="Directory containing grid-search .txt result files.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top rows to print/save.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="robust_worst_f1",
        choices=[
            "robust_worst_f1",
            "delta_worst_f1",
            "robust_median_f1",
            "delta_median_f1",
            "robust_best_f1",
            "delta_best_f1",
            "robust_biggest_f1",
            "delta_biggest_f1",
            "robust_smallest_f1",
            "delta_smallest_f1",
            "robust_macro_f1",
            "delta_macro_f1",
            "robust_weighted_f1",
            "delta_weighted_f1",
            "robust_worst_k_f1",
            "delta_worst_k_f1",

            # parity-based
            "robust_f1_entropy",
            "delta_f1_entropy",
            "robust_f1_variance",
            "delta_f1_variance",
            "robust_ppvp_disparity",
            "delta_ppvp_disparity",
            "robust_tprp_disparity",
            "delta_tprp_disparity",

            # Retained only for debugging
            "robust_tracked_f1",
            "delta_tracked_f1",
        ],
        help="Validation metric used to rank grid settings.",
    )
    parser.add_argument(
        "--biggest_domain",
        type=str,
        default=None,
        help=(
            "Name of the domain with the largest dataset, exactly as it appears "
            "in the Per-Domain F1s dictionary."
        ),
    )

    parser.add_argument(
        "--smallest_domain",
        type=str,
        default=None,
        help=(
            "Name of the domain with the smallest dataset, exactly as it appears "
            "in the Per-Domain F1s dictionary."
        ),
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.txt")))
    if not paths:
        raise SystemExit(f"No .txt files found in {args.results_dir}")

    rows = [
        parse_result_file(
            path,
            biggest_domain=args.biggest_domain,
            smallest_domain=args.smallest_domain,
        )
        for path in paths
    ]
    df = pd.DataFrame(rows)

    # keep failed/incomplete files visible in the full CSV, but rank only parsed rows.
    full_csv = os.path.join(args.results_dir, "grid_summary.csv")
    df.to_csv(full_csv, index=False)

    valid = df[df["parse_ok"]].copy()
    if valid.empty:
        raise SystemExit(
            "Found result files, but none had a parseable SUMMARY block. "
            f"Full parsed output written to {full_csv}"
        )

    # higher is better for F1 and entropy.
    # lower is better for variance/disparity metrics.
    lower_is_better = {
        "robust_f1_variance",
        "delta_f1_variance",
        "robust_ppvp_disparity",
        "delta_ppvp_disparity",
        "robust_tprp_disparity",
        "delta_tprp_disparity",
    }

    valid = valid.sort_values(args.sort_by, ascending=(args.sort_by in lower_is_better))

    display_cols = [
        "robust_worst_f1",
        "delta_worst_f1",
        "robust_median_f1",
        "delta_median_f1",
        "robust_best_f1",
        "delta_best_f1",

        "smallest_domain",
        "robust_smallest_f1",
        "delta_smallest_f1",

        "biggest_domain",
        "robust_biggest_f1",
        "delta_biggest_f1",
        "worst_k",
        "robust_worst_k_f1",
        "delta_worst_k_f1",

        "robust_macro_f1",
        "delta_macro_f1",
        "robust_weighted_f1",
        "delta_weighted_f1",
        "domains_non_decreased_pct",
        "num_domains_non_decreased",
        "num_domains_compared",

        # parity-based
        "robust_f1_entropy",
        "delta_f1_entropy",
        "robust_f1_variance",
        "delta_f1_variance",
        "robust_ppvp_disparity",
        "delta_ppvp_disparity",
        "robust_tprp_disparity",
        "delta_tprp_disparity",

        "robust_tracked_f1",
        "delta_tracked_f1",

        "batch_size",
        "lr",
        "robust_weight",
        "ema_alpha",
        "sampler",
        "weight_decay",
        "seed",
        "file",
    ]
    display_cols = [c for c in display_cols if c in valid.columns]

    top = valid[display_cols].head(args.top_k)

    top_csv = os.path.join(args.results_dir, f"grid_top_by_{args.sort_by}.csv")
    top.to_csv(top_csv, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)

    # print parsing summary
    print(f"\nParsed files: {len(df)}")
    print(f"Successfully parsed summaries: {len(valid)}")
    print(f"Full summary CSV: {full_csv}")
    print(f"Top-{args.top_k} CSV: {top_csv}")
    print(f"\nTop {args.top_k} settings ranked by {args.sort_by}:\n")
    print(top.to_string(index=False))

    print("\nBest setting:")
    best = valid.iloc[0]
    for key in ["batch_size", "lr", "robust_weight", "ema_alpha", "sampler", "weight_decay", "seed"]:
        if key in best:
            print(f"  {key}: {best[key]}")

    print(f"  selected_by: {args.sort_by}")
    print(f"  selection_score: {best[args.sort_by]}")
    print(f"  robust_worst_f1: {best.get('robust_worst_f1')}")
    print(f"  delta_worst_f1: {best.get('delta_worst_f1')}")
    print(f"  robust_macro_f1: {best.get('robust_macro_f1')}")
    print(f"  delta_macro_f1: {best.get('delta_macro_f1')}")
    print(f"  robust_weighted_f1: {best.get('robust_weighted_f1')}")
    print(f"  delta_weighted_f1: {best.get('delta_weighted_f1')}")
    print("  domains_non_decreased: "
        f"{int(best.get('num_domains_non_decreased'))}/"
        f"{int(best.get('num_domains_compared'))} "
        f"({best.get('domains_non_decreased_pct'):.1f}%)"
    )
    print(f"  worst_k: {best.get('worst_k')}")
    print(f"  robust_worst_k_f1: {best.get('robust_worst_k_f1')}")
    print(f"  delta_worst_k_f1: {best.get('delta_worst_k_f1')}")
    
    # parity-based
    print(f"  robust_f1_entropy: {best.get('robust_f1_entropy')}")
    print(f"  delta_f1_entropy: {best.get('delta_f1_entropy')}")
    print(f"  robust_f1_variance: {best.get('robust_f1_variance')}")
    print(f"  delta_f1_variance: {best.get('delta_f1_variance')}")
    print(f"  robust_ppvp_disparity: {best.get('robust_ppvp_disparity')}")
    print(f"  delta_ppvp_disparity: {best.get('delta_ppvp_disparity')}")
    print(f"  robust_tprp_disparity: {best.get('robust_tprp_disparity')}")
    print(f"  delta_tprp_disparity: {best.get('delta_tprp_disparity')}")

    if args.smallest_domain:
        print(f"  smallest_domain: {args.smallest_domain}")
        print(f"  robust_smallest_f1: {best.get('robust_smallest_f1')}")
        print(f"  delta_smallest_f1: {best.get('delta_smallest_f1')}")

    if args.biggest_domain:
        print(f"  biggest_domain: {args.biggest_domain}")
        print(f"  robust_biggest_f1: {best.get('robust_biggest_f1')}")
        print(f"  delta_biggest_f1: {best.get('delta_biggest_f1')}")


if __name__ == "__main__":
    main()
