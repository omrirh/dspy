"""
Results analysis and visualization for prompt optimization experiments.

Two modes
---------
1. Single-run summary (default)
   Scans --log-dir for completed runs (results.json), prints a table, and
   optionally saves per-dataset accuracy bar charts and an accuracy-vs-runtime
   scatter plot.

2. Matrix aggregation (--aggregate, used with --log-dir pointing at results_v1/)
   Groups runs by (dataset, model, optimizer), computes cross-seed statistics
   (mean, std, 95% CI), and prints a compact comparison table.  Also writes
   aggregate_stats.json next to matrix_summary.json.

Usage
-----
  # Summary of all runs in experiments/logs/
  python experiments/analyze_results.py

  # Filter by dataset or optimizer
  python experiments/analyze_results.py --dataset gsm8k --optimizer gepa_fewshot

  # Save bar charts + scatter plot
  python experiments/analyze_results.py --plot-dir experiments/plots

  # Statistical aggregation over seeds (results_v1/)
  python experiments/analyze_results.py \\
      --log-dir experiments/results_v1 \\
      --aggregate

  # Aggregation + plots
  python experiments/analyze_results.py \\
      --log-dir experiments/results_v1 \\
      --aggregate --plot-dir experiments/plots
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_runs(log_root: str, dataset_filter: str = None, optimizer_filter: str = None):
    """
    Walk log_root and collect all runs that have a results.json.
    Returns a list of dicts with flattened run metadata.
    """
    records = []
    log_path = Path(log_root)

    if not log_path.exists():
        print(f"Log directory not found: {log_root}")
        return records

    for run_dir in sorted(log_path.iterdir()):
        results_file = run_dir / "results.json"
        config_file  = run_dir / "config.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            results = json.load(f)

        config = {}
        if config_file.exists():
            with open(config_file) as f:
                raw = json.load(f)
                config = raw.get("args", raw)

        record = {
            "run_tag":        results.get("run_tag", run_dir.name),
            "dataset":        results.get("dataset",    config.get("dataset",    "?")),
            "optimizer":      results.get("optimizer",  config.get("optimizer",  "?")),
            "model":          results.get("model",      os.path.basename(config.get("model", "?"))),
            "auto":           results.get("auto",       config.get("auto",       "?")),
            "k_demos":        config.get("k_demos", results.get("k_demos", 0)),
            "seed":           results.get("seed", "?"),
            "test_score":     results.get("test_score",     float("nan")),
            "runtime_opt_s":  results.get("runtime_opt_s",  float("nan")),
            "runtime_eval_s": results.get("runtime_eval_s", float("nan")),
            "n_demos":        results.get("n_demos_total",
                              sum(results.get("demos_per_predictor", {}).values())),
            "optimized_instructions": results.get("optimized_instructions", {}),
        }

        if dataset_filter   and record["dataset"]   != dataset_filter:
            continue
        if optimizer_filter and record["optimizer"]  != optimizer_filter:
            continue

        records.append(record)

    return records


def load_runs_multi(log_roots: list[str], dataset_filter: str = None, optimizer_filter: str = None) -> list[dict]:
    """
    Load runs from multiple directories, deduplicating by (dataset, model, optimizer, seed).
    Later directories in the list take priority — use this to overlay corrected runs
    (e.g. results_v2) on top of a baseline matrix (e.g. results_v1).
    """
    # key → record; later dirs overwrite earlier ones for the same cell
    dedup: dict[tuple, dict] = {}
    for root in log_roots:
        for rec in load_runs(root, dataset_filter, optimizer_filter):
            key = (rec["dataset"], rec["model"], rec["optimizer"], rec["seed"])
            dedup[key] = rec
    return list(dedup.values())


# ---------------------------------------------------------------------------
# Single-run tabular summary
# ---------------------------------------------------------------------------

def print_summary(records):
    if not records:
        print("No completed runs found.")
        return

    col_w = [20, 14, 26, 8, 8, 10, 10, 10, 7]
    headers = ["Dataset", "Optimizer", "Model", "Budget", "k_demos",
               "Score", "Opt(s)", "Eval(s)", "Demos"]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + row_fmt.format(*headers))
    print(sep)
    for r in records:
        print(row_fmt.format(
            r["dataset"][:20],
            r["optimizer"][:14],
            r["model"][:26],
            r["auto"][:8],
            str(r["k_demos"])[:8],
            f"{r['test_score']:.4f}"[:10],
            f"{r['runtime_opt_s']:.1f}"[:10],
            f"{r['runtime_eval_s']:.1f}"[:10],
            str(r["n_demos"])[:7],
        ))
    print()


# ---------------------------------------------------------------------------
# Statistical aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else None

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _ci95(xs):
    """95% CI half-width using t-distribution (exact for n≤5, z≈1.96 for n>5)."""
    n = len(xs)
    if n < 2:
        return 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 1.96)
    return t * _std(xs) / math.sqrt(n)


def aggregate_by_group(records: list[dict]) -> dict:
    """
    Group records by (dataset, model, optimizer) and compute cross-seed statistics.

    Returns a dict keyed by '{dataset}|{model}|{optimizer}' with fields:
      seeds_run, n, mean_acc, std_acc, ci95_acc,
      min_acc, max_acc, median_opt_s, per_seed_scores
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["dataset"], r["model"], r["optimizer"])
        groups.setdefault(key, []).append(r)

    stats = {}
    for (dataset, model, optimizer), recs in sorted(groups.items()):
        scores       = [r["test_score"]          for r in recs if not math.isnan(r.get("test_score", float("nan")))]
        opt_times    = [r["runtime_opt_s"]        for r in recs if not math.isnan(r.get("runtime_opt_s", float("nan")))]
        metric_calls = [r["total_metric_calls"]   for r in recs if r.get("total_metric_calls") is not None]
        demo_counts  = [r["n_demos"]              for r in recs if r.get("n_demos") is not None]
        seeds        = sorted(r["seed"] for r in recs)

        sorted_times = sorted(opt_times)
        key_str = f"{dataset}|{model}|{optimizer}"
        stats[key_str] = {
            "dataset":   dataset,
            "model":     model,
            "optimizer": optimizer,
            "seeds_run": seeds,
            "n":         len(scores),
            # Accuracy statistics
            "mean_acc":  round(_mean(scores), 5)  if scores else None,
            "std_acc":   round(_std(scores),  5)  if scores else None,
            "ci95_acc":  round(_ci95(scores), 5)  if scores else None,
            "min_acc":   round(min(scores),   5)  if scores else None,
            "max_acc":   round(max(scores),   5)  if scores else None,
            # Runtime statistics (optimization phase only, seconds)
            "mean_opt_s":   round(_mean(opt_times), 1)   if opt_times else None,
            "std_opt_s":    round(_std(opt_times),  1)   if opt_times else None,
            "median_opt_s": round(sorted_times[len(sorted_times) // 2], 1) if sorted_times else None,
            "min_opt_s":    round(min(opt_times), 1)     if opt_times else None,
            "max_opt_s":    round(max(opt_times), 1)     if opt_times else None,
            # Demo counts (mean across seeds)
            "mean_demos": round(_mean(demo_counts), 1) if demo_counts else None,
            # Sample efficiency (GEPA / GEPAFewShot only; None for baseline / MIPROv2)
            "mean_metric_calls": round(_mean(metric_calls), 1) if metric_calls else None,
            "std_metric_calls":  round(_std(metric_calls),  1) if metric_calls else None,
            # Per-seed detail for downstream analysis / plots
            "per_seed_scores":        {str(r["seed"]): r["test_score"]         for r in recs},
            "per_seed_opt_s":         {str(r["seed"]): r["runtime_opt_s"]      for r in recs
                                       if "runtime_opt_s" in r},
            "per_seed_metric_calls":  {str(r["seed"]): r["total_metric_calls"] for r in recs
                                       if r.get("total_metric_calls") is not None},
        }
    return stats


def print_aggregate_table(stats: dict):
    """Print a compact comparison table of cross-seed statistics."""
    if not stats:
        print("No aggregated stats to display.")
        return

    def _fmt_min(secs):
        """Format seconds as 'xx.xx min'."""
        return f"{secs / 60:.2f} min" if secs is not None else "?"

    col_w = [6, 28, 14, 3, 7, 7, 7, 10, 10, 7]
    headers = ["Data", "Model", "Optimizer", "N", "Mean", "Std", "CI±95", "Med-opt", "Mean-opt", "Demos"]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep     = "  ".join("-" * w for w in col_w)

    print("\n" + row_fmt.format(*headers))
    print(sep)
    for entry in stats.values():
        mean_demos = entry.get("mean_demos")
        print(row_fmt.format(
            entry["dataset"][:6],
            entry["model"][:28],
            entry["optimizer"][:14],
            str(entry["n"]),
            f"{entry['mean_acc']:.4f}"  if entry["mean_acc"]  is not None else "?",
            f"{entry['std_acc']:.4f}"   if entry["std_acc"]   is not None else "?",
            f"{entry['ci95_acc']:.4f}"  if entry["ci95_acc"]  is not None else "?",
            _fmt_min(entry["median_opt_s"]),
            _fmt_min(entry["mean_opt_s"]),
            f"{mean_demos:.1f}"         if mean_demos is not None else "?",
        ))
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_comparison(records: list[dict], plot_dir: str, aggregate: bool = False):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib / pandas not installed — skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(records)

    optimizer_colors = {
        "baseline":    "#888888",
        "gepa":        "#4c72b0",
        "gepa_merge":  "#64a0d0",
        "gepa_fewshot":"#dd8452",
        "miprov2":     "#55a868",
    }

    # Bar chart: mean accuracy per optimizer per dataset (one chart per dataset×model)
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        optimizers = grp["optimizer"].unique()
        scores = [grp[grp["optimizer"] == opt]["test_score"].mean() for opt in optimizers]
        stds   = [grp[grp["optimizer"] == opt]["test_score"].std()  for opt in optimizers] \
                 if aggregate else [0] * len(optimizers)
        colors = [optimizer_colors.get(opt, "gray") for opt in optimizers]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            range(len(optimizers)), scores,
            tick_label=list(optimizers),
            color=colors,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=4,
        )
        ax.set_title(f"Test Accuracy — {dataset} / {model}")
        ax.set_ylabel("Accuracy (%)")
        top = min(105, max(scores) + max(stds or [0]) * 3 + 5)
        ax.set_ylim(0, top)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds or [0]) * 0.1 + 0.5,
                f"{score:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
        fig.tight_layout()
        model_slug = model.replace("/", "_").replace("-", "_")
        out = os.path.join(plot_dir, f"score_comparison_{dataset}_{model_slug}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)

    # Scatter: optimization runtime vs. accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"gepa": "o", "gepa_merge": "D", "gepa_fewshot": "s", "miprov2": "^", "baseline": "x"}
    for opt, grp in df.groupby("optimizer"):
        ax.scatter(
            grp["runtime_opt_s"], grp["test_score"],
            label=opt, marker=markers.get(opt, "o"),
            color=optimizer_colors.get(opt, "gray"), s=80, alpha=0.8,
        )
    ax.set_xlabel("Optimization runtime (s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. Optimization Cost")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(plot_dir, "accuracy_vs_runtime.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    log_dirs = args.log_dir  # now a list
    if len(log_dirs) == 1:
        records = load_runs(log_dirs[0], args.dataset, args.optimizer)
    else:
        records = load_runs_multi(log_dirs, args.dataset, args.optimizer)
        print(f"Loaded from {len(log_dirs)} directories (later dirs override earlier for same cell):")
        for d in log_dirs:
            print(f"  {d}")

    if args.aggregate:
        stats = aggregate_by_group(records)
        print_aggregate_table(stats)

        # Write aggregate_stats.json into the last (highest-priority) log dir
        out_path = Path(log_dirs[-1]) / "aggregate_stats.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Aggregate stats written: {out_path}")
    else:
        print_summary(records)

    if args.plot_dir:
        plot_score_comparison(records, args.plot_dir, aggregate=args.aggregate)


    if args.show_instructions:
        for r in records:
            print(f"\n{'='*60}")
            print(f"Run: {r['run_tag']}")
            for name, instr in r["optimized_instructions"].items():
                print(f"  [{name}] {instr[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize prompt optimization experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-dir", nargs="+", default=["experiments/logs"],
                        metavar="DIR",
                        help="One or more result directories to load from. "
                             "When multiple are given, later directories override "
                             "earlier ones for the same (dataset, model, optimizer, seed) cell. "
                             "Use this to overlay corrected runs (e.g. results_v2) on top of "
                             "a baseline matrix (e.g. results_v1).")
    parser.add_argument("--dataset",   default=None, choices=["gsm8k", "iris"])
    parser.add_argument("--optimizer", default=None,
                        choices=["baseline", "gepa", "gepa_merge", "gepa_fewshot", "miprov2"])
    parser.add_argument("--aggregate", action="store_true",
                        help="Group by (dataset, model, optimizer) and compute cross-seed "
                             "statistics (mean, std, 95%% CI).  Use with --log-dir pointing "
                             "at results_v1/.")
    parser.add_argument("--plot-dir",  default=None,
                        help="Directory to save comparison plots.")
    parser.add_argument("--show-instructions", action="store_true",
                        help="Print optimized instructions for each run.")
    args = parser.parse_args()
    main(args)
