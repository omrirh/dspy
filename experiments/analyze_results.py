"""
Results analysis and visualization for prompt optimization experiments.

Scans experiments/logs/ for completed runs (results.json), aggregates metrics,
and produces comparison tables and plots.

Usage
-----
# Summarize all completed runs
python experiments/analyze_results.py

# Filter by dataset or optimizer
python experiments/analyze_results.py --dataset gsm8k --optimizer gepa_fewshot

# Save plots to a directory
python experiments/analyze_results.py --plot-dir experiments/plots
"""
import argparse
import json
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
            "dataset":        config.get("dataset", "?"),
            "optimizer":      config.get("optimizer", "?"),
            "model":          os.path.basename(config.get("model", "?")),
            "auto":           config.get("auto", "?"),
            "k_demos":        config.get("k_demos", 0),
            "seed":           results.get("seed", "?"),
            "test_score":     results.get("test_score", float("nan")),
            "runtime_opt_s":  results.get("runtime_opt_s", float("nan")),
            "runtime_eval_s": results.get("runtime_eval_s", float("nan")),
            "n_demos":        sum(results.get("demos_per_predictor", {}).values()),
            "optimized_instructions": results.get("optimized_instructions", {}),
        }

        if dataset_filter   and record["dataset"]   != dataset_filter:
            continue
        if optimizer_filter and record["optimizer"]  != optimizer_filter:
            continue

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Tabular summary
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
# Plots
# ---------------------------------------------------------------------------

def plot_score_comparison(records, plot_dir: str):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib / pandas not installed — skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(records)

    optimizer_colors = {"gepa": "#4c72b0", "gepa_fewshot": "#dd8452", "miprov2": "#55a868"}

    # Bar chart: accuracy per optimizer, per dataset
    for dataset, grp in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4))
        optimizers = grp["optimizer"].unique()
        scores = [grp[grp["optimizer"] == opt]["test_score"].mean() for opt in optimizers]
        colors = [optimizer_colors.get(opt, "gray") for opt in optimizers]

        bars = ax.bar(range(len(optimizers)), scores, tick_label=list(optimizers), color=colors)
        ax.set_title(f"Test Accuracy — {dataset}")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        out = os.path.join(plot_dir, f"score_comparison_{dataset}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)

    # Scatter: runtime vs. accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"gepa": "o", "gepa_fewshot": "s", "miprov2": "^"}
    for opt, grp in df.groupby("optimizer"):
        ax.scatter(
            grp["runtime_opt_s"], grp["test_score"],
            label=opt, marker=markers.get(opt, "o"),
            color=optimizer_colors.get(opt, "gray"), s=80, alpha=0.8,
        )
    ax.set_xlabel("Optimization runtime (s)")
    ax.set_ylabel("Test accuracy")
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
    records = load_runs(args.log_dir, args.dataset, args.optimizer)
    print_summary(records)

    if args.plot_dir:
        plot_score_comparison(records, args.plot_dir)

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
    parser.add_argument("--log-dir",   default="experiments/logs")
    parser.add_argument("--dataset",   default=None, choices=["gsm8k", "iris"])
    parser.add_argument("--optimizer", default=None,
                        choices=["gepa", "gepa_fewshot", "miprov2"])
    parser.add_argument("--plot-dir",  default=None,
                        help="Directory to save comparison plots")
    parser.add_argument("--show-instructions", action="store_true",
                        help="Print optimized instructions for each run")
    args = parser.parse_args()
    main(args)
