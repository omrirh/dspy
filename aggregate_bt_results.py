"""
Aggregate and summarize BetterTogether prompt-optimizer results for the ClusterFewshot paper.

Reads all results_bt/<dataset>/<model>/<optimizer>/<seed>.json files, computes
cross-seed statistics, and prints paper-ready Markdown tables.

Metrics reported per (dataset, model, optimizer):
  - Accuracy mean ± std (sample std across seeds)
  - Delta vs standalone baseline (pp)
  - Compile time mean (minutes)
  - Per-seed scores for transparency

Usage
-----
  python aggregate_bt_results.py                               # default: results_bt/
  python aggregate_bt_results.py --results-dir results_bt      # explicit dir
  python aggregate_bt_results.py --datasets hotpotqa           # filter datasets
  python aggregate_bt_results.py --models Qwen2.5-32B-Instruct-AWQ  # filter models
  python aggregate_bt_results.py --write-json                  # also save aggregate.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stats helpers (sample std, 95% CI)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ci95(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 1.96)
    return t * _std(xs) / math.sqrt(n)


# ---------------------------------------------------------------------------
# Display maps
# ---------------------------------------------------------------------------

OPTIMIZER_DISPLAY = {
    "clusterfs": "ClusterFewshot",
    "miprov2":   "MIPROv2",
    "bfrs":      "BFRS",
    "baseline":  "Baseline",
}

OPTIMIZER_ORDER = ["ClusterFewshot", "BFRS", "MIPROv2"]

DATASET_DISPLAY = {
    "hotpotqa":           "HotPotQA",
    "iris":               "Iris",
    "gsm8k":              "GSM8K",
    "crop_recommendation": "CropRec",
}

MODEL_DISPLAY = {
    "Qwen2.5-32B-Instruct-AWQ": "Qwen2.5-32B-AWQ",
    "Qwen2.5-14B-Instruct":     "Qwen2.5-14B",
    "Qwen2.5-7B-Instruct":      "Qwen2.5-7B",
}


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str, datasets: list[str] | None = None, models: list[str] | None = None) -> dict:
    """
    Walk results_dir/<dataset>/<model>/<optimizer>/<seed>.json.
    Returns { (dataset, model, optimizer): [run_dict, ...] }
    """
    root = Path(results_dir)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if datasets and dataset not in datasets:
            continue

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if models and model not in models:
                continue

            # Standalone baseline score for this model+dataset
            bl_path = model_dir / "baseline" / "100.json"
            baseline_score = None
            if bl_path.exists():
                bl_data = json.loads(bl_path.read_text())
                baseline_score = bl_data["scores"].get("baseline")

            for opt_dir in sorted(model_dir.iterdir()):
                if not opt_dir.is_dir() or opt_dir.name == "baseline":
                    continue
                optimizer = opt_dir.name

                for seed_file in sorted(opt_dir.glob("*.json")):
                    try:
                        seed = int(seed_file.stem)
                    except ValueError:
                        continue

                    data = json.loads(seed_file.read_text())
                    scores = data.get("scores", {})
                    timing = data.get("timing_seconds", {})

                    run = {
                        "dataset":        dataset,
                        "model":          model,
                        "optimizer":      optimizer,
                        "seed":           seed,
                        "baseline_score": baseline_score,
                        "optimized":      scores.get("optimized"),
                        "compile_s":      timing.get("compile_total"),
                    }
                    groups[(dataset, model, optimizer)].append(run)

    return groups


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(groups: dict) -> list[dict]:
    """Compute cross-seed statistics for each (dataset, model, optimizer) group."""
    rows = []
    for (dataset, model, optimizer), runs in groups.items():
        def _collect(field):
            return [r[field] for r in runs if r.get(field) is not None]

        opt_scores  = _collect("optimized")
        compile_min = [r["compile_s"] / 60.0 for r in runs if r.get("compile_s")]
        bl_score    = runs[0]["baseline_score"] if runs else None

        seeds = sorted(r["seed"] for r in runs)
        rows.append({
            "dataset":           dataset,
            "model":             model,
            "optimizer":         optimizer,
            "seeds":             seeds,
            "n":                 len(opt_scores),
            "baseline":          bl_score,
            "mean_opt":          _mean(opt_scores),
            "std_opt":           _std(opt_scores),
            "ci95_opt":          _ci95(opt_scores),
            "delta_vs_bl":       (_mean(opt_scores) - bl_score) if (opt_scores and bl_score is not None) else None,
            "mean_compile_min":  _mean(compile_min),
            "per_seed_scores":   {str(r["seed"]): r["optimized"] for r in runs},
        })

    rows.sort(key=lambda r: (r["dataset"], r["model"], -(r["mean_opt"] or 0)))
    return rows


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _pp(v):
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}pp"


def _fmt_compile(v):
    return f"{v:.1f}" if v is not None else "—"


# ---------------------------------------------------------------------------
# Hardcoded GSM8K results (Qwen2.5-32B-Instruct-AWQ, reported in paper)
# ---------------------------------------------------------------------------
_GSM8K_RESULTS = {
    "Baseline":      {"acc": 86.72, "compile_min": None},
    "BFRS":          {"acc": 94.46, "compile_min": 194.7},
    "MIPROv2":       {"acc": 93.40, "compile_min": 143.8},
    "ClusterFewshot":{"acc": 94.80, "compile_min": 112.2},
}


def print_cross_dataset_table(rows: list[dict], baseline_scores: dict):
    """
    Single summary table: optimizer rows × dataset columns.
    Columns per dataset: Accuracy (%), Compile (min).
    GSM8K column is injected from hardcoded paper results.
    """
    all_datasets = sorted({r["dataset"] for r in rows})
    all_models   = sorted({r["model"]   for r in rows})

    index = {
        (r["dataset"], r["model"], OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])): r
        for r in rows
    }

    # Collect all numeric values per dataset column for winner detection.
    # col_key: "gsm8k" | dataset name.  value: {opt_display: (acc, compile_min)}
    col_data: dict[str, dict[str, tuple]] = {}

    gsm8k_col: dict[str, tuple] = {}
    for row_name, v in _GSM8K_RESULTS.items():
        gsm8k_col[row_name] = (v["acc"], v["compile_min"])
    col_data["gsm8k"] = gsm8k_col

    for d in all_datasets:
        dcol: dict[str, tuple] = {}
        for m in all_models:
            bl = baseline_scores.get((d, m))
            if bl is not None:
                dcol["Baseline"] = (bl, None)
            for opt in OPTIMIZER_ORDER:
                r = index.get((d, m, opt))
                if r and r["mean_opt"] is not None:
                    dcol[opt] = (r["mean_opt"], r["mean_compile_min"])
        col_data[d] = dcol

    def _best_acc(col_key):
        vals = {k: v[0] for k, v in col_data[col_key].items() if v[0] is not None}
        return max(vals, key=vals.get) if vals else None

    def _best_compile(col_key):
        vals = {k: v[1] for k, v in col_data[col_key].items()
                if k != "Baseline" and v[1] is not None}
        return min(vals, key=vals.get) if vals else None

    col_keys   = ["gsm8k"] + list(all_datasets)
    best_acc     = {ck: _best_acc(ck)     for ck in col_keys}
    best_compile = {ck: _best_compile(ck) for ck in col_keys}

    def _b(val_str, is_winner):
        return f"**{val_str}**" if is_winner else val_str

    # Header
    gsm8k_label = f"**{DATASET_DISPLAY.get('gsm8k', 'GSM8K')}**"
    dyn_labels  = [f"**{DATASET_DISPLAY.get(d, d)}**" for d in all_datasets]
    all_labels  = [gsm8k_label] + dyn_labels

    col_headers = " | ".join(f"{lbl} Acc (%) | Compile (min)" for lbl in all_labels)
    sep         = " | ".join(["--- | ---"] * len(all_labels))
    print(f"| Optimizer | {col_headers} |")
    print(f"| --- | {sep} |")

    # Baseline row
    gsm8k_bl  = _GSM8K_RESULTS["Baseline"]
    gsm8k_acc = _b(f"{gsm8k_bl['acc']:.2f}%", best_acc["gsm8k"] == "Baseline")
    bl_cells  = [f"{gsm8k_acc} | —"]
    for d in all_datasets:
        for m in all_models:
            bl = baseline_scores.get((d, m))
            if bl is None:
                bl_cells.append("— | —")
            else:
                bl_cells.append(f"{_b(f'{bl:.2f}%', best_acc[d] == 'Baseline')} | —")
    print(f"| Baseline | " + " | ".join(bl_cells) + " |")

    # Optimizer rows
    for opt in OPTIMIZER_ORDER:
        gsm8k_r = _GSM8K_RESULTS.get(opt)
        if gsm8k_r:
            acc_str  = _b(f"{gsm8k_r['acc']:.2f}%",      best_acc["gsm8k"]     == opt)
            comp_str = _b(f"{gsm8k_r['compile_min']:.1f}", best_compile["gsm8k"] == opt)
            gsm8k_cell = f"{acc_str} | {comp_str}"
        else:
            gsm8k_cell = "— | —"
        cells = [gsm8k_cell]

        for d in all_datasets:
            for m in all_models:
                r = index.get((d, m, opt))
                if r is None or r["mean_opt"] is None:
                    cells.append("— | —")
                    continue
                acc_str  = _b(f"{r['mean_opt']:.2f}%",               best_acc[d]     == opt)
                comp_str = _b(_fmt_compile(r["mean_compile_min"]),    best_compile[d] == opt)
                cells.append(f"{acc_str} | {comp_str}")

        print(f"| {opt} | " + " | ".join(cells) + " |")


def print_markdown_tables(rows: list[dict], baseline_scores: dict):
    """
    Print paper-ready Markdown tables.
    Outer grouping: dataset.  Inner grouping: model (one table per dataset × model).
    """
    # Group by (dataset, model)
    by_dm: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_dm[(r["dataset"], r["model"])].append(r)

    for (dataset, model), dm_rows in sorted(by_dm.items()):
        dl = DATASET_DISPLAY.get(dataset, dataset)
        ml = MODEL_DISPLAY.get(model, model)
        bl = baseline_scores.get((dataset, model))

        print(f"\n### {dl} — {ml}")
        print()

        # Sort rows by optimizer display order
        def _sort_key(r):
            d = OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])
            try:
                return OPTIMIZER_ORDER.index(d)
            except ValueError:
                return 99

        dm_rows.sort(key=_sort_key)

        print("| Optimizer | Accuracy (mean±std) | Δ vs Baseline | Compile (min) | Seeds |")
        print("|---|---|---|---|---|")

        if bl is not None:
            print(f"| Baseline | {bl:.2f}% | — | — | — |")

        for r in dm_rows:
            opt  = OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])
            mean = r["mean_opt"]
            std  = r["std_opt"]
            delta = r["delta_vs_bl"]
            comp  = r["mean_compile_min"]
            seeds = ", ".join(str(s) for s in r["seeds"])
            per_seed = "  ".join(f"s{s}={v:.2f}%" for s, v in sorted(r["per_seed_scores"].items()) if v is not None)

            acc_str = f"{mean:.2f}±{std:.2f}%" if mean is not None else "—"
            print(f"| {opt} | {acc_str} | {_pp(delta)} | {_fmt_compile(comp)} | {seeds} |")
            if per_seed:
                print(f"|  | *{per_seed}* | | | |")

        print()

    # Cross-dataset summary: compile efficiency
    print("\n---")
    print("\n### Compile efficiency summary (mean minutes per optimizer)\n")

    all_datasets = sorted({r["dataset"] for r in rows})
    all_models   = sorted({r["model"]   for r in rows})

    col_labels = " | ".join(
        f"{DATASET_DISPLAY.get(d, d)} / {MODEL_DISPLAY.get(m, m)}"
        for d in all_datasets for m in all_models
    )
    print(f"| Optimizer | {col_labels} |")
    print("|---|" + "---|" * len(all_datasets) * len(all_models))

    index = {(r["dataset"], r["model"], OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])): r for r in rows}
    for opt in OPTIMIZER_ORDER:
        cells = []
        for d in all_datasets:
            for m in all_models:
                r = index.get((d, m, opt))
                cells.append(_fmt_compile(r["mean_compile_min"] if r else None))
        print(f"| {opt} | " + " | ".join(cells) + " |")

    # Speedup table: ClusterFewshot vs BFRS vs MIPROv2
    print("\n---")
    print("\n### ClusterFewshot compile speedup vs BFRS and MIPROv2\n")
    print(f"| Dataset / Model | vs BFRS | vs MIPROv2 |")
    print("|---|---|---|")
    for d in all_datasets:
        for m in all_models:
            cfs_r   = index.get((d, m, "ClusterFewshot"))
            bfrs_r  = index.get((d, m, "BFRS"))
            mipro_r = index.get((d, m, "MIPROv2"))
            cfs_t   = cfs_r["mean_compile_min"]   if cfs_r   else None
            bfrs_t  = bfrs_r["mean_compile_min"]  if bfrs_r  else None
            mipro_t = mipro_r["mean_compile_min"] if mipro_r else None
            su_bfrs  = f"{bfrs_t/cfs_t:.2f}×"  if (cfs_t and bfrs_t)  else "—"
            su_mipro = f"{mipro_t/cfs_t:.2f}×" if (cfs_t and mipro_t) else "—"
            label = f"{DATASET_DISPLAY.get(d, d)} / {MODEL_DISPLAY.get(m, m)}"
            print(f"| {label} | {su_bfrs} | {su_mipro} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BetterTogether experiment results for paper tables."
    )
    parser.add_argument("--results-dir", default="results_bt",
                        help="Root directory containing <dataset>/<model>/<optimizer>/<seed>.json files.")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="DATASET",
                        help="Filter to specific dataset names (e.g. hotpotqa iris).")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                        help="Filter to specific model directory names (e.g. Qwen2.5-32B-Instruct-AWQ).")
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Print a single summary table: optimizer rows × dataset columns.")
    parser.add_argument("--write-json", action="store_true",
                        help="Write aggregate.json alongside the tables.")
    args = parser.parse_args()

    groups = load_results(args.results_dir, datasets=args.datasets, models=args.models)
    if not groups:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = aggregate(groups)

    # Collect baseline scores keyed by (dataset, model)
    baseline_scores: dict[tuple, float | None] = {}
    root = Path(args.results_dir)
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if args.datasets and dataset not in args.datasets:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if args.models and model not in args.models:
                continue
            bl_path = model_dir / "baseline" / "100.json"
            if bl_path.exists():
                d = json.loads(bl_path.read_text())
                baseline_scores[(dataset, model)] = d["scores"].get("baseline")

    if args.cross_dataset:
        print_cross_dataset_table(rows, baseline_scores)
    else:
        print_markdown_tables(rows, baseline_scores)

    if args.write_json:
        out = Path(args.results_dir) / "aggregate.json"
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nAggregate stats written: {out}")


if __name__ == "__main__":
    main()
