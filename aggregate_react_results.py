"""
Aggregate and summarize ReAct agent experiment results for the ClusterFewshot paper.

Reads all results_v2/<model>/<optimizer>/<seed>.json files, computes cross-seed
statistics, and prints paper-ready Markdown tables with trajectory quality metrics.

Metrics reported per (model, optimizer):
  - Accuracy mean ± std (sample std, 3 seeds)
  - Delta vs standalone baseline
  - Compile time (mean, minutes)
  - fin_tool% — fraction of trajectories terminated via Finish[] call
  - acc@≤2 — accuracy on fast-termination trajectories (≤2 steps); primary
              trajectory quality metric favoring ClusterFS's diversity-first selection
  - acc@3   — accuracy at modal trajectory length (step=3); complementary check

Usage
-----
  python aggregate_react_results.py                              # default: results_v2/
  python aggregate_react_results.py --results-dir results_v2    # explicit dir
  python aggregate_react_results.py --write-json                # also save aggregate.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stats helpers (sample std and 95% CI, consistent with academic convention)
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
# JSON loading
# ---------------------------------------------------------------------------

OPTIMIZER_DISPLAY = {
    "clusterfs": "ClusterFewshot",
    "miprov2":   "MIPROv2",
    "bfrs":      "BFRS",
    "baseline":  "Baseline",
}

MODEL_DISPLAY = {
    "Qwen2.5-7B-Instruct":    "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct":   "Qwen2.5-14B",
    "Llama-3.1-8B-Instruct":  "Llama-3.1-8B",
}


def _get_query_texts(search_queries: list) -> list[str]:
    """Extract plain-text search queries, skipping Finish[]-call entries."""
    texts = []
    for q in search_queries:
        if isinstance(q, dict) and "query" in q:
            texts.append(q["query"])
    return texts


def _trajectory_metrics(examples: list[dict]) -> dict:
    """Compute per-seed trajectory quality metrics from per_example_results."""
    n = len(examples)
    if not n:
        return {}

    finished = [e for e in examples if e["finished_via_tool"]]
    bucket_le2 = [e["score"] for e in examples if e["trajectory_steps"] <= 2]
    bucket_3   = [e["score"] for e in examples if e["trajectory_steps"] == 3]
    fin_scores = [e["score"] for e in finished]

    # Search query repeat rate (fraction of duplicated queries per multi-step trajectory)
    repeat_rates = []
    for e in examples:
        texts = _get_query_texts(e.get("search_queries", []))
        if len(texts) > 1:
            repeat_rates.append(1.0 - len(set(texts)) / len(texts))

    return {
        "fin_tool_pct": 100.0 * len(finished) / n,
        "acc_le2":      _mean(bucket_le2) if bucket_le2 else None,
        "acc_3":        _mean(bucket_3)   if bucket_3   else None,
        "acc_fin":      _mean(fin_scores) if fin_scores else None,
        "n_le2":        len(bucket_le2),
        "n_3":          len(bucket_3),
        "n_exhaust":    sum(1 for e in examples if not e["finished_via_tool"] and e["trajectory_steps"] == 20),
        "repeat_rate":  _mean(repeat_rates) if repeat_rates else None,
        "mean_steps":   _mean([e["trajectory_steps"] for e in examples]),
    }


def load_results(results_dir: str) -> dict:
    """
    Walk results_dir/<model>/<optimizer>/<seed>.json.
    Returns { (model, optimizer): [run_dict, ...] }
    where each run_dict contains scalar scores + per-seed trajectory metrics.
    """
    root = Path(results_dir)
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        # Standalone baseline
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
                if "_program" in seed_file.name:
                    continue
                try:
                    seed = int(seed_file.stem)
                except ValueError:
                    continue

                data = json.loads(seed_file.read_text())
                scores   = data.get("scores", {})
                timing   = data.get("timing_seconds", {})
                examples = data.get("per_example_results", [])

                run = {
                    "model":            model,
                    "optimizer":        optimizer,
                    "seed":             seed,
                    "baseline_score":   baseline_score,
                    "run_baseline":     scores.get("baseline"),
                    "optimized":        scores.get("optimized"),
                    "compile_s":        timing.get("compile_total"),
                    **_trajectory_metrics(examples),
                }
                groups[(model, optimizer)].append(run)

    return groups


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(groups: dict) -> list[dict]:
    """Compute cross-seed statistics for each (model, optimizer) group."""
    rows = []
    for (model, optimizer), runs in groups.items():
        def _collect(field):
            return [r[field] for r in runs if r.get(field) is not None]

        opt_scores  = _collect("optimized")
        compile_min = [r["compile_s"] / 60.0 for r in runs if r.get("compile_s")]
        bl_score    = runs[0]["baseline_score"] if runs else None

        traj = {}
        for field in ["fin_tool_pct", "acc_le2", "acc_3", "acc_fin", "repeat_rate", "mean_steps"]:
            vals = _collect(field)
            traj[field] = _mean(vals)
            traj[f"{field}_std"] = _std(vals)

        seeds = sorted(r["seed"] for r in runs)
        rows.append({
            "model":           model,
            "optimizer":       optimizer,
            "seeds":           seeds,
            "n":               len(opt_scores),
            "baseline":        bl_score,
            "mean_opt":        _mean(opt_scores),
            "std_opt":         _std(opt_scores),
            "ci95_opt":        _ci95(opt_scores),
            "delta_vs_bl":     (_mean(opt_scores) - bl_score) if (opt_scores and bl_score) else None,
            "mean_compile_min": _mean(compile_min),
            "per_seed_scores": {str(r["seed"]): r["optimized"] for r in runs},
            **traj,
        })

    # Sort: by model (alphabetical) then by mean accuracy descending
    rows.sort(key=lambda r: (r["model"], -(r["mean_opt"] or 0)))
    return rows


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

OPTIMIZER_ORDER = ["ClusterFewshot", "BFRS", "MIPROv2"]


def _pct(v, digits=2):
    if v is None:
        return "—"
    return f"{v:.{digits}f}%"


def _pp(v):
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}pp"


def _fmt_compile(v):
    if v is None:
        return "—"
    return f"{v:.1f}"


def print_markdown_tables(rows: list[dict], baseline_scores: dict):
    """Print paper-ready Markdown tables, one per model."""
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model, model_rows in sorted(by_model.items()):
        display = MODEL_DISPLAY.get(model, model)
        bl = baseline_scores.get(model)

        print(f"\n### {display}")
        print()

        # Sort by optimizer display order
        def _sort_key(r):
            d = OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])
            try:
                return OPTIMIZER_ORDER.index(d)
            except ValueError:
                return 99

        model_rows.sort(key=_sort_key)

        # Header
        print("| Optimizer | Accuracy (mean±std) | Δ vs Baseline | Compile (min) | fin_tool% | acc@≤2 | acc@3 |")
        print("|---|---|---|---|---|---|---|")

        # Baseline row
        if bl is not None:
            print(f"| Baseline | {bl:.2f}% | — | — | — | — | — |")

        for r in model_rows:
            opt = OPTIMIZER_DISPLAY.get(r["optimizer"], r["optimizer"])
            mean_opt = r["mean_opt"]
            std_opt  = r["std_opt"]
            delta    = r["delta_vs_bl"]
            compile_ = r["mean_compile_min"]
            fin      = r["fin_tool_pct"]
            acc_le2  = r["acc_le2"]
            acc_3    = r["acc_3"]

            acc_str = f"{mean_opt:.2f}±{std_opt:.2f}%" if mean_opt is not None else "—"
            fin_str = f"{fin:.1f}%" if fin is not None else "—"
            le2_str = f"{acc_le2:.4f}" if acc_le2 is not None else "—"
            a3_str  = f"{acc_3:.4f}"  if acc_3  is not None else "—"

            print(f"| {opt} | {acc_str} | {_pp(delta)} | {_fmt_compile(compile_)} | {fin_str} | {le2_str} | {a3_str} |")

        print()

    # Cross-model summary: compile efficiency
    print("\n---")
    print("\n### Compile efficiency summary (mean minutes across 3 seeds)\n")
    print("| Model | ClusterFewshot | BFRS | MIPROv2 | ClusterFS speedup vs BFRS | ClusterFS speedup vs MIPROv2 |")
    print("|---|---|---|---|---|---|")

    for model in sorted(by_model.keys()):
        display = MODEL_DISPLAY.get(model, model)
        model_rows = by_model[model]
        ct = {OPTIMIZER_DISPLAY.get(r["optimizer"]): r["mean_compile_min"] for r in model_rows}
        cfs = ct.get("ClusterFewshot")
        bfrs = ct.get("BFRS")
        mipro = ct.get("MIPROv2")
        speedup_bfrs  = f"{bfrs/cfs:.2f}×"  if (cfs and bfrs)  else "—"
        speedup_mipro = f"{mipro/cfs:.2f}×" if (cfs and mipro) else "—"
        print(f"| {display} | {_fmt_compile(cfs)} | {_fmt_compile(bfrs)} | {_fmt_compile(mipro)} | {speedup_bfrs} | {speedup_mipro} |")

    # acc@≤2 highlight table
    print("\n---")
    print("\n### acc@≤2 — Accuracy at fast termination (≤2 steps, mean across 3 seeds)\n")
    print("*Primary trajectory quality metric: diversity-first demos teach confident early answers.*\n")
    print("| Model | ClusterFewshot | BFRS | MIPROv2 | ClusterFS − BFRS | ClusterFS − MIPROv2 |")
    print("|---|---|---|---|---|---|")

    for model in sorted(by_model.keys()):
        display = MODEL_DISPLAY.get(model, model)
        model_rows = by_model[model]
        ct = {OPTIMIZER_DISPLAY.get(r["optimizer"]): r["acc_le2"] for r in model_rows}
        cfs = ct.get("ClusterFewshot")
        bfrs = ct.get("BFRS")
        mipro = ct.get("MIPROv2")

        def _diff(a, b):
            if a is None or b is None:
                return "—"
            d = a - b
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.4f}"

        cfs_s  = f"{cfs:.4f}"  if cfs  else "—"
        bfrs_s = f"{bfrs:.4f}" if bfrs else "—"
        mip_s  = f"{mipro:.4f}" if mipro else "—"
        print(f"| {display} | {cfs_s} | {bfrs_s} | {mip_s} | {_diff(cfs, bfrs)} | {_diff(cfs, mipro)} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate ReAct experiment results for paper tables.")
    parser.add_argument("--results-dir", default="results_v2",
                        help="Root directory containing <model>/<optimizer>/<seed>.json files.")
    parser.add_argument("--write-json", action="store_true",
                        help="Write aggregate.json alongside the tables.")
    args = parser.parse_args()

    groups = load_results(args.results_dir)
    if not groups:
        print(f"No results found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = aggregate(groups)

    # Standalone baseline scores keyed by model name
    baseline_scores = {}
    root = Path(args.results_dir)
    for model_dir in root.iterdir():
        bl_path = model_dir / "baseline" / "100.json"
        if bl_path.exists():
            d = json.loads(bl_path.read_text())
            baseline_scores[model_dir.name] = d["scores"].get("baseline")

    print_markdown_tables(rows, baseline_scores)

    if args.write_json:
        out = Path(args.results_dir) / "aggregate.json"
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nAggregate stats written: {out}")


if __name__ == "__main__":
    main()
