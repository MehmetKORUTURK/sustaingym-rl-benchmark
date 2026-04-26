"""
Action 3: SACLag lambda trajectory + cost-overshoot evidence.

Reviewer concern (counterverdict bullet 5 / minor): the SACLag staleness
mechanism is asserted in the paper but not isolated. This script extracts the
Lagrange multiplier (lambda) trajectory and the cost-vs-limit ratio over
training from existing SACLag and PPOLag progress.csv files (no retraining
required) and compares them.

Hypothesis (paper): SACLag's replay buffer retains high-cost transitions from
earlier policies, causing the cost critic to overestimate current cost,
inflating lambda, which triggers oscillatory over-correction.

Test: if the hypothesis is correct we should see
  (1) lambda trajectory: SACLag lambda inflates and oscillates; PPOLag lambda
      converges smoothly.
  (2) cost-vs-limit: SACLag overshoots the limit by 5-50x while PPOLag tracks.

Output:
  outputs/saclag/saclag_vs_ppolag_lambda.json
  outputs/saclag/saclag_vs_ppolag_report.md

Usage:
  python scripts/analysis/saclag_lambda_trajectory.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_ROOT = os.path.join(PROJECT_ROOT, "docs", "thesis", "logs", "logs_saferl_train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "saclag")


def find_progress_csvs(env_token: str, algo: str):
    """Find progress.csv for a given algorithm under an env directory."""
    pattern = os.path.join(
        LOG_ROOT,
        f"omnisafe_{env_token}",
        "**",
        f"*{algo}*",
        f"{algo}-*",
        "seed-*",
        "progress.csv",
    )
    return sorted(glob.glob(pattern, recursive=True))


def parse_cost_limit(path: str) -> float:
    """Pull the cost-limit number from the run directory name (CL_<num>)."""
    m = re.search(r"CL_([0-9]+(?:\.[0-9]+)?)", path)
    return float(m.group(1)) if m else float("nan")


def parse_seed(path: str) -> str:
    m = re.search(r"seed-([0-9]+)", path)
    return m.group(1) if m else "unknown"


def summarize_run(csv_path: str, algo: str) -> dict:
    """Extract lambda + cost trajectory from a progress.csv."""
    cl = parse_cost_limit(csv_path)
    seed = parse_seed(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"path": csv_path, "error": str(exc)}

    if "Metrics/LagrangeMultiplier" not in df.columns:
        return {"path": csv_path, "error": "no LagrangeMultiplier column"}

    lam = df["Metrics/LagrangeMultiplier"].to_numpy()
    cost = df["Metrics/EpCost"].to_numpy() if "Metrics/EpCost" in df.columns else np.array([])
    test_cost = df["Metrics/TestEpCost"].to_numpy() if "Metrics/TestEpCost" in df.columns else np.array([])
    epochs = df["Train/Epoch"].to_numpy() if "Train/Epoch" in df.columns else np.arange(len(lam))

    valid = ~np.isnan(lam)
    lam_valid = lam[valid]

    n_epochs = int(len(lam_valid))
    if n_epochs == 0:
        return {"path": csv_path, "error": "all NaN lambda"}

    last_quarter = lam_valid[3 * n_epochs // 4 :]

    # oscillation: count zero-crossings of (lambda_t - lambda_{t-1}) sign in last half
    diffs = np.diff(lam_valid[n_epochs // 2 :])
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))

    # cost overshoot relative to the cost limit
    if not np.isnan(cl) and len(cost) > 0:
        valid_cost = cost[~np.isnan(cost)]
        late_cost = valid_cost[len(valid_cost) // 2 :]
        cost_to_limit_mean_late = float(np.mean(late_cost) / cl) if cl > 0 else float("nan")
        cost_overshoot_ratio = float(np.max(valid_cost) / cl) if cl > 0 else float("nan")
    else:
        cost_to_limit_mean_late = float("nan")
        cost_overshoot_ratio = float("nan")

    return {
        "path": csv_path,
        "algo": algo,
        "cost_limit": cl,
        "seed": seed,
        "n_epochs": n_epochs,
        "lambda_min": float(np.min(lam_valid)),
        "lambda_max": float(np.max(lam_valid)),
        "lambda_mean": float(np.mean(lam_valid)),
        "lambda_late_mean": float(np.mean(last_quarter)),
        "lambda_late_std": float(np.std(last_quarter)),
        "lambda_late_cv": float(np.std(last_quarter) / np.mean(last_quarter))
        if np.mean(last_quarter) > 1e-12
        else float("nan"),
        "lambda_oscillation_sign_changes": sign_changes,
        "cost_to_limit_mean_late": cost_to_limit_mean_late,
        "cost_overshoot_ratio": cost_overshoot_ratio,
        "lambda_trajectory": lam_valid.tolist(),
        "cost_trajectory": cost[~np.isnan(cost)].tolist() if len(cost) > 0 else [],
        "test_cost_trajectory": test_cost[~np.isnan(test_cost)].tolist() if len(test_cost) > 0 else [],
    }


def aggregate(runs):
    """Aggregate across runs (per algorithm)."""
    by_algo = {}
    for r in runs:
        if "error" in r:
            continue
        by_algo.setdefault(r["algo"], []).append(r)

    summary = {}
    for algo, items in by_algo.items():
        summary[algo] = {
            "n_runs": len(items),
            "lambda_late_mean_avg": float(np.mean([r["lambda_late_mean"] for r in items])),
            "lambda_late_std_avg": float(np.mean([r["lambda_late_std"] for r in items])),
            "lambda_late_cv_avg": float(
                np.mean(
                    [r["lambda_late_cv"] for r in items if not np.isnan(r["lambda_late_cv"])]
                )
            ),
            "lambda_oscillation_sign_changes_avg": float(
                np.mean([r["lambda_oscillation_sign_changes"] for r in items])
            ),
            "cost_to_limit_late_avg": float(
                np.mean(
                    [
                        r["cost_to_limit_mean_late"]
                        for r in items
                        if not np.isnan(r["cost_to_limit_mean_late"])
                    ]
                )
            ),
            "cost_overshoot_ratio_max": float(
                np.max(
                    [
                        r["cost_overshoot_ratio"]
                        for r in items
                        if not np.isnan(r["cost_overshoot_ratio"])
                    ]
                )
            ),
        }
    return summary


def write_report(by_env_summary, by_env_runs, output_path):
    lines = []
    lines.append("# SACLag vs PPOLag: Lagrange-Multiplier and Cost Tracking")
    lines.append("")
    lines.append(
        "Action 3 of the no-RL-training revision plan. Extracts the lambda trajectory and"
        " cost-overshoot statistics from existing OmniSafe progress.csv logs (no retraining)."
    )
    lines.append("")
    for env, summary in by_env_summary.items():
        lines.append(f"## {env.title()}")
        lines.append("")
        lines.append("| Algorithm | n_runs | lambda_late mean | lambda_late std | lambda CV | osc.\\ sign-changes | cost/limit (late) | overshoot max |")
        lines.append("| --------- | ------ | ---------------- | --------------- | --------- | ------------------ | ----------------- | ------------- |")
        for algo in ["PPOLag", "SACLag"]:
            if algo not in summary:
                continue
            s = summary[algo]
            lines.append(
                f"| {algo} | {s['n_runs']} | "
                f"{s['lambda_late_mean_avg']:.3f} | "
                f"{s['lambda_late_std_avg']:.3f} | "
                f"{s['lambda_late_cv_avg']:.3f} | "
                f"{s['lambda_oscillation_sign_changes_avg']:.1f} | "
                f"{s['cost_to_limit_late_avg']:.2f}x | "
                f"{s['cost_overshoot_ratio_max']:.2f}x |"
            )
        lines.append("")

        # per-run breakdown of cost / cost_limit
        lines.append(f"### Per-run cost-vs-limit (late training mean) -- {env}")
        lines.append("")
        runs = by_env_runs[env]
        runs_sorted = sorted(runs, key=lambda r: (r["algo"], r["cost_limit"]))
        lines.append("| Algo | cost_limit | seed | lambda_late mean +/- std | osc.\\ sign-changes | cost/limit |")
        lines.append("| ---- | ---------- | ---- | ----------------------- | ------------------ | ---------- |")
        for r in runs_sorted:
            lines.append(
                f"| {r['algo']} | {r['cost_limit']:.1f} | {r['seed']} | "
                f"{r['lambda_late_mean']:.3f} +/- {r['lambda_late_std']:.3f} | "
                f"{r['lambda_oscillation_sign_changes']} | "
                f"{r['cost_to_limit_mean_late']:.2f}x |"
            )
        lines.append("")
    lines.append("## Mechanism interpretation")
    lines.append("")
    lines.append(
        "Per the paper's hypothesis, SACLag's replay buffer retains high-cost transitions"
        " from earlier policies, causing the cost critic to overestimate the current policy's"
        " cost, inflating lambda, which triggers oscillatory over-correction. The trace data"
        " above support the hypothesis if SACLag exhibits (a) a substantially higher late-training"
        " lambda coefficient of variation than PPOLag (oscillatory dynamics rather than convergence),"
        " (b) more sign-changes in d(lambda)/dt during the second half of training, and (c)"
        " systematically larger cost-to-limit ratios -- i.e., SACLag does not actually track the"
        " prescribed limit. PPOLag, computing cost estimates from on-policy fresh rollouts only,"
        " avoids the staleness loop and converges cleanly."
    )
    lines.append("")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    by_env_runs = {}
    by_env_summary = {}

    for env in ["building", "evcharging", "cogen"]:
        runs = []
        for algo in ["PPOLag", "SACLag"]:
            paths = find_progress_csvs(env, algo)
            for p in paths:
                rec = summarize_run(p, algo)
                runs.append(rec)
        valid = [r for r in runs if "error" not in r]
        by_env_runs[env] = valid
        by_env_summary[env] = aggregate(valid)
        print(f"\n=== {env.upper()} ===")
        print(f"  total progress.csv files found: {len(runs)}")
        print(f"  valid runs (with lambda data): {len(valid)}")
        for algo in ["PPOLag", "SACLag"]:
            if algo in by_env_summary[env]:
                s = by_env_summary[env][algo]
                print(
                    f"  {algo}: n={s['n_runs']}  lambda_late={s['lambda_late_mean_avg']:.3f}+/-{s['lambda_late_std_avg']:.3f}  "
                    f"CV={s['lambda_late_cv_avg']:.3f}  osc.sc={s['lambda_oscillation_sign_changes_avg']:.1f}  "
                    f"cost/limit={s['cost_to_limit_late_avg']:.2f}x  max overshoot={s['cost_overshoot_ratio_max']:.2f}x"
                )

    # save (drop heavy trajectories from JSON to keep file small)
    light = {}
    for env, runs in by_env_runs.items():
        light[env] = []
        for r in runs:
            light_r = {k: v for k, v in r.items() if k not in {"lambda_trajectory", "cost_trajectory", "test_cost_trajectory"}}
            light[env].append(light_r)

    with open(os.path.join(OUTPUT_DIR, "saclag_vs_ppolag_lambda.json"), "w") as fh:
        json.dump({"summary": by_env_summary, "runs": light}, fh, indent=2)

    write_report(
        by_env_summary,
        by_env_runs,
        os.path.join(OUTPUT_DIR, "saclag_vs_ppolag_report.md"),
    )

    print(f"\nSaved {OUTPUT_DIR}/saclag_vs_ppolag_lambda.json")
    print(f"Saved {OUTPUT_DIR}/saclag_vs_ppolag_report.md")


if __name__ == "__main__":
    main()
