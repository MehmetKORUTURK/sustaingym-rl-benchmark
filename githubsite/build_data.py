"""
Aggregate thesis logs into a compact JSON for the GitHub Pages site.

Reads:
  docs/thesis/logs/multiseed_stats.json
  docs/thesis/logs/multiseed_test_stats.json
  docs/thesis/logs/logs_baseline_test/<env>_<baseline>/.../episode_results.csv
  docs/thesis/logs/logs_marl_train/<env>_<algo>/.../metrics.csv
  docs/thesis/logs/logs_saferl_train/omnisafe_<env>/<run>/.../progress.csv

Writes:
  githubsite/site_data.json   (deep-merged, downsampled, ready for the browser)

Usage:
  python githubsite/build_data.py
"""
import csv
import glob
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "thesis" / "logs"
OUT  = Path(__file__).resolve().parent / "site_data.json"

ENVS = ["evcharging", "building", "cogen"]


# ---------- helpers ----------------------------------------------------
def downsample(points, n=80):
    if len(points) <= n:
        return points
    step = max(1, len(points) // n)
    return points[::step][:n]


def csv_stats(path, key="total_reward"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if key not in rows[0]:
        # fall back to second column
        key = list(rows[0].keys())[2] if len(rows[0]) >= 3 else list(rows[0].keys())[-1]
    vals = [float(r[key]) for r in rows]
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def first_match(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(str(p)))
        if hits:
            return hits[0]
    return None


# ---------- multiseed perturbation results -----------------------------
def load_perturbation():
    f = LOGS / "multiseed_stats.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        raw = json.load(fh)
    out = {}
    for combo, levels in raw.items():
        env, algo = combo.split("_", 1)
        out.setdefault(env, {}).setdefault(algo, {})
        for label, stats in levels.items():
            out[env][algo][label] = {
                "mean": round(stats["mean"], 4),
                "std":  round(stats["std"], 4),
                "n":    stats["n"],
            }
    return out


# ---------- baselines --------------------------------------------------
BASELINE_DIRS = {
    "evcharging": ["Greedy", "MPC", "OfflineOptimal", "Random"],
    "building":   ["MPC", "Random", "DoNothing"],
    "cogen":      ["Random", "DoNothing"],
}

def load_baselines():
    out = {}
    for env, names in BASELINE_DIRS.items():
        out[env] = {}
        for n in names:
            pat = LOGS / "logs_baseline_test" / f"{env}_{n}" / "*" / "episode_results.csv"
            f = first_match(pat)
            if not f:
                continue
            stats = csv_stats(f, "total_reward")
            if stats:
                out[env][n] = stats
    return out


# ---------- noise sweep (C_POST panel.png data) ------------------------
import re

CHANNEL_FROM_DIRNAME = re.compile(
    r"NOISE_(?P<obs>[\d.]+)_ACT_(?P<act>[\d.]+)_ENV_(?P<env>[\d.]+)"
)

def _classify_run(run_dir_name):
    """Return (channel, level) where channel is 'PS'/'PA'/'PD' and level is float.

    Matches the convention in stdrl_post_plot.py: a run sweeps exactly one channel.
    Returns (None, 0.0) for the all-zero baseline run.
    """
    m = CHANNEL_FROM_DIRNAME.search(run_dir_name)
    if not m:
        return None, None
    obs = float(m.group("obs"))
    act = float(m.group("act"))
    env = float(m.group("env"))
    nz = [(c, v) for c, v in (("PS", obs), ("PA", act), ("PD", env)) if v > 0]
    if not nz:
        return "baseline", 0.0
    if len(nz) > 1:
        return None, None  # combined-perturbation — ignored for the panel
    return nz[0][0], nz[0][1]


def load_noise_sweep():
    """Mirror C_POST/<env>/<algo>/panel.png: per-channel reward vs noise level."""
    out = {}
    for env in ENVS:
        out[env] = {}
        for algo in ("PPO", "SAC", "TD3"):
            algo_root = LOGS / "logs_std_test" / f"{env}_{algo}"
            if not algo_root.exists():
                continue
            buckets = {"PS": [], "PA": [], "PD": []}
            baseline_stats = None
            for run_dir in algo_root.iterdir():
                if not run_dir.is_dir():
                    continue
                channel, level = _classify_run(run_dir.name)
                if channel is None:
                    continue
                csv_path = run_dir / "episode_results.csv"
                if not csv_path.exists():
                    continue
                stats = csv_stats(csv_path, "total_reward")
                if stats is None:
                    continue
                if channel == "baseline":
                    baseline_stats = stats
                    continue
                buckets[channel].append({
                    "level": level,
                    "mean": stats["mean"],
                    "std":  stats["std"],
                    "n":    stats["n"],
                })
            for ch in buckets:
                buckets[ch].sort(key=lambda r: r["level"])
            if baseline_stats:
                # prepend the zero baseline to each channel for plotting
                for ch in buckets:
                    buckets[ch].insert(0, {
                        "level": 0.0,
                        "mean":  baseline_stats["mean"],
                        "std":   baseline_stats["std"],
                        "n":     baseline_stats["n"],
                    })
            if any(buckets.values()):
                out[env][algo] = buckets
    return out


# ---------- MARL training curves ---------------------------------------
MARL_ALGOS = ["PPO", "SAC", "APPO", "IMPALA"]

def load_marl_curves():
    out = {}
    for env in ENVS:
        out[env] = {}
        for algo in MARL_ALGOS:
            pat = LOGS / "logs_marl_train" / f"{env}_{algo}" / "*" / "metrics.csv"
            f = first_match(pat)
            if not f:
                continue
            with open(f, newline="") as fh:
                rows = list(csv.DictReader(fh))
            pts = [
                {"x": float(r["iteration"]), "y": float(r["mean_reward"])}
                for r in rows
                if r.get("mean_reward")
            ]
            if pts:
                out[env][algo] = downsample(pts, n=80)
    return out


# ---------- Safe RL learning curves (cost & reward) --------------------
def load_saferl():
    out = {}
    for env in ENVS:
        run_root = LOGS / "logs_saferl_train" / f"omnisafe_{env}"
        if not run_root.exists():
            continue
        out[env] = {}
        for run_dir in sorted(run_root.iterdir()):
            if not run_dir.is_dir():
                continue
            # name pattern: 20260417_220019_PPOLag_CL_5.0
            parts = run_dir.name.split("_")
            if len(parts) < 5:
                continue
            algo = parts[2]
            try:
                cl = float(parts[4])
            except ValueError:
                continue
            # progress.csv is two levels deep
            prog = first_match(run_dir / "*" / "*" / "progress.csv")
            if not prog:
                continue
            try:
                with open(prog, newline="") as fh:
                    rows = list(csv.DictReader(fh))
            except Exception:
                continue
            if not rows:
                continue
            # find reward / cost column (omnisafe varies)
            rkey = next((k for k in rows[0] if "Train/Reward" in k or "Reward" == k or "Metrics/EpRet" in k), None)
            ckey = next((k for k in rows[0] if "Train/Cost" in k or "Metrics/EpCost" in k), None)
            xkey = next((k for k in rows[0] if "Train/TotalSteps" in k or "Metrics/CurrentEpoch" in k or "epoch" in k.lower() or "step" in k.lower()), None)
            if not rkey or not ckey:
                continue
            pts = []
            for i, r in enumerate(rows):
                try:
                    pts.append({
                        "x": float(r[xkey]) if xkey else i,
                        "r": float(r[rkey]),
                        "c": float(r[ckey]),
                    })
                except (ValueError, KeyError):
                    pass
            if not pts:
                continue
            key = f"{algo}_CL{cl:g}"
            # keep only one per key (first one found)
            if key not in out[env]:
                out[env][key] = downsample(pts, n=80)
    return out


# ---------- baseline test stats (clean RL) -----------------------------
def load_clean_rl_test():
    f = LOGS / "multiseed_test_stats.json"
    if not f.exists():
        return {}
    with open(f) as fh:
        raw = json.load(fh)
    out = {}
    for combo, stats in raw.items():
        env, algo = combo.split("_", 1)
        out.setdefault(env, {})[algo] = {
            "mean": round(stats["mean"], 4),
            "std":  round(stats["std"], 4),
            "n":    stats["n"],
        }
    return out


# ---------- hyperparameters --------------------------------------------
HYPERPARAMS = {
    "evcharging": {
        "PPO": {"lr": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
                "policy": "MultiInputPolicy", "total_steps": 9_000_000, "framework": "SB3"},
        "SAC": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MultiInputPolicy", "total_steps": 9_000_000, "framework": "SB3"},
        "TD3": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MultiInputPolicy", "total_steps": 9_000_000, "framework": "SB3",
                "note": "Unstable in EV/Building"},
        "PPOLag":  {"cost_limits": [3, 5, 25, 1000], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 100.0},
        "CPO":     {"cost_limits": [3, 5, 25, 1000], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 100.0},
        "OnCRPO":  {"cost_limits": [3, 5, 25, 1000], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 100.0},
        "FOCOPS":  {"cost_limits": [3, 5, 25, 1000], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 100.0},
        "MARL_PPO":    {"workers": 10, "iterations": 32_000, "train_batch": 2880,
                        "rollout_fragment": 288, "lr": 3e-4, "framework": "Ray RLlib"},
        "MARL_SAC":    {"workers": 10, "iterations": 32_000, "train_batch": 2880,
                        "rollout_fragment": 288, "lr": 3e-4, "framework": "Ray RLlib"},
        "MARL_APPO":   {"workers": 10, "iterations": 32_000, "lr": 5e-5, "note": "lr hardcoded",
                        "framework": "Ray RLlib"},
        "MARL_IMPALA": {"workers": 10, "iterations": 32_000, "lr": 5e-5, "grad_clip": 5.0,
                        "framework": "Ray RLlib"},
    },
    "building": {
        "PPO": {"lr": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
                "policy": "MlpPolicy", "total_steps": 9_000_000, "framework": "SB3",
                "config": "OfficeSmall, Hot-Dry, Tucson, beta=0.5"},
        "SAC": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MlpPolicy", "total_steps": 9_000_000, "framework": "SB3"},
        "TD3": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MlpPolicy", "total_steps": 9_000_000, "framework": "SB3",
                "note": "Unstable in EV/Building"},
        "PPOLag":  {"cost_limits": [25, 50, 100, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 1.0, "network": "[128,128,128]"},
        "CPO":     {"cost_limits": [25, 50, 100, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 1.0, "network": "[128,128,128]"},
        "OnCRPO":  {"cost_limits": [25, 50, 100, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 1.0, "network": "[128,128,128]"},
        "FOCOPS":  {"cost_limits": [25, 50, 100, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 1.0, "network": "[128,128,128]"},
        "MARL_PPO":    {"workers": 4, "iterations": 32_000, "train_batch": 1152, "lr": 3e-4,
                        "rollout_fragment": 288, "framework": "Ray RLlib", "reward_beta": 0.5},
        "MARL_SAC":    {"workers": 4, "iterations": 32_000, "train_batch": 1152, "lr": 3e-4,
                        "rollout_fragment": 288, "framework": "Ray RLlib"},
        "MARL_APPO":   {"workers": 4, "iterations": 32_000, "lr": 5e-5, "framework": "Ray RLlib"},
        "MARL_IMPALA": {"workers": 4, "iterations": 32_000, "lr": 5e-5, "grad_clip": 5.0,
                        "framework": "Ray RLlib"},
    },
    "cogen": {
        "PPO": {"lr": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
                "policy": "MlpPolicy (Box)", "total_steps": 3_000_000, "framework": "SB3",
                "note": "Wrapped via MyCogenEnv (Dict->Box)"},
        "SAC": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MlpPolicy", "total_steps": 3_000_000, "framework": "SB3"},
        "TD3": {"lr": 3e-4, "buffer_size": 1_000_000, "batch_size": 256, "tau": 0.005, "gamma": 0.99,
                "policy": "MlpPolicy", "total_steps": 3_000_000, "framework": "SB3"},
        "PPOLag":  {"cost_limits": [10, 25, 50, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 0.005, "network": "[128,128,128]"},
        "CPO":     {"cost_limits": [10, 25, 50, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 0.005, "network": "[128,128,128]"},
        "OnCRPO":  {"cost_limits": [10, 25, 50, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 0.005, "network": "[128,128,128]"},
        "FOCOPS":  {"cost_limits": [10, 25, 50, 200], "steps_per_epoch": 2000, "total_steps": 6_000_000,
                    "framework": "OmniSafe", "cost_scale": 0.005, "network": "[128,128,128]"},
        "MARL_PPO":    {"workers": 4, "iterations": 3000, "train_batch": 4000,
                        "rollout_fragment": 200, "lr": 3e-4, "framework": "Ray RLlib"},
        "MARL_APPO":   {"workers": 4, "iterations": 3000, "lr": 5e-5, "framework": "Ray RLlib"},
        "MARL_IMPALA": {"workers": 4, "iterations": 3000, "lr": 1e-4, "grad_clip": 40.0,
                        "framework": "Ray RLlib", "note": "Cogen-specific tuning"},
    },
}


# ---------- environment metadata ---------------------------------------
META = {
    "evcharging": {
        "label": "EV Charging",
        "color": "rose",
        "kappa": 0.024,
        "steps_per_episode": 288,
        "delta_t_minutes": 5,
        "marl_agents": 54,
        "obs_dim": 146,
        "action_space": "[0,1]^54",
        "cmdp_cost": "Excess charge (network violations)",
        "data_source": "ACN-Data, Caltech",
    },
    "building": {
        "label": "Building HVAC",
        "color": "mint",
        "kappa": 0.145,
        "steps_per_episode": 288,
        "delta_t_minutes": 5,
        "marl_agents": 6,
        "obs_dim": "n+4",
        "action_space": "[-1,1]^6",
        "cmdp_cost": "Temperature deadband violation",
        "data_source": "ASHRAE 90.1-2019 OfficeSmall, Tucson, EnergyPlus",
    },
    "cogen": {
        "label": "Cogeneration",
        "color": "lavender",
        "kappa": 0.860,
        "steps_per_episode": 96,
        "delta_t_minutes": 15,
        "marl_agents": 4,
        "obs_dim": "variable",
        "action_space": "Dict (15-d, mixed cont/discrete)",
        "cmdp_cost": "Turbine ramp cost",
        "data_source": "ONNX surrogate trained on operational data",
    },
}


# ---------- assemble ---------------------------------------------------
def main():
    perturbation = load_perturbation()
    noise_sweep  = load_noise_sweep()
    baselines    = load_baselines()
    marl         = load_marl_curves()
    saferl       = load_saferl()
    clean_rl     = load_clean_rl_test()

    out = {}
    for env in ENVS:
        out[env] = {
            "meta":          META[env],
            "hyperparams":   HYPERPARAMS[env],
            "perturbation":  perturbation.get(env, {}),
            "noise_sweep":   noise_sweep.get(env, {}),
            "baselines":     baselines.get(env, {}),
            "clean_rl":      clean_rl.get(env, {}),
            "marl_curves":   marl.get(env, {}),
            "saferl_curves": saferl.get(env, {}),
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({size_kb:.1f} KB)")
    for env in ENVS:
        e = out[env]
        sweep_summary = ", ".join(
            f"{a}: {sum(len(v) for v in ch.values())}"
            for a, ch in e["noise_sweep"].items()
        )
        print(f"  {env:11s}  perturb={len(e['perturbation'])} algos  "
              f"sweep=[{sweep_summary}]  "
              f"baselines={len(e['baselines'])}  marl={len(e['marl_curves'])}  "
              f"saferl_runs={len(e['saferl_curves'])}")


if __name__ == "__main__":
    main()
