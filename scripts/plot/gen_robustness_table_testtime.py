"""
Generate test-time robustness ranking table from test evaluation logs.
No pandas/numpy dependency — pure stdlib.

Usage:
    python scripts/plot/gen_robustness_table_testtime.py
"""

import csv
import re
import os
import statistics
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parents[2] / "logs_std_test"

MODERATE = {
    "evcharging": {"obs": [0.1, 0.2], "action": [0.1, 0.2], "env": [0.1, 0.2]},
    "building":   {"obs": [0.05, 0.15], "action": [0.05, 0.15], "env": [0.1, 0.3]},
    "cogen":      {"obs": [3.0, 5.0], "action": [0.1, 0.2], "env": [1.0, 3.0]},
}

ENV_NAMES = {"evcharging": "EV", "building": "Bldg", "cogen": "Cogen"}
ALGOS = ["PPO", "SAC", "TD3"]


def parse_noise(dirname):
    result = {"obs": 0.0, "action": 0.0, "env": 0.0}
    m = re.search(r"NOISE_([\d.]+)", dirname)
    if m: result["obs"] = float(m.group(1))
    m = re.search(r"ACT_([\d.]+)", dirname)
    if m: result["action"] = float(m.group(1))
    m = re.search(r"ENV_([\d.]+)", dirname)
    if m: result["env"] = float(m.group(1))
    return result


def is_single_channel(noise):
    active = [ch for ch, v in noise.items() if v > 0]
    if len(active) == 0: return "clean"
    if len(active) == 1: return active[0]
    return None


def read_test_reward(run_dir):
    csv_path = run_dir / "episode_results.csv"
    if not csv_path.exists():
        return None
    try:
        rewards = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row["total_reward"]))
        if not rewards:
            return None
        return statistics.mean(rewards)
    except Exception:
        return None


def collect_test_data(env, algo):
    env_algo_dir = TEST_DIR / f"{env}_{algo}"
    if not env_algo_dir.exists():
        return {}
    data = {"obs": {}, "action": {}, "env": {}}
    clean_rewards = []
    for entry in sorted(os.listdir(env_algo_dir)):
        run_dir = env_algo_dir / entry
        if not run_dir.is_dir():
            continue
        noise = parse_noise(entry)
        channel = is_single_channel(noise)
        if channel is None:
            continue
        reward = read_test_reward(run_dir)
        if reward is None:
            continue
        if channel == "clean":
            clean_rewards.append(reward)
        else:
            level = noise[channel]
            if level not in data[channel] or reward > data[channel][level]:
                data[channel][level] = reward
    if clean_rewards:
        clean_best = statistics.mean(clean_rewards)
        for ch in data:
            data[ch][0.0] = clean_best
    return data


def fmt(val, p=2):
    if abs(val) < 0.1: return f"{val:.3f}"
    if abs(val) < 10: return f"{val:.{p}f}"
    return f"{val:.1f}"


def main():
    print("\n=== TEST-TIME ROBUSTNESS TABLE ===\n")

    header = f"{'Env':<6} {'Algo':<5} {'Clean':>8}"
    for ch in ["State (PS)", "Action (PA)", "Dynamics (PD)"]:
        header += f"  | {'Pert':>8} {'Delta':>8} {'%':>8}"
    print(header)
    print("-" * 105)

    latex_lines = []

    for env in ["evcharging", "building", "cogen"]:
        first = True
        for algo in ALGOS:
            data = collect_test_data(env, algo)
            if not data or not any(data[ch] for ch in data):
                continue

            clean = data.get("obs", {}).get(0.0)
            if clean is None: clean = data.get("action", {}).get(0.0)
            if clean is None: clean = data.get("env", {}).get(0.0)
            if clean is None: continue

            row = f"{ENV_NAMES[env] if first else '':.<6} {algo:<5} {fmt(clean):>8}"
            latex = f"& {algo} & ${fmt(clean)}$"

            for ch in ["obs", "action", "env"]:
                levels = MODERATE[env].get(ch, [])
                vals = [data[ch][lv] for lv in levels if lv in data[ch]]
                if vals:
                    avg = statistics.mean(vals)
                    delta = avg - clean
                    pct = (delta / abs(clean)) * 100 if clean != 0 else 0
                    row += f"  | {fmt(avg):>8} {delta:>+8.2f} {pct:>7.1f}%"
                    latex += f" & ${fmt(avg)}$ & ${delta:+.2f}$ & ${pct:.1f}\\%$"
                else:
                    row += f"  | {'---':>8} {'---':>8} {'---':>8}"
                    latex += " & --- & --- & ---"

            print(row)
            latex += " \\\\"
            latex_lines.append(latex)
            first = False
        print()

    print("\n=== LaTeX TABLE BODY ===\n")
    for line in latex_lines:
        print(line)


if __name__ == "__main__":
    main()
