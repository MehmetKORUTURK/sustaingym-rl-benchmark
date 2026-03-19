"""
baseline_testing.py — Evaluation script for non-RL baseline algorithms on SustainGym environments.

Runs baselines (Random, Greedy, MPC, OfflineOptimal, DoNothing) and produces
output in the same format as stdrl_testing.py so results can be plotted together.

Output: logs_baseline_test/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
  - test_config.json, episode_results.csv, evaluation_summary.txt, evaluation_results.txt

Usage:
    python scripts/test/baseline_testing.py --env evcharging --algo Greedy --n-eval 100
    python scripts/test/baseline_testing.py --env evcharging --algo MPC --n-eval 100
    python scripts/test/baseline_testing.py --env evcharging --algo OfflineOptimal --n-eval 50
    python scripts/test/baseline_testing.py --env building --algo MPC --n-eval 100
    python scripts/test/baseline_testing.py --env building --algo DoNothing --n-eval 100
    python scripts/test/baseline_testing.py --env cogen --algo Random --n-eval 100 --rm 300
"""

import os
import sys
import json
import argparse
import datetime
import time
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.plot.colored import color_text, Colors

# Reuse env creation and metric extraction from stdrl_testing
from scripts.test.stdrl_testing import (
    create_test_environment,
    _extract_step_metrics,
    _extract_episode_metrics,
    save_episode_csv,
    save_trajectory_csv,
    save_summary,
    save_legacy_results,
    _write_stat,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline Algorithm Registry
# ═══════════════════════════════════════════════════════════════════════════════

# Valid baselines per environment
VALID_BASELINES = {
    "evcharging": ["Random", "Greedy", "MPC", "OfflineOptimal"],
    "building":   ["Random", "MPC", "DoNothing"],
    "cogen":      ["Random", "DoNothing"],
}


def create_baseline(algo_name, env, env_type):
    """Instantiate the appropriate baseline algorithm.

    Returns a callable with interface: action = baseline.get_action(obs)
    and optional baseline.reset() between episodes.
    """
    if algo_name == "Random":
        return RandomBaseline(env, env_type)

    elif algo_name == "DoNothing":
        return DoNothingBaseline(env, env_type)

    elif algo_name == "Greedy":
        if env_type != "evcharging":
            print(color_text(f"Error: Greedy baseline only available for evcharging", Colors.RED))
            sys.exit(1)
        from algorithms.evcharging.baselines import GreedyAlgorithm
        return EVChargingBaselineWrapper(GreedyAlgorithm(env), "Greedy")

    elif algo_name == "MPC":
        if env_type == "evcharging":
            from algorithms.evcharging.baselines import MPC
            return EVChargingBaselineWrapper(MPC(env, lookahead=12), "MPC")
        elif env_type == "building":
            from algorithms.building.mpc_controller import MPCAgent
            return BuildingMPCWrapper(env, beta=0.5, pnorm=2)
        else:
            print(color_text(f"Error: MPC baseline not available for {env_type}", Colors.RED))
            sys.exit(1)

    elif algo_name == "OfflineOptimal":
        if env_type != "evcharging":
            print(color_text(f"Error: OfflineOptimal only available for evcharging", Colors.RED))
            sys.exit(1)
        from algorithms.evcharging.baselines import OfflineOptimal
        return EVChargingBaselineWrapper(OfflineOptimal(env), "OfflineOptimal")

    else:
        print(color_text(f"Error: Unknown baseline '{algo_name}'", Colors.RED))
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline Wrappers (unified interface)
# ═══════════════════════════════════════════════════════════════════════════════

class RandomBaseline:
    """Random action baseline for any environment."""
    def __init__(self, env, env_type):
        self.env = env
        self.env_type = env_type
        self.name = "Random"

    def get_action(self, obs):
        return self.env.action_space.sample()

    def reset(self):
        pass


class DoNothingBaseline:
    """Minimal-action baseline.
    Building: zero action = no HVAC (building drifts to outdoor temp).
    Cogen: lower-bound action = minimum power operation.
    """
    def __init__(self, env, env_type):
        self.env = env
        self.env_type = env_type
        self.name = "DoNothing"
        # Use action_space.low to respect non-zero lower bounds (e.g. Cogen discrete actions)
        if env_type == "building":
            self._action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            self._action = env.action_space.low.copy().astype(np.float32)

    def get_action(self, obs):
        return self._action.copy()

    def reset(self):
        pass


class EVChargingBaselineWrapper:
    """Wraps algorithms.evcharging.baselines classes to unified interface."""
    def __init__(self, algo, name):
        self.algo = algo
        self.name = name

    def get_action(self, obs):
        return self.algo.get_action(obs)

    def reset(self):
        self.algo.reset()


class BuildingMPCWrapper:
    """Wraps Building MPCAgent to unified interface.

    MPCAgent.predict() expects env.B_d but current BuildingEnv stores
    BD_d (combined B and D matrices). This wrapper patches the attribute
    before calling predict().
    """
    def __init__(self, env, beta=0.5, pnorm=2):
        self.env = env
        self.beta = beta
        self.pnorm = pnorm
        self.name = "MPC"
        self.agent = None

    def get_action(self, obs):
        # Lazy-init MPC agent (needs env to be reset first)
        if self.agent is None:
            from algorithms.building.mpc_controller import MPCAgent
            # Patch: MPCAgent expects env.B_d, but env has BD_d
            # BD_d columns: [Occupower, ground_temp, out_temp, action..., ghi]
            # This matches exactly what MPCAgent expects from B_d
            if not hasattr(self.env, 'B_d') and hasattr(self.env, 'BD_d'):
                self.env.B_d = self.env.BD_d
            self.agent = MPCAgent(self.env, beta=self.beta, pnorm=self.pnorm)
        action, _ = self.agent.predict()
        return action

    def reset(self):
        # Re-create agent after each reset (epoch changes)
        self.agent = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Loop (mirrors stdrl_testing.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_baseline(baseline, env, n_episodes, env_type, seed=42,
                      collect_trajectories=False):
    """Run evaluation episodes collecting per-episode metrics.

    Mirrors stdrl_testing.py:evaluate_with_data_collection() output format.
    """
    episode_data = []
    trajectory_data = [] if collect_trajectories else None

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        obs, info = env.reset(seed=ep_seed)
        baseline.reset()

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_infos = []

        if collect_trajectories:
            ep_trajectory = []

        while not done:
            action = baseline.get_action(obs)

            # Clip action to env bounds (safety)
            if hasattr(env.action_space, 'low'):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward_val, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            info = deepcopy(info)

            ep_reward += float(reward_val)
            ep_steps += 1
            ep_infos.append(info)

            if collect_trajectories:
                step_data = {"step": ep_steps, "reward": float(reward_val)}
                step_data.update(_extract_step_metrics(info, env_type))
                ep_trajectory.append(step_data)

        # Aggregate episode metrics (identical to stdrl_testing.py)
        ep_record = {
            "episode": ep_idx,
            "seed": ep_seed,
            "total_reward": float(ep_reward),
            "episode_length": ep_steps,
        }
        ep_record.update(_extract_episode_metrics(ep_infos, env_type, ep_reward))
        episode_data.append(ep_record)

        if collect_trajectories:
            trajectory_data.append(ep_trajectory)

        # Progress reporting
        if (ep_idx + 1) % max(1, n_episodes // 10) == 0 or ep_idx == 0:
            print(color_text(
                f"  Episode {ep_idx + 1}/{n_episodes} | "
                f"Reward: {ep_reward:.4f} | Steps: {ep_steps}",
                Colors.CYAN))

    return episode_data, trajectory_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Output & Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(env, algo, noise, noise_action, noise_env=0.0):
    """Create logging directory:
       logs_baseline_test/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = (f"./logs_baseline_test/{env}_{algo}/"
              f"{timestamp}_NOISE_{noise}_ACT_{noise_action}_ENV_{noise_env}")
    os.makedirs(logdir, exist_ok=True)
    return logdir, timestamp


def save_config(logdir, args, algo):
    """Save full test configuration as JSON for reproducibility."""
    config = {
        "env": args.env,
        "algo": algo,
        "baseline": True,
        "model_path": f"baseline:{algo}",
        "noise": args.noise,
        "noise_action": args.noise_action,
        "noise_env": args.noise_env,
        "n_eval": args.n_eval,
        "deterministic": True,
        "seed": args.seed,
        "rm": args.rm,
        "reward_beta": args.reward_beta,
        "vec_normalize_path": None,
        "collect_trajectories": args.collect_trajectories,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    config_path = os.path.join(logdir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate non-RL baseline algorithms on SustainGym environments")

    all_baselines = sorted(set(b for bs in VALID_BASELINES.values() for b in bs))

    parser.add_argument("--env", type=str, default="evcharging",
                        choices=["cogen", "evcharging", "building"],
                        help="Environment to test")
    parser.add_argument("--algo", type=str, required=True,
                        choices=all_baselines,
                        help="Baseline algorithm to run")

    # Noise — for robustness testing
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Observation noise level")
    parser.add_argument("--noise-action", type=float, default=0.0,
                        help="Action noise level")
    parser.add_argument("--noise-env", type=float, default=0.0,
                        help="Environment noise level")

    # Evaluation
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Number of evaluation episodes")

    # Environment-specific
    parser.add_argument("--rm", type=int, default=300,
                        help="Renewables magnitude for cogen")
    parser.add_argument("--reward-beta", type=float, default=0.5,
                        help="Reward beta for building")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--collect-trajectories", action="store_true",
                        help="Collect per-timestep trajectory data")

    args = parser.parse_args()

    # Validate algo is valid for this env
    if args.algo not in VALID_BASELINES[args.env]:
        valid = ", ".join(VALID_BASELINES[args.env])
        print(color_text(
            f"Error: '{args.algo}' not available for {args.env}. Valid: {valid}",
            Colors.RED))
        sys.exit(1)

    return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_arguments()

    ENV = args.env
    ALGO = args.algo
    NOISE = args.noise
    NOISE_ACTION = args.noise_action
    NOISE_ENV = args.noise_env
    N_EVAL = args.n_eval
    SEED = args.seed

    # Setup logging
    logdir, timestamp = setup_logging(ENV, ALGO, NOISE, NOISE_ACTION, NOISE_ENV)

    # Save configuration
    config = save_config(logdir, args, ALGO)

    # Print configuration
    print("\n" + "=" * 80)
    print(color_text("BASELINE EVALUATION CONFIGURATION", Colors.GREEN))
    print("=" * 80)
    print(f"  Environment:        {ENV}")
    print(f"  Baseline:           {ALGO}")
    print(f"  Noise (obs):        {NOISE}")
    print(f"  Noise (action):     {NOISE_ACTION}")
    print(f"  Noise (env):        {NOISE_ENV}")
    print(f"  Episodes:           {N_EVAL}")
    print(f"  Seed:               {SEED}")
    if ENV == "cogen":
        print(f"  Renewables Mag:     {args.rm}")
    if ENV == "building":
        print(f"  Reward Beta:        {args.reward_beta}")
    print(f"  Collect Traject.:   {args.collect_trajectories}")
    print(f"  Output Directory:   {logdir}")
    print("=" * 80 + "\n")

    np.random.seed(SEED)

    # Create environment
    print(color_text("[1/3] Creating test environment...", Colors.GREEN))
    env = create_test_environment(
        ENV, noise=NOISE, noise_action=NOISE_ACTION, noise_env=NOISE_ENV,
        rm=args.rm, reward_beta=args.reward_beta, seed=SEED)

    # Create baseline
    print(color_text(f"[2/3] Creating {ALGO} baseline...", Colors.GREEN))
    baseline = create_baseline(ALGO, env, ENV)

    # Run evaluation
    print(color_text(f"\n[3/3] Evaluating {ALGO} for {N_EVAL} episodes...", Colors.GREEN))
    start_time = time.time()

    episode_data, trajectory_data = evaluate_baseline(
        baseline=baseline,
        env=env,
        n_episodes=N_EVAL,
        env_type=ENV,
        seed=SEED,
        collect_trajectories=args.collect_trajectories,
    )

    duration = time.time() - start_time

    # Save results (reuse stdrl_testing.py functions)
    print(color_text(f"\nSaving results to {logdir}...", Colors.GREEN))
    episode_df = save_episode_csv(logdir, episode_data, ENV)

    if args.collect_trajectories and trajectory_data:
        save_trajectory_csv(logdir, trajectory_data, ENV)

    mean_reward = episode_df["total_reward"].mean()
    std_reward = episode_df["total_reward"].std()

    save_summary(logdir, episode_df, config, ENV, duration)
    save_legacy_results(logdir, mean_reward, std_reward)

    # Print final summary
    print("\n" + "=" * 80)
    print(color_text("BASELINE EVALUATION RESULTS", Colors.GREEN))
    print("=" * 80)
    print(f"  Baseline:        {ALGO}")
    print(f"  Mean Reward:     {mean_reward:.6f} +/- {std_reward:.6f}")
    print(f"  Median Reward:   {episode_df['total_reward'].median():.6f}")
    ci95 = 1.96 * std_reward / np.sqrt(N_EVAL)
    print(f"  95% CI:          [{mean_reward - ci95:.6f}, {mean_reward + ci95:.6f}]")
    print(f"  Episodes:        {N_EVAL}")
    print(f"  Duration:        {duration:.1f}s ({duration / N_EVAL:.2f}s/episode)")
    print(f"  Output Dir:      {logdir}")
    print("=" * 80)

    # Env-specific highlights (same as stdrl_testing.py)
    if ENV == "evcharging" and "total_profit" in episode_df.columns:
        print(f"  Profit:          {episode_df['total_profit'].mean():.4f} +/- "
              f"{episode_df['total_profit'].std():.4f}")
        print(f"  Carbon Cost:     {episode_df['total_carbon_cost'].mean():.4f} +/- "
              f"{episode_df['total_carbon_cost'].std():.4f}")
        print(f"  Excess Charge:   {episode_df['total_excess_charge'].mean():.6f} +/- "
              f"{episode_df['total_excess_charge'].std():.6f}")
    elif ENV == "building" and "total_cost_usd" in episode_df.columns:
        print(f"  Comfort Cost:    {episode_df['total_comfort_cost'].mean():.4f} +/- "
              f"{episode_df['total_comfort_cost'].std():.4f}")
        print(f"  Power Cost:      {episode_df['total_power_cost'].mean():.4f} +/- "
              f"{episode_df['total_power_cost'].std():.4f}")
        print(f"  Total USD Cost:  {episode_df['total_cost_usd'].mean():.4f} +/- "
              f"{episode_df['total_cost_usd'].std():.4f}")
    elif ENV == "cogen" and "total_fuel_costs" in episode_df.columns:
        print(f"  Fuel Cost:       {episode_df['total_fuel_costs'].mean():.4f} +/- "
              f"{episode_df['total_fuel_costs'].std():.4f}")
        if "total_dyn_cv_costs" in episode_df.columns:
            print(f"  Dyn CV Cost:     {episode_df['total_dyn_cv_costs'].mean():.4f} +/- "
                  f"{episode_df['total_dyn_cv_costs'].std():.4f}")
        if "total_non_delivery_cost" in episode_df.columns:
            print(f"  Non-Delivery:    {episode_df['total_non_delivery_cost'].mean():.4f} +/- "
                  f"{episode_df['total_non_delivery_cost'].std():.4f}")

    print("=" * 80 + "\n")

    print(color_text("Output files:", Colors.GREEN))
    print(f"  1. {logdir}/episode_results.csv       (per-episode metrics)")
    if args.collect_trajectories:
        print(f"  2. {logdir}/trajectory_data.csv        (per-timestep data)")
    print(f"  3. {logdir}/evaluation_summary.txt     (paper-ready statistics)")
    print(f"  4. {logdir}/evaluation_results.txt     (legacy: mean +/- std)")
    print(f"  5. {logdir}/test_config.json           (reproducibility config)")
    print()
