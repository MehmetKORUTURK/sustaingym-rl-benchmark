"""
stdrl_testing.py — Evaluation script for trained RL models on SustainGym environments.

Evaluates best_model.zip (or any checkpoint) with full noise injection support,
VecNormalize loading, per-episode data collection, and CSV export for paper-ready
figure generation.

Naming convention matches training: logs_std_test/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
"""

import os
import sys
import json
import argparse
import datetime
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from copy import deepcopy

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    VecNormalize = None

from envs.evcharging import EVChargingEnv, GMMsTraceGenerator
from sustaingym.envs.cogen import CogenEnv
from envs.cogen.MyCogenEnv import MyCogenEnv
from envs.building import BuildingEnv, ParameterGenerator
from scripts.plot.colored import color_text, Colors


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL models on SustainGym environments")

    parser.add_argument("--env", type=str, default="evcharging",
                        choices=["cogen", "evcharging", "building"],
                        help="Environment to test")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file (.zip)")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["PPO", "SAC", "TD3"],
                        help="Override algorithm type (auto-detected from path if omitted)")

    # Noise — must match training configuration
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Observation/state noise level (must match training)")
    parser.add_argument("--noise-action", type=float, default=0.0,
                        help="Action noise level (must match training)")
    parser.add_argument("--noise-env", type=float, default=0.0,
                        help="Environment noise level. "
                             "EVCharging: MOER disturbance std. "
                             "Cogen: scales TAMB/PAMB/RHAMB + ONNX model noise. "
                             "Building: scales out_temp/ground_temp/ghi noise.")

    # Evaluation
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (default: True, matching training eval)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (overrides --deterministic)")

    # Environment-specific
    parser.add_argument("--rm", type=int, default=300,
                        help="Renewables magnitude for cogen environment")
    parser.add_argument("--reward-beta", type=float, default=0.5,
                        help="Reward beta for building environment (must match training)")

    # Normalization
    parser.add_argument("--vec-normalize-path", type=str, default=None,
                        help="Path to vec_normalize.pkl (auto-detected from model_path dir if omitted)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Output
    parser.add_argument("--collect-trajectories", action="store_true",
                        help="Collect per-timestep trajectory data (larger output)")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm Detection
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def determine_algorithm(model_path, specified_algo=None):
    if specified_algo and specified_algo in ALGO_CLASSES:
        return specified_algo
    for name in ALGO_CLASSES:
        if name in model_path:
            return name
    print(color_text("Error: Could not determine algorithm from path. Use --algo.", Colors.RED))
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  VecNormalize Auto-Detection
# ═══════════════════════════════════════════════════════════════════════════════

def find_vec_normalize(model_path, explicit_path=None):
    """Auto-detect vec_normalize.pkl from the same directory as the model."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    model_dir = os.path.dirname(model_path)
    candidate = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(candidate):
        return candidate

    # Also check parent directory (in case model is in a subdirectory)
    parent_candidate = os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl")
    if os.path.exists(parent_candidate):
        return parent_candidate

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Creation
# ═══════════════════════════════════════════════════════════════════════════════

def create_test_environment(env_type, noise=0.0, noise_action=0.0, noise_env=0.0,
                            rm=300, reward_beta=0.5, seed=42):
    """Create the test environment with identical configuration to training.

    noise_env semantics per environment:
        evcharging: scalar float — std of Gaussian noise added to actual MOER in reward calc
        cogen:      scalar float — applied uniformly as {"temperature": X, "pressure": X, "humidity": X}
                    (MyCogenEnv pushes this to base CogenEnv so it affects ONNX inputs + reward)
        building:   not supported (ignored with a warning)
    """
    if env_type == "cogen":
        noise_env_arg = noise_env if noise_env > 0 else None
        env = MyCogenEnv(CogenEnv(renewables_magnitude=rm),
                         noise=noise, noise_action=noise_action,
                         noise_env=noise_env_arg)

    elif env_type == "evcharging":
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')
        env = EVChargingEnv(gmmg,
                            noise=noise if noise > 0 else None,
                            noise_action=noise_action if noise_action > 0 else None,
                            noise_env=noise_env if noise_env > 0 else None)

    elif env_type == "building":
        # Convert scalar noise_env to dict (matching stdrl_training.py scaling)
        building_noise_env = None
        if noise_env > 0:
            building_noise_env = {
                "out_temp": 1.0 * noise_env,      # 1 °C std dev per unit
                "ground_temp": 0.5 * noise_env,    # 0.5 °C std dev per unit
                "ghi": 50.0 * noise_env,           # 50 W/m² std dev per unit
            }
            print(color_text(f"[INFO] Building environment noise config: {building_noise_env}", Colors.GREEN))
        params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry',
                                    location='Tucson', reward_beta=reward_beta)
        env = BuildingEnv(params,
                          noise=noise if noise > 0 else None,
                          noise_action=noise_action if noise_action > 0 else None,
                          noise_env=building_noise_env)
    else:
        print(color_text(f"Error: Unknown environment '{env_type}'", Colors.RED))
        sys.exit(1)

    env.reset(seed=seed)
    return env


def load_vecnormalize_stats(vec_normalize_path, env):
    """Load VecNormalize stats for manual observation normalization.

    Instead of wrapping the env in VecEnv (which causes auto-reset mutation
    of info dicts like reward_breakdown), we load the stats and apply
    normalization manually in the evaluation loop.
    """
    if not vec_normalize_path or VecNormalize is None:
        return None

    # Load stats via a temporary DummyVecEnv (required by VecNormalize.load)
    tmp_vec = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize.load(vec_normalize_path, tmp_vec)
    vec_norm.training = False
    vec_norm.norm_reward = False
    print(color_text(f"[INFO] VecNormalize stats loaded from {vec_normalize_path}", Colors.GREEN))
    return vec_norm


def make_model_env(env, vec_norm_ref):
    """Create an env suitable for model.load() — must match the trained obs space.

    If training used VecNormalize, the loaded vec_norm_ref already has the
    correct observation space (adjusted bounds from the saved pickle), so we
    reuse it directly.  Otherwise we wrap the raw env in a plain DummyVecEnv.
    """
    if vec_norm_ref is not None:
        return vec_norm_ref  # already a VecNormalize wrapping a DummyVecEnv
    return DummyVecEnv([lambda: env])


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom Evaluation Loop (Per-Episode Data Collection)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_obs(obs, vec_norm):
    """Manually normalize an observation using VecNormalize running stats.

    Replicates VecNormalize.normalize_obs() without requiring a VecEnv wrapper.
    This avoids the DummyVecEnv auto-reset problem that mutates info dicts.
    """
    if vec_norm is None:
        return obs

    if isinstance(obs, dict):
        # Dict obs: normalize each key independently (MultiInputPolicy)
        return {k: np.clip(
            (v - vec_norm.obs_rms[k].mean) / np.sqrt(vec_norm.obs_rms[k].var + vec_norm.epsilon),
            -vec_norm.clip_obs, vec_norm.clip_obs
        ).astype(np.float32) for k, v in obs.items()}
    else:
        # Flat obs: single RunningMeanStd
        return np.clip(
            (obs - vec_norm.obs_rms.mean) / np.sqrt(vec_norm.obs_rms.var + vec_norm.epsilon),
            -vec_norm.clip_obs, vec_norm.clip_obs
        ).astype(np.float32)


def evaluate_with_data_collection(model, env, n_episodes, deterministic=True,
                                  env_type="evcharging", collect_trajectories=False,
                                  seed=42, vec_norm=None):
    """Run evaluation episodes on the RAW env, collecting per-episode metrics.

    Uses the raw Gymnasium env directly (not VecEnv) so that info dicts
    like reward_breakdown are never mutated by auto-reset.  Observations
    are normalized manually via ``vec_norm`` (loaded VecNormalize stats).

    Returns:
        episode_data: list of dicts, one per episode, with env-specific metrics
        trajectory_data: list of lists of per-step dicts (only if collect_trajectories=True)
    """
    episode_data = []
    trajectory_data = [] if collect_trajectories else None

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        obs, info = env.reset(seed=ep_seed)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_rewards = []
        ep_infos = []

        if collect_trajectories:
            ep_trajectory = []

        while not done:
            # Normalize obs for the model, then wrap in batch dim [1, ...]
            norm_obs = _normalize_obs(obs, vec_norm)
            if isinstance(norm_obs, dict):
                model_obs = {k: v[np.newaxis] for k, v in norm_obs.items()}
            else:
                model_obs = norm_obs[np.newaxis]

            action, _ = model.predict(model_obs, deterministic=deterministic)
            # Remove batch dim from action
            action = action.squeeze(0)

            obs, reward_val, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Deep-copy info to be safe (prevents any remaining reference issues)
            info = deepcopy(info)

            ep_reward += float(reward_val)
            ep_steps += 1
            ep_rewards.append(float(reward_val))
            ep_infos.append(info)

            if collect_trajectories:
                step_data = {"step": ep_steps, "reward": float(reward_val)}
                step_data.update(_extract_step_metrics(info, env_type))
                ep_trajectory.append(step_data)

        # Aggregate episode metrics
        ep_record = {
            "episode": ep_idx,
            "seed": ep_seed,
            "total_reward": float(ep_reward),
            "episode_length": ep_steps,
        }

        # Add env-specific episode-level metrics from final info
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


def _extract_step_metrics(info, env_type):
    """Extract per-timestep metrics from info dict (env-specific)."""
    metrics = {}

    if env_type == "evcharging":
        rb = info.get("reward_breakdown", {})
        metrics["profit_cumul"] = rb.get("profit", 0.0)
        metrics["carbon_cost_cumul"] = rb.get("carbon_cost", 0.0)
        metrics["excess_charge_cumul"] = rb.get("excess_charge", 0.0)

    elif env_type == "building":
        rb = info.get("reward_breakdown", {})
        metrics["comfort_level_cumul"] = rb.get("comfort_level", 0.0)
        metrics["power_consumption_cumul"] = rb.get("power_consumption", 0.0)
        metrics["cost_usd"] = info.get("cost_usd", 0.0)
        zone_temp = info.get("zone_temperature", None)
        if zone_temp is not None:
            metrics["mean_zone_temp"] = float(np.mean(zone_temp))
            metrics["max_zone_temp"] = float(np.max(zone_temp))
            metrics["min_zone_temp"] = float(np.min(zone_temp))

    elif env_type == "cogen":
        # CogenEnv info keys: fuel_costs(dict), ramp_costs(dict),
        # dyn_cv_costs(dict), non_delivery_cost(float)
        for key in ["fuel_costs", "ramp_costs", "dyn_cv_costs", "non_delivery_cost"]:
            if key in info:
                val = info[key]
                if isinstance(val, dict):
                    metrics[key] = float(sum(val.values()))
                else:
                    metrics[key] = float(val)

    return metrics


def _extract_episode_metrics(ep_infos, env_type, total_reward):
    """Extract episode-level summary metrics from the list of step infos."""
    metrics = {}

    if not ep_infos:
        return metrics

    final_info = ep_infos[-1]

    if env_type == "evcharging":
        rb = final_info.get("reward_breakdown", {})
        metrics["total_profit"] = float(rb.get("profit", 0.0))
        metrics["total_carbon_cost"] = float(rb.get("carbon_cost", 0.0))
        metrics["total_excess_charge"] = float(rb.get("excess_charge", 0.0))
        metrics["max_profit"] = float(final_info.get("max_profit", 0.0))
        if metrics["max_profit"] > 0:
            metrics["profit_ratio"] = metrics["total_profit"] / metrics["max_profit"]

    elif env_type == "building":
        rb = final_info.get("reward_breakdown", {})
        metrics["total_comfort_cost"] = float(rb.get("comfort_level", 0.0))
        metrics["total_power_cost"] = float(rb.get("power_consumption", 0.0))

        # Aggregate cost_usd across all steps
        total_cost_usd = sum(info.get("cost_usd", 0.0) for info in ep_infos)
        metrics["total_cost_usd"] = float(total_cost_usd)

        # Temperature statistics across episode
        zone_temps = [info.get("zone_temperature") for info in ep_infos
                      if info.get("zone_temperature") is not None]
        if zone_temps:
            all_temps = np.array(zone_temps)
            metrics["mean_temp"] = float(np.mean(all_temps))
            metrics["std_temp"] = float(np.std(all_temps))
            metrics["max_temp"] = float(np.max(all_temps))
            metrics["min_temp"] = float(np.min(all_temps))

    elif env_type == "cogen":
        # Sum costs across all timesteps
        cost_keys = ["fuel_costs", "ramp_costs", "dyn_cv_costs", "non_delivery_cost"]
        for key in cost_keys:
            vals = []
            for info in ep_infos:
                if key in info:
                    v = info[key]
                    vals.append(float(sum(v.values())) if isinstance(v, dict) else float(v))
            if vals:
                metrics[f"total_{key}"] = sum(vals)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Output & Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(env, algo, noise, noise_action, noise_env=0.0):
    """Create logging directory matching training naming convention:
       logs_std_test/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = f"./logs_std_test/{env}_{algo}/{timestamp}_NOISE_{noise}_ACT_{noise_action}_ENV_{noise_env}"
    os.makedirs(logdir, exist_ok=True)
    return logdir, timestamp


def save_config(logdir, args, algo, vec_normalize_path):
    """Save full test configuration as JSON for reproducibility."""
    config = {
        "env": args.env,
        "algo": algo,
        "model_path": os.path.abspath(args.model_path),
        "noise": args.noise,
        "noise_action": args.noise_action,
        "noise_env": args.noise_env,
        "n_eval": args.n_eval,
        "deterministic": not args.stochastic,
        "seed": args.seed,
        "rm": args.rm,
        "reward_beta": args.reward_beta,
        "vec_normalize_path": vec_normalize_path,
        "collect_trajectories": args.collect_trajectories,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    config_path = os.path.join(logdir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def save_episode_csv(logdir, episode_data, env_type):
    """Save per-episode data to CSV for paper-ready figure generation."""
    df = pd.DataFrame(episode_data)
    csv_path = os.path.join(logdir, "episode_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(color_text(f"[OUTPUT] Episode results CSV: {csv_path}", Colors.GREEN))
    return df


def save_trajectory_csv(logdir, trajectory_data, env_type):
    """Save per-timestep trajectory data to CSV (one row per timestep per episode)."""
    rows = []
    for ep_idx, ep_traj in enumerate(trajectory_data):
        for step_data in ep_traj:
            step_data["episode"] = ep_idx
            rows.append(step_data)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(logdir, "trajectory_data.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(color_text(f"[OUTPUT] Trajectory data CSV: {csv_path}", Colors.GREEN))
    return df


def save_summary(logdir, episode_df, config, env_type, duration_seconds):
    """Save paper-ready summary statistics text file."""
    summary_path = os.path.join(logdir, "evaluation_summary.txt")

    rewards = episode_df["total_reward"]
    n_eps = len(episode_df)

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("--- Configuration ---\n")
        f.write(f"Environment:        {config['env']}\n")
        f.write(f"Algorithm:          {config['algo']}\n")
        f.write(f"Model Path:         {config['model_path']}\n")
        f.write(f"Noise (obs):        {config['noise']}\n")
        f.write(f"Noise (action):     {config['noise_action']}\n")
        f.write(f"Noise (env):        {config['noise_env']}\n")
        f.write(f"Episodes:           {config['n_eval']}\n")
        f.write(f"Deterministic:      {config['deterministic']}\n")
        f.write(f"Seed:               {config['seed']}\n")
        f.write(f"VecNormalize:       {config['vec_normalize_path'] or 'None'}\n")
        if config['env'] == 'cogen':
            f.write(f"Renewables Mag:     {config['rm']}\n")
        if config['env'] == 'building':
            f.write(f"Reward Beta:        {config['reward_beta']}\n")
        f.write(f"Duration:           {duration_seconds:.1f}s\n")
        f.write("\n")

        # Reward statistics
        f.write("--- Reward Statistics ---\n")
        f.write(f"Mean:               {rewards.mean():.6f}\n")
        f.write(f"Std:                {rewards.std():.6f}\n")
        f.write(f"Median:             {rewards.median():.6f}\n")
        f.write(f"Min:                {rewards.min():.6f}\n")
        f.write(f"Max:                {rewards.max():.6f}\n")
        f.write(f"Q25:                {rewards.quantile(0.25):.6f}\n")
        f.write(f"Q75:                {rewards.quantile(0.75):.6f}\n")
        f.write(f"IQR:                {rewards.quantile(0.75) - rewards.quantile(0.25):.6f}\n")
        ci95 = 1.96 * rewards.std() / np.sqrt(n_eps)
        f.write(f"95% CI:             [{rewards.mean() - ci95:.6f}, {rewards.mean() + ci95:.6f}]\n")
        f.write("\n")

        # Episode length statistics
        lengths = episode_df["episode_length"]
        f.write("--- Episode Length ---\n")
        f.write(f"Mean:               {lengths.mean():.1f}\n")
        f.write(f"Std:                {lengths.std():.1f}\n")
        f.write(f"Min:                {lengths.min()}\n")
        f.write(f"Max:                {lengths.max()}\n")
        f.write("\n")

        # Env-specific metrics
        f.write(f"--- Environment-Specific Metrics ({env_type}) ---\n")
        if env_type == "evcharging":
            _write_stat(f, episode_df, "total_profit", "Total Profit ($)")
            _write_stat(f, episode_df, "total_carbon_cost", "Carbon Cost ($)")
            _write_stat(f, episode_df, "total_excess_charge", "Excess Charge ($)")
            _write_stat(f, episode_df, "max_profit", "Max Possible Profit ($)")
            _write_stat(f, episode_df, "profit_ratio", "Profit Ratio")
        elif env_type == "building":
            _write_stat(f, episode_df, "total_comfort_cost", "Comfort Cost")
            _write_stat(f, episode_df, "total_power_cost", "Power Cost")
            _write_stat(f, episode_df, "total_cost_usd", "Total Cost (USD)")
            _write_stat(f, episode_df, "mean_temp", "Mean Zone Temp (C)")
            _write_stat(f, episode_df, "max_temp", "Max Zone Temp (C)")
            _write_stat(f, episode_df, "min_temp", "Min Zone Temp (C)")
        elif env_type == "cogen":
            _write_stat(f, episode_df, "total_fuel_costs", "Fuel Cost")
            _write_stat(f, episode_df, "total_ramp_costs", "Ramp Cost")
            _write_stat(f, episode_df, "total_dyn_cv_costs", "Dynamic CV Cost")
            _write_stat(f, episode_df, "total_non_delivery_cost", "Non-Delivery Cost")

        f.write("\n" + "=" * 80 + "\n")

    print(color_text(f"[OUTPUT] Evaluation summary: {summary_path}", Colors.GREEN))


def _write_stat(f, df, col, label):
    """Write mean ± std for a column if it exists."""
    if col in df.columns and df[col].notna().any():
        vals = df[col].dropna()
        f.write(f"  {label}:\n")
        f.write(f"    Mean:  {vals.mean():.6f}  Std: {vals.std():.6f}\n")
        f.write(f"    Range: [{vals.min():.6f}, {vals.max():.6f}]\n")


# Also save backward-compatible evaluation_results.txt
def save_legacy_results(logdir, mean_reward, std_reward):
    results_file = os.path.join(logdir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_arguments()

    # Resolve settings
    ENV = args.env
    NOISE = args.noise
    NOISE_ACTION = args.noise_action
    NOISE_ENV = args.noise_env
    N_EVAL = args.n_eval
    SEED = args.seed
    DETERMINISTIC = not args.stochastic

    # Determine algorithm
    ALGO = determine_algorithm(args.model_path, args.algo)

    # Setup logging (matching training naming convention)
    logdir, timestamp = setup_logging(ENV, ALGO, NOISE, NOISE_ACTION, NOISE_ENV)

    # Auto-detect VecNormalize
    vec_normalize_path = find_vec_normalize(args.model_path, args.vec_normalize_path)

    # Save configuration
    config = save_config(logdir, args, ALGO, vec_normalize_path)

    # Print configuration summary
    print("\n" + "=" * 80)
    print(color_text("EVALUATION CONFIGURATION", Colors.GREEN))
    print("=" * 80)
    print(f"  Environment:        {ENV}")
    print(f"  Algorithm:          {ALGO}")
    print(f"  Model Path:         {args.model_path}")
    print(f"  Noise (obs):        {NOISE}")
    print(f"  Noise (action):     {NOISE_ACTION}")
    print(f"  Noise (env):        {NOISE_ENV}")
    print(f"  Episodes:           {N_EVAL}")
    print(f"  Deterministic:      {DETERMINISTIC}")
    print(f"  Seed:               {SEED}")
    if ENV == "cogen":
        print(f"  Renewables Mag:     {args.rm}")
    if ENV == "building":
        print(f"  Reward Beta:        {args.reward_beta}")
    print(f"  VecNormalize:       {vec_normalize_path or 'Not found (raw obs)'}")
    print(f"  Collect Traject.:   {args.collect_trajectories}")
    print(f"  Output Directory:   {logdir}")
    print("=" * 80 + "\n")

    # Set random seed
    np.random.seed(SEED)

    # Validate model path
    if not os.path.exists(args.model_path):
        print(color_text(f"Error: Model file not found at {args.model_path}", Colors.RED))
        sys.exit(1)

    # Create test environment
    print(color_text("[1/4] Creating test environment...", Colors.GREEN))
    base_env = create_test_environment(
        ENV, noise=NOISE, noise_action=NOISE_ACTION, noise_env=NOISE_ENV,
        rm=args.rm, reward_beta=args.reward_beta, seed=SEED)

    # Load VecNormalize stats (if training used it)
    vec_norm_ref = load_vecnormalize_stats(vec_normalize_path, base_env)

    # Build env for model loading (model expects VecEnv observation shape)
    model_env = make_model_env(base_env, vec_norm_ref)

    # Load model
    print(color_text("[2/4] Loading trained model...", Colors.GREEN))
    try:
        algo_class = ALGO_CLASSES[ALGO]
        model = algo_class.load(args.model_path, env=model_env, device="cpu")
        print(color_text(f"  Model loaded: {ALGO} from {args.model_path}", Colors.GREEN))
    except Exception as e:
        print(color_text(f"Error loading model: {e}", Colors.RED))
        sys.exit(1)

    # Run evaluation — uses raw env directly (no VecEnv auto-reset mutation)
    print(color_text(f"\n[3/4] Evaluating for {N_EVAL} episodes "
                     f"(deterministic={DETERMINISTIC})...", Colors.GREEN))
    start_time = time.time()

    episode_data, trajectory_data = evaluate_with_data_collection(
        model=model,
        env=base_env,
        vec_norm=vec_norm_ref,
        n_episodes=N_EVAL,
        deterministic=DETERMINISTIC,
        env_type=ENV,
        collect_trajectories=args.collect_trajectories,
        seed=SEED,
    )

    duration = time.time() - start_time

    # Save results
    print(color_text(f"\n[4/4] Saving results to {logdir}...", Colors.GREEN))
    episode_df = save_episode_csv(logdir, episode_data, ENV)

    if args.collect_trajectories and trajectory_data:
        save_trajectory_csv(logdir, trajectory_data, ENV)

    # Summary statistics
    mean_reward = episode_df["total_reward"].mean()
    std_reward = episode_df["total_reward"].std()

    save_summary(logdir, episode_df, config, ENV, duration)
    save_legacy_results(logdir, mean_reward, std_reward)

    # Print final summary
    print("\n" + "=" * 80)
    print(color_text("EVALUATION RESULTS", Colors.GREEN))
    print("=" * 80)
    print(f"  Mean Reward:     {mean_reward:.6f} +/- {std_reward:.6f}")
    print(f"  Median Reward:   {episode_df['total_reward'].median():.6f}")
    ci95 = 1.96 * std_reward / np.sqrt(N_EVAL)
    print(f"  95% CI:          [{mean_reward - ci95:.6f}, {mean_reward + ci95:.6f}]")
    print(f"  Episodes:        {N_EVAL}")
    print(f"  Duration:        {duration:.1f}s ({duration / N_EVAL:.2f}s/episode)")
    print(f"  Output Dir:      {logdir}")
    print("=" * 80)

    # Env-specific highlights
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
