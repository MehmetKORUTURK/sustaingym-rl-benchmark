"""
marl_testing.py — Evaluation script for trained MARL models on SustainGym environments.

Loads a Ray RLlib checkpoint and runs N evaluation episodes on the PettingZoo
multi-agent environment.  Produces per-episode CSV and paper-ready summary
statistics (mean, std, CI95, quantiles), matching the output format of
stdrl_testing.py.

Output directory:
    logs_marl_test/{env}_{algo}[_SHARED]/{timestamp}_NOISE_0_ACT_0_ENV_0/

Usage (single run):
    python marl_testing.py --env evcharging --algo PPO --n-eval 100
    python marl_testing.py --env building --algo SAC --shared-policy --n-eval 100
    python marl_testing.py --env cogen --algo IMPALA --rm 300 --n-eval 100

Usage (run all 19 configs):
    python marl_testing.py --run-all --n-eval 100
"""
from __future__ import annotations
import sys
from pathlib import Path
# Support running from repo root (python marl_testing.py) or from scripts/test/
if Path(__file__).resolve().parent.name == "test":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

try:
    from scripts.plot.colored import color_text, Colors
except ImportError:
    from ttp.colored import color_text, Colors

# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Defaults (mirrors marl_training.py)
# ═══════════════════════════════════════════════════════════════════════════════

ENV_DEFAULTS = {
    "evcharging": {
        "algos": ["PPO", "SAC", "APPO", "IMPALA"],
        "supports_shared": True,
        "num_workers": 0,  # eval: no parallel workers
    },
    "building": {
        "algos": ["PPO", "SAC", "APPO", "IMPALA"],
        "supports_shared": True,
        "num_workers": 0,
    },
    "cogen": {
        "algos": ["PPO", "APPO", "IMPALA"],
        "supports_shared": False,
        "num_workers": 0,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Arguments
# ═══════════════════════════════════════════════════════════════════════════════

# All 19 MARL configurations: (env, algo, shared, checkpoint_path)
ALL_CONFIGS = [
    ("evcharging", "PPO", False, "logs_marl_train/evcharging_PPO/2026-03-09_00-27-27_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("evcharging", "PPO", True, "logs_marl_train/evcharging_PPO_SHARED/2026-03-09_12-31-13_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("evcharging", "SAC", False, "logs_marl_train/evcharging_SAC/2026-03-20_13-37-25_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_90"),
    ("evcharging", "SAC", True, "logs_marl_train/evcharging_SAC_SHARED/2026-03-09_12-31-13_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("evcharging", "APPO", False, "logs_marl_train/evcharging_APPO/2026-03-20_20-04-49_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_90"),
    ("evcharging", "APPO", True, "logs_marl_train/evcharging_APPO_SHARED/2026-03-20_16-03-13_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_90"),
    ("evcharging", "IMPALA", False, "logs_marl_train/evcharging_IMPALA/2026-03-10_14-50-10_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("evcharging", "IMPALA", True, "logs_marl_train/evcharging_IMPALA_SHARED/2026-03-20_13-50-33_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("building", "PPO", False, "logs_marl_train/building_PPO/2026-03-20_23-38-22_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("building", "PPO", True, "logs_marl_train/building_PPO_SHARED/2026-03-21_08-44-06_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_90"),
    ("building", "SAC", False, "logs_marl_train/building_SAC/2026-03-21_08-37-12_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("building", "SAC", True, "logs_marl_train/building_SAC_SHARED/2026-03-12_11-41-26_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_9990"),
    ("building", "APPO", False, "logs_marl_train/building_APPO/2026-03-21_01-56-57_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("building", "APPO", True, "logs_marl_train/building_APPO_SHARED/2026-03-21_09-56-32_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_90"),
    ("building", "IMPALA", False, "logs_marl_train/building_IMPALA/2026-03-21_06-13-24_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("building", "IMPALA", True, "logs_marl_train/building_IMPALA_SHARED/2026-03-21_15-51-41_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("cogen", "PPO", False, "logs_marl_train/cogen_PPO/2026-03-15_20-44-19_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("cogen", "APPO", False, "logs_marl_train/cogen_APPO/2026-03-15_21-24-43_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
    ("cogen", "IMPALA", False, "logs_marl_train/cogen_IMPALA/2026-03-15_21-24-43_NOISE_0_ACT_0_ENV_0/checkpoints/checkpoint_iter_990"),
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MARL models on SustainGym environments")

    parser.add_argument("--env", type=str,
                        choices=["evcharging", "building", "cogen"],
                        help="Environment to evaluate")
    parser.add_argument("--algo", type=str,
                        help="RLlib algorithm used for training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if omitted)")
    parser.add_argument("--shared-policy", action="store_true",
                        help="Use shared policy mode (must match training)")

    parser.add_argument("--run-all", action="store_true",
                        help="Run all 19 MARL configs sequentially")
    parser.add_argument("--n-eval", type=int, default=100,
                        help="Number of evaluation episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--rm", type=int, default=300,
                        help="Renewables magnitude, cogen only (default: 300)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: 0 for CPU-only eval)")

    args = parser.parse_args()

    if not args.run_all and (args.env is None or args.algo is None):
        parser.error("--env and --algo are required unless using --run-all")

    return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment & Policy Setup (mirrors marl_training.py)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_env_and_config(args):
    """Register the env, build an AlgorithmConfig matching training, return
    (config, sample_env, env_config_dict)."""

    defaults = ENV_DEFAULTS[args.env]
    env_config_dict = {}

    if args.env == "evcharging":
        from envs.evcharging import GMMsTraceGenerator, MultiAgentEVChargingEnv

        def env_creator(env_config):
            trace_gen = GMMsTraceGenerator('caltech', 'Summer 2019')
            env = MultiAgentEVChargingEnv(
                trace_gen, periods_delay=0, moer_forecast_steps=36,
                project_action_in_env=True, discrete=False, verbose=0)
            return ParallelPettingZooEnv(env)

        register_env("marl_env", env_creator)
        sample_env = MultiAgentEVChargingEnv(
            GMMsTraceGenerator('caltech', 'Summer 2019'),
            project_action_in_env=True)

    elif args.env == "building":
        from envs.building import MultiAgentBuildingEnv, ParameterGenerator

        def env_creator(env_config):
            params = ParameterGenerator(
                building='OfficeSmall', weather='Hot_Dry', location='Tucson',
                reward_beta=0.5)
            env = MultiAgentBuildingEnv(params)
            return ParallelPettingZooEnv(env)

        register_env("marl_env", env_creator)
        sample_params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson',
            reward_beta=0.5)
        sample_env = MultiAgentBuildingEnv(sample_params)

    elif args.env == "cogen":
        from envs.cogen import MultiAgentCogenEnv

        def env_creator(env_config):
            rm = env_config.get("renewables_magnitude", 300)
            env = MultiAgentCogenEnv(renewables_magnitude=rm)
            return ParallelPettingZooEnv(env)

        register_env("marl_env", env_creator)
        sample_env = MultiAgentCogenEnv(renewables_magnitude=args.rm)
        env_config_dict = {"renewables_magnitude": args.rm}

    # Policy mapping (must match training exactly)
    agent_ids = sample_env.possible_agents

    if args.shared_policy:
        ref_agent = agent_ids[0]
        policies = {
            "shared_policy": (
                None,
                sample_env.observation_spaces[ref_agent],
                sample_env.action_spaces[ref_agent],
                {},
            )
        }
        policy_ids = ["shared_policy"]

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return "shared_policy"
    else:
        policies = {
            f"policy_{aid}": (
                None,
                sample_env.observation_spaces[aid],
                sample_env.action_spaces[aid],
                {},
            )
            for aid in agent_ids
        }
        policy_ids = list(policies.keys())

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return f"policy_{agent_id}"

    del sample_env

    # Build AlgorithmConfig matching training
    obs_filter = "MeanStdFilter" if args.env == "building" else "NoFilter"

    common_multi_agent = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": policy_ids,
    }

    # Model config must match training (fcnet_hiddens=[64,64])
    model_config = {"fcnet_hiddens": [64, 64]}

    if args.algo == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        config = (
            PPOConfig()
            .environment(env="marl_env", env_config=env_config_dict,
                         disable_env_checking=True)
            .framework("torch")
            .training(model=model_config)
            .resources(num_gpus=args.num_gpus, num_gpus_per_worker=0)
            .multi_agent(**common_multi_agent)
            .rollouts(num_rollout_workers=0, observation_filter=obs_filter,
                      enable_connectors=True)
            .debugging(seed=args.seed)
        )
    elif args.algo == "SAC":
        from ray.rllib.algorithms.sac import SACConfig
        config = (
            SACConfig()
            .environment(env="marl_env", env_config=env_config_dict,
                         disable_env_checking=True)
            .framework("torch")
            .training(model=model_config)
            .resources(num_gpus=args.num_gpus, num_gpus_per_worker=0)
            .multi_agent(**common_multi_agent)
            .rollouts(num_rollout_workers=0, observation_filter=obs_filter,
                      enable_connectors=True)
            .debugging(seed=args.seed)
        )
    elif args.algo == "APPO":
        from ray.rllib.algorithms.appo import APPOConfig
        config = (
            APPOConfig()
            .environment(env="marl_env", env_config=env_config_dict,
                         disable_env_checking=True)
            .framework("torch")
            .training(model=model_config)
            .resources(num_gpus=args.num_gpus, num_gpus_per_worker=0)
            .multi_agent(**common_multi_agent)
            .rollouts(num_rollout_workers=0, observation_filter=obs_filter,
                      enable_connectors=True)
            .debugging(seed=args.seed)
        )
    elif args.algo == "IMPALA":
        from ray.rllib.algorithms.impala import ImpalaConfig
        config = (
            ImpalaConfig()
            .environment(env="marl_env", env_config=env_config_dict,
                         disable_env_checking=True)
            .framework("torch")
            .training(model=model_config)
            .resources(num_gpus=args.num_gpus, num_gpus_per_worker=0)
            .multi_agent(**common_multi_agent)
            .rollouts(num_rollout_workers=0, observation_filter=obs_filter,
                      enable_connectors=True)
            .debugging(seed=args.seed)
        )
    else:
        print(color_text(f"Error: Unknown algorithm '{args.algo}'", Colors.RED))
        sys.exit(1)

    return config, agent_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  Manual Evaluation Loop (bypasses RLlib evaluate() for full control)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_marl(algo, env_name, n_episodes, seed, rm=300):
    """Run n_episodes on a fresh PettingZoo env using the trained algo's
    compute_actions, collecting per-episode total reward and length."""

    # Build a raw PettingZoo env (not wrapped in ParallelPettingZooEnv)
    if env_name == "evcharging":
        from envs.evcharging import GMMsTraceGenerator, MultiAgentEVChargingEnv
        trace_gen = GMMsTraceGenerator('caltech', 'Summer 2019')
        env = MultiAgentEVChargingEnv(
            trace_gen, periods_delay=0, moer_forecast_steps=36,
            project_action_in_env=True, discrete=False, verbose=0)
    elif env_name == "building":
        from envs.building import MultiAgentBuildingEnv, ParameterGenerator
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson',
            reward_beta=0.5)
        env = MultiAgentBuildingEnv(params)
    elif env_name == "cogen":
        from envs.cogen import MultiAgentCogenEnv
        env = MultiAgentCogenEnv(renewables_magnitude=rm)

    episode_data = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        obs, infos = env.reset(seed=ep_seed)

        ep_reward = 0.0
        ep_steps = 0
        done_agents = set()

        while env.agents:  # PettingZoo: env.agents is empty when done
            # Compute actions for all active agents
            actions = {}
            for agent_id in env.agents:
                if agent_id in obs:
                    # Support both old and new RLlib API
                    try:
                        mapping_fn = algo.config.multi_agent_config["policy_mapping_fn"]
                    except AttributeError:
                        mapping_fn = algo.config.policy_mapping_fn
                    policy_id = mapping_fn(agent_id, None, None)
                    action = algo.compute_single_action(
                        obs[agent_id], policy_id=policy_id, explore=False)
                    actions[agent_id] = action

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Sum rewards across all agents (matches training: shared reward / n_agents)
            step_reward = sum(rewards.values())
            ep_reward += step_reward
            ep_steps += 1

            # Track done agents
            for aid in list(env.agents):
                if terminations.get(aid, False) or truncations.get(aid, False):
                    done_agents.add(aid)

        ep_record = {
            "episode": ep_idx,
            "seed": ep_seed,
            "total_reward": float(ep_reward),
            "episode_length": ep_steps,
        }
        episode_data.append(ep_record)

        # Progress reporting
        if (ep_idx + 1) % max(1, n_episodes // 10) == 0 or ep_idx == 0:
            print(color_text(
                f"  Episode {ep_idx + 1}/{n_episodes} | "
                f"Reward: {ep_reward:.4f} | Steps: {ep_steps}",
                Colors.CYAN))

    env.close()
    return episode_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Output & Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(env, algo, shared_policy):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    shared_str = "_SHARED" if shared_policy else ""
    logdir = (f"./logs_marl_test/{env}_{algo}{shared_str}/"
              f"{timestamp}_NOISE_0_ACT_0_ENV_0")
    os.makedirs(logdir, exist_ok=True)
    return logdir, timestamp


def save_config(logdir, args, timestamp):
    config = {
        "env": args.env,
        "algo": args.algo,
        "checkpoint": os.path.abspath(args.checkpoint),
        "shared_policy": args.shared_policy,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "rm": args.rm,
        "num_gpus": args.num_gpus,
        "timestamp": timestamp,
    }
    config_path = os.path.join(logdir, "test_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def save_episode_csv(logdir, episode_data):
    df = pd.DataFrame(episode_data)
    csv_path = os.path.join(logdir, "episode_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(color_text(f"[OUTPUT] Episode results CSV: {csv_path}", Colors.GREEN))
    return df


def save_summary(logdir, episode_df, config, duration_seconds):
    summary_path = os.path.join(logdir, "evaluation_summary.txt")

    rewards = episode_df["total_reward"]
    n_eps = len(episode_df)

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MARL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Environment:        {config['env']}\n")
        f.write(f"Algorithm:          {config['algo']}\n")
        f.write(f"Checkpoint:         {config['checkpoint']}\n")
        f.write(f"Shared Policy:      {config['shared_policy']}\n")
        f.write(f"Episodes:           {config['n_eval']}\n")
        f.write(f"Seed:               {config['seed']}\n")
        if config['env'] == 'cogen':
            f.write(f"Renewables Mag:     {config['rm']}\n")
        f.write(f"Duration:           {duration_seconds:.1f}s\n")
        f.write("\n")

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

        lengths = episode_df["episode_length"]
        f.write("--- Episode Length ---\n")
        f.write(f"Mean:               {lengths.mean():.1f}\n")
        f.write(f"Std:                {lengths.std():.1f}\n")
        f.write(f"Min:                {lengths.min()}\n")
        f.write(f"Max:                {lengths.max()}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(color_text(f"[OUTPUT] Evaluation summary: {summary_path}", Colors.GREEN))


def save_legacy_results(logdir, mean_reward, std_reward):
    results_file = os.path.join(logdir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_single(args):
    """Run a single MARL evaluation."""

    # Check checkpoint is provided
    if args.checkpoint is None:
        print(color_text(f"Error: No checkpoint specified for {args.env}_{args.algo}"
                         f"{'_SHARED' if args.shared_policy else ''}", Colors.RED))
        return False

    # Validate
    defaults = ENV_DEFAULTS[args.env]
    if args.algo not in defaults["algos"]:
        print(color_text(f"Error: {args.algo} not supported for {args.env}. "
                         f"Choose from: {defaults['algos']}", Colors.RED))
        return False
    if args.shared_policy and not defaults["supports_shared"]:
        print(color_text(f"Error: --shared-policy not supported for {args.env}.", Colors.RED))
        return False
    if not os.path.exists(args.checkpoint):
        print(color_text(f"Error: Checkpoint not found at {args.checkpoint}", Colors.RED))
        return False

    # Setup logging
    logdir, timestamp = setup_logging(args.env, args.algo, args.shared_policy)
    config = save_config(logdir, args, timestamp)

    policy_mode = "shared" if args.shared_policy else "independent"

    print("\n" + "=" * 80)
    print(color_text(f"MARL EVAL: {args.env} / {args.algo} / {policy_mode}", Colors.GREEN))
    print("=" * 80)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Episodes:    {args.n_eval}")
    print(f"  Output:      {logdir}")
    print("=" * 80 + "\n")

    # Build config and restore checkpoint
    print(color_text("[1/3] Building algorithm and restoring checkpoint...", Colors.GREEN))
    algo_config, agent_ids = setup_env_and_config(args)
    algo = algo_config.build()
    algo.restore(args.checkpoint)
    print(color_text(f"  Checkpoint restored. Agents: {agent_ids}", Colors.GREEN))

    # Run evaluation
    print(color_text(f"\n[2/3] Evaluating for {args.n_eval} episodes...", Colors.GREEN))
    np.random.seed(args.seed)
    start_time = time.time()

    episode_data = evaluate_marl(
        algo=algo,
        env_name=args.env,
        n_episodes=args.n_eval,
        seed=args.seed,
        rm=args.rm,
    )

    duration = time.time() - start_time

    # Save results
    print(color_text(f"\n[3/3] Saving results to {logdir}...", Colors.GREEN))
    episode_df = save_episode_csv(logdir, episode_data)

    mean_reward = episode_df["total_reward"].mean()
    std_reward = episode_df["total_reward"].std()

    save_summary(logdir, episode_df, config, duration)
    save_legacy_results(logdir, mean_reward, std_reward)

    # Print final summary
    ci95 = 1.96 * std_reward / np.sqrt(args.n_eval)
    print(color_text(
        f"  Result: {mean_reward:.4f} +/- {std_reward:.4f} "
        f"[{mean_reward - ci95:.4f}, {mean_reward + ci95:.4f}] "
        f"({duration:.0f}s)", Colors.GREEN))

    # Cleanup Ray between runs
    algo.stop()

    return True


if __name__ == "__main__":
    args = parse_arguments()

    if args.run_all:
        # Run all 19 configs sequentially
        print(color_text(f"Running all {len(ALL_CONFIGS)} MARL evaluation configs", Colors.GREEN))
        print(f"Episodes per config: {args.n_eval}\n")

        ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus)

        results = []
        for i, (env, algo, shared, ckpt) in enumerate(ALL_CONFIGS):
            suffix = "_SHARED" if shared else ""
            mode = "shared" if shared else "ind"
            print(color_text(f"\n[{i+1}/{len(ALL_CONFIGS)}] {env}_{algo}{suffix}", Colors.GREEN))

            # Create a namespace mimicking parsed args
            run_args = argparse.Namespace(
                env=env, algo=algo, shared_policy=shared, checkpoint=ckpt,
                n_eval=args.n_eval, seed=args.seed, rm=args.rm,
                num_gpus=args.num_gpus,
            )

            success = run_single(run_args)
            results.append((f"{env}_{algo}_{mode}", success))

        ray.shutdown()

        # Summary
        print("\n" + "=" * 80)
        print(color_text("ALL MARL EVALUATIONS COMPLETE", Colors.GREEN))
        print("=" * 80)
        for name, ok in results:
            status = "PASS" if ok else "FAIL"
            color = Colors.GREEN if ok else Colors.RED
            print(color_text(f"  {status}: {name}", color))
        passed = sum(1 for _, ok in results if ok)
        print(f"\n  {passed}/{len(results)} completed successfully.")
        print(f"  Results in: logs_marl_test/")

    else:
        # Single run — need checkpoint from --checkpoint or from ALL_CONFIGS lookup
        if args.checkpoint is None:
            # Try to find from ALL_CONFIGS
            suffix = "_SHARED" if args.shared_policy else ""
            key = f"{args.env}_{args.algo}{suffix}"
            for env, algo_name, shared, ckpt in ALL_CONFIGS:
                cfg_suffix = "_SHARED" if shared else ""
                if f"{env}_{algo_name}{cfg_suffix}" == key:
                    args.checkpoint = ckpt
                    break

        ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus)
        run_single(args)
        ray.shutdown()

    print(color_text("\nDone. Results in: logs_marl_test/", Colors.GREEN))
