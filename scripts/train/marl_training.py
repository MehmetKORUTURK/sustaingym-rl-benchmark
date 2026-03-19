"""
Unified MARL training script for all SustainGym environments using Ray RLlib.

Supports:
    EVCharging  -> PPO, SAC, APPO, IMPALA  (54 agents, 32k iterations)
    Building    -> PPO, SAC, APPO, IMPALA  (~5 agents, 32k iterations)
    Cogen       -> PPO, APPO, IMPALA             (4 agents, 750 iterations)

Policy modes:
    --shared-policy   Use a single shared policy for all agents (MAPPO/MASAC).
                      Only for EVCharging and Building (same obs/action space).
                      Cogen agents have different action spaces, so sharing
                      is not possible.

Usage:
    python marl_training.py --env evcharging --algo APPO --seed 42
    python marl_training.py --env building --algo SAC --num-iterations 32000
    python marl_training.py --env cogen --algo PPO --rm 300
    python marl_training.py --env evcharging --algo PPO --shared-policy  # MAPPO
    python marl_training.py --env building --algo IMPALA --num-gpus 0
    python marl_training.py --env evcharging --algo IMPALA --num-workers 10
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

# ── Environment Defaults ─────────────────────────────────────────────────

ENV_DEFAULTS = {
    "evcharging": {
        "algo": "APPO",
        "algos": ["PPO", "SAC", "APPO", "IMPALA"],
        "num_iterations": 32_000,
        "num_workers": 10,
        "supports_shared": True,
    },
    "building": {
        "algo": "SAC",
        "algos": ["PPO", "SAC", "APPO", "IMPALA"],
        "num_iterations": 32_000,
        "num_workers": 4,
        "supports_shared": True,
    },
    "cogen": {
        "algo": "PPO",
        "algos": ["PPO", "APPO", "IMPALA"],
        "num_iterations": 3_000,
        "num_workers": 4,
        "supports_shared": False,  # agents have different action spaces
    },
}

# ── CLI Arguments ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Unified MARL training for SustainGym environments (Ray RLlib)")
parser.add_argument("--env", type=str, required=True,
                    choices=["evcharging", "building", "cogen"],
                    help="Environment to train on")
parser.add_argument("--algo", type=str, default=None,
                    help="RLlib algorithm (default: env-specific, see ENV_DEFAULTS)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--num-iterations", type=int, default=None,
                    help="Number of training iterations (default: env-specific)")
parser.add_argument("--num-workers", type=int, default=None,
                    help="Number of rollout workers (default: env-specific)")
parser.add_argument("--checkpoint-freq", type=int, default=10,
                    help="Save checkpoint every N iterations (default: 10)")
parser.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs (default: auto-detect, 0 for CPU-only)")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 3e-4)")
parser.add_argument("--rm", type=int, default=300,
                    help="Renewables magnitude, cogen only (default: 300)")
parser.add_argument("--shared-policy", action="store_true",
                    help="Use a single shared policy for all agents (MAPPO/MASAC)")
args = parser.parse_args()

# ── Resolve Defaults ─────────────────────────────────────────────────────

defaults = ENV_DEFAULTS[args.env]

if args.algo is None:
    args.algo = defaults["algo"]
if args.algo not in defaults["algos"]:
    parser.error(f"Algorithm '{args.algo}' not supported for {args.env}. "
                 f"Choose from: {defaults['algos']}")
if args.num_iterations is None:
    args.num_iterations = defaults["num_iterations"]
if args.num_workers is None:
    args.num_workers = defaults["num_workers"]

if args.shared_policy and not defaults["supports_shared"]:
    parser.error(f"--shared-policy not supported for {args.env}. "
                 "Cogen agents have different action spaces.")

# ── GPU Detection ────────────────────────────────────────────────────────

if args.num_gpus is not None:
    num_gpus = args.num_gpus
else:
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0

if num_gpus == 0:
    print("[MARL] No GPU detected. Running in CPU-only mode.")

# ── Output Directory ─────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shared_str = "_SHARED" if args.shared_policy else ""
log_dir = (f"./logs_marl_train/{args.env}_{args.algo}{shared_str}/"
           f"{timestamp}_NOISE_0_ACT_0_ENV_0")
os.makedirs(log_dir, exist_ok=True)

# ── Configuration Summary ────────────────────────────────────────────────

policy_mode = "shared" if args.shared_policy else "independent"

# Compute actual lr (some algos/envs use hardcoded values, not args.lr)
actual_lr = args.lr
if args.algo == "APPO":
    actual_lr = 5e-5  # all envs, conservative for async
elif args.algo == "IMPALA":
    if args.env == "evcharging":
        actual_lr = 5e-5
    elif args.env == "building":
        actual_lr = 5e-5
    else:
        actual_lr = 1e-4
elif args.algo == "PPO" and args.env == "building":
    actual_lr = 1e-4

print("\n" + "=" * 70)
print(f"MARL TRAINING CONFIGURATION - {args.env.upper()}")
print("=" * 70)
print(f"  Environment:      {args.env}")
print(f"  Algorithm:        {args.algo}")
print(f"  Policy mode:      {policy_mode}")
print(f"  Seed:             {args.seed}")
print(f"  Num iterations:   {args.num_iterations:,}")
print(f"  Num workers:      {args.num_workers}")
print(f"  Num GPUs:         {num_gpus}")
print(f"  Learning rate:    {actual_lr}")
if args.env == "cogen":
    print(f"  Renewables mag:   {args.rm}")
print(f"  Checkpoint freq:  {args.checkpoint_freq}")
print(f"  Log directory:    {log_dir}")
print("=" * 70 + "\n")

# ── Environment Factory ──────────────────────────────────────────────────

np.random.seed(args.seed)

if args.env == "evcharging":
    from envs.evcharging import GMMsTraceGenerator, MultiAgentEVChargingEnv

    def env_creator(env_config):
        trace_gen = GMMsTraceGenerator('caltech', 'Summer 2019')
        env = MultiAgentEVChargingEnv(
            trace_gen,
            periods_delay=0,
            moer_forecast_steps=36,
            project_action_in_env=True,
            discrete=False,
            verbose=0,
        )
        return ParallelPettingZooEnv(env)

    register_env("marl_env", env_creator)

    sample_env = MultiAgentEVChargingEnv(
        GMMsTraceGenerator('caltech', 'Summer 2019'),
        project_action_in_env=True,
    )
    env_config_dict = {}

elif args.env == "building":
    from envs.building import MultiAgentBuildingEnv, ParameterGenerator

    def env_creator(env_config):
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson',
            reward_beta=0.5,
        )
        # normalize_reward=True (default) — keeps reward in [-1, 0]
        # MeanStdFilter on obs provides the gradient signal instead
        env = MultiAgentBuildingEnv(params)
        return ParallelPettingZooEnv(env)

    register_env("marl_env", env_creator)

    sample_params = ParameterGenerator(
        building='OfficeSmall', weather='Hot_Dry', location='Tucson',
        reward_beta=0.5,
    )
    sample_env = MultiAgentBuildingEnv(sample_params)
    env_config_dict = {}

elif args.env == "cogen":
    from envs.cogen import MultiAgentCogenEnv

    def env_creator(env_config):
        rm = env_config.get("renewables_magnitude", 300)
        env = MultiAgentCogenEnv(renewables_magnitude=rm)
        return ParallelPettingZooEnv(env)

    register_env("marl_env", env_creator)

    sample_env = MultiAgentCogenEnv(renewables_magnitude=args.rm)
    env_config_dict = {"renewables_magnitude": args.rm}

# ── Policy Setup ─────────────────────────────────────────────────────────

agent_ids = sample_env.possible_agents

if args.shared_policy:
    # Shared policy: single network trained with all agents' data
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
    # Independent policies: one network per agent
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

print(f"[MARL] {len(agent_ids)} agents: {agent_ids}")
print(f"[MARL] Policy mode: {policy_mode} ({len(policies)} policies)")

# ── Ray Init & Algorithm Configuration ───────────────────────────────────

ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

common_multi_agent = {
    "policies": policies,
    "policy_mapping_fn": policy_mapping_fn,
    "policies_to_train": policy_ids,
}

# Observation normalization: Building has large scale differences
# (temps ~20 vs GHI/Occupower ~1000), so MeanStdFilter is critical
obs_filter = "MeanStdFilter" if args.env == "building" else "NoFilter"

# Algorithm-specific hyperparameters per environment
# Follows CLAUDE.md recommended defaults

if args.algo == "APPO":
    from ray.rllib.algorithms.appo import APPOConfig

    if args.env == "evcharging":
        appo_params = dict(
            train_batch_size=2880, num_sgd_iter=2, lr=5e-5,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.005, grad_clip=5.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288
    elif args.env == "building":
        appo_params = dict(
            train_batch_size=1152, num_sgd_iter=3, lr=5e-5,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.05, grad_clip=5.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288
    elif args.env == "cogen":
        appo_params = dict(
            train_batch_size=4000, num_sgd_iter=5, lr=5e-5,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, grad_clip=5.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 200

    config = (
        APPOConfig()
        .environment(env="marl_env", env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length=rollout_frag,
                  observation_filter=obs_filter, enable_connectors=True)
        .training(**appo_params)
        .debugging(seed=args.seed)
    )

elif args.algo == "PPO":
    from ray.rllib.algorithms.ppo import PPOConfig

    if args.env == "cogen":
        ppo_params = dict(
            train_batch_size=4000, sgd_minibatch_size=128,
            num_sgd_iter=10, lr=args.lr,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, grad_clip=0.5,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 200
    elif args.env == "evcharging":
        # 10 workers * 288 = 2880
        ppo_params = dict(
            train_batch_size=2880, sgd_minibatch_size=1024,
            num_sgd_iter=10,
            lr=args.lr, gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.005, grad_clip=0.5,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288
    elif args.env == "building":
        # 4 workers * 288 = 1152
        ppo_params = dict(
            train_batch_size=1152, sgd_minibatch_size=128,
            num_sgd_iter=3,
            lr=1e-4, gamma=0.99, lambda_=0.95, clip_param=0.1,
            entropy_coeff=0.05, grad_clip=0.5,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288

    config = (
        PPOConfig()
        .environment(env="marl_env", env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length=rollout_frag,
                  observation_filter=obs_filter, enable_connectors=True)
        .training(**ppo_params)
        .debugging(seed=args.seed)
    )

elif args.algo == "SAC":
    from ray.rllib.algorithms.sac import SACConfig
    config = (
        SACConfig()
        .environment(env="marl_env", env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length="auto",
                  observation_filter=obs_filter, enable_connectors=True)
        .training(
            train_batch_size=256, n_step=1,
            lr=args.lr, gamma=0.99, grad_clip=1.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        .debugging(seed=args.seed)
    )

elif args.algo == "IMPALA":
    from ray.rllib.algorithms.impala import ImpalaConfig

    if args.env == "evcharging":
        # Shared policy: use fewer workers (3) to reduce staleness,
        # batch=864 (3*288) for divisibility. Independent: normal 10 workers.
        if args.shared_policy:
            ev_impala_batch = 864   # 3 workers * 288
            ev_impala_workers = 3
        else:
            ev_impala_batch = 2880  # 10 workers * 288
            ev_impala_workers = args.num_workers
        impala_params = dict(
            train_batch_size=ev_impala_batch, lr=5e-5,
            gamma=0.99, entropy_coeff=0.005,
            vtrace=True, vtrace_clip_rho_threshold=0.5,
            vtrace_clip_pg_rho_threshold=0.5,
            grad_clip=5.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288
    elif args.env == "building":
        impala_params = dict(
            train_batch_size=1152, lr=5e-5,
            gamma=0.99, entropy_coeff=0.05,
            vtrace=True, vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            grad_clip=5.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 288
    elif args.env == "cogen":
        impala_params = dict(
            train_batch_size=4000, lr=1e-4,
            gamma=0.99, entropy_coeff=0.01,
            vtrace=True, vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            grad_clip=40.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        rollout_frag = 200

    # EVCharging shared IMPALA uses fewer workers to reduce policy staleness
    impala_workers = args.num_workers
    if args.env == "evcharging" and args.shared_policy:
        impala_workers = 3

    config = (
        ImpalaConfig()
        .environment(env="marl_env", env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=impala_workers,
                  rollout_fragment_length=rollout_frag,
                  observation_filter=obs_filter, enable_connectors=True)
        .training(**impala_params)
        .debugging(seed=args.seed)
    )

# ── Save Config JSON ─────────────────────────────────────────────────────

run_config = {
    "env": args.env,
    "algo": args.algo,
    "shared_policy": args.shared_policy,
    "policy_mode": policy_mode,
    "seed": args.seed,
    "num_iterations": args.num_iterations,
    "num_workers": args.num_workers,
    "num_gpus": num_gpus,
    "lr": actual_lr,
    "checkpoint_freq": args.checkpoint_freq,
    "timestamp": timestamp,
}
if args.env == "cogen":
    run_config["renewables_magnitude"] = args.rm

config_path = os.path.join(log_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2, default=str)
print(f"[MARL] Config saved to {config_path}")

# ── Training Loop ────────────────────────────────────────────────────────

trainer = config.build()

metrics_history = []
metrics_path = os.path.join(log_dir, "metrics.csv")
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

start_time = time.time()

try:
    for i in range(args.num_iterations):
        result = trainer.train()

        # Extract metrics safely
        mean_reward = result.get("episode_reward_mean", np.nan)
        mean_length = result.get("episode_len_mean", np.nan)

        loss_val = np.nan
        try:
            learner_info = result.get("info", {}).get("learner", {})
            if policy_ids[0] in learner_info:
                loss_val = learner_info[policy_ids[0]].get(
                    "learner_stats", {}
                ).get("total_loss", np.nan)
        except (KeyError, TypeError, AttributeError):
            pass

        elapsed = time.time() - start_time
        metrics_history.append({
            "iteration": i + 1,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "loss": loss_val,
            "elapsed_seconds": elapsed,
        })

        # Console output every 10 iterations
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Iter {i+1:>6d}/{args.num_iterations}] "
                  f"reward={mean_reward:>10.2f}  "
                  f"length={mean_length:>8.1f}  "
                  f"loss={loss_val:>10.4f}  "
                  f"elapsed={elapsed/60:>7.1f}min")

        # Write CSV periodically
        if (i + 1) % 10 == 0 or (i + 1) == args.num_iterations:
            pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)

        # Checkpoint
        if (i + 1) % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{i+1}")
            trainer.save(ckpt_path)
            print(f"[Checkpoint] Saved at iteration {i+1}")

except KeyboardInterrupt:
    print("\n[MARL] Training interrupted by user.")
except Exception as e:
    print(f"\n[MARL] Error during training: {e}")
    raise
finally:
    # Always save final metrics
    if metrics_history:
        pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)

    # Training summary
    total_time = time.time() - start_time
    completed = len(metrics_history)
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - {args.env.upper()} MARL ({args.algo}, {policy_mode})")
    print("=" * 70)
    print(f"  Iterations completed: {completed}/{args.num_iterations}")
    print(f"  Training duration:    {total_time/60:.1f} minutes "
          f"({total_time/3600:.1f} hours)")
    if metrics_history:
        last = metrics_history[-1]
        print(f"  Final mean reward:    {last['mean_reward']:.2f}")
        print(f"  Final mean length:    {last['mean_length']:.1f}")
    print(f"  Metrics saved to:     {metrics_path}")
    print(f"  Checkpoints in:       {checkpoint_dir}")
    print("=" * 70 + "\n")

    ray.shutdown()
