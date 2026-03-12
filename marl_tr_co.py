"""
MARL training script for Cogen using Ray RLlib.

Usage:
    python marl_tr_co.py --algo PPO --seed 42 --num-iterations 100
    python marl_tr_co.py --algo IMPALA --rm 300 --num-workers 4
    python marl_tr_co.py --algo APPO --rm 300

Note: Only PPO, APPO, and IMPALA are supported because MultiAgentCogenEnv
uses Dict action spaces with mixed continuous/discrete sub-spaces, which
SAC cannot handle. Parameter sharing (--shared-policy) is also not
supported because agents have different action spaces.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from envs.cogen import MultiAgentCogenEnv

# ── CLI Arguments ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="MARL training for Cogen (Ray RLlib)")
parser.add_argument("--algo", type=str, default="PPO",
                    choices=["PPO", "APPO", "IMPALA"],
                    help="RLlib algorithm (default: PPO, only on-policy algos for Dict action space)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--num-iterations", type=int, default=750,
                    help="Number of training iterations (default: 100)")
parser.add_argument("--num-workers", type=int, default=4,
                    help="Number of rollout workers (default: 4)")
parser.add_argument("--checkpoint-freq", type=int, default=10,
                    help="Save checkpoint every N iterations (default: 10)")
parser.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs (default: auto-detect, 0 for CPU-only)")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 3e-4)")
parser.add_argument("--rm", type=int, default=300,
                    help="Renewables magnitude (default: 300)")
args = parser.parse_args()

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
log_dir = f"./logs_marl_train/cogen_{args.algo}/{timestamp}_NOISE_0_ACT_0_ENV_0"
os.makedirs(log_dir, exist_ok=True)

# ── Configuration Summary ────────────────────────────────────────────────

# Compute actual lr (APPO/IMPALA use hardcoded values, not args.lr)
actual_lr = args.lr
if args.algo == "APPO":
    actual_lr = 1e-4
elif args.algo == "IMPALA":
    actual_lr = 1e-4

print("\n" + "=" * 70)
print("MARL TRAINING CONFIGURATION - Cogen")
print("=" * 70)
print(f"  Algorithm:        {args.algo}")
print(f"  Policy mode:      independent")
print(f"  Seed:             {args.seed}")
print(f"  Num iterations:   {args.num_iterations:,}")
print(f"  Num workers:      {args.num_workers}")
print(f"  Num GPUs:         {num_gpus}")
print(f"  Learning rate:    {actual_lr}")
print(f"  Renewables mag:   {args.rm}")
print(f"  Checkpoint freq:  {args.checkpoint_freq}")
print(f"  Log directory:    {log_dir}")
print("=" * 70 + "\n")

# ── Environment Factory & Policy Setup ───────────────────────────────────

np.random.seed(args.seed)

env_config_dict = {"renewables_magnitude": args.rm}


def env_creator(env_config):
    """Create a ParallelPettingZoo-wrapped Cogen environment."""
    rm = env_config.get("renewables_magnitude", 300)
    env = MultiAgentCogenEnv(renewables_magnitude=rm)
    return ParallelPettingZooEnv(env)


register_env("multi_agent_cogen", env_creator)

sample_env = MultiAgentCogenEnv(renewables_magnitude=args.rm)
agent_ids = sample_env.possible_agents  # ['GT1', 'GT2', 'GT3', 'ST']

# Independent policies only (agents have different action spaces)
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

# ── Ray Init & Algorithm Configuration ───────────────────────────────────

ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

common_multi_agent = {
    "policies": policies,
    "policy_mapping_fn": policy_mapping_fn,
    "policies_to_train": policy_ids,
}

if args.algo == "PPO":
    config = (
        PPOConfig()
        .environment(env="multi_agent_cogen",
                     env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length=200, enable_connectors=True)
        .training(
            train_batch_size=4000, sgd_minibatch_size=128,
            num_sgd_iter=10, lr=args.lr,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, grad_clip=0.5,
            model={"fcnet_hiddens": [64, 64]},
        )
        .debugging(seed=args.seed)
    )
elif args.algo == "APPO":
    config = (
        APPOConfig()
        .environment(env="multi_agent_cogen",
                     env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length=200, enable_connectors=True)
        .training(
            train_batch_size=4000, num_sgd_iter=10, lr=1e-4,
            gamma=0.99, lambda_=0.95, clip_param=0.2,
            entropy_coeff=0.01, grad_clip=40.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        .debugging(seed=args.seed)
    )
elif args.algo == "IMPALA":
    config = (
        ImpalaConfig()
        .environment(env="multi_agent_cogen",
                     env_config=env_config_dict,
                     disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
        .multi_agent(**common_multi_agent)
        .rollouts(num_rollout_workers=args.num_workers,
                  rollout_fragment_length=200, enable_connectors=True)
        .training(
            train_batch_size=4000, lr=1e-4,
            gamma=0.99, entropy_coeff=0.01,
            vtrace=True, vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
            grad_clip=40.0,
            model={"fcnet_hiddens": [64, 64]},
        )
        .debugging(seed=args.seed)
    )

# ── Save Config JSON ─────────────────────────────────────────────────────

run_config = {
    "env": "cogen",
    "algo": args.algo,
    "shared_policy": False,
    "policy_mode": "independent",
    "seed": args.seed,
    "num_iterations": args.num_iterations,
    "num_workers": args.num_workers,
    "num_gpus": num_gpus,
    "lr": actual_lr,
    "renewables_magnitude": args.rm,
    "checkpoint_freq": args.checkpoint_freq,
    "timestamp": timestamp,
}
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

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Iter {i+1:>6d}/{args.num_iterations}] "
                  f"reward={mean_reward:>10.2f}  "
                  f"length={mean_length:>8.1f}  "
                  f"loss={loss_val:>10.4f}  "
                  f"elapsed={elapsed/60:>7.1f}min")

        if (i + 1) % 10 == 0 or (i + 1) == args.num_iterations:
            pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)

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
    if metrics_history:
        pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)

    total_time = time.time() - start_time
    completed = len(metrics_history)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Cogen MARL")
    print("=" * 70)
    print(f"  Algorithm:            {args.algo}")
    print(f"  Renewables mag:       {args.rm}")
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
