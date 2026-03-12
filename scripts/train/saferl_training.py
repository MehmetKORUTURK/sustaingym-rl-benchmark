"""
Unified OmniSafe (Safe RL / CMDP) training script for all three environments.

Usage:
    python saferl_training.py --env evcharging --algo OnCRPO --climit 1
    python saferl_training.py --env building   --algo OnCRPO --climit 1
    python saferl_training.py --env cogen      --algo OnCRPO --rm 300 --climit 1
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import argparse
import os
from datetime import datetime
from typing import Any, ClassVar

import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


# ── CLI Arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OmniSafe Safe RL training")
parser.add_argument("--env", type=str, required=True,
                    choices=["evcharging", "building", "cogen"],
                    help="Environment to train on")
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--noise", type=float, default=None, help="Observation noise scale")
parser.add_argument("--noise_act", type=float, default=None, help="Action noise scale")
parser.add_argument("--noise_env", type=float, default=None, help="Environment noise scale")
parser.add_argument("--climit", type=float, default=1.0, help="Cost limit for CMDP")
parser.add_argument("--tsteps", type=int, default=None,
                    help="Total training steps (default: env-specific)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--cost_scale", type=float, default=None,
                    help="Cost scaling factor (default: env-specific)")
# Cogen-specific
parser.add_argument("--rm", type=int, default=300, help="Renewables magnitude (cogen only)")
args = parser.parse_args()

# ── Environment defaults ─────────────────────────────────────────────────────
ENV_DEFAULTS = {
    "evcharging": {"tsteps": 6_000_000, "cost_scale": 100.0, "steps_per_epoch": 2000},
    "building":   {"tsteps": 6_000_000, "cost_scale": 1.0,   "steps_per_epoch": 2000},
    "cogen":      {"tsteps": 6_000_000, "cost_scale": 0.005, "steps_per_epoch": 2000},
}

defaults = ENV_DEFAULTS[args.env]
TRAIN_STEPS = args.tsteps or defaults["tsteps"]
COST_SCALE = args.cost_scale if args.cost_scale is not None else defaults["cost_scale"]
STEPS_PER_EPOCH = defaults["steps_per_epoch"]


# ── Helper ───────────────────────────────────────────────────────────────────
def _flatten_obs_dict(obs_dict: dict) -> np.ndarray:
    """Convert a dict of scalar/array observations into a single 1D float32 array."""
    parts = []
    for val in obs_dict.values():
        arr = np.asarray(val, dtype=np.float32).flatten()
        parts.append(arr)
    return np.concatenate(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CMDP Wrappers (one per environment)
# ══════════════════════════════════════════════════════════════════════════════

# ── EVCharging ───────────────────────────────────────────────────────────────
@env_register
@env_unregister
class SustaingymEVChargingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymEVCharging-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, num_envs: int = 1,
                 device: torch.device = DEVICE_CPU, **kwargs: Any) -> None:
        super().__init__(env_id)
        from envs.evcharging import EVChargingEnv, GMMsTraceGenerator
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')
        self._env = EVChargingEnv(
            gmmg, noise=args.noise, noise_action=args.noise_act,
            noise_env=args.noise_env, safe_rl=True, **kwargs,
        )
        self._num_envs = num_envs
        self._device = device
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._prev_cumulative_excess = 0.0

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = _flatten_obs_dict(obs)
        self._prev_cumulative_excess = 0.0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self):
        return self._env.max_timestep

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def set_seed(self, seed: int):
        self._env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy())
        obs = _flatten_obs_dict(obs)
        # Per-step cost: delta of cumulative excess_charge
        curr = self._env._reward_breakdown['excess_charge']
        cost = (curr - self._prev_cumulative_excess) * COST_SCALE
        self._prev_cumulative_excess = curr
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated))
        return obs, reward, cost, terminated, truncated, info


# ── Building ─────────────────────────────────────────────────────────────────
@env_register
@env_unregister
class SustaingymBuildingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymBuilding-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, num_envs: int = 1,
                 device: torch.device = DEVICE_CPU, **kwargs: Any) -> None:
        super().__init__(env_id)
        from envs.building import BuildingEnv, ParameterGenerator
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson',
            reward_beta=0.5)
        self._env = BuildingEnv(
            params, noise=args.noise, noise_action=args.noise_act,
            noise_env=args.noise_env, **kwargs,
        )
        self._num_envs = num_envs
        self._device = device
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_action = np.zeros(self._action_space.shape, dtype=np.float32)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self):
        return self._env.episode_len

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def set_seed(self, seed: int):
        self._env.reset(seed=seed)

    def step(self, action):
        act_np = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(act_np)
        # Cost = mean HVAC ramping (action change rate) across AC-enabled zones
        # Fully orthogonal to reward: reward measures energy + comfort,
        # but CMDP cost penalizes rapid HVAC switching (equipment wear)
        ramp = np.abs(act_np - self._prev_action) * self._env.ac_map
        num_zones = max(1, int(self._env.ac_map.sum()))
        cost = float(ramp.sum() / num_zones) * COST_SCALE
        self._prev_action = act_np.copy()
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated))
        return obs, reward, cost, terminated, truncated, info


# ── Cogen ────────────────────────────────────────────────────────────────────
@env_register
@env_unregister
class SustaingymCogenCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymCogen-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, env_id: str, num_envs: int = 1,
                 device: torch.device = DEVICE_CPU, **kwargs: Any) -> None:
        super().__init__(env_id)
        from sustaingym.envs.cogen import CogenEnv
        from envs.cogen.MyCogenEnv import MyCogenEnv
        self._env = MyCogenEnv(
            CogenEnv(renewables_magnitude=args.rm),
            noise=args.noise or 0.0, noise_action=args.noise_act or 0.0,
            noise_env=args.noise_env, safe_rl=True, **kwargs,
        )
        self._num_envs = num_envs
        self._device = device
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = _flatten_obs_dict(obs)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self):
        return self._env.timesteps_per_day

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def set_seed(self, seed: int):
        self._env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy())
        obs = _flatten_obs_dict(obs)
        # Cost = turbine ramping stress (equipment wear/safety)
        # Orthogonal to reward: reward is dominated by fuel + delivery,
        # agent ignores small ramp_penalty=2 in reward but CMDP enforces smooth operation
        cost = sum(info['ramp_costs'].values()) * COST_SCALE
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated))
        return obs, reward, cost, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
ENV_ID_MAP = {
    "evcharging": "SustaingymEVCharging-v0",
    "building": "SustaingymBuilding-v0",
    "cogen": "SustaingymCogen-v0",
}

env_id = ENV_ID_MAP[args.env]
noise_str = f"DS_{args.noise}_DA_{args.noise_act}_DE_{args.noise_env}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("runs", f"omnisafe_{args.env}", f"{timestamp}_{args.algo}_CL_{args.climit}")

print(f"[SafeRL] env={args.env}  algo={args.algo}  noise={noise_str}  "
      f"cost_limit={args.climit}  cost_scale={COST_SCALE}  "
      f"steps={TRAIN_STEPS}  seed={args.seed}")
print(f"[SafeRL] log_dir={log_dir}")

custom_cfgs = {
    'seed': args.seed,
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': STEPS_PER_EPOCH,
        'update_iters': 10,
    },
    'logger_cfgs': {
        'log_dir': log_dir,
    },
}

# Lagrangian algorithms use lagrange_cfgs.cost_limit
# Non-Lagrangian algorithms use algo_cfgs.cost_limit
# Off-policy algorithms need different config structure
OFF_POLICY_ALGOS = {'SACLag', 'TD3Lag', 'DDPGLag', 'SACPID', 'TD3PID', 'DDPGPID',
                    'DDPGCBF', 'SACRCBF', 'CRABS'}
LAGRANGIAN_ALGOS = {'PPOLag', 'TRPOLag', 'FOCOPS', 'CUP', 'PPOSaute', 'PPOSimmerPID',
                    'PPOEarlyTerminated', 'SACLag', 'TD3Lag', 'DDPGLag'}

if args.algo in OFF_POLICY_ALGOS:
    # Off-policy: replace on-policy specific configs but keep steps_per_epoch
    custom_cfgs['algo_cfgs'] = {
        'steps_per_epoch': STEPS_PER_EPOCH,
        'batch_size': 256,
        'update_cycle': 100,  # Was 1; accumulate fresh data before updating to reduce stale-buffer bias
    }

if args.algo in LAGRANGIAN_ALGOS:
    custom_cfgs['lagrange_cfgs'] = {
        'cost_limit': args.climit,
    }
else:
    custom_cfgs['algo_cfgs']['cost_limit'] = args.climit

# Building and Cogen use larger networks
if args.env in ("building", "cogen"):
    if args.algo not in OFF_POLICY_ALGOS:
        custom_cfgs['algo_cfgs']['batch_size'] = 1024
    # No reward_normalize/cost_normalize: BuildingEnv already normalizes to [-1,0]
    # and Cogen reward is already scaled by 1e7 in MyCogenEnv
    custom_cfgs['model_cfgs'] = {
        'actor': {'hidden_sizes': [128, 128, 128], 'activation': 'relu'},
        'critic': {'hidden_sizes': [128, 128, 128], 'activation': 'relu'},
    }

agent = omnisafe.Agent(args.algo, env_id, custom_cfgs=custom_cfgs)
agent.learn()

print(f"[SafeRL] Training completed. Results in: {log_dir}")
