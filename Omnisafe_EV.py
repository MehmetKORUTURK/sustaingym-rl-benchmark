from __future__ import annotations
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from typing import Any, ClassVar

from envs.evcharging import EVChargingEnv, GMMsTraceGenerator

import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


# ── CLI Arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OmniSafe Safe RL training for EVCharging")
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--noise", type=float, default=None, help="Observation noise scale")
parser.add_argument("--noise_act", type=float, default=None, help="Action noise scale")
parser.add_argument("--noise_env", type=float, default=None, help="Environment noise scale")
parser.add_argument("--climit", type=float, default=1.0, help="Cost limit for CMDP")
parser.add_argument("--tsteps", type=int, default=6_000_192, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--cost_scale", type=float, default=100.0, help="Cost scaling factor")
args = parser.parse_args()

ALGO = args.algo
NOISE = args.noise
NOISE_ACTION = args.noise_act
NOISE_ENV = args.noise_env
COST_LIMIT = args.climit
TRAIN_STEPS = args.tsteps
SEED = args.seed
COST_SCALE = args.cost_scale


# ── CMDP Wrapper ─────────────────────────────────────────────────────────────
@env_register
@env_unregister
class SustaingymEVChargingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymEVCharging-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')
        # safe_rl=True so env sets obs_space to flat Box
        self._env = EVChargingEnv(
            gmmg,
            noise=NOISE,
            noise_action=NOISE_ACTION,
            noise_env=NOISE_ENV,
            safe_rl=True,
            **kwargs,
        )

        self._num_envs = num_envs
        self._device = device
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

        # Track cumulative excess_charge for per-step cost delta
        self._prev_cumulative_excess = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        # Flatten dict obs to list (env returns dict regardless of safe_rl)
        obs = [x for v in obs.values() for x in v]
        # Reset per-episode cost tracker
        self._prev_cumulative_excess = 0.0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.max_timestep

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def set_seed(self, seed: int) -> None:
        self._env.reset(seed=seed)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy()
        )
        # Flatten dict obs
        obs = [x for v in obs.values() for x in v]

        # Per-step cost: delta of cumulative excess_charge from reward breakdown
        curr_cumulative = self._env._reward_breakdown['excess_charge']
        step_excess = curr_cumulative - self._prev_cumulative_excess
        self._prev_cumulative_excess = curr_cumulative
        cost = step_excess * COST_SCALE

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        return obs, reward, cost, terminated, truncated, info


# ── Training ─────────────────────────────────────────────────────────────────
noise_str = f"DS_{NOISE}_DA_{NOISE_ACTION}_DE_{NOISE_ENV}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("runs", "omnisafe_evcharging", f"{ALGO}_{noise_str}_{timestamp}")

print(f"[OmniSafe-EV] algo={ALGO}  noise={noise_str}  "
      f"cost_limit={COST_LIMIT}  steps={TRAIN_STEPS}  seed={SEED}")
print(f"[OmniSafe-EV] log_dir={log_dir}")

custom_cfgs = {
    'seed': SEED,
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': 288,
        'update_iters': 10,
        'cost_limit': COST_LIMIT,
    },
    'logger_cfgs': {
        'log_dir': log_dir,
    },
}

agent = omnisafe.Agent(ALGO, 'SustaingymEVCharging-v0', custom_cfgs=custom_cfgs)
agent.learn()
