from __future__ import annotations
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from typing import Any, ClassVar

from sustaingym.envs.cogen import CogenEnv
from envs.cogen.MyCogenEnv import MyCogenEnv

import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


# ── CLI Arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OmniSafe Safe RL training for Cogen")
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--rm", type=int, default=300, help="Renewables magnitude")
parser.add_argument("--noise", type=float, default=0.0, help="Observation noise scale")
parser.add_argument("--noise_act", type=float, default=0.0, help="Action noise scale")
parser.add_argument("--noise_env", type=float, default=None, help="Environment noise scale")
parser.add_argument("--climit", type=float, default=1.0, help="Cost limit for CMDP")
parser.add_argument("--tsteps", type=int, default=2_000_064, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--cost_scale", type=float, default=1e-7, help="Cost scaling factor")
args = parser.parse_args()

ALGO = args.algo
RM = args.rm
NOISE = args.noise
NOISE_ACTION = args.noise_act
NOISE_ENV = args.noise_env
COST_LIMIT = args.climit
TRAIN_STEPS = args.tsteps
SEED = args.seed
COST_SCALE = args.cost_scale


def _flatten_obs_dict(obs_dict: dict) -> np.ndarray:
    """Convert a dict of scalar/array observations into a single 1D float32 array."""
    parts = []
    for val in obs_dict.values():
        arr = np.asarray(val, dtype=np.float32).flatten()
        parts.append(arr)
    return np.concatenate(parts)


# ── CMDP Wrapper ─────────────────────────────────────────────────────────────
@env_register
@env_unregister
class SustaingymCogenCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymCogen-v0']
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
        self._env = MyCogenEnv(
            CogenEnv(renewables_magnitude=RM, noise_env=NOISE_ENV),
            noise=NOISE,
            noise_action=NOISE_ACTION,
            safe_rl=True,
            **kwargs,
        )

        self._num_envs = num_envs
        self._device = device
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        obs = _flatten_obs_dict(obs)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.timesteps_per_day

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

        # Flatten dict obs to 1D array
        obs = _flatten_obs_dict(obs)

        # Cost = dynamic operating constraint violations (scaled)
        cost = sum(info['dyn_cv_costs'].values()) * COST_SCALE

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        return obs, reward, cost, terminated, truncated, info


# ── Training ─────────────────────────────────────────────────────────────────
noise_str = f"DS_{NOISE}_DA_{NOISE_ACTION}_DE_{NOISE_ENV}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("runs", "omnisafe_cogen", f"{ALGO}_{noise_str}_{timestamp}")

print(f"[OmniSafe-Cogen] algo={ALGO}  rm={RM}  noise={noise_str}  "
      f"cost_limit={COST_LIMIT}  steps={TRAIN_STEPS}  seed={SEED}")
print(f"[OmniSafe-Cogen] log_dir={log_dir}")

custom_cfgs = {
    'seed': SEED,
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': 96,
        'update_iters': 10,
        'cost_limit': COST_LIMIT,
        'batch_size': 1024,
        'reward_normalize': True,
        'cost_normalize': True,
    },
    'logger_cfgs': {
        'log_dir': log_dir,
    },
    'model_cfgs': {
        'actor': {
            'hidden_sizes': [128, 128, 128],
            'activation': 'relu',
        },
        'critic': {
            'hidden_sizes': [128, 128, 128],
            'activation': 'relu',
        },
    },
}

agent = omnisafe.Agent(ALGO, 'SustaingymCogen-v0', custom_cfgs=custom_cfgs)
agent.learn()

print(f"[OmniSafe-Cogen] Training completed. Results in: {log_dir}")
