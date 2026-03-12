from __future__ import annotations
import torch
import argparse
import os
from datetime import datetime
from typing import Any, ClassVar

from envs.building import BuildingEnv, ParameterGenerator

import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


# ── CLI Arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OmniSafe Safe RL training for Building")
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--noise", type=float, default=None, help="Observation noise scale")
parser.add_argument("--noise_act", type=float, default=None, help="Action noise scale")
parser.add_argument("--noise_env", type=float, default=None, help="Environment noise scale")
parser.add_argument("--climit", type=float, default=1.0, help="Cost limit for CMDP")
parser.add_argument("--tsteps", type=int, default=6_000_192, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--cost_scale", type=float, default=1.0, help="Cost scaling factor")
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
class SustaingymBuildingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymBuilding-v0']
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
        params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self._env = BuildingEnv(
            params,
            noise=NOISE,
            noise_action=NOISE_ACTION,
            noise_env=NOISE_ENV,
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
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.episode_len

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

        # Cost = comfort violation only (temperature exceeding threshold)
        # Energy cost is already penalized in reward, so we don't double-penalize
        cost = info['cost_breakdown']['comfort_level'] * COST_SCALE

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        return obs, reward, cost, terminated, truncated, info


# ── Training ─────────────────────────────────────────────────────────────────
noise_str = f"DS_{NOISE}_DA_{NOISE_ACTION}_DE_{NOISE_ENV}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("runs", "omnisafe_building", f"{ALGO}_{noise_str}_{timestamp}")

print(f"[OmniSafe-Building] algo={ALGO}  noise={noise_str}  "
      f"cost_limit={COST_LIMIT}  steps={TRAIN_STEPS}  seed={SEED}")
print(f"[OmniSafe-Building] log_dir={log_dir}")

custom_cfgs = {
    'seed': SEED,
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': 288,
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

agent = omnisafe.Agent(ALGO, 'SustaingymBuilding-v0', custom_cfgs=custom_cfgs)
agent.learn()

print(f"[OmniSafe-Building] Training completed. Results in: {log_dir}")
