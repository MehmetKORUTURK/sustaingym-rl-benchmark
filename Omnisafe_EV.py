from __future__ import annotations
import torch
import random
import numpy as np
import argparse
import os
# SustainGym imports
# from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator
from envs.evcharging import EVChargingEnv, GMMsTraceGenerator

# OmniSafe core
from typing import Any, ClassVar
import gymnasium
import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU



parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--lag", type=str, default="lagr")
parser.add_argument("--safe_rl", action="store_true", help="Enable SAFE_RL specific code")
parser.add_argument("--noise", type=float, default=None)
parser.add_argument("--noise_act", type=float, default=None)
parser.add_argument("--climit", type=float, default=1)
parser.add_argument("--tsteps", type=int, default=3_000_096*2)
args = parser.parse_args()

ALGO = args.algo # 
LAG = args.lag # "lagr", "no_lagr"
SAFE_RL = args.safe_rl
NOISE = args.noise
NOISE_ACTION = args.noise_act
COST_LIMIT = args.climit
TRAIN_STEPS = args.tsteps


##############################################################################
# EVCharging environment registered with OmniSafe
##############################################################################
@env_register
@env_unregister
class SustaingymEVChargingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymEVCharging-v0']
    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        # Create the data generator
        gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')
        self._env = EVChargingEnv(gmmg, noise=NOISE, noise_action=NOISE_ACTION, safe_rl=SAFE_RL,**kwargs)
        
        # Additional CMDP and device-related initialization
        self._num_envs = num_envs
        self._device = device

        # Set the action and observation spaces using the parent class attributes
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        obs = [x for v in obs.values() for x in v]
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    
    @property
    def max_episode_steps(self) -> int | None:
        return self._env.max_timestep

    def render(self) -> Any:
        # Return the image rendered by the environment
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

    def set_seed(self, seed: int) -> None:
        pass

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action.detach().cpu().numpy())
        obs = [x for v in obs.values() for x in v]
        
        # cost = np.zeros_like(reward)  # Replace with meaningful safety cost computation
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        cost = ((# (self._env.carbon_cost)*100
            (self._env.excess_charge)*100))
        
        print(f"\033[93mOnCRBO with noise = DS_{NOISE}_DA_{NOISE_ACTION} total tstep {TRAIN_STEPS} cost limit {COST_LIMIT} with current constraints at tstep {self._env.t} :{cost}\033[0m")
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        # print(f"\033[92mOnCRBO reward : {reward}\033[0m")

        # print(reward)
        return obs, reward, cost, terminated, truncated, info
        
# Register the environment

print(f"\033[92mRunning total tstep {TRAIN_STEPS} with {ALGO} in with noise = DS_{NOISE}_DA_{NOISE_ACTION}\033[0m")

    
env_id = 'SustaingymEVCharging-v0'
# Training with OmniSafe
custom_cfgs = {
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': 288,
        'update_iters': 10,
        'cost_limit': COST_LIMIT,
        # 'distance': 0.05,
        # 'lam_c': 0.99,
    },
    # 'logger_cfgs': {
    #     'log_dir': f"./runs/{ALGO}_{0.0 if NOISE is None else NOISE}",
    # },
    # 'lagrange_cfgs': {
    #     'cost_limit': COST_LIMIT,
    # },

}
agent = omnisafe.Agent(f"{ALGO}", 'SustaingymEVCharging-v0', custom_cfgs=custom_cfgs)
agent.learn()


        