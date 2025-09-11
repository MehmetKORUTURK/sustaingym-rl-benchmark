from __future__ import annotations
import torch
import argparse
# SustainGym imports
from envs.building import BuildingEnv, ParameterGenerator
# OmniSafe core
from typing import Any, ClassVar
import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU
from ttp.colored import color_text, Colors


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
# Building environment registered with OmniSafe
##############################################################################
@env_register
@env_unregister
class SustaingymBuildingCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymBuilding-v0']
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
        params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self._env = BuildingEnv(params, noise=NOISE)
        
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
       
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    
    @property
    def max_episode_steps(self) -> int | None:
        return self._env.episode_len

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

## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cost = self._env.total_cost
        # cost = 0
        print(
            color_text(f"{ALGO}", Colors.BOLD + Colors.BLUE) + " " +
            color_text("with noise =", Colors.YELLOW) + " " +
            color_text(f"DS_{NOISE}", Colors.CYAN) + " " +
            color_text(f"DA_{NOISE_ACTION}", Colors.MAGENTA) + " " +
            color_text("| total tstep:", Colors.YELLOW) + " " +
            color_text(f"{TRAIN_STEPS}", Colors.GREEN) + " " +
            color_text("| cost limit:", Colors.YELLOW) + " " +
            color_text(f"{COST_LIMIT}", Colors.RED) + " " +
            color_text("| current constraints at tstep:", Colors.YELLOW) + " " +
            color_text(f"{self._env.num_epoch_runs}", Colors.MAGENTA) + " " +
            color_text(":", Colors.YELLOW) +
            color_text(f"{cost}", Colors.BLUE))
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        # print(f"\033[92mOnCRBO reward : {reward}\033[0m")

        # print(reward)
        return obs, reward, cost, terminated, truncated, info
        
# Register the environment

print(color_text(
    f"Running {ALGO} for total training steps: {TRAIN_STEPS} "
    f"| Noise: DS_{NOISE} | DA_{NOISE_ACTION}", Colors.GREEN))
    
env_id = 'SustaingymBuilding-v0'
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
        # 'entropy_coef': 0.0,
        'batch_size': 1024,
        'reward_normalize': True,
        'cost_normalize': True,
        'max_grad_norm': 0.001, #! deneme
    },
    # 'lagrange_cfgs': {
    #     'cost_limit': COST_LIMIT,
    
    # },
    'model_cfgs': {
        'actor': {
            'hidden_sizes': [128, 128, 128],
            'activation': 'relu',
            'lr': 1e-7,
        },
        'critic': {
            'hidden_sizes': [128, 128, 128],
            'activation': 'relu',
            'lr': 1e-7,
        }
    }
}
agent = omnisafe.Agent(f"{ALGO}", 'SustaingymBuilding-v0', custom_cfgs=custom_cfgs)
agent.learn()

print(color_text(f"Running Building Env | Total Tstep: {TRAIN_STEPS} | {ALGO} | Noise: DS_{NOISE} | DA_{NOISE_ACTION}", Colors.GREEN))
print(color_text(f"Training completed", Colors.CYAN))
        