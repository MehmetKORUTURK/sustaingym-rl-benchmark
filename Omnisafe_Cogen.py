
from __future__ import annotations
import torch
import numpy as np
import argparse
from ttp.colored import color_text, Colors
# SustainGym imports
from sustaingym.envs.cogen import CogenEnv
from envs.cogen.MyCogenEnv import MyCogenEnv
# OmniSafe core
from typing import Any, ClassVar
import omnisafe
from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="OnCRPO")
parser.add_argument("--rm", type=int, default=300)
parser.add_argument("--lag", type=str, default="lagr")
parser.add_argument("--safe_rl", action="store_true", help="Enable SAFE_RL specific code")
parser.add_argument("--noise", type=float, default=None)
parser.add_argument("--noise_act", type=float, default=None)
parser.add_argument("--climit", type=float, default=1)
parser.add_argument("--tsteps", type=int, default=1_000_032*2)
args = parser.parse_args()

ALGO = args.algo
RM = args.rm
LAG = args.lag
SAFE_RL = args.safe_rl
NOISE = args.noise
NOISE_ACTION = args.noise_act
COST_LIMIT = args.climit
TRAIN_STEPS = args.tsteps


##############################################################################
# Helper function to flatten a dictionary of observations
##############################################################################
def _flatten_obs_dict(obs_dict: dict) -> np.ndarray:
    """
    Convert a dict of scalar/array observations into a single 1D float32 array.
    """
    obs_list = []
    for val in obs_dict.values():
        # Ensure 'val' is at least a 1D array (turn scalars into shape (1,))
        arr = np.array(val, ndmin=1)
        obs_list.extend(arr.flatten())
    return np.array(obs_list, dtype=np.float32)


##############################################################################
# Building environment registered with OmniSafe
##############################################################################
@env_register
@env_unregister
class SustaingymCogenCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['SustaingymCogen-v0']
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
        self._env = MyCogenEnv(CogenEnv(renewables_magnitude=RM), safe_rl=SAFE_RL)

        # Additional CMDP and device-related initialization
        self._num_envs = num_envs
        self._device = device

        # Set the action and observation spaces
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        print(f"info: {info}")
        # Flatten the dictionary observation into a 1D array of shape (obs_dim,)
        obs_array = _flatten_obs_dict(obs)
        # Expand to (1, obs_dim) so OmniSafe's normalizer sees a batch dimension
        obs_array = np.expand_dims(obs_array, axis=0)  # shape: (1, obs_dim)

        return torch.as_tensor(obs_array, dtype=torch.float32, device=self._device), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Convert the action to NumPy if needed
        action_np = action.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = self._env.step(action_np)
        

        # print(f"info: {info}")
        # ##TODO:   
        # The cost is the sum of all values in the dyn_cv_costs dictionary
         # Debug: Print the contents of the info dictionary
        print("\n=== INFO DICTIONARY CONTENTS ===")
        for key, value in info.items():
            print(f"Key: {key}")
            if isinstance(value, dict):
                print("  Nested dictionary with keys:", list(value.keys()))
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  Value: {value}")
        print("===============================\n")
        if 'dyn_cv_costs' in info:
            cost = sum(info['dyn_cv_costs'].values())
            print("Cost from dyn_cv_costs:", cost)
            Total = -(sum(info['fuel_costs'].values()) + sum(info['ramp_costs'].values()) + info['non_delivery_cost'])
            print("Total cost:", Total)
            print(f"reward: {reward}")

        else:
            # Fallback in case the info doesn't contain what we expect
            cost = 3.0
            print("Warning: dyn_cv_costs not found in info dictionary!")
        # ##TODO:   

        # Flatten the dictionary observation to (obs_dim,) and expand to (1, obs_dim)
        obs_array = _flatten_obs_dict(obs)
        obs_array = np.expand_dims(obs_array, axis=0)  # shape: (1, obs_dim)

        # If your environment returns scalars for reward, cost, etc., expand them to shape (1,)
        if np.isscalar(reward):
            reward = np.array([reward], dtype=np.float32)
        if np.isscalar(terminated):
            terminated = np.array([terminated], dtype=np.float32)
        if np.isscalar(truncated):
            truncated = np.array([truncated], dtype=np.float32)

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # cost = 0
        cost = (cost*(1))/10000000
        print(color_text(f"{ALGO}", Colors.BOLD + Colors.BLUE) + " " +
                color_text("with noise =", Colors.YELLOW) + " " +
                color_text(f"DS_{NOISE}", Colors.CYAN) + " " +
                color_text(f"DA_{NOISE_ACTION}", Colors.MAGENTA) + " " +
                color_text("| total tstep:", Colors.YELLOW) + " " +
                color_text(f"{TRAIN_STEPS}", Colors.GREEN) + " " +
                color_text("| cost limit:", Colors.YELLOW) + " " +
                color_text(f"{COST_LIMIT}", Colors.RED) + " " +
                color_text("| current constraints at tstep:", Colors.YELLOW) + " " +
                color_text(f"{obs['Time']*96}", Colors.MAGENTA) + " " +
                color_text("| current constraint cost:", Colors.YELLOW) + " " +
                color_text(f"{cost}", Colors.BLUE))

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # Convert everything to Torch tensors
        obs_torch = torch.as_tensor(obs_array, dtype=torch.float32, device=self._device)
        reward_torch = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        cost_torch = torch.as_tensor(cost, dtype=torch.float32, device=self._device)
        terminated_torch = torch.as_tensor(terminated, dtype=torch.float32, device=self._device)
        truncated_torch = torch.as_tensor(truncated, dtype=torch.float32, device=self._device)

        return obs_torch, reward_torch, cost_torch, terminated_torch, truncated_torch, info

    @property
    def max_episode_steps(self) -> int | None:
        return self._env.timesteps_per_day

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def set_seed(self, seed: int) -> None:
        pass


print(color_text(
    f"Running {ALGO} for total training steps: {TRAIN_STEPS} "
    f"| Noise: DS_{NOISE} | DA_{NOISE_ACTION}", Colors.GREEN))


env_id = 'SustaingymCogen-v0'
custom_cfgs = {
    'train_cfgs': {
        'total_steps': TRAIN_STEPS,
    },
    'algo_cfgs': {
        'steps_per_epoch': 96,
        'update_iters': 10,
        'cost_limit': COST_LIMIT,
        # 'distance': 0.05,
        # 'lam_c': 0.99,
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
    

agent = omnisafe.Agent(f"{ALGO}", env_id, custom_cfgs=custom_cfgs)
agent.learn()

print(color_text(f"Running Cogen Env | Total Tstep: {TRAIN_STEPS} | {ALGO} | Noise: DS_{NOISE} | DA_{NOISE_ACTION}", Colors.GREEN))
print(color_text(f"Training completed", Colors.CYAN))
        