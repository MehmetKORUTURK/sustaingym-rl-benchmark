
from gymnasium import Wrapper, spaces
import gymnasium as gym
import numpy as np
from copy import deepcopy

class MyCogenEnv(Wrapper):
    def __init__(self, env,
                 noise: float = 0.0,
                 noise_action: float = 0.0,
                 noise_env: dict = None,
                 safe_rl: bool = False,
                 verbose: bool = False,
                 reward_scale: float = 1e7):
        """Wrapper for CogenEnv with noise injection.

        Args:
            env: Base CogenEnv to wrap
            noise: Observation noise scale (0.0=no noise, 1.0=default, 2.0=double)
            noise_action: Action noise std dev (e.g., 0.05 for 5% noise)
            noise_env: Environmental noise dict {"temperature": 1.0, "pressure": 0.01, "humidity": 0.02}
            safe_rl: If True, flatten observation space to Box
            verbose: Print noise configuration
            reward_scale: Reward scaling factor (default 1e7)
        """
        super(MyCogenEnv, self).__init__(env)
        self.safe_rl = safe_rl
        self.verbose = verbose

        # Observation noise: Simple scaling of base ratios
        # noise=1.0 gives realistic sensor/forecast noise
        # noise=0.0 disables all observation noise
        self.noise = noise
        if noise > 0.0:
            self.noise_obs_config = {
                "ambient_current": 0.02 * noise,   # Current sensor noise
                "ambient_forecast": 0.05 * noise,  # Forecast uncertainty
                "targets": 0.01 * noise,           # Demand uncertainty
                "prices": 0.03 * noise,            # Price uncertainty
            }
        else:
            self.noise_obs_config = {
                "ambient_current": 0.0,
                "ambient_forecast": 0.0,
                "targets": 0.0,
                "prices": 0.0,
            }

        # Action noise parameter
        self.noise_action = noise_action

        # Environmental noise: delegate to the base CogenEnv so that it
        # affects the ONNX model inputs and reward computation path.
        default_env_noise = {
            "temperature": 0.0,  # Ambient temperature (F)
            "pressure": 0.0,     # Ambient pressure (psia)
            "humidity": 0.0,     # Ambient relative humidity (fraction)
            "model": 0.0,        # ONNX model output uncertainty (fraction)
        }
        if noise_env is None:
            self.noise_env = default_env_noise
        elif isinstance(noise_env, dict):
            self.noise_env = {**default_env_noise, **noise_env}
        else:
            # If scalar provided, scale each noise type appropriately
            self.noise_env = {
                "temperature": 2.0 * noise_env,   # 2 F per unit
                "pressure": 0.1 * noise_env,      # 0.1 psia per unit
                "humidity": 0.05 * noise_env,      # 0.05 fraction per unit
                "model": 0.15 * noise_env,         # 15% model uncertainty per unit (increased from 4%)
            }
        # Push to underlying CogenEnv so noise is applied before reward calc
        env.noise_env = self.noise_env

        # Reward scaling factor
        self.reward_scale = reward_scale
        self.timesteps_per_day = env.timesteps_per_day

        # Verbose tracking (print noise info only once per episode)
        self._verbose_printed = False

        # Adjust observation space by removing 'Prev_Action' and flattening
        self.observation_space = self._adjust_observation_space(env.observation_space)

        # Keep Dict space reference for per-key noise clipping (even if safe_rl
        # replaces self.observation_space with a flat Box below).
        self._obs_dict_space = self.observation_space

        # For safe RL algorithms, flatten Dict space to Box space
        if self.safe_rl:
            self.observation_space = gym.spaces.Box(
                low=np.array([low for v in self._obs_dict_space.values() for low in v.low]),
                high=np.array([high for v in self._obs_dict_space.values() for high in v.high]),
                dtype=np.float32
            )

        self.action_space = self._flatten_action_space(env.action_space)

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)

        # Fix: CogenEnv uses seed % n_days which can select the last day,
        # causing IndexError in _forecast_from_time when it needs day+1.
        # Clamp to n_days-2 to match the seed=None branch (high=n_days-1, exclusive).
        if self.env.current_day >= self.env.n_days - 1:
            self.env.current_day = self.env.current_day % (self.env.n_days - 1)
            # Re-fetch obs with corrected day (use _get_obs which exists in
            # both local and package CogenEnv; env noise not needed at reset)
            self.env.obs = self.env._get_obs()
            obs = self.env.obs

        # Reset verbose flag for new episode
        self._verbose_printed = False

        # Environmental noise is now handled inside CogenEnv (before reward).
        # Flatten Prev_Action into top-level keys, then apply observation noise
        # so the agent's first observation is noisy too (sensor/forecast uncertainty).
        obs = self._remove_prev_action(obs)
        obs = self._apply_observation_noise(obs)

        return obs, infos
    

    def step(self, action):
        # Apply action noise (actuator uncertainty)
        if self.noise_action > 0.0:
            action = self._apply_action_noise(action)

        # Convert to Dict format and step environment
        original_action = self._unflatten_action(action)
        obs, reward, done, terminal, info = self.env.step(original_action)

        # Environmental noise is now handled inside CogenEnv (before reward).
        # Remove Prev_Action and apply observation noise (sensor uncertainty).
        obs = self._remove_prev_action(obs)

        # Diagnostic: log noise delta once per episode
        if self.verbose and not self._verbose_printed:
            print(f"[MyCogenEnv] Noise config: noise={self.noise}, "
                  f"noise_action={self.noise_action}, noise_env={self.noise_env}")
            if self.noise > 0.0:
                clean_copy = deepcopy(obs)
                obs = self._apply_observation_noise(obs)
                deltas = []
                for key in clean_copy:
                    c = clean_copy.get(key)
                    n = obs.get(key)
                    if isinstance(c, np.ndarray) and isinstance(n, np.ndarray):
                        rms = float(np.sqrt(np.mean((n - c) ** 2)))
                        mean_abs = float(np.mean(np.abs(c))) if np.any(c) else 1.0
                        pct = 100.0 * rms / mean_abs if mean_abs > 0 else 0.0
                        deltas.append(f"{key}: rms={rms:.4f} ({pct:.1f}%)")
                if deltas:
                    print(f"[MyCogenEnv] Noise delta: {', '.join(deltas)}")
            else:
                obs = self._apply_observation_noise(obs)
            self._verbose_printed = True
        else:
            obs = self._apply_observation_noise(obs)

        return obs, reward / self.reward_scale, done, terminal, info
    
    def _apply_environmental_noise(self, obs):
        """Apply environmental noise to current ambient conditions (TAMB[0], PAMB[0], RHAMB[0])."""
        if all(std == 0.0 for std in self.noise_env.values()):
            return obs

        obs = deepcopy(obs)

        # Temperature noise
        if self.noise_env["temperature"] > 0.0 and "TAMB" in obs:
            obs["TAMB"] = obs["TAMB"].copy()
            obs["TAMB"][0] += self.env.np_random.normal(0, self.noise_env["temperature"])

        # Pressure noise
        if self.noise_env["pressure"] > 0.0 and "PAMB" in obs:
            obs["PAMB"] = obs["PAMB"].copy()
            obs["PAMB"][0] += self.env.np_random.normal(0, self.noise_env["pressure"])

        # Humidity noise (clipped to [0, 1])
        if self.noise_env["humidity"] > 0.0 and "RHAMB" in obs:
            obs["RHAMB"] = obs["RHAMB"].copy()
            obs["RHAMB"][0] += self.env.np_random.normal(0, self.noise_env["humidity"])
            obs["RHAMB"][0] = np.clip(obs["RHAMB"][0], 0.0, 1.0)

        return obs

    def _apply_observation_noise(self, obs):
        """Apply multiplicative Gaussian noise to observations."""
        if self.noise == 0.0:
            return obs

        obs = deepcopy(obs)

        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                continue

            # Ambient observations: different noise for current vs forecast
            if key in ["TAMB", "PAMB", "RHAMB"]:
                noisy_value = value.copy()
                sigma_current = self.noise_obs_config["ambient_current"]
                sigma_forecast = self.noise_obs_config["ambient_forecast"]

                # Current (index 0): small sensor noise
                if sigma_current > 0.0:
                    noise = self.env.np_random.normal(0, sigma_current * np.abs(value[0]))
                    noisy_value[0] += noise

                # Forecasts (index 1+): larger uncertainty, grows with horizon
                if sigma_forecast > 0.0:
                    for i in range(1, len(value)):
                        horizon_factor = 1.0 + 0.2 * (i - 1)
                        noise = self.env.np_random.normal(0, sigma_forecast * horizon_factor * np.abs(value[i]))
                        noisy_value[i] += noise

                obs[key] = noisy_value

            # Targets: demand uncertainty
            elif key in ["Target_Power", "Target_Steam"]:
                sigma = self.noise_obs_config["targets"]
                if sigma > 0.0:
                    noise = self.env.np_random.normal(0, sigma * np.abs(value), size=value.shape)
                    obs[key] = value + noise

            # Prices: market uncertainty
            elif key in ["Energy_Price", "Gas_Price"]:
                sigma = self.noise_obs_config["prices"]
                if sigma > 0.0:
                    noise = self.env.np_random.normal(0, sigma * np.abs(value), size=value.shape)
                    obs[key] = value + noise

        # Clip to valid ranges (uses stored Dict space so clipping works
        # regardless of whether safe_rl flattened self.observation_space).
        for key in obs:
            if key in self._obs_dict_space.spaces:
                space = self._obs_dict_space.spaces[key]
                if isinstance(space, spaces.Box):
                    obs[key] = np.clip(obs[key], space.low, space.high)

        return obs

    def _apply_action_noise(self, action):
        """Apply action noise: multiplicative for continuous, flip probability for discrete."""
        action = np.asarray(action).flatten().copy()
        index = 0

        for key, space in self.env.action_space.spaces.items():
            if isinstance(space, spaces.Box):
                # Continuous: multiplicative Gaussian noise
                size = np.prod(space.shape)
                noise = self.env.np_random.normal(0, self.noise_action, size=size) * action[index:index + size]
                action[index:index + size] += noise
                action[index:index + size] = np.clip(
                    action[index:index + size],
                    self.action_space.low[index:index + size],
                    self.action_space.high[index:index + size]
                )
                index += size

            elif isinstance(space, spaces.Discrete):
                # Discrete: Bernoulli flip
                start = getattr(space, 'start', 0)
                if self.env.np_random.random() < self.noise_action:
                    if space.n == 2:
                        # Binary: flip 0↔1
                        action[index] = start + (1 - int(action[index] - start))
                    else:
                        # Multi-valued: ±1 offset
                        offset = self.env.np_random.choice([-1, 1])
                        action[index] = np.clip(action[index] + offset, start, start + space.n - 1)
                index += 1

        return action

    def _remove_prev_action(self, obs):
        """Flatten Prev_Action dict into main observation."""
        obs = deepcopy(obs)
        for key, value in obs['Prev_Action'].items():
            obs[key] = value
        del obs['Prev_Action']
        return obs
    
    def _adjust_observation_space(self, original_space):
        """Remove 'Prev_Action' nesting by flattening its contents into main Dict space."""
        # Copy all non-Prev_Action spaces
        new_spaces = {
            key: value for key, value in original_space.spaces.items() if key != 'Prev_Action'
        }

        # Unpack 'Prev_Action' dict into main space, converting Discrete to Box
        for key, value in original_space['Prev_Action'].spaces.items():
            if isinstance(value, spaces.Discrete):
                new_spaces[key] = spaces.Box(low=0, high=value.n - 1, shape=(1,), dtype=np.float32)
            else:
                new_spaces[key] = value

        return spaces.Dict(new_spaces)

    def _flatten_action_space(self, original_space):
        """Flatten Dict action space into a single continuous Box space.

        CRITICAL FIX: Handle 'start' parameter for Discrete spaces correctly.
        E.g., CT_NrBays is Discrete(12, start=1) → bounds should be [1, 12], not [0, 11]
        """
        low = []
        high = []

        for key, space in original_space.spaces.items():
            if isinstance(space, spaces.Box):
                low.extend(space.low)
                high.extend(space.high)
            elif isinstance(space, spaces.Discrete):
                # FIXED: Handle 'start' parameter for discrete spaces
                start = getattr(space, 'start', 0)
                low.extend([start])
                high.extend([start + space.n - 1])

        return spaces.Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32))

    def _unflatten_action(self, action):
        """Convert flattened Box action back to original Dict format.

        CRITICAL FIX: For discrete actions, use rounding + clipping instead of truncation.
        Truncation (int()) creates severe bias toward lower values and breaks with noise:
        - int(0.92) = 0 (should be 1!)
        - With noise: 1.0 + N(0, 0.05) can become 0.95 → int() = 0 (wrong!)

        Proper handling:
        - Round to nearest integer: np.rint()
        - Clip to valid range: [start, start + n - 1]
        """
        action = np.asarray(action).flatten()
        original_action = {}
        index = 0

        for key, space in self.env.action_space.spaces.items():
            if isinstance(space, spaces.Box):
                size = np.prod(space.shape)
                original_action[key] = action[index:index + size].reshape(space.shape)
                index += size
            elif isinstance(space, spaces.Discrete):
                # FIXED: Use rounding + clipping for discrete actions
                # Handle 'start' parameter (e.g., CT_NrBays is Discrete(12, start=1))
                start = getattr(space, 'start', 0)
                val = np.rint(action[index])  # Round to nearest integer
                val = int(np.clip(val, start, start + space.n - 1))  # Clip to valid range
                original_action[key] = val
                index += 1

        return original_action

