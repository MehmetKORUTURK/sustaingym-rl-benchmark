"""
The module implements the BuildingEnv class.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from sklearn import linear_model
from ttp.colored import color_text, Colors


class BuildingEnv(gym.Env):
    """BuildingEnv class (single-agent).

    RC fizik modeli + nonlinear occupancy kazancı ile çok bölgeli bina simülasyonu.
    Gymnasium v0.28–0.29 ile uyumludur.
    """

    # Occupancy nonlinear coefficients (EnergyPlus EngRef v23.1 p.1299)
    OCCU_COEF = [
        6.461927,
        0.946892,
        0.0000255737,
        0.0627909,
        0.0000589172,
        0.19855,
        0.000940018,
        0.00000149532,
    ]
    OCCU_COEF_LINEAR = 7.139322  # not used directly, kept for reference

    DISCRETE_LENGTH = 100
    SCALING_FACTOR = 24

    state: np.ndarray

    def __init__(
        self,
        parameters: dict[str, Any],
        noise: float | None = None,
        noise_action: float | None = None,
        noise_env: dict | None = None,
        *,
        verbose: bool = False,
    ):
        """
        parameters:
          n, zones, target, out_temp, ground_temp, ghi, metabolism,
          reward_beta, reward_pnorm, ac_map, max_power, temp_range,
          is_continuous_action, time_resolution (sec), episode_len, A, B, D
          + optional:
            comfort_deadband: float (°C), default 0.0
            normalize_reward: bool, default True

        noise_env: dict with keys {"out_temp", "ground_temp", "ghi"} (std devs).
            Perturbs external conditions before RC dynamics and reward computation.
            out_temp/ground_temp in °C, ghi in W/m².
        """
        super().__init__()

        self.parameters = parameters
        self.verbose = verbose

        self.noise = noise
        self.noise_action = noise_action

        _default_env_noise = {"out_temp": 0.0, "ground_temp": 0.0, "ghi": 0.0}
        if noise_env is None:
            self.noise_env = None
        elif isinstance(noise_env, dict):
            self.noise_env = {**_default_env_noise, **noise_env}
        else:
            self.noise_env = {
                "out_temp": float(noise_env),
                "ground_temp": float(noise_env),
                "ghi": float(noise_env),
            }

        self.n = int(parameters["n"])
        self.zones = parameters["zones"]
        self.target = np.asarray(parameters["target"], dtype=float)
        self.out_temp = np.asarray(parameters["out_temp"], dtype=float)
        self.ground_temp = np.asarray(parameters["ground_temp"], dtype=float)
        self.ghi = np.asarray(parameters["ghi"], dtype=float)
        self.metabolism = np.asarray(parameters["metabolism"], dtype=float)
        self.ac_map = np.asarray(parameters["ac_map"], dtype=bool)
        self.maxpower = float(parameters["max_power"])
        self.temp_range = tuple(parameters["temp_range"])
        self.reward_pnorm = float(parameters["reward_pnorm"])
        self.is_continuous_action = bool(parameters["is_continuous_action"])
        self.timestep = int(parameters["time_resolution"])  # seconds
        self.episode_len = int(parameters["episode_len"])
        self.Occupower = 0.0  # W
        self.datadriven = False
        self.length_of_weather = len(self.out_temp)

        # === new reward controls ===
        self.comfort_deadband = float(parameters.get("comfort_deadband", 0.0))
        self.normalize_reward = bool(parameters.get("normalize_reward", True))

        # Action space
        self.Qlow = -self.ac_map.astype(np.float32)  # [-1 or 0]
        self.Qhigh = self.ac_map.astype(np.float32)  # [ 1 or 0]

        if self.is_continuous_action:
            self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        else:
            nvec = np.where(self.ac_map, 2 * self.DISCRETE_LENGTH, 1).astype(np.int64)
            self.action_space = gym.spaces.MultiDiscrete(nvec)

        # Observation space (T [°C], GHI [W/m^2], Occupower [W])
        min_T, max_T = self.temp_range
        self.heat_max = 1000.0  # cap for GHI/Occupower

        self.low = np.concatenate(
            [
                np.ones(self.n) * min_T,  # zone temps
                [min_T],                  # outdoor
                [min_T],                  # ground
                [0.0],                    # GHI
                [0.0],                    # Occupower
            ]
        ).astype(np.float32)

        self.high = np.concatenate(
            [
                np.ones(self.n) * max_T,
                [max_T],
                [max_T],
                [self.heat_max],
                [self.heat_max],
            ]
        ).astype(np.float32)

        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)

        # Reward weights
        self.q_rate = (1 - float(parameters["reward_beta"])) * self.SCALING_FACTOR
        self.error_rate = float(parameters["reward_beta"])

        # Tracking
        self._reward_breakdown = {"comfort_level": 0.0, "power_consumption": 0.0}
        self.rewardsum = 0.0
        self.statelist: list[np.ndarray] = []
        self.actionlist: list[np.ndarray] = []
        self.epoch = 0
        self.num_epoch_runs = 0

        self.X_new = self.target.copy()

        # Discretize linear dynamics (ZOH)
        A = np.asarray(parameters["A"], dtype=float)
        B = np.asarray(parameters["B"], dtype=float)
        D = np.asarray(parameters["D"], dtype=float)
        BD = np.hstack((D[:, np.newaxis], B))
        self.A_d = expm(A * self.timestep)
        self.BD_d = LA.inv(A) @ (self.A_d - np.eye(self.A_d.shape[0])) @ BD

    # ---- helpers ----
    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(
            np.float32
        )

    # ---------- Core Gym API ----------
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Discrete -> continuous mapping
        if not self.is_continuous_action:
            scaled = np.zeros_like(self.Qlow, dtype=np.float32)
            L = float(self.DISCRETE_LENGTH)
            for i in range(self.n):
                if self.ac_map[i]:
                    idx = float(action[i])  # 0..2L-1
                    scaled[i] = (idx / L) - 1.0  # -> [-1,1]
                else:
                    scaled[i] = 0.0
            action = scaled

        # Optional action noise
        if self.noise_action is not None:
            if self.verbose:
                print(color_text(f"noise_action level: {self.noise_action}", Colors.GREEN))
            noise = self.np_random.normal(0, self.noise_action, size=action.shape) * action
            action = action + noise

        # clip action into valid range
        action = np.clip(action, self.Qlow, self.Qhigh)

        self.statelist.append(self.state)
        terminated = False  # time-limit only

        # External conditions (with optional environmental noise)
        ot = float(self.out_temp[self.epoch])
        gt = float(self.ground_temp[self.epoch])
        gh = float(self.ghi[self.epoch])
        if self.noise_env is not None:
            if self.noise_env["out_temp"] > 0:
                ot += self.np_random.normal(0, self.noise_env["out_temp"])
            if self.noise_env["ground_temp"] > 0:
                gt += self.np_random.normal(0, self.noise_env["ground_temp"])
            if self.noise_env["ghi"] > 0:
                gh = max(0.0, gh + self.np_random.normal(0, self.noise_env["ghi"]))

        # Inputs
        X = self.state[: self.n].T
        Y = np.insert(np.append(action, gh), 0, ot).T
        Y = np.insert(Y, 0, gt).T

        avg_temp = float(np.mean(self.state[: self.n]))
        meta = float(self.metabolism[self.epoch])

        # Occupancy sensible heat (W) and clip
        occ_w = self._calc_occupower(avg_temp, meta)
        self.Occupower = float(np.clip(occ_w, 0.0, self.heat_max))

        if self.datadriven:
            Y = np.insert(Y, 0, meta).T
            Y = np.insert(Y, 0, meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T
        else:
            Y = np.insert(Y, 0, self.Occupower).T

        # Next temps
        X_new = self.A_d @ X + self.BD_d @ Y

        # Info cost (USD)
        total_cost, cost_breakdown = self.calculate_total_cost(
            action, X_new, self.target, self.ac_map, self.maxpower
        )
        self.total_cost = total_cost

        # ---------- Reward (normalized) ----------
        p = self.reward_pnorm
        raw_err = (X_new - self.target) * self.ac_map

        # deadband
        if self.comfort_deadband > 0.0:
            e_pen = np.maximum(np.abs(raw_err) - self.comfort_deadband, 0.0)
        else:
            e_pen = np.abs(raw_err)

        M_ac = max(1, int(self.ac_map.sum()))
        act_term = LA.norm(action, p) / (M_ac ** (1.0 / p))
        err_term = LA.norm(e_pen, p) / (M_ac ** (1.0 / p))

        reward = -(self.q_rate * act_term + self.error_rate * err_term)

        # scale roughly into [-1, 0]
        if self.normalize_reward:
            worst_act = 1.0
            worst_err = 2.0 if self.comfort_deadband == 0 else max(0.5, 2.0 - self.comfort_deadband)
            denom = self.q_rate * worst_act + self.error_rate * worst_err
            if denom > 0:
                reward = np.clip(reward / denom, -1.0, 0.0)

        self.rewardsum += reward
        self._reward_breakdown["comfort_level"] -= err_term * self.error_rate
        self._reward_breakdown["power_consumption"] -= act_term * self.q_rate

        # Observation (occupower W) - use perturbed external conditions
        self.X_new = X_new
        obs = np.concatenate(
            [
                X_new,
                [ot, gt, gh, self.Occupower],
            ]
        )
        self.state = self._clip_obs(obs)

        # log physical power (W)
        self.actionlist.append(action * self.maxpower)

        # time advance
        self.epoch += 1
        self.num_epoch_runs += 1
        if self.epoch >= self.length_of_weather:
            self.epoch = 0
        truncated = self.num_epoch_runs >= self.episode_len

        # Optional state noise + clip
        if self.noise is not None and isinstance(self.state, np.ndarray):
            if self.verbose:
                print(color_text(f"noise level: {self.noise}", Colors.GREEN))
            s_noise = self.np_random.normal(0, self.noise * np.abs(self.state), size=self.state.shape)
            self.state = self._clip_obs(self.state + s_noise)

        info = self._get_info()
        info["cost_usd"] = float(total_cost)
        info["cost_breakdown"] = {k: float(v) for k, v in cost_breakdown.items()}

        return self.state, float(reward), terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if seed is None:
            self.epoch = self.np_random.integers(low=0, high=self.length_of_weather - 1)
        else:
            num_days_normalizer = ((self.episode_len * self.timestep) // 86_400) * 365
            self.epoch = int((seed / max(1, num_days_normalizer)) * self.length_of_weather)
            self.epoch = min(self.epoch, self.length_of_weather - 1)

        self.num_epoch_runs = 0
        self.statelist = []
        self.actionlist = []

        T_initial = self.target if options is None else options.get("T_initial", self.target)
        T_initial = np.asarray(T_initial, dtype=float)

        avg_temp = float(np.mean(T_initial))
        meta = float(self.metabolism[self.epoch])
        occ_w = self._calc_occupower(avg_temp, meta)
        self.Occupower = float(np.clip(occ_w, 0.0, self.heat_max))

        # External conditions (with optional environmental noise for initial obs)
        ot = float(self.out_temp[self.epoch])
        gt = float(self.ground_temp[self.epoch])
        gh = float(self.ghi[self.epoch])
        if self.noise_env is not None:
            if self.noise_env["out_temp"] > 0:
                ot += self.np_random.normal(0, self.noise_env["out_temp"])
            if self.noise_env["ground_temp"] > 0:
                gt += self.np_random.normal(0, self.noise_env["ground_temp"])
            if self.noise_env["ghi"] > 0:
                gh = max(0.0, gh + self.np_random.normal(0, self.noise_env["ghi"]))

        self.X_new = T_initial
        obs = np.concatenate(
            [
                T_initial,
                [ot, gt, gh, self.Occupower],
            ]
        )
        self.state = self._clip_obs(obs)

        self.rewardsum = 0.0
        for k in self._reward_breakdown:
            self._reward_breakdown[k] = 0.0

        return self.state, self._get_info()

    # ---------- Info helpers ----------
    def _get_info(self, all: bool = False) -> dict[str, Any]:
        if all:
            return {
                "zone_temperature": self.X_new,
                "out_temperature": self.out_temp[self.epoch].reshape(-1,),
                "ghi": self.ghi[self.epoch].reshape(-1,),
                "ground_temperature": self.ground_temp[self.epoch].reshape(-1,),
                "reward_breakdown": self._reward_breakdown,
            }
        else:
            return {"zone_temperature": self.X_new, "reward_breakdown": self._reward_breakdown}

    def _calc_occupower(self, temp: float, meta: float) -> float:
        """Sensible heat gain from occupants (W)."""
        c = self.OCCU_COEF
        heat = (
            c[0]
            + c[1] * meta
            + c[2] * meta**2
            - c[3] * temp * meta
            + c[4] * temp * meta**2
            - c[5] * temp**2
            + c[6] * temp**2 * meta
            - c[7] * temp**2 * meta**2
        )
        return float(np.clip(heat, 0.0, self.heat_max))

    # ---------- Data-driven fit ----------
    def train(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Fit linear residual model from data (data-driven mode)."""
        current_state, next_state = [], []
        for i in range(len(states) - 1):
            X = states[i]
            Y = np.insert(np.append(actions[i] / self.maxpower, self.ghi[i]), 0, self.out_temp[i]).T
            Y = np.insert(Y, 0, self.ground_temp[i]).T

            avg_temp = float(np.mean(X))
            meta = float(self.metabolism[i])
            _ = self._calc_occupower(avg_temp, meta)

            Y = np.insert(Y, 0, meta).T
            Y = np.insert(Y, 0, meta**2).T
            Y = np.insert(Y, 0, avg_temp).T
            Y = np.insert(Y, 0, avg_temp**2).T

            current_state.append(np.concatenate((X, Y), axis=0))
            next_state.append(states[i + 1])

        model = linear_model.LinearRegression(fit_intercept=False, positive=True)
        beta = model.fit(np.array(current_state), np.array(next_state)).coef_
        self.A_d = beta[:, : self.n]
        self.BD_d = beta[:, self.n :]
        self.datadriven = True

    # ---------- Cost model (info only) ----------
    def calculate_total_cost(self, action, X_new, target, ac_map, maxpower):
        """USD cost = energy (kWh) * price + comfort violation charge."""
        PRICE_PER_KWH = 0.15
        TEMP_VIOLATION_THRESHOLD = 2.0  # degC
        VIOLATION_WEIGHT = 0.05  # $ per degree

        # comfort
        error = (X_new - target) * ac_map
        excess_temp = np.maximum(np.abs(error) - TEMP_VIOLATION_THRESHOLD, 0.0)
        excess_charge = float(np.sum(excess_temp) * VIOLATION_WEIGHT)

        # energy
        hours = self.timestep / 3600.0
        power_kW = float(np.sum(np.abs(action)) * maxpower / 1000.0)
        energy_kWh = power_kW * hours
        power_cost = float(energy_kWh * PRICE_PER_KWH)

        total_cost = power_cost + excess_charge
        return total_cost, {"power_consumption": power_cost, "comfort_level": excess_charge}
