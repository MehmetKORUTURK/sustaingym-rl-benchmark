# CLAUDE.md - RL Benchmark on SustainGym Environments

## Project Overview

**Master's thesis** by Mehmet Koruturk (Virginia Tech, RoleLab) benchmarking RL on three SustainGym environments across three axes:

1. **Perturbation Robustness** — state (PS), action (PA), dynamics (PD) noise channels
2. **Safe RL** — constrained optimization via OmniSafe (CMDP formulation)
3. **Multi-Agent RL** — single vs multi-agent via Ray RLlib + PettingZoo

**Repo**: `MehmetKORUTURK/sustaingym-rl-benchmark` | **Runtime**: Virginia Tech ARC HPC (SLURM, A100/H200 GPUs)

---

## Three Environments

### 1. EVCharging (`envs/evcharging/`)

- **Episode**: 288 timesteps (5-min, 24h day), 54 charging stations
- **Obs**: Dict{timestep, est_departures, demands, prev_moer, forecasted_moer} | **Policy**: `MultiInputPolicy`
- **Action**: Box(54) normalized [0,1] → [0,32] Amps
- **Reward**: `profit - carbon_cost - excess_charge` (USD/timestep)
- **Safe RL cost**: excess_charge delta (network violations), cost_scale=100
- **MARL**: 54 agents (per-station), shared reward / num_agents

### 2. Building (`envs/building/`)

- **Config**: `OfficeSmall`, `Hot_Dry`, `Tucson`, `reward_beta=0.5`
- **Episode**: 288 timesteps (5-min, 1 day), RC thermal model
- **Obs**: Box(n+4) [zone_temps, outdoor_temp, ground_temp, ghi, occupower] | **Policy**: `MlpPolicy`
- **Action**: Box(n) continuous [-1,1] per zone (HVAC)
- **Reward**: `-(q_rate * ||action|| + error_rate * ||temp_error||)` normalized to [-1, 0]
- **Safe RL cost**: deadband temp violation `mean(max(0, |temp_error| - 1.0) * ac_map)`, cost_scale=1.0. Orthogonal: reward penalizes all error smoothly, CMDP enforces hard 1°C boundary.
- **MARL**: 6 agents (all zones, ac_map=1 default includes ATTIC), shared reward / num_agents

### 3. Cogen (`envs/cogen/`)

- **Episode**: 96 timesteps/day, ONNX surrogate model
- **Obs**: Dict{Time, Prev_Action(15), TAMB, PAMB, RHAMB, targets, prices} | **Policy**: `MultiInputPolicy`
- **Action**: Dict 15 keys (3 GT × 4 + ST + condenser + cooling) — mixed continuous/discrete
- **Reward**: `-(fuel + ramp + non_delivery + constraint_violation)` scaled by 1e7
- **Wrapper**: `MyCogenEnv` flattens Dict→Box, handles noise, reward/1e7
- **Safe RL cost**: `sum(ramp_costs)` (turbine wear), cost_scale=0.005. Orthogonal: reward=fuel+delivery, CMDP=smooth operation.
- **MARL**: GT1, GT2, GT3, ST (4 agents), per-agent rewards

---

## Perturbation Channels (PS/PA/PD)

| Type | Param | Where Applied |
| ---- | ----- | ------------- |
| State (PS) | `--noise` | After env step, before agent sees obs |
| Action (PA) | `--noise-action` | Before env processes action |
| Dynamics (PD) | `--noise-env` | In reward/dynamics computation |

### Scales Tested

- **EVCharging**: obs/env 0.0–0.60 (10 levels), act 0.0–0.60 (9 levels)
- **Building**: obs/act/env 0.0–0.40 (9 levels)
- **Cogen**: obs/env 0–5.0 (7 levels). MyCogenEnv auto-scales per type: temp=2F, pressure=0.1psia, humidity=0.05, model=15% per unit. Model noise dominant (multiplicative on all 29 ONNX outputs).

### Building `--noise-env` Scaling

Scalar → dict in `stdrl_training.py`: out_temp=1.0°C, ground_temp=0.5°C, GHI=50 W/m² (multiplied by noise_env value)

---

## Training Defaults

| Parameter | Cogen | EVCharging | Building |
| --------- | ----- | ---------- | -------- |
| Total steps | 3M | 9M | 9M |
| Eval freq | 1M | 1M | 1M |
| Checkpoint freq | 2.5M | 2.5M | 2.5M |
| Algos (SB3) | PPO, SAC, TD3 | PPO, SAC, TD3 | PPO, SAC, TD3 |

### MARL (Ray RLlib v2.10.0)

- **EVCharging**: 10 workers, 32k iter, train_batch=2880, rollout_fragment=288
- **Building**: 4 workers, 32k iter, train_batch=1152, rollout_fragment=288, reward_beta=0.5
- **Cogen**: 4 workers, 3000 iter, train_batch=4000, rollout_fragment=200 (no SAC — mixed action spaces)
- **APPO**: lr=5e-5 hardcoded (ignores --lr). **IMPALA**: EV/BU lr=5e-5/grad_clip=5.0, CO lr=1e-4/grad_clip=40.0
- `--shared-policy` → MAPPO/MASAC (EV/Building only; Cogen agents have different action spaces)
- Batch divisibility: train_batch must be divisible by (num_workers × rollout_fragment)

### Safe RL (OmniSafe)

| Param | EVCharging | Building | Cogen |
| ----- | ---------- | -------- | ----- |
| Total steps | 6M | 6M | 6M |
| Steps/epoch | 2000 | 2000 | 2000 |
| Cost scale | 100.0 | 1.0 | 0.005 |
| Cost limits | 3, 5, 25, 1000 | 25, 50, 100, 200 | 10, 25, 50, 200 |
| Algos | PPOLag, CPO, OnCRPO, FOCOPS | same | same |
| Network | default | [128,128,128] | [128,128,128] |

- **Lagrangian** (PPOLag, FOCOPS): cost_limit via `lagrange_cfgs`
- **Trust-region** (CPO, OnCRPO): cost_limit via `algo_cfgs`
- **SACLag REMOVED**: Off-policy Lagrangian fails across all 3 envs — replay buffer staleness breaks lambda tracking. Report as negative finding.

---

## Common Gotchas

1. **Cogen ONNX lazy load**: Loaded in `reset()` not `__init__()` — ONNX sessions can't be pickled across Ray workers
2. **Cogen discrete actions**: Use `np.rint()` not `int()` — truncation causes severe bias
3. **MyCogenEnv `safe_rl=True`**: Flattens Dict obs to Box for OmniSafe. EVCharging also flattens Dict→Box with `safe_rl=True`.
4. **Building commented code**: `multiagent_env.py` and `env.py` have multiple commented-out versions; active code is at the bottom
5. **Building reproducibility**: Noise uses `self.np_random` (seeded), NOT `np.random`
6. **Building reward_beta=0.5**: Used consistently in stdrl, saferl, and MARL
7. **VecNormalize sync**: Must sync train→eval stats; `SyncVecNormalizeCallback` handles this
8. **OmniSafe double normalization**: Do NOT enable `reward_normalize`/`cost_normalize` for Building/Cogen — already normalized. Causes severe instability.
9. **OmniSafe steps_per_epoch**: Use 2000+. `steps_per_epoch=288` (1 episode) = extreme variance
10. **OmniSafe loose constraints**: CPO/OnCRPO diverge with very loose limits (>4x episode cost). PPOLag degrades gracefully (lambda→0).
11. **EVCharging obs noise**: Creates deep copies to avoid mutating internal state arrays
12. **CogenEnv env noise**: Model noise (15% per unit on all 29 ONNX outputs) dominates; ambient-only noise has negligible reward impact
13. **Building deadband cost**: Previous attempts failed: (1) raw temp_error overlapped reward, (2) HVAC ramping too weak. Deadband=1°C creates genuine orthogonality.
14. **MARL reward scaling**: EV/Building divide by num_agents. Cogen per-agent costs / 1e7.
15. **Cogen MARL obs bounds**: Uses `-inf/+inf` (ONNX outputs can exceed nominal bounds)

---

## Conda Environments on ARC

- `stdrl_train` — EVCharging + Building
- `stdrl_train_co` — Cogen (separate due to ONNX/dependency conflicts)
