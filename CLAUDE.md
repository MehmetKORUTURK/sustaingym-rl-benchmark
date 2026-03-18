# CLAUDE.md - RL Benchmark on SustainGym Environments

## Project Overview

This is a **Master's thesis research project** by Mehmet Koruturk (Virginia Tech, RoleLab) benchmarking reinforcement learning algorithms on three SustainGym environments with three experimental axes:

1. **Noise Robustness** - observation noise, action noise, environment noise
2. **Safe RL** - constrained optimization via OmniSafe (CMDP formulation)
3. **Multi-Agent RL** - single-agent vs multi-agent comparison via Ray RLlib + PettingZoo

**Repo**: `MehmetKORUTURK/sustaingym-rl-benchmark` | **Branch**: `main` | **Runtime**: Virginia Tech ARC HPC (SLURM, A100/H200 GPUs)

---

## Three Environments

### 1. EVCharging (`envs/evcharging/`)
- **Source**: ACN-Data / ACN-Sim (Caltech), GMM-based trace generation
- **Episode**: 24-hour day (288 timesteps, 5-min resolution)
- **Obs**: Dict{timestep, est_departures(n), demands(n), prev_moer(1), forecasted_moer(36)} where n=54 stations
- **Action**: Box(n) normalized pilot signals [0,1] -> scaled to [0,32] Amps
- **Reward**: `profit - carbon_cost - excess_charge` (USD per timestep)
- **Policy type**: `MultiInputPolicy` (Dict obs space)
- **Noise implementation**: obs noise on MOER/departures/demands, action noise pre-clipping, env noise on actual MOER in reward
- **Safe RL cost**: per-step `excess_charge` delta from `_reward_breakdown` (network constraint violations), scaled by `cost_scale` (default 100)
- **MARL agents**: Each charging station is an agent (54 agents), PettingZoo ParallelEnv, shared reward / num_agents

### 2. Building (`envs/building/`)
- **Source**: EnergyPlus RC thermal model, ASHRAE 90.1-2019 prototype buildings
- **Default config**: `OfficeSmall`, `Hot_Dry` weather, `Tucson` location, `reward_beta=0.5`
- **Episode**: 288 timesteps (1 day at 5-min resolution)
- **Obs**: Box(n+4) = [zone_temps(n), outdoor_temp, ground_temp, ghi, occupower]
- **Action**: Box(n) continuous [-1,1] per zone (HVAC: negative=cool, positive=heat)
- **Reward**: `-(q_rate * ||action||_p + error_rate * ||temp_error||_p)` normalized to [-1, 0]
- **Policy type**: `MlpPolicy` (flat Box obs space)
- **Noise implementation**: obs noise = proportional Gaussian (`noise * |state|`), action noise = proportional Gaussian (`noise_action * |action|`), env noise = dict with `out_temp`, `ground_temp`, `ghi` std (applied to external conditions before RC dynamics)
- **Safe RL cost**: deadband temperature violation across AC-enabled zones, computed as `mean(max(0, |temp_error| - 1.0) * ac_map)`, scaled by `cost_scale` (default 1.0). Deadband=1°C: temp errors within ±1°C are "free", only violations beyond 1°C count. Orthogonal to reward: reward penalizes ALL temp error + energy smoothly, CMDP enforces a hard comfort boundary.
- **MARL agents**: Each AC-enabled zone is an agent, PettingZoo ParallelEnv, shared reward / num_agents

### 3. Cogen (`envs/cogen/`)
- **Source**: ONNX surrogate model of a cogeneration plant
- **Episode**: 96 timesteps per day
- **Obs**: Dict{Time, Prev_Action(15 keys), TAMB, PAMB, RHAMB, Target_Power, Target_Steam, Energy_Price, Gas_Price} each with forecast_horizon+1 steps
- **Action**: Dict with 15 keys (3 gas turbines x 4 params + steam turbine + condenser + cooling bays) - mixed continuous/discrete
- **Reward**: `-(fuel_cost + ramp_cost + non_delivery_penalty + constraint_violation)` scaled by 1e7
- **Wrapper**: `MyCogenEnv` flattens Dict action/obs to Box, handles noise, reward scaling
- **Policy type**: `MultiInputPolicy` (Dict obs after MyCogenEnv removes Prev_Action nesting)
- **Noise implementation**: obs noise (ambient sensor/forecast/targets/prices with configurable ratios), action noise (multiplicative for Box, Bernoulli flip for Discrete), env noise = ambient perturbation (TAMB/PAMB/RHAMB) + ONNX model output uncertainty (multiplicative noise on all 29 model outputs, simulating sim-to-real gap)
- **Safe RL cost**: `sum(info['ramp_costs'].values())` per-step (turbine ramping stress / equipment wear), scaled by `cost_scale` (default 0.005). Orthogonal tradeoff: reward dominated by fuel+delivery, agent ignores small ramp_penalty=2 in reward, CMDP enforces smooth turbine operation.
- **MARL agents**: GT1, GT2, GT3, ST (4 agents), each controls its sub-actions, per-agent reward based on individual costs

---

## Key Files

### Training Scripts
| File | Purpose | Framework | Algorithms |
|------|---------|-----------|------------|
| `scripts/train/stdrl_training.py` | Standard single-agent training | Stable-Baselines3 | PPO, SAC, TD3 |
| `scripts/test/stdrl_testing.py` | Model evaluation/testing | Stable-Baselines3 | PPO, SAC, TD3 |
| `scripts/train/marl_training.py` | MARL training - unified | Ray RLlib | PPO, SAC, APPO, IMPALA |
| `scripts/train/marl_tr_ev.py` | MARL training - EVCharging | Ray RLlib | PPO, SAC, APPO, IMPALA |
| `scripts/train/marl_tr_bu.py` | MARL training - Building | Ray RLlib | PPO, SAC, APPO, IMPALA |
| `scripts/train/marl_tr_co.py` | MARL training - Cogen | Ray RLlib | PPO, APPO, IMPALA |
| `scripts/train/saferl_training.py` | Safe RL - all envs (unified) | OmniSafe | PPOLag, CPO, OnCRPO, FOCOPS |
| `scripts/slurm/arc_saferl_all.sh` | Safe RL run commands (48 runs) | OmniSafe | 4 algo x 3 env x 4 climit |

### Environment Files
| File | Description |
|------|-------------|
| `envs/evcharging/env.py` | EVChargingEnv (single-agent, Gymnasium) |
| `envs/evcharging/multiagent_env.py` | MultiAgentEVChargingEnv (PettingZoo ParallelEnv) |
| `envs/building/env.py` | BuildingEnv (single-agent, RC thermal model) |
| `envs/building/multiagent_env.py` | MultiAgentBuildingEnv (PettingZoo, multiple commented-out versions) |
| `envs/cogen/env.py` | CogenEnv (original SustainGym, modified with noise_env) |
| `envs/cogen/MyCogenEnv.py` | MyCogenEnv wrapper (flattens Dict spaces, noise injection, reward scaling) |
| `envs/cogen/multiagent_env.py` | MultiAgentCogenEnv (active version: SAC-compatible with continuous Box actions) |
| `envs/cogen/env_marl.py` | Alternate CogenEnv for MARL (flattened obs, different reward) |
| `envs/building/utils.py` | ParameterGenerator, HTM parser, RC model construction |
| `envs/building/stochastic_generator.py` | StochasticUncontrollableGenerator for Building env |
| `envs/evcharging/utils.py` | EV charging utility functions |
| `envs/evcharging/event_generation.py` | GMM-based EV arrival/departure trace generation |
| `envs/evcharging/discrete_action_wrapper.py` | Discrete action wrapper for EVChargingEnv |

### SLURM Scripts (Training)
| File | Env | Notes |
|------|-----|-------|
| `scripts/slurm/arc_slurm_ev.sh` | EVCharging | Conda env: `stdrl_train`, partition: `normal_q`, constraint: `intel&avx512`, 48 CPUs, 40G RAM, 3-day limit |
| `scripts/slurm/arc_slurm_bu.sh` | Building | Conda env: `stdrl_train`, partition: `normal_q`, constraint: `intel&avx512`, 48 CPUs, 40G RAM, 3-day limit |
| `scripts/slurm/arc_slurm_co.sh` | Cogen | Conda env: `stdrl_train_co`, partition: `normal_q`, constraint: `intel&avx512`, 48 CPUs, 40G RAM, 3-day limit |

### SLURM Scripts (Testing) - to be created on ARC
| File | Env | Notes |
|------|-----|-------|
| `scripts/slurm/arc_slurm_test_ev.sh` | EVCharging | Part A: noise-trained at matching noise, Part B: baseline robustness |
| `scripts/slurm/arc_slurm_test_bu.sh` | Building | Part A: noise-trained at matching noise, Part B: baseline robustness |
| `scripts/slurm/arc_slurm_test_co.sh` | Cogen | Part A: noise-trained at matching noise, Part B: baseline robustness |

### Plot Scripts
| File | Env/Purpose | Output Dir |
|------|-------------|------------|
| `scripts/plot/stdrl_plot_ev.py` | EVCharging training curves (DS/DA/DE) | `./graphs/C_STDRL/` |
| `scripts/plot/stdrl_plot_bu.py` | Building training curves (DS/DA/DE) | `./graphs/C_STDRL/` |
| `scripts/plot/stdrl_plot_co.py` | Cogen training curves (DS/DA/DE) | `./graphs/C_STDRL/` |
| `scripts/plot/stdrl_post_plot.py` | Post-training bar/line/heatmap charts | `./graphs/C_POST/` |
| `scripts/plot/omni_plot.py` | OmniSafe Safe RL dual-panel (reward+cost) | `./graphs/C_SRL/{env}/` |
| `scripts/plot/marl_plot.py` | MARL training curves | `./graphs/C_MARL/` |

**Plot usage**:
```bash
python scripts/plot/stdrl_plot_bu.py --algo PPO --dt DE --auto_ylim
python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot all
python scripts/plot/omni_plot.py --env building --t_steps 20000 --w_size 500 --climit 5
python scripts/plot/marl_plot.py --env evcharging --t_steps 32000 --w_size 500 --auto_ylim
```

---

## CLI Usage

### Standard RL Training
```bash
# Basic training
python scripts/train/stdrl_training.py --env evcharging --algo SAC --seed 42

# With observation noise
python scripts/train/stdrl_training.py --env evcharging --algo SAC --noise 0.1 --seed 42

# With action noise
python scripts/train/stdrl_training.py --env building --algo PPO --noise-action 0.05 --seed 42

# With VecNormalize (recommended)
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.1 --seed 42

# Quick test (10k steps)
python scripts/train/stdrl_training.py --env cogen --algo PPO --quick-test
```

**Key args**: `--env {cogen,evcharging,building}`, `--algo {PPO,SAC,TD3}`, `--noise FLOAT`, `--noise-action FLOAT`, `--noise-env FLOAT`, `--use-vecnormalize`, `--norm-reward`, `--seed INT`, `--quick-test`, `--no-eval`, `--rm INT` (cogen), `--eval-freq INT`, `--n-eval-episodes INT`, `--checkpoint-freq INT`

### MARL Training (Ray RLlib)
```bash
# Unified script (like saferl_training.py)
python scripts/train/marl_training.py --env evcharging --algo APPO --seed 42
python scripts/train/marl_training.py --env building --algo SAC --num-iterations 32000
python scripts/train/marl_training.py --env cogen --algo IMPALA --rm 300
python scripts/train/marl_training.py --env evcharging --algo PPO --shared-policy  # MAPPO

# Individual scripts
python scripts/train/marl_tr_ev.py --algo APPO --seed 42 --num-iterations 32000
python scripts/train/marl_tr_bu.py --algo SAC --shared-policy  # MASAC
python scripts/train/marl_tr_co.py --algo PPO --rm 300 --num-iterations 750
```

**Key args**: `--env {cogen,evcharging,building}`, `--algo`, `--seed INT`, `--num-iterations INT`, `--num-workers INT`, `--checkpoint-freq INT`, `--num-gpus INT`, `--lr FLOAT`, `--rm INT` (cogen only), `--shared-policy` (EV/Building only)

**Policy modes**: `--shared-policy` uses a single shared network for all agents (MAPPO/MASAC pattern). Without it, each agent gets its own independent policy. Cogen does not support shared policy (agents have different action spaces).

### Safe RL Training (OmniSafe)
```bash
# Unified script for all environments (4 algos x 3 envs x 4 climit = 48 runs)
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 1 --seed 42
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 50 --seed 42
python scripts/train/saferl_training.py --env cogen --algo OnCRPO --climit 25 --seed 42

# With noise
python scripts/train/saferl_training.py --env evcharging --algo FOCOPS --noise 0.1 --climit 1
python scripts/train/saferl_training.py --env evcharging --algo CPO --noise_act 0.1 --noise_env 0.05 --climit 1
```

**Key args**: `--env {cogen,evcharging,building}`, `--algo {PPOLag,CPO,OnCRPO,FOCOPS}` (SACLag removed — see negative finding), `--noise FLOAT`, `--noise_act FLOAT`, `--noise_env FLOAT`, `--climit FLOAT`, `--tsteps INT`, `--seed INT`, `--cost_scale FLOAT`, `--rm INT` (cogen only)

### Model Testing
```bash
# Basic testing (100 episodes default)
python scripts/test/stdrl_testing.py --env evcharging --model_path path/to/model.zip --algo SAC --n-eval 100

# With noise injection (test robustness of baseline model)
python scripts/test/stdrl_testing.py --env evcharging --model_path path/to/model.zip --algo SAC --noise 0.1

# With VecNormalize (auto-detected from model directory, or explicit path)
python scripts/test/stdrl_testing.py --env building --model_path path/to/model.zip --algo SAC --vec-normalize-path path/to/vec_normalize.pkl

# With environment noise
python scripts/test/stdrl_testing.py --env cogen --model_path path/to/model.zip --algo PPO --noise-env 2.0

# Collect per-step trajectories
python scripts/test/stdrl_testing.py --env evcharging --model_path path/to/model.zip --algo SAC --collect-trajectories
```

**Key args**: `--env`, `--model_path`, `--algo`, `--noise FLOAT`, `--noise-action FLOAT`, `--noise-env FLOAT`, `--n-eval INT`, `--vec-normalize-path PATH`, `--reward-beta FLOAT` (building), `--rm INT` (cogen), `--collect-trajectories`, `--stochastic`, `--seed INT`

---

## Noise System Architecture

Each environment supports three independent noise channels:

| Noise Type | Parameter | Where Applied | Effect |
|-----------|-----------|---------------|--------|
| **Observation** | `--noise` | After env step, before agent sees obs | Simulates sensor uncertainty |
| **Action** | `--noise-action` | Before env processes action | Simulates actuator imprecision |
| **Environment** | `--noise-env` | In reward/dynamics computation | Simulates real-world stochasticity / sim-to-real gap |

### Noise Scales Tested

- **EVCharging obs**: 0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60
- **EVCharging act**: 0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60
- **EVCharging env**: 0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60
- **Building obs**: 0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40
- **Building act**: 0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40
- **Building env**: 0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40
- **Cogen obs**: 0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 (MyCogenEnv scales internally: noise * base_ratio)
- **Cogen env**: 0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0 (scales: temp=2F, pressure=0.1psia, humidity=0.05, model=15% per unit)

### Cogen Environment Noise Scaling (`--noise-env` per unit)
| Component | Scale | noise_env=1.0 | noise_env=3.0 | What it affects |
|-----------|-------|---------------|---------------|-----------------|
| Temperature | 2.0 F | 2 F std | 6 F std | ONNX model input (TAMB) |
| Pressure | 0.1 psia | 0.1 psia std | 0.3 psia std | ONNX model input (PAMB) |
| Humidity | 0.05 frac | 0.05 std | 0.15 std | ONNX model input (RHAMB) |
| Model | 15% | 15% output noise | 45% output noise | All ONNX outputs (fuel, power, steam, limits) |

Model noise is the dominant component: multiplicative perturbation on all 29 ONNX model outputs, directly affecting fuel costs, power/steam delivery, and constraint boundaries.

### Building Environment Noise Scaling (`--noise-env` per unit)
Scalar `noise_env` converted to dict in `scripts/train/stdrl_training.py`:
| Component | Scale | noise_env=0.1 | noise_env=0.3 | What it affects |
|-----------|-------|---------------|---------------|-----------------|
| Out temp | 1.0 C | 0.1 C std | 0.3 C std | Outdoor temperature |
| Ground temp | 0.5 C | 0.05 C std | 0.15 C std | Ground temperature |
| GHI | 50.0 W/m² | 5.0 W/m² std | 15.0 W/m² std | Solar irradiance |

---

## Training Defaults

| Parameter | Cogen | EVCharging | Building |
|-----------|-------|------------|----------|
| Total steps | 3,000,000 | 9,000,000 | 9,000,000 |
| Eval freq | 1,000,000 | 1,000,000 | 1,000,000 |
| Checkpoint freq | 2,500,000 | 2,500,000 | 2,500,000 |
| Device | CPU | CPU | CPU |
| Seed | 42 | 42 | 42 |

### Algorithm Hyperparameters (SB3)
- **PPO**: lr=3-5e-5, n_steps=1024-2048, batch=128-256, gamma=0.99-0.995, ent_coef=0.001-0.005
- **SAC**: lr=3e-4, buffer=1M, learning_starts=10k, batch=256, gamma=0.99, ent_coef=auto
- **TD3**: lr=3e-4, buffer=1M, learning_starts=10k, batch=256, policy_delay=2

### MARL Hyperparameters (Ray RLlib)
- **EVCharging**: PPO/SAC/APPO/IMPALA (default APPO), train_batch=2880, rollout_fragment=288, 10 workers, 32k iterations, entropy_coeff=0.005, sgd_minibatch=1024 (PPO)
- **Building**: PPO/SAC/APPO/IMPALA (default SAC), train_batch=1152 (PPO/APPO/IMPALA), rollout_fragment=288, 4 workers, 32k iterations, entropy_coeff=0.01, sgd_minibatch=128 (PPO), reward_beta=0.5
- **Cogen**: PPO/APPO/IMPALA (default PPO), train_batch=4000, rollout_fragment=200, 4 workers, 3000 iterations, entropy_coeff=0.01, sgd_minibatch=128 (PPO)
- **PPO**: lr=3e-4 (from --lr), grad_clip=0.5, num_sgd_iter=10, gamma=0.99, lambda=0.95, clip=0.2
- **SAC**: lr=3e-4 (from --lr), train_batch=256, n_step=1, grad_clip=1.0, rollout_fragment="auto", model=[64,64]
- **APPO**: lr=1e-4 (hardcoded), grad_clip=40.0, num_sgd_iter=10, gamma=0.99, lambda=0.95, clip=0.2
- **IMPALA**: V-trace based, vtrace_clip_rho=1.0, vtrace_clip_pg_rho=1.0. EVCharging/Building: lr=5e-5, grad_clip=5.0 (reduced for V-trace stability with many agents). Cogen: lr=1e-4, grad_clip=40.0 (stable with 4 agents)
- **Ray RLlib**: v2.10.0, uses `ImpalaConfig` (not `IMPALAConfig`), TD3 removed (deprecated)
- **Parameter sharing**: `--shared-policy` maps all agents to one network (EV/Building only; Cogen agents have different action spaces)
- **Reward scaling**: Building divides reward by num_agents, EVCharging divides reward by num_agents, Cogen per-agent costs scaled by 1/1e7 (matching MyCogenEnv single-agent)
- **Batch divisibility**: train_batch must be divisible by (num_workers × rollout_fragment). EV: 2880/(10×288)=1, BU: 1152/(4×288)=1, CO: 4000/(4×200)=5
- **LR logging**: APPO/IMPALA hardcode lr (ignoring --lr flag). `actual_lr` computed and saved to config.json + console summary

### Safe RL Hyperparameters (OmniSafe)

| Parameter | EVCharging | Building | Cogen |
|-----------|-----------|----------|-------|
| Total steps | 6,000,000 | 6,000,000 | 6,000,000 |
| Steps per epoch | 2000 | 2000 | 2000 |
| Cost scale | 100.0 | 1.0 | 0.005 |
| Cost limits tested | 1, 5, 25, 1000 | 50, 100, 200, 500 | 10, 25, 50, 200 |
| Algorithms | PPOLag, CPO, OnCRPO, FOCOPS | same | same |
| Network (Building/Cogen) | default | [128,128,128] ReLU | [128,128,128] ReLU |
| Batch size (on-policy, Building/Cogen) | default | 1024 | 1024 |

**Safe RL config details**:
- **Lagrangian algos** (PPOLag, FOCOPS): cost_limit via `lagrange_cfgs`
- **Non-Lagrangian algos** (CPO, OnCRPO): cost_limit via `algo_cfgs`
- **SACLag REMOVED**: Off-policy Lagrangian fails consistently across all 3 environments — replay buffer staleness prevents lambda from correctly tracking policy cost. Tested with update_cycle=1 and 100, neither worked. This is a known limitation of off-policy CMDP methods and should be reported as a negative finding in the thesis: "Off-policy Lagrangian methods (SACLag) fail to satisfy cost constraints due to fundamental incompatibility between replay buffer data distribution and Lagrange multiplier updates."
- **Building/Cogen**: No `reward_normalize` or `cost_normalize` (BuildingEnv already normalizes reward to [-1,0], Cogen reward already scaled by 1e7)
- **Building Safe RL**: Uses `reward_beta=0.5` (matching stdrl), CMDP cost = deadband temperature violation `mean(max(0, |temp_error| - 1.0) * ac_map)` with 1°C deadband. Orthogonal to reward: reward penalizes ALL temp error + energy smoothly, CMDP enforces hard comfort boundary (no zone >1°C from target). Limits TBD after calibration (placeholder: 50/100/200/500).
- **Cogen Safe RL**: CMDP cost = `sum(ramp_costs)` (turbine ramping stress/equipment wear, orthogonal: reward dominated by fuel+delivery, ramp_penalty=2 too small for agent to care, CMDP enforces smooth operation)

---

## Output Structure

```
logs_std_train/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
  ├── best_model.zip                       # Best model from EvalCallback
  ├── {env}_{algo}_final.zip               # Final model after training
  ├── {env}_{algo}_checkpoint_*_steps.zip  # Periodic checkpoints
  ├── eval/                                # Evaluation logs (evaluations.npz)
  ├── vec_normalize.pkl                    # VecNormalize stats (if --use-vecnormalize)
  └── events.out.*                         # TensorBoard logs

logs_std_test/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
  ├── test_config.json         # Full reproducibility config (JSON)
  ├── episode_results.csv      # Per-episode metrics
  ├── evaluation_summary.txt   # Paper-ready statistics (mean, std, CI95, quantiles)
  ├── evaluation_results.txt   # Legacy format (mean +/- std)
  └── trajectory_data.csv      # Per-step metrics (if --collect-trajectories)

logs_marl_train/{env}_{algo}/{timestamp}_NOISE_0_ACT_0_ENV_0/
  ├── config.json         # Full training config (JSON)
  ├── metrics.csv         # Per-iteration metrics (iteration, mean_reward, mean_length, loss, elapsed_seconds)
  └── checkpoints/        # Periodic RLlib checkpoints

logs_saferl_train/omnisafe_{env}/{timestamp}_{algo}_CL_{climit}/
  └── ...                      # OmniSafe logger output (Safe RL)
```

---

## Important Implementation Notes

### MyCogenEnv Wrapper (Cogen-specific)
- Removes `Prev_Action` nesting from obs (flattens into top-level keys)
- Flattens Dict action space to single continuous Box
- Handles Discrete action encoding: `Discrete(2)` -> `[0,1]`, `Discrete(12, start=1)` -> `[1,12]`
- Uses `np.rint()` + `np.clip()` for discrete action decoding (NOT truncation)
- Reward scaled by `1 / reward_scale` (default 1e7) to stabilize training
- `safe_rl=True` further flattens Dict obs to Box for OmniSafe compatibility
- Scalar `noise_env` auto-scales per type: `temp=2F, pressure=0.1psia, humidity=0.05, model=15%` per unit

### EVCharging Safe RL
- `safe_rl=True` flattens Dict obs to Box (concatenates all values)
- Cost signal: per-step `excess_charge` delta from `_reward_breakdown`, scaled by `cost_scale` (default 100)
- OmniSafe wraps as CMDP with `max_episode_steps = 288`

### Building Env Specifics
- RC thermal model with nonlinear occupancy (8-coefficient polynomial from EnergyPlus)
- `reward_beta` controls tradeoff: comfort (temperature error) vs energy (action magnitude)
- `reward_beta=0.5` used consistently in both stdrl and saferl training
- `normalize_reward=True` clips reward to [-1, 0] for stable training
- Zones without AC (`ac_map=0`) have fixed 0 action
- Noise injection uses seeded `self.np_random` (not global `np.random`) for reproducibility

### Multi-Agent Environments
- All use PettingZoo `ParallelEnv` interface
- All share global reward divided by number of agents
- EVCharging MARL uses `project_action_in_env=True` for network constraint satisfaction
- Cogen MARL: Dict action spaces (mixed continuous/discrete), only PPO/APPO/IMPALA supported (no SAC)
- Cogen MARL obs space uses `-inf/+inf` bounds (ONNX outputs can exceed nominal bounds)
- Cogen MARL reward scaled by `/ 1e7` to match MyCogenEnv single-agent wrapper
- Building MARL: `reward_beta=0.5` in ParameterGenerator (matching stdrl), reward divided by num_agents
- Building multiagent_env.py: each AC-enabled zone is an agent, action mapped by agent ID (not enumerate index)

---

## Dependencies

- `stable-baselines3` - Standard RL (PPO, SAC, TD3)
- `ray[rllib]` - Multi-agent RL
- `pettingzoo` - Multi-agent environment API
- `omnisafe` - Safe RL (CMDP algorithms like OnCRPO)
- `gymnasium` - Environment API
- `sustaingym` - Base environment implementations (installed as package)
- `acnportal` - EV charging simulation
- `cvxpy` + `mosek` - Action projection optimization
- `onnxruntime` - Cogen plant surrogate model
- `pvlib` - Solar/weather data processing
- `scipy`, `sklearn` - Interpolation, linear models

### Conda Environments on ARC
- `stdrl_train` - For EVCharging and Building training
- `stdrl_train_co` - For Cogen training (separate due to ONNX/dependency conflicts)

---

## Common Gotchas

1. **Cogen ONNX model**: Loaded lazily in `reset()` (not `__init__`) because ONNX sessions can't be pickled across Ray workers
2. **VecNormalize sync**: Must sync train->eval env stats before evaluation; `SyncVecNormalizeCallback` handles this
3. **Cogen discrete actions**: Use `np.rint()` not `int()` for decoding - truncation causes severe bias
4. **Building commented code**: `multiagent_env.py` and `env.py` have multiple commented-out versions; the active code is at the bottom of each file
5. **SLURM conda**: Scripts auto-detect conda base and source `conda.sh` before activating environments
6. **CogenEnv env noise**: Passed through to base env via `env.noise_env = self.noise_env`; includes ambient perturbation (TAMB/PAMB/RHAMB) AND model output noise (multiplicative on all 29 ONNX outputs). Model noise is the dominant component — ambient-only noise has negligible reward impact due to ONNX model insensitivity
7. **EVCharging obs noise**: Creates deep copies to avoid mutating internal state arrays
8. **Building reproducibility**: `envs/building/env.py` noise injection uses `self.np_random` (seeded RNG), NOT global `np.random` — fixed for reproducibility across seeds
9. **OmniSafe double normalization**: Do NOT enable `reward_normalize` or `cost_normalize` for Building/Cogen — BuildingEnv already normalizes reward to [-1,0] and Cogen reward is already scaled by 1e7. Double normalization causes severe training instability
10. **OmniSafe steps_per_epoch**: Must be large enough to collect meaningful gradient estimates. `steps_per_epoch=288` (1 episode) causes extreme variance; use 2000+ for all envs
11. **OmniSafe loose constraint divergence**: Trust-region CMDP algorithms (CPO, OnCRPO) can diverge with very loose cost limits (e.g., climit=1000). When constraint is nearly inactive, trust-region update becomes unstable. PPOLag handles this gracefully (lambda→0 = unconstrained PPO). Avoid cost limits >4x initial episode cost for trust-region methods.
12. **scripts/plot/omni_plot.py filenames**: PNG filenames include unique algo names extracted from experiment labels (e.g., `PPOLag_CPO` not `PPOLag_1_PPOLag_5`)
13. **Building deadband temp cost**: Building Safe RL uses deadband=1°C: `cost = mean(max(0, |temp_error| - 1.0) * ac_map) * cost_scale`. Previous attempts: (1) raw temp_error had partial overlap with reward's comfort term → no PPOLag limit differentiation; (2) HVAC ramping was too weak a signal (cost collapsed to ~7 regardless of limit). Deadband approach creates genuine orthogonality: reward penalizes all deviation smoothly, cost enforces hard boundary.
