# SustainGym RL Benchmark

[![Project Page](https://img.shields.io/badge/Project-Page-861f41?style=flat-square&logo=githubpages)](https://mehmetkoruturk.github.io/sustaingym-rl-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

🌐 **Project page:** [mehmetkoruturk.github.io/sustaingym-rl-benchmark](https://mehmetkoruturk.github.io/sustaingym-rl-benchmark/)

A comprehensive reinforcement learning benchmark on [SustainGym](https://github.com/chrisyeh96/sustaingym) environments, evaluating RL algorithms across three experimental axes: **perturbation robustness**, **safe RL**, and **multi-agent RL**.

> Virginia Tech — [RoleLab](https://rolelab.org/)

## Environments

| Environment | Domain | Episode | Agents (MARL) | Obs Space | Action Space |
|-------------|--------|---------|---------------|-----------|--------------|
| **EVCharging** | EV charging scheduling | 288 steps (24h, 5-min) | 54 (per station) | Dict (MOER, demands, departures) | Box(54) pilot signals |
| **Building** | HVAC control | 288 steps (24h, 5-min) | Per AC-zone | Box(n+4) zone temps + weather | Box(n) heat/cool [-1,1] |
| **Cogen** | Cogeneration plant | 96 steps (24h, 15-min) | 4 (GT1, GT2, GT3, ST) | Dict (ambient, targets, prices) | Dict (15 keys, mixed) |

## Experimental Axes

### 1. Noise Robustness
Three independent noise channels per environment:

| Channel | Where Applied | Simulates |
|---------|---------------|-----------|
| **Observation** (`--noise`) | After env step, before agent | Sensor uncertainty |
| **Action** (`--noise-action`) | Before env processes action | Actuator imprecision |
| **Environment** (`--noise-env`) | In dynamics/reward computation | Sim-to-real gap |

### 2. Safe RL (Constrained MDP)
Algorithms from [OmniSafe](https://github.com/PKU-Alignment/omnisafe): **PPOLag**, **CPO**, **OnCRPO**, **FOCOPS**

| Environment | CMDP Cost Signal | Tradeoff |
|-------------|-----------------|----------|
| EVCharging | Excess charge (network violations) | Profit vs. grid safety |
| Building | HVAC ramping (action change rate) | Energy/comfort vs. equipment wear |
| Cogen | Turbine ramp costs | Fuel/delivery vs. smooth operation |

### 3. Multi-Agent RL
[Ray RLlib](https://docs.ray.io/en/latest/rllib/) + [PettingZoo](https://pettingzoo.farama.org/) ParallelEnv.
Algorithms: **PPO**, **SAC**, **APPO**, **IMPALA** with optional `--shared-policy` (parameter sharing).

## Installation

### Prerequisites
- Python 3.9+
- [MOSEK](https://www.mosek.com/downloads/) license (for EVCharging action projection)

### Setup

```bash
git clone https://github.com/MehmetKORUTURK/sustaingym-rl-benchmark.git
cd sustaingym-rl-benchmark
pip install -e ".[all]"
```

Or install only what you need:
```bash
pip install -e .                  # Core (SB3 + standard RL)
pip install -e ".[saferl]"       # + OmniSafe
pip install -e ".[marl]"         # + Ray RLlib + PettingZoo
pip install -e ".[evcharging]"   # + ACN-Portal, CVXPY
```

### Data Setup

Environment data files are not included in the repository due to size. Download and place them as follows:

```
data/
├── building/          # ASHRAE 90.1-2019 prototype building tables + EPW weather files
│   ├── ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm
│   ├── USA_AZ_Tucson-*.epw
│   └── ...
├── cogen/
│   ├── onnx_model/    # ONNX surrogate model for cogeneration plant
│   │   └── model.onnx
│   └── ambients_data/ # Ambient conditions + gas price data
│       ├── ambients_wind=300.0.pkl
│       └── Henry_Hub_Natural_Gas_Spot_Price.csv
├── evcharging/
│   ├── acn_data/      # ACN-Data traces (Caltech/JPL)
│   │   ├── caltech/
│   │   └── jpl/
│   └── gmms/          # Pre-trained GMM models for trace generation
│       ├── caltech/
│       └── jpl/
└── moer/              # Marginal operating emissions rate data
    └── *.csv
```

**Building data**: Download ASHRAE 90.1 prototype building HTML tables from [EnergyPlus](https://www.energyplus.net/weather) and TMY3 EPW weather files from the same source.

**Cogen data**: The ONNX surrogate model and ambient data are provided with the [SustainGym package](https://github.com/chrisyeh96/sustaingym). Install `sustaingym` and copy from its data directory.

**EVCharging data**: ACN-Data traces can be downloaded via `acnportal`. GMM models can be generated using `envs/evcharging/train_gmm_model.py`. MOER data is sourced from [WattTime](https://www.watttime.org/).

## Usage

### Standard RL Training (Stable-Baselines3)

```bash
# PPO on EVCharging (baseline, no noise)
python scripts/train/stdrl_training.py --env evcharging --algo PPO --seed 42

# SAC on Building with observation noise
python scripts/train/stdrl_training.py --env building --algo SAC --noise 0.1 --use-vecnormalize --seed 42

# TD3 on Cogen with all noise channels
python scripts/train/stdrl_training.py --env cogen --algo TD3 --noise 1.0 --noise-action 0.5 --noise-env 2.0 --seed 42
```

**Algorithms**: PPO, SAC, TD3

### Safe RL Training (OmniSafe)

```bash
# PPOLag on EVCharging with cost limit 5
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 5 --seed 42

# OnCRPO on Building with cost limit 50
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 50 --seed 42

# CPO on Cogen with cost limit 25
python scripts/train/saferl_training.py --env cogen --algo CPO --climit 25 --seed 42
```

**Algorithms**: PPOLag, CPO, OnCRPO, FOCOPS

### Multi-Agent RL Training (Ray RLlib)

```bash
# APPO on EVCharging (independent policies)
python scripts/train/marl_training.py --env evcharging --algo APPO --seed 42

# MAPPO on Building (shared policy)
python scripts/train/marl_training.py --env building --algo PPO --shared-policy --seed 42

# PPO on Cogen
python scripts/train/marl_training.py --env cogen --algo PPO --seed 42
```

**Algorithms**: PPO, SAC, APPO, IMPALA

### Evaluation

```bash
# Test a trained model
python scripts/test/stdrl_testing.py --env evcharging --model_path logs_std_train/evcharging_SAC/.../best_model.zip --algo SAC

# Test robustness: apply noise to a baseline model
python scripts/test/stdrl_testing.py --env evcharging --model_path path/to/model.zip --algo SAC --noise 0.2
```

## Output Structure

```
logs_std_train/{env}_{algo}/{timestamp}_NOISE_{n}_ACT_{a}_ENV_{e}/
├── best_model.zip           # Best model (EvalCallback)
├── {env}_{algo}_final.zip   # Final model
├── eval/evaluations.npz     # Evaluation logs
└── events.out.*             # TensorBoard logs

runs/omnisafe_{env}/{algo}_CL_{climit}_{noise}_{timestamp}/
└── ...                      # OmniSafe logs

logs_marl_train/{env}_{algo}/{timestamp}/
├── config.json              # Training config
├── metrics.csv              # Per-iteration metrics
└── checkpoints/             # RLlib checkpoints
```

## Plotting

```bash
# Standard RL training curves
python scripts/plot/stdrl_plot_bu.py --algo PPO --dt DE --auto_ylim

# Post-training analysis (bar charts, heatmaps)
python scripts/plot/stdrl_post_plot.py --env evcharging --algo SAC --noise-type obs --plot all

# Safe RL dual-panel (reward + cost)
python scripts/plot/omni_plot.py --env building --t_steps 20000 --w_size 500 --climit 5

# MARL training curves
python scripts/plot/marl_plot.py --env evcharging --t_steps 32000 --w_size 500 --auto_ylim
```

## HPC (SLURM)

SLURM scripts are provided for Virginia Tech ARC:

- `scripts/slurm/arc_slurm_ev.sh` / `arc_slurm_bu.sh` / `arc_slurm_co.sh` — Standard RL training
- `scripts/slurm/arc_saferl_all.sh` — Safe RL batch commands (48 runs)
- `scripts/slurm/arc_marl_all.sh` — MARL batch commands

## Project Structure

```
sustaingym-rl-benchmark/
├── envs/                              # Environment implementations
│   ├── evcharging/                    # EV charging (single + multi-agent)
│   ├── building/                      # Building HVAC (single + multi-agent)
│   └── cogen/                         # Cogeneration plant (single + multi-agent)
├── algorithms/                        # Baseline algorithms (MPC, greedy)
├── scripts/
│   ├── train/                         # Training scripts
│   │   ├── stdrl_training.py          # Standard RL (SB3: PPO, SAC, TD3)
│   │   ├── saferl_training.py         # Safe RL (OmniSafe: PPOLag, CPO, OnCRPO, FOCOPS)
│   │   ├── marl_training.py           # Multi-agent RL (RLlib, unified)
│   │   └── marl_tr_ev/bu/co.py       # Per-env MARL scripts
│   ├── test/
│   │   └── stdrl_testing.py           # Model evaluation
│   ├── plot/                          # Plotting scripts
│   │   ├── stdrl_plot_ev/bu/co.py     # Training curves
│   │   ├── stdrl_post_plot.py         # Post-training analysis
│   │   ├── omni_plot.py               # Safe RL plots
│   │   └── marl_plot.py               # MARL plots
│   └── slurm/                         # SLURM job scripts (Virginia Tech ARC)
├── requirements.txt
├── setup.py
└── LICENSE
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{koruturk2026sustainrlbench,
    title={{SustainRL-Bench}: Benchmarking Reinforcement Learning on
           {SustainGym} across Perturbation Robustness, Safe RL,
           and Multi-Agent RL},
    author={Koruturk, Mehmet and Sel, Bilgehan and Jin, Ming},
    year={2026},
    institution={Virginia Polytechnic Institute and State University},
    url={https://mehmetkoruturk.github.io/sustaingym-rl-benchmark/}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [SustainGym](https://github.com/chrisyeh96/sustaingym) for the base environment implementations
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [OmniSafe](https://github.com/PKU-Alignment/omnisafe), and [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for RL frameworks
- Virginia Tech Advanced Research Computing (ARC) for computational resources
