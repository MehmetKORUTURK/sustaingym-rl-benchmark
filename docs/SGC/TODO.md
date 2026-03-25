# SmartGridComm 2026 Paper — Master TODO

## Conference Info
- **Conference**: IEEE SmartGridComm 2026
- **Format**: 6-page IEEE double-column
- **Source**: Thesis Ch.4 (Methodology) + Ch.5 (Results) condensed
- **Deadline**: TBD (typically June/July submission)
- **Recommended Focus**: Paper A — Perturbation Robustness (strongest axis)

---

## Paper Strategy Decision

Three possible papers from the thesis. Pick **one** for SmartGridComm:

| Paper | Focus | Strongest Data | Risk |
|-------|-------|---------------|------|
| **A: Perturbation Robustness** | 3-channel noise framework, degradation curves, cross-env | Ch.5 Sec 5.1-5.4 complete | Low — data exists, needs multi-seed |
| **B: Safe RL for Energy** | CMDP formulation, orthogonality finding, PPOLag | Ch.5 Sec 5.5 complete for Cogen+EV | Medium — Building negative result weakens |
| **C: MARL for Energy** | Single vs multi-agent, scalability | Ch.5 Sec 5.6 INCOMPLETE | High — data missing |

**Recommendation**: Paper A. Remaining sections assume Paper A unless noted.

---

# PHASE 1: NEW EXPERIMENTS (Priority Order)

## 1.1 CRITICAL — Multi-Seed Training Runs

Current thesis: single seed (42). Paper requires minimum 3, ideally 5 seeds.
Seeds: `{42, 43, 44, 45, 46}` (seed 42 already done — reuse existing models).

### EVCharging — PPO (4 new seeds x 4 configs = 16 runs)

```bash
# Baseline (clean) — seeds 43-46 only (42 exists)
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --seed 46

# Obs noise 0.1 — seeds 43-46
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.1 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.1 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.1 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.1 --seed 46

# Obs noise 0.3 — seeds 43-46
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.3 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.3 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.3 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise 0.3 --seed 46

# Act noise 0.1 — seeds 43-46
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise-action 0.1 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise-action 0.1 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise-action 0.1 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --noise-action 0.1 --seed 46
```

### EVCharging — SAC (4 new seeds x 4 configs = 16 runs)

```bash
# Baseline (clean)
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --seed 46

# Obs noise 0.1
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.1 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.1 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.1 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.1 --seed 46

# Obs noise 0.3
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.3 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.3 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.3 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise 0.3 --seed 46

# Act noise 0.1
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise-action 0.1 --seed 43
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise-action 0.1 --seed 44
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise-action 0.1 --seed 45
python scripts/train/stdrl_training.py --env evcharging --algo SAC --use-vecnormalize --noise-action 0.1 --seed 46
```

### Building — PPO (4 new seeds x 4 configs = 16 runs)

```bash
# Baseline (clean)
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --seed 43
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --seed 44
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --seed 45
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --seed 46

# Obs noise 0.05
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.05 --seed 43
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.05 --seed 44
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.05 --seed 45
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.05 --seed 46

# Obs noise 0.15
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.15 --seed 43
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.15 --seed 44
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.15 --seed 45
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise 0.15 --seed 46

# Act noise 0.05
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise-action 0.05 --seed 43
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise-action 0.05 --seed 44
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise-action 0.05 --seed 45
python scripts/train/stdrl_training.py --env building --algo PPO --use-vecnormalize --noise-action 0.05 --seed 46
```

### Building — SAC (4 new seeds x 4 configs = 16 runs)

```bash
# Baseline (clean)
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --seed 43
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --seed 44
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --seed 45
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --seed 46

# Obs noise 0.05
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.05 --seed 43
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.05 --seed 44
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.05 --seed 45
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.05 --seed 46

# Obs noise 0.15
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.15 --seed 43
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.15 --seed 44
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.15 --seed 45
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise 0.15 --seed 46

# Act noise 0.05
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise-action 0.05 --seed 43
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise-action 0.05 --seed 44
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise-action 0.05 --seed 45
python scripts/train/stdrl_training.py --env building --algo SAC --use-vecnormalize --noise-action 0.05 --seed 46
```

### Cogen — PPO (4 new seeds x 4 configs = 16 runs, conda: stdrl_train_co)

```bash
# Baseline (clean)
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --seed 46

# Obs noise 1.0
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 1.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 1.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 1.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 1.0 --seed 46

# Obs noise 3.0
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 3.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 3.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 3.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise 3.0 --seed 46

# Env noise 1.0
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise-env 1.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise-env 1.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise-env 1.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo PPO --use-vecnormalize --rm 300 --noise-env 1.0 --seed 46
```

### Cogen — SAC (4 new seeds x 4 configs = 16 runs, conda: stdrl_train_co)

```bash
# Baseline (clean)
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --seed 46

# Obs noise 1.0
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 1.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 1.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 1.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 1.0 --seed 46

# Obs noise 3.0
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 3.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 3.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 3.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise 3.0 --seed 46

# Env noise 1.0
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise-env 1.0 --seed 43
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise-env 1.0 --seed 44
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise-env 1.0 --seed 45
python scripts/train/stdrl_training.py --env cogen --algo SAC --use-vecnormalize --rm 300 --noise-env 1.0 --seed 46
```

### Multi-Seed Training Summary

| Env | Algo | Configs | New seeds | Runs | Approx hours/run | Total hours |
|-----|------|---------|-----------|------|-------------------|-------------|
| EV  | PPO  | 4       | 4         | 16   | ~8h (9M steps)    | 128h        |
| EV  | SAC  | 4       | 4         | 16   | ~6h (9M steps)    | 96h         |
| BU  | PPO  | 4       | 4         | 16   | ~8h (9M steps)    | 128h        |
| BU  | SAC  | 4       | 4         | 16   | ~6h (9M steps)    | 96h         |
| CO  | PPO  | 4       | 4         | 16   | ~3h (3M steps)    | 48h         |
| CO  | SAC  | 4       | 4         | 16   | ~3h (3M steps)    | 48h         |
| **Total** | | | | **96** | | **~544h CPU** |

On ARC normal_q (48 CPUs): ~12 parallel jobs reasonable -> ~45h wall-clock.

---

## 1.2 CRITICAL — Combined Perturbation Experiments

Real-world systems have simultaneous noise on all channels. Currently each channel tested independently.
No new training needed — test existing clean-trained models under combined noise.

### Combined Perturbation Testing Commands

```bash
# ── EVCharging: clean-trained PPO tested under combined noise ──
# Use best_model.zip from clean training (seed 42)
MODEL_EV_PPO="logs_std_train/evcharging_PPO/2026-02-05-20-06-35_DS_0.0_DA_0.0/best_model.zip"

# Light combined
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO --noise 0.05 --noise-action 0.05 --noise-env 0.05 --n-eval 200
# Moderate combined
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO --noise 0.1 --noise-action 0.1 --noise-env 0.1 --n-eval 200
# Heavy combined
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO --noise 0.3 --noise-action 0.2 --noise-env 0.2 --n-eval 200
# Asymmetric: strong obs, weak others
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO --noise 0.3 --noise-action 0.05 --noise-env 0.05 --n-eval 200
# Asymmetric: strong env, weak others
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO --noise 0.05 --noise-action 0.05 --noise-env 0.3 --n-eval 200

# ── EVCharging: clean-trained SAC ──
MODEL_EV_SAC="logs_std_train/evcharging_SAC/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path $MODEL_EV_SAC --noise 0.05 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path $MODEL_EV_SAC --noise 0.1 --noise-action 0.1 --noise-env 0.1 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path $MODEL_EV_SAC --noise 0.3 --noise-action 0.2 --noise-env 0.2 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path $MODEL_EV_SAC --noise 0.3 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path $MODEL_EV_SAC --noise 0.05 --noise-action 0.05 --noise-env 0.3 --n-eval 200

# ── Building: clean-trained PPO ──
MODEL_BU_PPO="logs_std_train/building_PPO/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO --noise 0.03 --noise-action 0.03 --noise-env 0.03 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO --noise 0.05 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO --noise 0.1 --noise-action 0.1 --noise-env 0.1 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO --noise 0.15 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO --noise 0.05 --noise-action 0.05 --noise-env 0.15 --n-eval 200

# ── Building: clean-trained SAC ──
MODEL_BU_SAC="logs_std_train/building_SAC/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env building --algo SAC --model_path $MODEL_BU_SAC --noise 0.03 --noise-action 0.03 --noise-env 0.03 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path $MODEL_BU_SAC --noise 0.05 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path $MODEL_BU_SAC --noise 0.1 --noise-action 0.1 --noise-env 0.1 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path $MODEL_BU_SAC --noise 0.15 --noise-action 0.05 --noise-env 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path $MODEL_BU_SAC --noise 0.05 --noise-action 0.05 --noise-env 0.15 --n-eval 200

# ── Cogen: clean-trained PPO ──
MODEL_CO_PPO="logs_std_train/cogen_PPO/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env cogen --algo PPO --model_path $MODEL_CO_PPO --rm 300 --noise 1.0 --noise-action 0.1 --noise-env 1.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo PPO --model_path $MODEL_CO_PPO --rm 300 --noise 3.0 --noise-action 0.2 --noise-env 2.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo PPO --model_path $MODEL_CO_PPO --rm 300 --noise 3.0 --noise-action 0.05 --noise-env 0.5 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo PPO --model_path $MODEL_CO_PPO --rm 300 --noise 0.5 --noise-action 0.05 --noise-env 3.0 --n-eval 200

# ── Cogen: clean-trained SAC ──
MODEL_CO_SAC="logs_std_train/cogen_SAC/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC --rm 300 --noise 1.0 --noise-action 0.1 --noise-env 1.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC --rm 300 --noise 3.0 --noise-action 0.2 --noise-env 2.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC --rm 300 --noise 3.0 --noise-action 0.05 --noise-env 0.5 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC --rm 300 --noise 0.5 --noise-action 0.05 --noise-env 3.0 --n-eval 200
```

**Total combined tests**: 6 models x ~5 configs x 200 episodes = ~30 test runs (~minutes each, no GPU needed)

**Analysis**: Compare `degradation_combined` vs `degradation_obs + degradation_act + degradation_env` to measure interaction effects (super-additive = compounding, sub-additive = saturation).

---

## 1.3 CRITICAL — Noise-Trained vs Clean-Trained Cross-Evaluation

Key question: Does training under noise improve deployment robustness?
TD3 Cogen already shows noise-as-regularization. Systematically test this.

### Cross-Evaluation Matrix Testing Commands

Test noise-trained models at **different** noise levels than they were trained on.

```bash
# ── EVCharging: model trained at obs=0.1, test at obs=0.0 / 0.05 / 0.2 / 0.3 ──
MODEL_EV_PPO_N01="logs_std_train/evcharging_PPO/<TIMESTAMP>_NOISE_0.1_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N01 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N01 --noise 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N01 --noise 0.2 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N01 --noise 0.3 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N01 --noise 0.5 --n-eval 200

# ── EVCharging: model trained at obs=0.3, test at other levels ──
MODEL_EV_PPO_N03="logs_std_train/evcharging_PPO/<TIMESTAMP>_NOISE_0.3_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N03 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N03 --noise 0.1 --n-eval 200
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path $MODEL_EV_PPO_N03 --noise 0.5 --n-eval 200

# ── Building: model trained at obs=0.05, test at obs=0.0 / 0.1 / 0.15 / 0.3 ──
MODEL_BU_PPO_N005="logs_std_train/building_PPO/<TIMESTAMP>_NOISE_0.05_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N005 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N005 --noise 0.1 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N005 --noise 0.15 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N005 --noise 0.3 --n-eval 200

# ── Building: model trained at obs=0.15, test at other levels ──
MODEL_BU_PPO_N015="logs_std_train/building_PPO/<TIMESTAMP>_NOISE_0.15_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N015 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N015 --noise 0.05 --n-eval 200
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path $MODEL_BU_PPO_N015 --noise 0.3 --n-eval 200

# ── Cogen: model trained at obs=1.0, test at other levels ──
MODEL_CO_SAC_N1="logs_std_train/cogen_SAC/<TIMESTAMP>_NOISE_1.0_ACT_0.0_ENV_0.0/best_model.zip"

python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC_N1 --rm 300 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC_N1 --rm 300 --noise 3.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo SAC --model_path $MODEL_CO_SAC_N1 --rm 300 --noise 5.0 --n-eval 200

# ── Cogen: TD3 noise-as-regularization confirmation ──
MODEL_CO_TD3_N10="logs_std_train/cogen_TD3/<TIMESTAMP>_NOISE_10.0_ACT_0.0_ENV_0.0/best_model.zip"
MODEL_CO_TD3_CLEAN="logs_std_train/cogen_TD3/<TIMESTAMP>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

# Compare clean-trained vs noise-trained at various test noise levels
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --model_path $MODEL_CO_TD3_CLEAN --rm 300 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --model_path $MODEL_CO_TD3_CLEAN --rm 300 --noise 5.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --model_path $MODEL_CO_TD3_N10 --rm 300 --noise 0.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --model_path $MODEL_CO_TD3_N10 --rm 300 --noise 5.0 --n-eval 200
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --model_path $MODEL_CO_TD3_N10 --rm 300 --noise 15.0 --n-eval 200
```

**Total cross-eval tests**: ~30 runs (minutes each, CPU only)

**Expected output**: Cross-evaluation heatmap (train_noise x test_noise -> reward). Key insight: diagonal = matched, off-diagonal = generalization.

---

## 1.4 IMPORTANT — Statistical Significance Tests

No formal hypothesis testing exists. Need p-values for algorithm comparisons.

### Script to Create: `scripts/analysis/stat_tests.py`

```python
# Required analysis:
# 1. For each (env, noise_level): Welch t-test PPO vs SAC, PPO vs TD3, SAC vs TD3
# 2. Bonferroni correction for multiple comparisons
# 3. Cohen's d effect size
# 4. Output: LaTeX table with p-values and significance stars
#
# Input: episode_results.csv files from logs_std_test/
# Output: tables/stat_significance.tex
#
# Libraries: scipy.stats.ttest_ind, numpy
```

---

## 1.5 IMPORTANT — Computation Cost Reporting

Reviewer will ask "how much compute?". Need wall-clock and CPU-hour numbers.

### Commands to Extract from ARC

```bash
# Get wall-clock from SLURM logs
sacct -u $USER --format=JobID,JobName,Elapsed,MaxRSS,State -S 2026-02-01 | grep STDRL

# Or parse from training logs: start/end time
for dir in logs_std_train/*/*; do
  if [ -f "$dir/events.out.tfevents."* ]; then
    echo "$dir: $(stat -c '%Y' "$dir/events.out.tfevents."*)"
  fi
done
```

Report in paper: "Total compute: X CPU-hours on Intel Xeon (48 cores, normal_q partition). No GPU required for SB3 training."

---

# PHASE 2: ANALYSIS & PLOTTING

## 2.1 IQM + rliable Pipeline

### Script to Create: `scripts/analysis/compute_iqm.py`

```bash
# Uses rliable library (pip install rliable)
# Input: multi-seed test results from Phase 1
# Output: IQM scores, stratified bootstrap CIs, performance profiles
pip install rliable
```

### Script to Create: `scripts/plot/sgc_figures.py`

Paper needs exactly 5-6 figures:

| Fig | Content | Data Source |
|-----|---------|-------------|
| Fig 1 | Framework architecture | TikZ diagram (create) |
| Fig 2 | Degradation curves: 3 envs x 2 algos, IQM + CI bands | Multi-seed test results |
| Fig 3 | Cross-env heatmap: algo x env x noise_type -> degradation % | `noise_heatmap.py` exists |
| Fig 4 | Combined vs individual perturbation interaction | Combined test results (Sec 1.2) |
| Fig 5 | Noise-trained vs clean-trained cross-eval matrix | Cross-eval results (Sec 1.3) |
| Fig 6 | (Optional) Safe RL dual-panel Cogen PPOLag | Existing saferl logs |

### Script to Create: `scripts/plot/sgc_degradation.py`

```bash
# Replaces thesis's per-algo violin plots with compact degradation curves
# X-axis: noise level, Y-axis: normalized reward (IQM), bands: CI from 5 seeds
# 3 subplots (1 per env), 2 lines per subplot (PPO, SAC), 3 line styles (obs/act/env)
python scripts/plot/sgc_degradation.py --seeds 42,43,44,45,46 --output docs/SGC/figures/
```

## 2.2 Normalized Metrics

Define and compute:
- **Normalized Performance**: `R_norm = (R - R_random) / (R_best_baseline - R_random)` per env
- **Degradation Ratio**: `D(sigma) = (R(sigma) - R(0)) / |R(0)|`
- **Sensitivity Coefficient**: `kappa = -dD/d_sigma |_{sigma=0}` (linear fit at low noise)

---

# PHASE 3: POTENTIAL REVIEWER QUESTIONS & MITIGATIONS

## Q1: "Only 1 seed per config" (CRITICAL)
- **Mitigation**: Phase 1.1 multi-seed runs
- **Fallback**: If time-constrained, run 3 seeds instead of 5 (minimum acceptable)

## Q2: "RL underperforms Greedy heuristic in EV Charging"
- **Mitigation**: Frame as finding, not weakness
- **Talking points**:
  - Greedy ignores carbon cost (optimizes only profit)
  - RL learns carbon-aware behavior at cost of raw profit
  - EV Charging reward has 3 conflicting objectives (profit, carbon, violations)
  - Key insight: "Simple heuristics can outperform RL when reward decomposition favors greedy strategies"
- **Optional**: Re-run EV PPO with higher entropy_coef (0.01-0.02) or longer training (15M steps)

```bash
# Optional: improved EV PPO
python scripts/train/stdrl_training.py --env evcharging --algo PPO --use-vecnormalize --evcharging_train_steps 15000000 --seed 42
```

## Q3: "No combined perturbation experiments"
- **Mitigation**: Phase 1.2 (test-time only, no new training needed)

## Q4: "No domain randomization / robust RL baseline"
- **Mitigation**: Phase 1.3 noise-trained cross-eval = "poor man's domain randomization"
- **Talking points**:
  - Training at fixed noise level is a point sample of domain randomization
  - Cross-eval matrix shows whether noise-trained agents generalize
  - If they do, this motivates full domain randomization as future work
- **Optional**: If time allows, train with uniform-sampled noise per episode

```bash
# Requires code change in stdrl_training.py to sample noise ~ Uniform(0, sigma_max) per episode
# Not implemented yet — mark as future work if no time
```

## Q5: "No comparison with robust RL methods (RARL, etc.)"
- **Mitigation**: Out of scope for 6-page paper. Cite Pinto (2017) RARL and Gu (2025) Robust-Gymnasium. Position our work as "characterization" not "solution".
- **Talking point**: "We provide the degradation baseline that robust RL methods should be evaluated against"

## Q6: "Building Safe RL doesn't work — weakens the Safe RL story"
- **Mitigation**: If Paper A (perturbation only), drop Safe RL entirely. If including Safe RL, lead with Cogen (strong positive) and present Building as "design lesson: orthogonality requirement".

## Q7: "Cogen PPO is robust but performs terribly (reward -1.76 vs SAC -0.03)"
- **Mitigation**: Explicitly discuss the **robustness-performance tradeoff**:
  - PPO's conservative strategy (low actions) = robust but suboptimal
  - SAC's aggressive strategy (near constraint boundary) = optimal but fragile
  - This IS the key finding: "There is a fundamental tension between clean-condition optimality and perturbation robustness"
  - Practitioner guidance: "Choose PPO for safety-critical deployment, SAC for controlled environments"

## Q8: "How generalizable beyond SustainGym?"
- **Mitigation**:
  - Framework is environment-agnostic (noise injection is a wrapper)
  - SustainGym uses physics-based models grounded in real data
  - 3 diverse environments show pattern generality
  - Cite that findings align with general RL robustness literature (Dulac-Arnold 2021)

## Q9: "What about partial observability? Your noise model is simplistic."
- **Mitigation**:
  - Gaussian noise is standard in robustness literature (Dulac-Arnold 2021, Kirk 2023)
  - Proportional noise matches realistic sensor error models
  - Mixed noise model in EV Charging (different per component) shows flexibility
  - Structured noise (delays, dropouts, bias) is future work

## Q10: "Why not test on real hardware / real building / real charging network?"
- **Mitigation**: Simulation-only is standard for benchmark papers. Cite SustainGym's real-world data grounding. Add "sim-to-real transfer" as future work.

---

# PHASE 4: PAPER WRITING

## Proposed Paper Structure (6 pages, IEEE SmartGridComm)

```
Title: "How Robust Are RL Controllers for Smart Grid Systems?
        A Systematic Three-Channel Perturbation Analysis"

I. Introduction (0.75 page)
   - RL promise for energy systems
   - Gap: no systematic perturbation characterization
   - Contributions: 3-channel framework, cross-env analysis, actionable findings

II. Related Work (0.5 page)
   - RL for energy (SustainGym, CityLearn, Grid2Op)
   - Robustness in RL (Dulac-Arnold, RARL, Robust-Gymnasium)
   - Positioning: first multi-env perturbation benchmark for energy RL

III. Perturbation Framework (1 page)
   - 3 environments (table with key specs)
   - 3 noise channels (formal definitions, Eq 1-3)
   - Physical motivation for each noise model per env

IV. Experimental Setup (0.5 page)
   - Algorithms: PPO, SAC (drop TD3 — negative finding in 1 sentence)
   - Multi-seed protocol (5 seeds, IQM)
   - Noise levels per env (table)
   - Baselines: Random, Greedy/MPC/DoNothing

V. Results (2.5 pages)
   V-A. Clean-condition performance + non-RL baselines (Table)
   V-B. Per-channel degradation curves (Fig 2 — key figure)
   V-C. Cross-environment comparison (Fig 3 — heatmap)
   V-D. Combined perturbation (Fig 4 — interaction analysis)
   V-E. Noise-trained generalization (Fig 5 — cross-eval)
   V-F. Key findings (numbered list, 4-5 findings)

VI. Conclusion (0.5 page)
   - Summary: env structure > algorithm choice
   - Practical guidelines for practitioners
   - Future: robust RL, combined noise training, real-world deployment

References (~0.25 page, ~20-25 refs)
```

## Key Tables (4 tables max)

| Table | Content |
|-------|---------|
| Table I | Environment specs (obs/act dim, episode length, reward range, noise model) |
| Table II | Baseline performance: IQM +/- CI, 5 seeds, with non-RL baselines |
| Table III | Degradation summary: % degradation at moderate noise per env x algo x channel |
| Table IV | Cross-eval matrix: train_noise x test_noise -> reward (Building PPO, most interesting) |

## Key Figures (5-6 figures max)

| Fig | Type | What it shows |
|-----|------|---------------|
| Fig 1 | Architecture diagram | 3 envs, 3 noise channels, evaluation pipeline |
| Fig 2 | Line plot (3x1 panel) | Degradation curves: noise level vs normalized reward, 2 algos, 3 channels |
| Fig 3 | Heatmap (3x3 grid) | env x algo -> degradation %, colored cells, 3 sub-heatmaps per channel |
| Fig 4 | Bar chart | Combined vs sum-of-individual degradation (super/sub-additive analysis) |
| Fig 5 | Heatmap (NxN) | Cross-eval: train_noise x test_noise -> reward for Building PPO |
| Fig 6 | (Optional) Violin | Reward distributions at key noise levels (most informative single figure) |

---

# PHASE 5: CHECKLIST BEFORE SUBMISSION

- [ ] All experiments use 5 seeds with IQM reporting
- [ ] Combined perturbation results included
- [ ] Cross-evaluation (noise-trained vs clean-trained) included
- [ ] Statistical significance tests (Welch t-test + Bonferroni) for all algo comparisons
- [ ] Computation cost reported (CPU-hours, hardware specs)
- [ ] Non-RL baselines in comparison table
- [ ] Figures are IEEE-compliant (readable in grayscale, proper font sizes)
- [ ] Code repository link (anonymized if double-blind, check SmartGridComm policy)
- [ ] All `<TIMESTAMP>` placeholders in commands replaced with actual paths
- [ ] Paper fits in 6 pages including references

---

# SLURM BATCH SCRIPT TEMPLATES

## Template: Multi-Seed Training (EV/Building)

```bash
#!/bin/bash -l
#SBATCH -J SGC_MULTISEED_{ENV}_{ALGO}_S{SEED}
#SBATCH -A llmAlignment
#SBATCH --partition=normal_q
#SBATCH --constraint=intel&avx512
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=40G
#SBATCH -t 3-00:00:00
#SBATCH -o slurmlogs/%x-%j.out
#SBATCH -e slurmlogs/%x-%j.err

set -e
cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate stdrl_train

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ── Run 4 configs sequentially (1 seed, all noise levels) ──
python scripts/train/stdrl_training.py --env {ENV} --algo {ALGO} --use-vecnormalize --seed {SEED}
python scripts/train/stdrl_training.py --env {ENV} --algo {ALGO} --use-vecnormalize --noise {N1} --seed {SEED}
python scripts/train/stdrl_training.py --env {ENV} --algo {ALGO} --use-vecnormalize --noise {N2} --seed {SEED}
python scripts/train/stdrl_training.py --env {ENV} --algo {ALGO} --use-vecnormalize --noise-action {A1} --seed {SEED}
```

## Template: Multi-Seed Training (Cogen)

```bash
#!/bin/bash -l
#SBATCH -J SGC_MULTISEED_CO_{ALGO}_S{SEED}
#SBATCH -A llmAlignment
#SBATCH --partition=normal_q
#SBATCH --constraint=intel&avx512
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=40G
#SBATCH -t 2-00:00:00
#SBATCH -o slurmlogs/%x-%j.out
#SBATCH -e slurmlogs/%x-%j.err

set -e
cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate stdrl_train_co  # NOTE: different conda env for cogen

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

python scripts/train/stdrl_training.py --env cogen --algo {ALGO} --use-vecnormalize --rm 300 --seed {SEED}
python scripts/train/stdrl_training.py --env cogen --algo {ALGO} --use-vecnormalize --rm 300 --noise {N1} --seed {SEED}
python scripts/train/stdrl_training.py --env cogen --algo {ALGO} --use-vecnormalize --rm 300 --noise {N2} --seed {SEED}
python scripts/train/stdrl_training.py --env cogen --algo {ALGO} --use-vecnormalize --rm 300 --noise-env {E1} --seed {SEED}
```

## Concrete SLURM Submission Plan (24 jobs)

```bash
# Each job runs 1 seed x 4 configs (sequential within job, parallel across jobs)
# EV+BU use stdrl_train, CO uses stdrl_train_co

# EV PPO seeds 43-46 (4 jobs)
sbatch scripts/slurm/sgc_ev_ppo_s43.sh
sbatch scripts/slurm/sgc_ev_ppo_s44.sh
sbatch scripts/slurm/sgc_ev_ppo_s45.sh
sbatch scripts/slurm/sgc_ev_ppo_s46.sh

# EV SAC seeds 43-46 (4 jobs)
sbatch scripts/slurm/sgc_ev_sac_s43.sh
sbatch scripts/slurm/sgc_ev_sac_s44.sh
sbatch scripts/slurm/sgc_ev_sac_s45.sh
sbatch scripts/slurm/sgc_ev_sac_s46.sh

# BU PPO seeds 43-46 (4 jobs)
sbatch scripts/slurm/sgc_bu_ppo_s43.sh
sbatch scripts/slurm/sgc_bu_ppo_s44.sh
sbatch scripts/slurm/sgc_bu_ppo_s45.sh
sbatch scripts/slurm/sgc_bu_ppo_s46.sh

# BU SAC seeds 43-46 (4 jobs)
sbatch scripts/slurm/sgc_bu_sac_s43.sh
sbatch scripts/slurm/sgc_bu_sac_s44.sh
sbatch scripts/slurm/sgc_bu_sac_s45.sh
sbatch scripts/slurm/sgc_bu_sac_s46.sh

# CO PPO seeds 43-46 (4 jobs, stdrl_train_co)
sbatch scripts/slurm/sgc_co_ppo_s43.sh
sbatch scripts/slurm/sgc_co_ppo_s44.sh
sbatch scripts/slurm/sgc_co_ppo_s45.sh
sbatch scripts/slurm/sgc_co_ppo_s46.sh

# CO SAC seeds 43-46 (4 jobs, stdrl_train_co)
sbatch scripts/slurm/sgc_co_sac_s43.sh
sbatch scripts/slurm/sgc_co_sac_s44.sh
sbatch scripts/slurm/sgc_co_sac_s45.sh
sbatch scripts/slurm/sgc_co_sac_s46.sh
```

---

# TIMELINE

| Week | Task | Deliverable |
|------|------|-------------|
| Week 1 | Submit 24 SLURM jobs for multi-seed training | 96 new training runs on ARC |
| Week 1 | Run combined perturbation tests (Sec 1.2) + cross-eval (Sec 1.3) | ~60 test runs (CPU, fast) |
| Week 2 | Collect results, run IQM analysis, stat tests | Tables + p-values |
| Week 2 | Create paper figures (sgc_figures.py) | 5-6 publication-quality figures |
| Week 3 | Write paper draft | 6-page IEEE format |
| Week 3 | Internal review with advisor | Feedback |
| Week 4 | Revisions + final polish | Submission-ready PDF |
