# SmartGridComm Deep Dive: Novelty, Strategy & Gap Analysis

**Project**: Multi-Paradigm RL Benchmark for Sustainable Energy Systems
**Author**: Mehmet Koruturk, Virginia Tech (RoleLab)
**Target**: IEEE SmartGridComm 2026 (Deadline: June 19, 2026)
**Date**: March 3, 2026
**Version**: 2.0 — Deep Dive Edition

---

## Table of Contents

1. [Novelty & Contributions: Detailed Assessment](#1-novelty--contributions-detailed-assessment)
2. [Gap Analysis: Comprehensive Audit](#2-gap-analysis-comprehensive-audit)
3. [Strategic Plan: Step-by-Step Execution Roadmap](#3-strategic-plan-step-by-step-execution-roadmap)
4. [Experiment Design: Detailed Protocols](#4-experiment-design-detailed-protocols)
5. [Baseline Implementation Guide](#5-baseline-implementation-guide)
6. [Evaluation & Reporting Framework](#6-evaluation--reporting-framework)
7. [Paper Writing Blueprint](#7-paper-writing-blueprint)
8. [Risk Assessment & Contingency Plans](#8-risk-assessment--contingency-plans)

---

## 1. Novelty & Contributions: Detailed Assessment

### 1.1 Honest Novelty Audit

Before claiming novelty, we must be precise about what already exists and what does not.

**What is NOT novel (do not overclaim):**

| Component | Prior Work | Citation |
|-----------|-----------|----------|
| EVCharging environment | SustainGym (NeurIPS 2023) | Yeh et al. 2023 |
| Building environment | SustainGym (NeurIPS 2023) | Yeh et al. 2023 |
| Cogeneration environment | SustainGym (NeurIPS 2023) | Yeh et al. 2023 |
| PPO, SAC, TD3 algorithms | Well-established | Schulman 2017; Haarnoja 2018; Fujimoto 2018 |
| OmniSafe framework | Published | Ji et al. 2023 |
| Ray RLlib for MARL | Published | Liang et al. 2018 |
| PettingZoo API | Published | Terry et al. 2021 |
| RL for EV charging (general) | Multiple papers | Wan 2019; Qian 2019; Wang 2023 |
| RL for building HVAC (general) | Multiple papers | Yu 2021; Brandi 2022; Zhang 2019 |
| Safe RL theory (CMDP) | Well-established | Altman 1999; Achiam 2017 |
| MARL for buildings | CityLearn, GridLearn | Vazquez-Canteli 2019; Pigott 2022 |

**What IS potentially novel (but must be validated):**

| Candidate Novelty | Novelty Type | Confidence | Evidence Required |
|-------------------|-------------|-----------|-------------------|
| **Unified 3-env x 3-paradigm benchmark** | Protocol/Framework | HIGH | Verify no prior paper does this exact combination |
| **Noise decomposition (obs/action/env) for energy RL** | Methodology | HIGH | Verify no prior paper systematically decomposes noise channels in energy domains |
| **Safe RL applied to EV charging** | Application | MEDIUM | Verify no prior OmniSafe/CMDP work on ACN-based EV charging |
| **Safe RL applied to cogeneration** | Application | HIGH | Very niche domain; unlikely prior constrained RL work here |
| **Cross-paradigm comparison on same envs** | Empirical | HIGH | No paper compares standard RL vs safe RL vs MARL head-to-head |
| **Distribution shift analysis for energy RL** | Methodology | MEDIUM | Some papers study generalization; but structured noise decomposition is rare |
| **Multi-agent cogen control** | Application | HIGH | 4-agent decomposition of cogen plant is likely novel |

### 1.2 Detailed Novelty-Contribution Mapping

This section defines a precise "If-Then" logic: for each action we take, what novelty/contribution can we claim.

---

#### Novelty Claim N1: First Unified Multi-Paradigm Energy RL Benchmark

**What we must do:**
1. Complete standard RL training (PPO, SAC) across all 3 environments with 5 seeds
2. Complete safe RL training (OnCRPO + 2 others) across all 3 environments with 5 seeds
3. Complete MARL training across all 3 environments with 5 seeds
4. Present all results in a single unified table with identical metrics

**What we can then claim:**
> "To the best of our knowledge, this is the first benchmark that systematically evaluates standard RL, constrained (safe) RL, and multi-agent RL on the same set of energy system environments under a unified evaluation protocol."

**Why this is credible:**
- SustainGym (Yeh et al. 2023) provided environments + basic PPO baselines only
- CityLearn (Vazquez-Canteli 2019) focuses on MARL for buildings only, no safe RL
- No existing paper spans EV + Building + Cogen AND standard + safe + MARL

**Strength**: HIGH — this is the paper's anchor novelty claim

---

#### Novelty Claim N2: Structured Noise Robustness Decomposition

**What we must do:**
1. Define a formal noise model with three orthogonal channels:
   - Observation noise: `o_noisy = o_true + epsilon_o`, where `epsilon_o ~ N(0, sigma_o * |o_true|)`
   - Action noise: `a_noisy = a_true + epsilon_a`, where `epsilon_a ~ N(0, sigma_a)`
   - Environment noise: dynamics perturbation `s' = f(s, a) + epsilon_e`
2. Run experiments varying each channel independently (other two held at 0)
3. Run experiments varying all channels simultaneously
4. Show decomposed impact profiles differ across environments

**What we can then claim:**
> "We introduce a structured noise decomposition framework for energy RL evaluation, decomposing uncertainty into observation, action, and environment channels. Our analysis reveals that [finding: e.g., environment noise dominates in EV charging while observation noise dominates in cogeneration], providing actionable guidance for practitioners prioritizing robustness improvements."

**Why this is credible:**
- Dulac-Arnold et al. (2021) define real-world RL challenges but don't apply decomposition to energy domains
- Most energy RL papers add noise as a single unstructured perturbation
- The decomposition into obs/action/env is a methodological contribution

**Strength**: HIGH — this is the paper's methodological novelty

---

#### Novelty Claim N3: Safe RL Benchmark for EV Charging and Cogeneration

**What we must do:**
1. Formally define CMDP formulations for each environment:
   - EVCharging: cost = network constraint violations (excess charge)
   - Building: cost = temperature comfort violations
   - Cogen: cost = dynamic operational constraint violations
2. Run 3-4 constrained RL algorithms (OnCRPO, CPO, PPOLag) with varying cost limits
3. Generate Pareto curves (reward vs. constraint cost)
4. Compare constraint satisfaction rate vs. unconstrained baselines

**What we can then claim:**
> "We present the first constrained RL benchmark for EV charging and cogeneration control, demonstrating that CMDP-based algorithms can reduce constraint violations by X% while maintaining Y% of unconstrained performance."

**Why this is credible:**
- Safe RL surveys (Garcia 2015; Xu 2023) note the gap between theory and energy applications
- OmniSafe (Ji et al. 2023) provides algorithms but no energy domain benchmarks
- No published CMDP results exist for ACN-based EV charging or ONNX-based cogeneration

**Strength**: MEDIUM-HIGH — dependent on generating meaningful Pareto tradeoffs

---

#### Novelty Claim N4: Cross-Domain MARL Scalability Comparison

**What we must do:**
1. Train MARL on 3 environments with different agent counts:
   - EVCharging: 54 agents (one per charging station)
   - Building: variable (depends on zone count, typically 3-6)
   - Cogen: 4 agents (GT1, GT2, GT3, ST)
2. Compare MARL vs. single-agent performance on identical tasks
3. Analyze how reward decomposition affects learning (shared vs. individual)
4. Report communication overhead and training time comparisons

**What we can then claim:**
> "Our cross-domain MARL analysis reveals that multi-agent decomposition improves performance in [env] but degrades in [env], with the benefit correlating to [finding: e.g., natural agent decomposability / action independence]."

**Why this is credible:**
- Most MARL energy papers study a single domain
- The comparison across 3 domains with different agent structures is novel

**Strength**: MEDIUM — findings may be environment-specific rather than generalizable

---

#### Novelty Claim N5: RL vs. Non-RL Baseline Comparison Under Uncertainty

**What we must do:**
1. Implement rule-based baselines for all 3 environments
2. Implement at least one optimization baseline (LP or simple MPC) for at least one env
3. Evaluate all baselines and RL algorithms under identical noise conditions
4. Show when RL outperforms baselines and when it does not

**What we can then claim:**
> "We demonstrate that deep RL consistently outperforms rule-based baselines across all environments, but optimization-based approaches remain competitive in low-noise settings. RL's advantage grows with increasing uncertainty, suggesting RL is most valuable when model accuracy degrades."

**Why this is credible:**
- This is the question every reviewer will ask
- Answering it honestly (including when RL loses) adds credibility
- The finding that "RL advantage grows with noise" would be actionable for practitioners

**Strength**: HIGH — addresses the most fundamental reviewer concern

---

### 1.3 Contribution Statement (Final Draft)

Based on the above analysis, here is the paper's contribution statement:

> We present **SustainRL-Bench**, a comprehensive benchmark for reinforcement learning in sustainable energy systems. Our contributions are:
>
> **(C1) A unified multi-paradigm evaluation protocol** that systematically compares standard RL (PPO, SAC), constrained safe RL (OnCRPO, CPO, PPOLag), and cooperative multi-agent RL across three diverse SustainGym environments—EV charging, building HVAC, and cogeneration—using standardized metrics and statistical methodology.
>
> **(C2) A structured noise robustness framework** that decomposes uncertainty into observation, action, and environment noise channels, revealing environment-specific vulnerability profiles and providing actionable guidance for deployment prioritization.
>
> **(C3) The first safe RL benchmark for EV charging and cogeneration control**, formulating CMDP cost signals for network constraint violations and operational limits, and characterizing the reward-safety Pareto frontier across domains.
>
> **(C4) A cross-domain comparative analysis** showing that [key finding about when RL beats baselines, when MARL helps, and when safe RL is worth the cost], grounded in [X seeds × Y conditions = Z total experiments] across [total training hours].
>
> **(C5) An open-source benchmark suite** with reproducible training scripts, evaluation protocols, and pre-trained models, enabling the community to extend and compare against our results.

### 1.4 Positioning Against Closest Competitors

| Paper | What They Do | What We Add |
|-------|-------------|------------|
| **SustainGym** (Yeh 2023) | Defines 3 envs + basic PPO baseline | Multi-paradigm benchmark (safe RL + MARL) + noise robustness + non-RL baselines |
| **CityLearn** (Vazquez-Canteli 2019/2024) | MARL benchmark for buildings only | 3 domains instead of 1 + safe RL axis + noise decomposition |
| **MARL for CityLearn** (arXiv 2026) | Multi-KPI MARL on CityLearn | We cover EV/Cogen too + safe RL + structured noise |
| **GridLearn** (Pigott 2022) | MARL for grid-aware buildings | Single domain; we are multi-domain multi-paradigm |
| **BOPTEST** (Blum 2021) | Building control benchmark with MPC | Building-only; we add RL + MARL + safe RL across domains |
| **OmniSafe** (Ji 2023) | Safe RL framework + Safety-Gym | Robotics benchmark; we apply to energy systems |
| **Dulac-Arnold** (2021) | Defines real-world RL challenges | Framework paper; we instantiate for energy domains |

**Our unique position**: The intersection of {multi-domain} x {multi-paradigm} x {noise decomposition}. No existing paper occupies this intersection.

---

## 2. Gap Analysis: Comprehensive Audit

### 2.1 Experiment Completion Matrix (Detailed)

This matrix tracks every experiment needed for the paper at granular level.

#### Standard RL Experiments

| Env | Algo | Clean (noise=0) | Obs Noise Sweep | Act Noise Sweep | Env Noise Sweep | 5 Seeds | Status |
|-----|------|-----------------|-----------------|-----------------|-----------------|---------|--------|
| **EVCharging** | PPO | ✅ | ✅ (10 levels) | ✅ | ⚠️ In progress | ❓ | 70% |
| **EVCharging** | SAC | ✅ | ✅ (10 levels) | ✅ | ⚠️ In progress | ❓ | 70% |
| **EVCharging** | TD3 | ✅ | ✅ | ✅ | ❌ | ❓ | 60% |
| **Building** | PPO | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | ❌ | ❓ | 30% |
| **Building** | SAC | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | ❌ | ❓ | 30% |
| **Building** | TD3 | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ | ❓ | 20% |
| **Cogen** | PPO | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ | ❓ | 20% |
| **Cogen** | SAC | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ | ❓ | 20% |
| **Cogen** | TD3 | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ | ❓ | 15% |

**Minimum for paper (scope reduction)**:
- Drop TD3 entirely (keep PPO + SAC only) → saves 33% compute
- Use 5 noise levels instead of 10: {0.0, 0.05, 0.10, 0.20, 0.40}
- Required: 3 envs × 2 algos × 5 noise levels × 3 noise types × 5 seeds = **450 training runs** for standard RL alone

#### Safe RL Experiments

| Env | Algorithm | Cost Limit 0.5 | Cost Limit 1.0 | Cost Limit 5.0 | 5 Seeds | Status |
|-----|-----------|---------------|---------------|---------------|---------|--------|
| **EVCharging** | OnCRPO | ✅ | ✅ | ❌ | ❓ | 60% |
| **EVCharging** | CPO | ✅ | ✅ | ❌ | ❓ | 60% |
| **EVCharging** | PPOLag | ✅ | ✅ | ❌ | ❓ | 60% |
| **Building** | OnCRPO | ⚠️ | ❌ | ❌ | ❌ | 15% |
| **Building** | CPO | ❌ | ❌ | ❌ | ❌ | 0% |
| **Building** | PPOLag | ❌ | ❌ | ❌ | ❌ | 0% |
| **Cogen** | OnCRPO | ⚠️ | ❌ | ❌ | ❌ | 10% |
| **Cogen** | CPO | ❌ | ❌ | ❌ | ❌ | 0% |
| **Cogen** | PPOLag | ❌ | ❌ | ❌ | ❌ | 0% |

**Minimum for paper**:
- 3 safe RL algorithms × 3 envs × 2 cost limits × 5 seeds = **90 training runs**
- Can reduce to 2 algorithms if compute-limited: OnCRPO + PPOLag

#### MARL Experiments

| Env | Algorithm | Agents | Iterations | Converged? | 5 Seeds | Status |
|-----|-----------|--------|------------|-----------|---------|--------|
| **EVCharging** | APPO | 54 | 32,000 | ❓ | ❌ | 30% |
| **Building** | SAC | ~5 | 32,000 | ❓ | ❌ | 30% |
| **Cogen** | PPO | 4 | 100 | ❌ (too few) | ❌ | 10% |

**Minimum for paper**:
- 3 envs × 1 algo × 5 seeds = **15 training runs**
- Cogen needs >> 100 iterations (at least 1,000-5,000)
- Must show convergence curves

#### Baseline Experiments

| Env | Do-Nothing | Rule-Based | Optimization | Status |
|-----|-----------|-----------|--------------|--------|
| **EVCharging** | ❌ | ❌ | ❌ | 0% |
| **Building** | ❌ | ❌ | ❌ | 0% |
| **Cogen** | ❌ | ❌ | ❌ | 0% |

**Minimum for paper**:
- 3 envs × 2 baselines (Do-Nothing + Rule-Based) × 5 seeds = **30 evaluation runs** (fast, no training)
- Optimization baseline: at least 1 env × 5 seeds = **5 runs** (nice-to-have)

### 2.2 Code-Level Gaps

| File | Issue | Impact | Fix Required | Effort |
|------|-------|--------|-------------|--------|
| `stdrl_testing.py` | Domain metrics (cost, CO₂, violations) not extracted | Cannot report Table III | Add env-specific metric extraction in eval loop | Low |
| `Omnisafe_Building.py` | `noise_action` arg parsed but never passed to env | Building safe RL ignores action noise | Add `noise_action=NOISE_ACT` to env constructor | Trivial |
| `Omnisafe_Cogen.py` | `print(info)` on every step | Floods logs, slows training on ARC | Remove or gate behind `--verbose` | Trivial |
| `marl_tr_co.py` | Metrics file named `building_env_metrics_` | Misleading output filename | Rename to `cogen_env_metrics_` | Trivial |
| `marl_tr_co.py` | Only 100 iterations | Insufficient for convergence | Increase to 1,000-5,000 | Config change |
| `ttp/marl_plot.py` | All code commented out | Cannot visualize MARL results | Reactivate with current paths | Low |
| All MARL envs | No domain-specific metric extraction | Cannot compute cost/CO₂/violations for MARL | Add to MARL env `info` dict | Medium |
| `stdrl_training.py` | No mechanism to batch multiple seeds | Must manually launch 5x per config | Add `--seeds 42,43,44,45,46` support | Low |
| No file exists | No baseline implementations | Fatal gap | Create `baselines.py` or per-env baseline scripts | Medium |
| No file exists | No unified results aggregation script | Cannot produce paper tables automatically | Create `aggregate_results.py` | Medium |

### 2.3 Infrastructure Gaps

| Gap | What's Needed | Why | Priority |
|-----|--------------|-----|----------|
| **Multi-seed launcher** | Script or SLURM array job that runs 5 seeds automatically | Henderson et al. (2018) requires ≥5 seeds for publishable results | P0 |
| **Results aggregation pipeline** | Script that reads all `evaluation_results.txt` / `episode_results.csv` and produces unified tables | Cannot manually collate 500+ experiments | P0 |
| **Baseline implementations** | `baselines/{donothing,rulebased,optimization}.py` per env | Reviewer's first question: "Why RL?" | P0 |
| **Domain metric extraction** | Per-env code to extract kWh, $, CO₂, violation% from env `info` | Reward alone is not a publishable metric | P0 |
| **Figure generation pipeline** | Unified plotting script that reads aggregated results and produces paper-ready figures | Manual matplotlib scripting for 5+ figures is error-prone | P1 |
| **Experiment tracker** | Spreadsheet or script tracking which {env, algo, noise, seed} combos are complete | With 500+ runs, manual tracking fails | P1 |

### 2.4 What to Cut (Scope Reduction for Feasibility)

The full experimental matrix is enormous. Here is what to cut without losing publishability:

| Cut | Saves | Risk |
|-----|-------|------|
| **Drop TD3** — keep only PPO + SAC | 33% of standard RL runs | Low — TD3 rarely used in energy literature |
| **Reduce noise levels** from 10 to 5: {0, 0.05, 0.10, 0.20, 0.40} | 50% of noise sweep runs | Low — 5 points still show degradation curve |
| **Reduce safe RL algorithms** from 15 to 3: OnCRPO, CPO, PPOLag | Huge savings | Low — 3 representative algorithms suffice |
| **Reduce safe RL cost limits** to 2: {1.0, 5.0} | 33% of safe RL runs | Low — two points define a tradeoff |
| **Single MARL algo per env** | Already the case | None |
| **Skip optimization baseline** if time-constrained | Saves implementation time | MEDIUM — nice-to-have but not fatal |

**After scope reduction, minimum experiment count:**
- Standard RL: 3 envs × 2 algos × 5 noise × 3 types × 5 seeds = 450 runs
- Safe RL: 3 envs × 3 algos × 2 limits × 5 seeds = 90 runs
- MARL: 3 envs × 1 algo × 5 seeds = 15 runs
- Baselines: 3 envs × 2 baselines × 5 seeds = 30 eval runs
- Distribution shift: 3 envs × 2 algos × 5 test noise × 5 seeds = 150 eval runs (no training)
- **Total: ~735 runs** (many are fast evaluation-only runs)

---

## 3. Strategic Plan: Step-by-Step Execution Roadmap

### 3.1 Phase 0: Infrastructure Setup (March 3-10, 2026) — 1 week

**Goal**: Build the tooling needed to run 735+ experiments efficiently.

#### Step 0.1: Multi-Seed Support
**File to modify**: `stdrl_training.py`
```
Add --seeds argument: comma-separated list (e.g., --seeds 42,43,44,45,46)
Loop over seeds in main(), launching sequential training per seed
Alternative: Create a SLURM array job launcher script
```
**Expected output**: Can launch `python stdrl_training.py --env evcharging --algo SAC --noise 0.1 --seeds 42,43,44,45,46` and get 5 independent runs.

#### Step 0.2: Experiment Tracker
**New file**: `experiment_tracker.py`
```
Scan logs_std_train/ and logs_std_test/ directories
Cross-reference against the full experiment matrix
Output: CSV showing which {env, algo, noise_type, noise_level, seed} combinations are complete
```

#### Step 0.3: Domain Metric Extraction
**File to modify**: `stdrl_testing.py`

Add per-environment metric extraction after each episode:

**EVCharging metrics**:
- `total_profit` ($) — from `info['reward_breakdown']['profit']`
- `total_carbon_cost` ($) — from `info['reward_breakdown']['carbon_cost']`
- `total_excess_charge` — from `info['reward_breakdown']['excess_charge']`
- `constraint_violation_rate` (%) — timesteps with excess_charge > 0 / total timesteps
- `demand_satisfaction_rate` (%) — energy delivered / energy demanded
- `peak_pilot_signal` (A) — max(action) per episode

**Building metrics**:
- `total_energy_consumption` (kWh) — sum of |action| * scaling_factor
- `mean_temperature_error` (°C) — mean |T_zone - T_setpoint| across zones/steps
- `max_temperature_violation` (°C) — max deviation from comfort band
- `comfort_violation_rate` (%) — timesteps where any zone > threshold / total
- `peak_hvac_demand` — max(|action|) per episode

**Cogen metrics**:
- `total_fuel_cost` ($) — from env info
- `total_ramp_cost` ($) — from env info
- `non_delivery_penalty` ($) — from env info
- `constraint_violation_count` — count of dyn_cv > 0
- `power_delivery_rate` (%) — actual_power / target_power
- `steam_delivery_rate` (%) — actual_steam / target_steam

#### Step 0.4: Bug Fixes
- Fix `Omnisafe_Building.py`: pass `noise_action` to env constructor
- Fix `Omnisafe_Cogen.py`: remove debug print statements
- Fix `marl_tr_co.py`: rename metrics filename from `building_` to `cogen_`

#### Step 0.5: SLURM Job Templates
Create SLURM array job scripts for batch experiments:
```bash
# arc_slurm_batch.sh — Example for EVCharging noise sweep
#SBATCH --array=0-49  # 2 algos × 5 noise × 5 seeds
ALGOS=(PPO SAC)
NOISES=(0.0 0.05 0.10 0.20 0.40)
SEEDS=(42 43 44 45 46)
# Compute indices from $SLURM_ARRAY_TASK_ID
```

**Deliverable**: All infrastructure ready for Phase 1 batch launches.

---

### 3.2 Phase 1: Baseline Implementation (March 10-24, 2026) — 2 weeks

**Goal**: Implement and evaluate all non-RL baselines.

#### Step 1.1: Do-Nothing Baselines

**New file**: `baselines/donothing.py`

**EVCharging Do-Nothing**: Always charge at maximum rate
```python
action = np.ones(54)  # Max pilot signal for all stations
```

**Building Do-Nothing**: Set all HVAC to midpoint (0)
```python
action = np.zeros(n_zones)  # No heating or cooling
```

**Cogen Do-Nothing**: Nominal operating point (mid-range all controls)
```python
action = (action_space.high + action_space.low) / 2  # Midpoint
```

**Expected results**: These should be the worst performers, establishing a floor.

#### Step 1.2: Rule-Based Baselines

**New file**: `baselines/rulebased.py`

**EVCharging Rule-Based — Proportional to Demand**:
```python
# Allocate charging power proportional to remaining demand
action = demands / max(demands.max(), 1e-6)  # Normalized by max demand
# Clip to respect station-level limits
action = np.clip(action, 0, 1)
```

**EVCharging Rule-Based — LLLF (Least Laxity Last First)**:
```python
# Priority: stations with least time remaining get most power
laxity = est_departures - current_timestep
# Inverse laxity gives priority (urgent stations first)
priority = 1.0 / np.maximum(laxity, 1)
action = priority / priority.sum()  # Normalize to sum=1 within network budget
```

**Building Rule-Based — Deadband Controller**:
```python
# Simple thermostat: cool if too hot, heat if too cold
for zone in range(n_zones):
    if zone_temp[zone] > setpoint + 1.0:  # Too hot
        action[zone] = -0.5  # Cool
    elif zone_temp[zone] < setpoint - 1.0:  # Too cold
        action[zone] = 0.5  # Heat
    else:
        action[zone] = 0.0  # Deadband — do nothing
```

**Cogen Rule-Based — Load Following**:
```python
# Match gas turbine output proportionally to target power
target_ratio = target_power / max_power
for gt in [GT1, GT2, GT3]:
    gt.fuel_flow = target_ratio * gt.max_fuel
    gt.igv = target_ratio * gt.max_igv
# Steam turbine follows remaining steam demand
st.valve = target_steam / max_steam
```

**Expected results**: Should beat Do-Nothing significantly. RL should beat these, but the margin matters.

#### Step 1.3: Optimization Baseline (If Time Permits)

**Priority env**: EVCharging (LP formulation is natural)

**EVCharging LP Baseline**:
```python
import cvxpy as cp
# Decision variables: pilot signals for each station
x = cp.Variable(54, nonneg=True)
# Objective: minimize carbon cost - maximize profit
# subject to: network constraints, station limits
prob = cp.Problem(
    cp.Minimize(carbon_price * cp.sum(x) - profit_per_kwh * cp.sum(x)),
    [x <= 1.0, x >= 0.0, network_constraint(x) <= capacity]
)
prob.solve()
action = x.value
```

**Note**: This requires access to future carbon prices (perfect information). The comparison to RL (which sees only forecasts) is inherently unfair but illustrative — it shows the ceiling.

#### Step 1.4: Evaluate All Baselines
```bash
# For each baseline × env × 5 seeds:
python stdrl_testing.py --env evcharging --baseline donothing --n_eval 100 --seed 42
python stdrl_testing.py --env evcharging --baseline rulebased --n_eval 100 --seed 42
# etc.
```

Alternatively, write a dedicated `evaluate_baselines.py` that wraps the eval loop without SB3.

**Deliverable**: Baseline results for all 3 envs with domain metrics.

---

### 3.3 Phase 2: Complete Standard RL Experiments (March 24 - April 21, 2026) — 4 weeks

**Goal**: Fill all gaps in the standard RL experiment matrix.

#### Step 2.1: Priority Ordering

Prioritize experiments by information value:

1. **Building + Cogen clean baselines** (PPO, SAC, noise=0, 5 seeds) — establishes performance floor
2. **Building + Cogen obs noise sweeps** (5 levels, 5 seeds) — directly extends EVCharging results
3. **EVCharging env noise sweep** (5 levels, 5 seeds) — completes the third noise channel
4. **Building + Cogen action noise sweeps** (5 levels, 5 seeds) — completes second channel
5. **Building + Cogen env noise sweeps** (5 levels, 5 seeds) — completes third channel

#### Step 2.2: SLURM Batch Submission Strategy

Given ARC HPC constraints (limited GPU nodes, queue times):

**Week 1 (March 24-31)**: Submit Building clean + obs noise jobs (highest priority)
**Week 2 (March 31-April 7)**: Submit Cogen clean + obs noise jobs; Building action noise
**Week 3 (April 7-14)**: Submit EVCharging env noise; Building env noise
**Week 4 (April 14-21)**: Submit Cogen action + env noise; any reruns for failed jobs

**Parallel utilization**: Each SLURM job takes 1-5 days. With 5 seeds per config, use SLURM array jobs to parallelize across seeds.

#### Step 2.3: Quality Checkpoints

After each batch completes:
1. Check for NaN rewards or collapsed policies
2. Verify training curves show convergence
3. Spot-check that noise level 0.0 matches previous clean results
4. Log any anomalies in experiment tracker

**Deliverable**: Complete standard RL results for Table II of the paper.

---

### 3.4 Phase 3: Safe RL & MARL Experiments (April 21 - May 12, 2026) — 3 weeks

#### Step 3.1: Safe RL Completion

**Priority**: Building and Cogen safe RL (EVCharging already extensive)

**Scope**: 3 algorithms (OnCRPO, CPO, PPOLag) × 2 cost limits (1.0, 5.0) × 5 seeds per env

**Before submitting to ARC**:
1. Verify `Omnisafe_Building.py` now passes `noise_action` correctly (Step 0.4 fix)
2. Verify `Omnisafe_Cogen.py` no longer prints on every step (Step 0.4 fix)
3. Run one quick local test per env to confirm no crashes

**Key output**: Pareto curves showing reward vs. constraint cost for each env

#### Step 3.2: MARL Completion

**Priority**: Increase Cogen MARL iterations from 100 to at least 2,000

**Modifications needed**:
- `marl_tr_co.py`: Change `stop={"training_iteration": 100}` to `2000`
- All MARL scripts: Add seed support (currently hardcoded)

**Key output**: MARL vs. single-agent comparison bars with error bars

#### Step 3.3: Distribution Shift Experiments

**Protocol**: Train-Clean-Test-Noisy (TCTN)
1. Take best clean (noise=0) model from standard RL (already trained)
2. Evaluate on noisy environments: noise = {0.05, 0.10, 0.20, 0.40}
3. No additional training needed — evaluation only

This tests whether policies trained in ideal conditions survive deployment uncertainty.

```bash
# Example: evaluate clean SAC model under obs noise
python stdrl_testing.py --env evcharging --model_path logs_std_train/evcharging_SAC/best_model.zip \
    --algo SAC --noise 0.1 --n_eval 100 --seed 42
```

**Key output**: Degradation curves for TCTN (train-clean-test-noisy) protocol

**Deliverable**: Complete safe RL + MARL + distribution shift results.

---

### 3.5 Phase 4: Analysis & Figure Generation (May 12-26, 2026) — 2 weeks

#### Step 4.1: Results Aggregation

**New file**: `aggregate_results.py`

This script should:
1. Walk `logs_std_test/` directories
2. Parse `episode_results.csv` and `evaluation_summary.txt`
3. Compute per-config statistics: mean, std, 95% CI, IQM (interquartile mean)
4. Output unified CSV: `results/unified_results.csv`

Columns: `env, algo, paradigm, noise_type, noise_level, seed, metric_name, metric_value`

#### Step 4.2: Figure Generation

**Figure 1: Benchmark Taxonomy** (draw manually or with matplotlib)
```
                    SustainRL-Bench
                   /       |        \
            EVCharging  Building   Cogen
            /    |    \
       Std RL  Safe RL  MARL
       /    \
    PPO    SAC
     |
  Noise: {obs, action, env} × {0, 0.05, 0.10, 0.20, 0.40}
```

**Figure 2: Training Curves** (3 panels, one per env)
- X-axis: training steps
- Y-axis: episode reward
- Lines: one per algorithm
- Shading: IQR across 5 seeds
- Use `ttp/stdrl_plot_*.py` as template

**Figure 3: Noise Degradation Curves** (3x3 grid)
- Rows: environments (EV, Building, Cogen)
- Columns: noise types (obs, action, env)
- X-axis: noise level
- Y-axis: normalized reward (% of clean baseline)
- Lines: one per algorithm
- Error bars: 95% CI

**Figure 4: Safe RL Pareto Fronts** (3 panels)
- X-axis: cumulative constraint cost per episode
- Y-axis: episode reward
- Points: one per algorithm × cost limit
- Pareto frontier highlighted
- Include unconstrained RL as reference point

**Figure 5: MARL vs Single-Agent** (grouped bar chart)
- Groups: environments
- Bars: single-agent best, MARL
- Error bars: 95% CI
- Color coding by paradigm

**Figure 6 (if space): Distribution Shift** (3 panels)
- X-axis: test noise level
- Y-axis: reward
- Lines: "trained at noise=X" for X in {0, 0.1, 0.2}
- Shows gap between in-distribution and out-of-distribution

#### Step 4.3: Table Generation

**Table I: Environment Specifications**

| Property | EVCharging | Building | Cogen |
|----------|-----------|----------|-------|
| Episode length | 288 | 288 | 96 |
| Obs space | Dict (146-d) | Box (n+4) | Dict (~300-d) |
| Action space | Box (54) | Box (n) | Box (15) |
| Reward range | variable | [-1, 0] | variable |
| # MARL agents | 54 | 3-6 | 4 |
| Safe RL cost signal | excess_charge | temp_violation | dyn_cv_costs |
| Data source | ACN-Data (Caltech) | ASHRAE 90.1 RC | ONNX surrogate |

**Table II: Main Results** (reward, mean ± 95% CI)

| Paradigm | Algorithm | EVCharging | Building | Cogen |
|----------|-----------|-----------|----------|-------|
| Baseline | Do-Nothing | X ± Y | X ± Y | X ± Y |
| Baseline | Rule-Based | X ± Y | X ± Y | X ± Y |
| Std RL | PPO | X ± Y | X ± Y | X ± Y |
| Std RL | SAC | X ± Y | X ± Y | X ± Y |
| Safe RL | OnCRPO (c=1) | X ± Y | X ± Y | X ± Y |
| Safe RL | PPOLag (c=1) | X ± Y | X ± Y | X ± Y |
| MARL | APPO/SAC/PPO | X ± Y | X ± Y | X ± Y |

**Table III: Domain-Specific Metrics**

| Metric | Baseline | PPO | SAC | OnCRPO | MARL |
|--------|----------|-----|-----|--------|------|
| *EVCharging* | | | | | |
| Profit ($) | X | X | X | X | X |
| CO₂ cost ($) | X | X | X | X | X |
| Violation rate (%) | X | X | X | X | X |
| *Building* | | | | | |
| Energy (kWh) | X | X | X | X | X |
| Temp error (°C) | X | X | X | X | X |
| Comfort viol. (%) | X | X | X | X | X |
| *Cogen* | | | | | |
| Fuel cost ($) | X | X | X | X | X |
| Constraint viol. | X | X | X | X | X |
| Delivery rate (%) | X | X | X | X | X |

**Table IV: Robustness — Performance Retention Under Noise**

| Env | Algo | Clean | Obs 0.1 | Obs 0.2 | Act 0.1 | Act 0.2 | Env 0.1 | Env 0.2 |
|-----|------|-------|---------|---------|---------|---------|---------|---------|
| EV | PPO | 100% | X% | X% | X% | X% | X% | X% |
| EV | SAC | 100% | X% | X% | X% | X% | X% | X% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

(Values as percentage of clean baseline reward)

**Deliverable**: All paper figures and tables generated.

---

### 3.6 Phase 5: Paper Writing (May 26 - June 12, 2026) — 2.5 weeks

#### Step 5.1: Draft Structure (Week 1)

Write first draft following the blueprint in Section 7 below. Focus on:
1. Introduction + contribution statement (C1-C5)
2. Environment descriptions (lean — point to SustainGym paper for details)
3. Experimental setup (algorithms, hyperparameters, evaluation protocol)

#### Step 5.2: Results Integration (Week 1-2)

Insert actual numbers into Tables II-IV and Figures 2-6. Write analysis paragraphs:
- One paragraph per major finding
- Each finding should reference specific table/figure entries
- Avoid over-interpreting noise in results

#### Step 5.3: Related Work + Discussion (Week 2)

Position against literature (Section 1.4 of this document). Discuss:
- When does RL beat baselines? (Finding from N5)
- Is the safe RL cost worth it? (Finding from N3)
- When does MARL help? (Finding from N4)
- Which noise type matters most? (Finding from N2)

#### Step 5.4: Revision (Week 2.5)

- Cut to 6 pages (+ 2-page appendix if needed)
- Verify all claims are supported by data
- Run IEEE template formatting check
- Internal review by advisor

**Deliverable**: Submission-ready paper.

---

### 3.7 Phase 6: Finalization (June 12-19, 2026) — 1 week

- Final proofreading
- Verify all figures are 300 DPI, Times New Roman labels
- Prepare supplementary material (code link, extended tables)
- Upload to IEEE submission system
- **Submit by June 19, 2026**

---

## 4. Experiment Design: Detailed Protocols

### 4.1 Standard RL Evaluation Protocol

```
INPUT: Trained model M, environment E, noise config N, number of episodes K=100
OUTPUT: Per-episode metrics

FOR seed in [42, 43, 44, 45, 46]:
    FOR noise_type in [obs, action, env]:
        FOR noise_level in [0.0, 0.05, 0.10, 0.20, 0.40]:
            env = make_env(E, noise_type=noise_type, noise_level=noise_level)
            rewards, metrics = evaluate(M, env, n_episodes=K)
            RECORD: {env, algo, noise_type, noise_level, seed,
                     mean_reward, std_reward, domain_metrics}
```

### 4.2 Safe RL Evaluation Protocol

```
INPUT: CMDP algorithm A, environment E, cost limit C
OUTPUT: Reward + cost metrics

FOR seed in [42, 43, 44, 45, 46]:
    FOR cost_limit in [1.0, 5.0]:
        model = train_omnisafe(A, E, cost_limit=C, seed=seed)
        rewards, costs, metrics = evaluate_safe(model, E, n_episodes=K)
        RECORD: {env, algo, cost_limit, seed,
                 mean_reward, mean_cost, cost_violation_rate,
                 domain_metrics}

ANALYSIS:
    - Plot Pareto front: (mean_cost, mean_reward) for each (algo, cost_limit)
    - Compute cost reduction vs. unconstrained: (cost_unconstrained - cost_safe) / cost_unconstrained
    - Compute reward retention: reward_safe / reward_unconstrained
```

### 4.3 MARL Evaluation Protocol

```
INPUT: MARL policy P, environment E
OUTPUT: Per-episode metrics comparable to single-agent

FOR seed in [42, 43, 44, 45, 46]:
    env = make_multiagent_env(E)
    model = train_rllib(P, env, seed=seed, iterations=N)

    # Evaluate using same metrics as single-agent
    rewards, metrics = evaluate_marl(model, env, n_episodes=K)

    # Additionally report:
    # - Per-agent reward variance (coordination quality)
    # - Training wall-clock time vs. single-agent
    # - Communication overhead (if applicable)

    RECORD: {env, algo, seed, mean_reward, per_agent_variance,
             training_time, domain_metrics}

COMPARISON:
    - Direct bar comparison vs. best single-agent (same env)
    - Statistical significance test (Welch's t-test or bootstrap)
```

### 4.4 Distribution Shift Protocol (Train-Clean-Test-Noisy)

```
INPUT: Best clean model M_clean (trained at noise=0), environment E
OUTPUT: Degradation profile

FOR seed in [42, 43, 44, 45, 46]:
    FOR noise_type in [obs, action, env]:
        FOR test_noise in [0.0, 0.05, 0.10, 0.20, 0.40]:
            env = make_env(E, noise_type=noise_type, noise_level=test_noise)
            rewards = evaluate(M_clean, env, n_episodes=100)
            retention = mean(rewards) / clean_baseline_reward
            RECORD: {env, algo, noise_type, test_noise, seed, retention}

ANALYSIS:
    - Plot retention curves (y: % of clean performance, x: test noise)
    - Compare to models trained WITH noise (in-distribution results)
    - Highlight "robustness gap": in-distribution vs. out-of-distribution
```

### 4.5 Noise Decomposition Ablation

```
INPUT: Environment E, Algorithm A
OUTPUT: Independent vs. joint noise analysis

# Phase A: Independent noise channels
FOR noise_type in [obs, action, env]:
    FOR noise_level in [0.0, 0.05, 0.10, 0.20, 0.40]:
        train and evaluate with ONLY this noise type active

# Phase B: Joint noise (all channels active simultaneously)
FOR noise_level in [0.0, 0.05, 0.10, 0.20, 0.40]:
    train and evaluate with ALL three noise types at same level

ANALYSIS:
    - Compare: reward_drop(obs_only) vs. reward_drop(act_only) vs. reward_drop(env_only)
    - Check additivity: does joint_drop ≈ obs_drop + act_drop + env_drop?
    - Or is there interaction: joint_drop >> sum of individual drops?
    - This reveals whether noise effects are independent or compounding
```

---

## 5. Baseline Implementation Guide

### 5.1 Architecture

Create a unified baseline evaluation framework:

```
baselines/
├── __init__.py
├── base.py           # BasePolicy class with evaluate() method
├── donothing.py      # Do-Nothing policies per env
├── rulebased.py      # Rule-based policies per env
├── optimization.py   # LP/MPC policies (optional)
└── evaluate.py       # CLI entry point for baseline evaluation
```

### 5.2 Base Class

```python
class BaselinePolicy:
    """Base class for non-RL baseline policies."""

    def __init__(self, env_name: str):
        self.env_name = env_name

    def predict(self, obs) -> np.ndarray:
        """Return action given observation. Override in subclasses."""
        raise NotImplementedError

    def evaluate(self, env, n_episodes=100, seeds=None):
        """Evaluate policy on environment, return per-episode metrics."""
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]
        all_results = []
        for seed in seeds:
            env.seed(seed)
            for ep in range(n_episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_metrics = {}
                done = False
                while not done:
                    action = self.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                all_results.append({
                    'seed': seed, 'episode': ep,
                    'reward': episode_reward,
                    **extract_domain_metrics(info, self.env_name)
                })
        return pd.DataFrame(all_results)
```

### 5.3 Detailed Do-Nothing Implementations

**EVCharging Do-Nothing — Max Charging**:
```python
class EVChargingDoNothing(BaselinePolicy):
    def predict(self, obs):
        # Charge every station at maximum rate
        return np.ones(54)  # All stations at max
```
Rationale: This maximizes immediate charging but ignores carbon costs and network constraints. Expected to have high profit but high carbon cost and high excess charge.

**EVCharging Do-Nothing — Zero Charging**:
```python
class EVChargingZero(BaselinePolicy):
    def predict(self, obs):
        return np.zeros(54)  # No charging at all
```
Rationale: Establishes absolute floor. Zero profit, zero carbon, zero violations.

**Building Do-Nothing — Off**:
```python
class BuildingDoNothing(BaselinePolicy):
    def predict(self, obs):
        return np.zeros(n_zones)  # HVAC off
```
Rationale: Zero energy consumption but maximum temperature discomfort.

**Cogen Do-Nothing — Shutdown**:
```python
class CogenDoNothing(BaselinePolicy):
    def predict(self, obs):
        return np.zeros(15)  # All controls at minimum
```
Rationale: Minimal fuel cost but zero power/steam delivery.

### 5.4 Detailed Rule-Based Implementations

**EVCharging — LLLF (Least Laxity Last First)**:
```python
class EVLLLF(BaselinePolicy):
    def predict(self, obs):
        # obs is a dict: {timestep, est_departures, demands, prev_moer, forecasted_moer}
        timestep = obs['timestep']
        departures = obs['est_departures']
        demands = obs['demands']

        # Laxity = time remaining before departure
        laxity = departures - timestep
        laxity = np.maximum(laxity, 1)  # Avoid division by zero

        # Urgency = demand / time remaining (higher = more urgent)
        urgency = demands / laxity

        # Normalize to [0, 1]
        if urgency.max() > 0:
            action = urgency / urgency.max()
        else:
            action = np.zeros(54)

        return action
```

**Building — PID-like Thermostat**:
```python
class BuildingThermostat(BaselinePolicy):
    def __init__(self, setpoint=22.0, deadband=1.0, gain=0.5):
        super().__init__('building')
        self.setpoint = setpoint
        self.deadband = deadband
        self.gain = gain

    def predict(self, obs):
        zone_temps = obs[:n_zones]  # First n values are zone temperatures
        error = zone_temps - self.setpoint
        action = np.zeros(n_zones)
        for i in range(n_zones):
            if error[i] > self.deadband:    # Too hot
                action[i] = -self.gain       # Cool
            elif error[i] < -self.deadband:  # Too cold
                action[i] = self.gain        # Heat
            # else: within deadband, do nothing
        return np.clip(action, -1, 1)
```

**Cogen — Load Following**:
```python
class CogenLoadFollowing(BaselinePolicy):
    def predict(self, obs):
        # Extract target power and steam from obs
        target_power = obs['Target_Power'][0]
        target_steam = obs['Target_Steam'][0]

        # Equal load sharing across 3 gas turbines
        gt_load = target_power / (3 * max_gt_power)
        gt_load = np.clip(gt_load, 0.3, 1.0)  # Min load 30%

        # Set each GT: [fuel_flow, igv, speed, on/off]
        gt_action = [gt_load, gt_load, 0.5, 1.0]  # On, proportional
        st_action = [target_steam / max_steam]       # Steam valve

        action = np.concatenate([gt_action]*3 + [st_action, [0.5, 0.5]])
        return np.clip(action, 0, 1)
```

---

## 6. Evaluation & Reporting Framework

### 6.1 Statistical Methodology

Following Henderson et al. (2018) and Agarwal et al. (2021):

**Required statistics per experiment configuration**:
1. **Mean** across seeds and episodes
2. **Standard deviation** across seeds
3. **95% Confidence Interval**: bootstrap with 10,000 resamples
4. **Interquartile Mean (IQM)**: mean of the middle 50% of runs (robust to outliers)

**Code snippet for bootstrap CI**:
```python
import numpy as np

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(data), lower, upper

def interquartile_mean(data):
    """Compute IQM (mean of middle 50%)."""
    q25, q75 = np.percentile(data, [25, 75])
    mask = (data >= q25) & (data <= q75)
    return np.mean(data[mask])
```

**Significance testing** (for pairwise comparisons):
- Use Welch's t-test (unequal variance) for comparing two algorithms
- Report p-values; consider significant at p < 0.05
- For multiple comparisons, apply Bonferroni correction

### 6.2 Normalized Performance Metrics

To compare across environments with different reward scales:

**Performance Retention** (for noise robustness):
```
retention(noise) = reward(noise) / reward(clean) × 100%
```

**Performance Improvement over Baseline**:
```
improvement = (reward_RL - reward_baseline) / |reward_baseline| × 100%
```

**Safe RL Efficiency**:
```
constraint_reduction = (cost_unconstrained - cost_safe) / cost_unconstrained × 100%
reward_retention = reward_safe / reward_unconstrained × 100%
```

**MARL Gain**:
```
marl_gain = (reward_MARL - reward_single_agent) / |reward_single_agent| × 100%
```

### 6.3 Visualization Standards

All figures should follow IEEE publication standards:

- **Font**: Times New Roman or similar serif font, minimum 8pt in figures
- **Resolution**: 300 DPI minimum for raster; prefer vector (PDF/SVG)
- **Colors**: Use colorblind-friendly palette (e.g., from Seaborn's "colorblind")
- **Line styles**: Vary both color AND line style (solid, dashed, dotted) for accessibility
- **Error representation**: Shaded regions for training curves; error bars for bar charts
- **Labels**: Include units on all axes; use consistent naming across figures
- **Legend**: Place outside plot area if it obscures data

### 6.4 Figure Specifications

**Figure 1: Benchmark Taxonomy (Full-Page Width)**
- Type: Hierarchical diagram
- Content: 3 environments → 3 paradigms each → noise axes → algorithms
- Style: Boxes with rounded corners, color-coded by environment
- Tools: Draw with matplotlib, TikZ, or draw.io

**Figure 2: Training Curves (3-Panel, Full-Width)**
- Layout: 3 subplots side-by-side (EV | Building | Cogen)
- X-axis: Training timesteps
- Y-axis: Mean episode reward (smoothed, window=10 episodes)
- Lines: PPO (blue solid), SAC (red dashed), Do-Nothing (gray dotted), Rule-Based (green dash-dot)
- Shading: IQR across 5 seeds (alpha=0.2)
- Each subplot: ~4 lines × 2 baselines = 4 curves

**Figure 3: Noise Degradation (3x3 Grid, Full-Width)**
- Layout: 3 rows (EV, Building, Cogen) × 3 columns (obs, action, env noise)
- X-axis: Noise level [0, 0.05, 0.10, 0.20, 0.40]
- Y-axis: Performance retention (% of clean baseline)
- Lines: PPO, SAC
- Error bars: 95% CI
- Horizontal reference line at 100% (clean performance)
- Key insight callout box on most affected quadrant

**Figure 4: Safe RL Pareto Fronts (3-Panel, Full-Width)**
- Layout: 3 subplots (EV | Building | Cogen)
- X-axis: Mean episode constraint cost
- Y-axis: Mean episode reward
- Points: (cost, reward) for each (algorithm × cost_limit)
- Markers: Different shape per algorithm (circle, triangle, square)
- Connect Pareto-optimal points with dashed line
- Star marker: unconstrained RL reference point

**Figure 5: Cross-Paradigm Comparison (Grouped Bar Chart, Full-Width)**
- Groups: 3 environments
- Bars per group: Do-Nothing, Rule-Based, PPO, SAC, OnCRPO, MARL
- Y-axis: Normalized reward (% of best achievable)
- Error bars: 95% CI
- Color code: gray (baselines), blue (std RL), orange (safe RL), green (MARL)

---

## 7. Paper Writing Blueprint

### 7.1 Title Options

1. "SustainRL-Bench: A Multi-Paradigm Benchmark for Reinforcement Learning in Sustainable Energy Systems"
2. "Benchmarking Standard, Safe, and Multi-Agent RL Across Sustainable Energy Domains"
3. "How Robust is Energy RL? A Cross-Domain Benchmark of Standard, Safe, and Multi-Agent Approaches"

**Recommendation**: Option 1 — gives the benchmark a name (important for citations) and is descriptive.

### 7.2 Abstract Template (150 words)

> Reinforcement learning (RL) is increasingly applied to energy system control, yet evaluations are fragmented across isolated environments, algorithms, and settings. We present SustainRL-Bench, the first benchmark systematically evaluating standard RL, constrained (safe) RL, and multi-agent RL across three diverse energy domains: EV charging, building HVAC, and cogeneration. Our evaluation protocol introduces a structured noise decomposition framework that independently varies observation, action, and environment uncertainty to assess policy robustness. Across [X] experiments spanning [Y] algorithm configurations and [Z] seeds, we find that: (1) [key finding about algorithm rankings], (2) [key finding about noise vulnerability], (3) [key finding about safe RL tradeoffs], and (4) [key finding about MARL benefits]. We release our benchmark suite, including training scripts, evaluation tools, and pre-trained models, to facilitate reproducible research in energy RL. Code available at [URL].

### 7.3 Section-by-Section Writing Guide

#### I. Introduction (0.75 pages)

**Paragraph 1**: Motivation (3-4 sentences)
- Energy systems are critical for sustainability
- RL has shown promise but evaluations are fragmented
- No unified benchmark compares paradigms across domains

**Paragraph 2**: Problem statement (3-4 sentences)
- Define the three axes: paradigm (std/safe/MARL), domain (EV/building/cogen), robustness (obs/action/env noise)
- State the research questions (Q1-Q5 from previous report)

**Paragraph 3**: Contributions (5 bullet points)
- C1-C5 from Section 1.3

**Paragraph 4**: Paper organization (2 sentences)
- "The remainder of this paper is organized as follows..."

#### II. Related Work (0.75 pages)

Organize into 4 sub-paragraphs:
1. RL for energy systems (cite [B1-B4, EV1-EV4, CG1, MG1-MG2])
2. Safe RL and constrained optimization (cite [SR1-SR7])
3. MARL for energy (cite [MA1-MA6])
4. RL benchmarks and robustness (cite [BF1-BF6, RB1-RB5])

Close with: "Unlike prior work that addresses one domain or one paradigm, we provide a unified evaluation across all three."

#### III. Problem Formulation & Environments (1.0 page)

**III-A**: MDP formulation (standard RL): (S, A, P, R, gamma)
**III-B**: CMDP formulation (safe RL): (S, A, P, R, C, d, gamma) where C is cost, d is limit
**III-C**: Dec-POMDP formulation (MARL): per-agent obs, shared reward
**III-D**: Noise model formalization:
- Define obs noise: o_noisy = o + N(0, sigma_o * |o|)
- Define action noise: a_noisy = clip(a + N(0, sigma_a), a_low, a_high)
- Define env noise: s' = f(s, a; theta + N(0, sigma_e))
**III-E**: Environment descriptions (brief — point to SustainGym paper)
- Table I: Environment specifications (see Section 4.3 above)

#### IV. Experimental Setup (0.75 pages)

**IV-A**: Algorithms and hyperparameters
- Standard RL: PPO, SAC (Table of key hyperparameters per env)
- Safe RL: OnCRPO, CPO, PPOLag (cost limits: 1.0, 5.0)
- MARL: APPO (EV), SAC (Building), PPO (Cogen)
- Baselines: Do-Nothing, Rule-Based (describe briefly)

**IV-B**: Evaluation protocol
- 5 seeds, 100 eval episodes per config
- Bootstrap 95% CIs
- Domain-specific metrics (brief description)
- Noise levels: {0, 0.05, 0.10, 0.20, 0.40}

**IV-C**: Compute infrastructure
- Virginia Tech ARC HPC, A100/H200 GPUs
- Total training compute: [X] GPU-hours
- Frameworks: SB3, OmniSafe, Ray RLlib, PettingZoo

#### V. Results & Analysis (2.0 pages)

**V-A**: Standard RL Performance (0.5 pages)
- Table II: main results
- Figure 2: training curves
- Key finding: which algorithm performs best per env

**V-B**: Noise Robustness (0.5 pages)
- Table IV: performance retention under noise
- Figure 3: degradation curves
- Key finding: which noise type is most damaging per env

**V-C**: Safe RL Analysis (0.4 pages)
- Figure 4: Pareto fronts
- Key finding: constraint reduction vs. reward cost

**V-D**: Multi-Agent RL (0.3 pages)
- Figure 5: MARL vs. single-agent
- Key finding: when does MARL help?

**V-E**: Cross-Paradigm Insights (0.3 pages)
- Synthesize findings across V-A through V-D
- "Practitioner's guide": which paradigm to use when

#### VI. Conclusion (0.25 pages)
- Summarize key findings
- Limitations (env fidelity, algorithm coverage)
- Future work (sim-to-real, more algorithms, larger scales)

### 7.4 Review Anticipation: Likely Reviewer Questions

| Question | Our Answer |
|----------|-----------|
| "Why not test more algorithms?" | We chose representative algorithms from each family. PPO (on-policy), SAC (off-policy), OnCRPO/CPO (constrained). Extending is future work. |
| "Why use SustainGym and not CityLearn?" | SustainGym covers 3 diverse domains (EV, building, cogen); CityLearn is buildings-only. Our contribution is the cross-domain protocol, not the environments. |
| "The environments are not new" | Correct. Our contribution is the benchmark protocol and findings, not the environments. We cite SustainGym properly. |
| "Why not include MPC baseline?" | We include rule-based baselines that establish RL's advantage. MPC requires perfect-information assumptions; the comparison is inherently unfair. We discuss this limitation. |
| "5 seeds is not enough" | 5 seeds with bootstrap CIs is standard practice per Henderson et al. (AAAI 2018). We also report IQM per Agarwal et al. (NeurIPS 2021). |
| "What about sim-to-real transfer?" | Acknowledged as limitation. Our noise robustness analysis addresses part of this concern by measuring policy degradation under realistic uncertainty. |
| "What is the practical impact?" | Our findings provide actionable guidance: e.g., "use SAC over PPO for actuator-noisy environments" or "safe RL reduces violations by X% at Y% reward cost." |

---

## 8. Risk Assessment & Contingency Plans

### 8.1 Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| ARC HPC queue delays | HIGH | Cannot complete experiments | Submit jobs early; use preemptable partition; prioritize highest-value experiments |
| Building/Cogen training diverges | MEDIUM | Missing env results | Tune hyperparameters on quick-test first; have backup hyperparameter configs |
| Safe RL (OmniSafe) crashes on Building/Cogen | MEDIUM | Missing safe RL results | Test locally first with small config; fall back to PPOLag (most stable) |
| MARL Cogen doesn't converge in 2000 iterations | MEDIUM | Weak MARL results for Cogen | Report partial results with honest discussion; focus MARL claim on EV + Building |
| Insufficient time for paper writing | MEDIUM | Poor writing quality | Start writing introduction and related work NOW (Phase 0-1); don't wait for all results |
| Advisor review delays | LOW | Late submission | Share drafts incrementally; don't wait for "perfect" version |

### 8.2 Result Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| RL doesn't beat rule-based baseline | LOW-MEDIUM | Undermines motivation | Report honestly; RL may still win under noise; frame as "when is RL worth the complexity?" |
| Safe RL shows negligible improvement | MEDIUM | Weak C3 claim | Try more cost limits; if genuinely no benefit, report as finding: "unconstrained RL already satisfies constraints on [env]" |
| MARL is worse than single-agent everywhere | MEDIUM | Weak C4 claim | Frame as finding: "naive agent decomposition hurts; careful decomposition needed" — this is actually valuable |
| Noise has no effect on any algorithm | LOW | Undermines C2 claim | Very unlikely — but if so, report as "energy RL policies are surprisingly robust" |
| Results are inconsistent across seeds | MEDIUM | Wide CIs, no clear conclusions | Use IQM (robust to outliers); increase to 10 seeds if feasible; focus analysis on consistent trends |

### 8.3 Scope Contingency Plan

If compute budget is insufficient for all experiments:

**Tier 1 (absolute minimum — still publishable)**:
- Standard RL: 3 envs × 2 algos × clean only × 5 seeds (30 runs)
- Baselines: 3 envs × 2 baselines × 5 seeds (30 eval runs)
- Noise robustness: obs noise only, 3 levels, on EVCharging and Building (60 runs)
- Safe RL: EVCharging only, 2 algorithms, 2 cost limits (20 runs)
- MARL: EVCharging only (5 runs)
- **Total: ~145 runs** — publishable but weaker

**Tier 2 (target — strong submission)**:
- Full matrix as described in Section 2.1
- **Total: ~735 runs**

**Tier 3 (aspirational — top-tier submission)**:
- Full matrix + optimization baselines + 10 seeds + distribution shift + all noise combinations
- **Total: ~2000+ runs**

---

## Appendix: Complete Experiment Checklist

### Checklist for Paper Readiness

- [ ] **Infrastructure**
  - [ ] Multi-seed support added to training scripts
  - [ ] Experiment tracker created
  - [ ] Domain metric extraction added to `stdrl_testing.py`
  - [ ] Bug fixes applied (Omnisafe_Building, Omnisafe_Cogen, marl_tr_co)
  - [ ] SLURM array job templates created
  - [ ] Results aggregation script created

- [ ] **Baselines**
  - [ ] Do-Nothing implemented for EVCharging
  - [ ] Do-Nothing implemented for Building
  - [ ] Do-Nothing implemented for Cogen
  - [ ] Rule-Based implemented for EVCharging (LLLF)
  - [ ] Rule-Based implemented for Building (Thermostat)
  - [ ] Rule-Based implemented for Cogen (Load Following)
  - [ ] All baselines evaluated with 5 seeds

- [ ] **Standard RL**
  - [ ] EVCharging PPO/SAC: clean + obs/act/env noise × 5 seeds
  - [ ] Building PPO/SAC: clean + obs/act/env noise × 5 seeds
  - [ ] Cogen PPO/SAC: clean + obs/act/env noise × 5 seeds

- [ ] **Safe RL**
  - [ ] EVCharging: OnCRPO + CPO + PPOLag × 2 cost limits × 5 seeds
  - [ ] Building: OnCRPO + CPO + PPOLag × 2 cost limits × 5 seeds
  - [ ] Cogen: OnCRPO + CPO + PPOLag × 2 cost limits × 5 seeds

- [ ] **MARL**
  - [ ] EVCharging: APPO × sufficient iterations × 5 seeds
  - [ ] Building: SAC × sufficient iterations × 5 seeds
  - [ ] Cogen: PPO × sufficient iterations × 5 seeds

- [ ] **Analysis**
  - [ ] Distribution shift experiments (TCTN protocol)
  - [ ] Noise decomposition ablation
  - [ ] Results aggregated into unified CSV
  - [ ] Bootstrap CIs computed for all metrics
  - [ ] Cross-paradigm comparison table populated

- [ ] **Figures & Tables**
  - [ ] Figure 1: Benchmark taxonomy
  - [ ] Figure 2: Training curves (3 panels)
  - [ ] Figure 3: Noise degradation (3×3 grid)
  - [ ] Figure 4: Safe RL Pareto fronts (3 panels)
  - [ ] Figure 5: Cross-paradigm bar chart
  - [ ] Table I: Environment specifications
  - [ ] Table II: Main results (reward ± CI)
  - [ ] Table III: Domain-specific metrics
  - [ ] Table IV: Robustness analysis

- [ ] **Paper**
  - [ ] Introduction + contributions drafted
  - [ ] Related work completed
  - [ ] Problem formulation (MDP, CMDP, Dec-POMDP, noise model)
  - [ ] Experimental setup documented
  - [ ] Results + analysis written
  - [ ] Discussion + conclusion
  - [ ] Formatted to IEEE 6-page template
  - [ ] Internal review by advisor
  - [ ] Submitted by June 19, 2026

---

*Deep Dive Report v2.0 — Generated March 3, 2026*
*Consistent with CLAUDE.md project context and previous smartgridcomm_benchmark_report.md*
