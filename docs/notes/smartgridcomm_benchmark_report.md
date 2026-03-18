# SmartGridComm Benchmark Paper: Feasibility Assessment & Strategic Plan

**Project**: Benchmarking RL for Sustainable Energy Systems on SustainGym
**Author**: Mehmet Koruturk, Virginia Tech (RoleLab)
**Target Venue**: IEEE SmartGridComm 2026 (Submission Deadline: June 19, 2026)
**Date**: March 3, 2026

---

## Table of Contents

1. [Publication Feasibility Assessment](#1-publication-feasibility-assessment)
2. [Literature Review](#2-literature-review)
3. [Gap Analysis](#3-gap-analysis)
4. [Strategic Plan for a Strong Paper](#4-strategic-plan-for-a-strong-paper)
5. [Novelty & Contributions](#5-novelty--contributions)
6. [Recommended Paper Structure](#6-recommended-paper-structure)

---

## 1. Publication Feasibility Assessment

### 1.1 SmartGridComm Venue Profile

IEEE SmartGridComm (International Conference on Communications, Control, and Computing Technologies for Smart Grids) is a mid-tier IEEE conference indexed in IEEE Xplore. Key characteristics:

- **Format**: 6-page double-column IEEE format (+ optional 2-page extension)
- **Scope**: Communications, control, and computing for smart grids
- **Acceptance rate**: Approximately 30-40% (competitive but not top-tier ML venue)
- **Review criteria**: Technical rigor, relevance to smart grid, clarity, novelty, reproducibility
- **Audience**: Power systems + communications + computing intersection community
- **Submission deadline for 2026**: June 19, 2026

### 1.2 Fit Assessment: Is 3 Envs x {RL, Safe RL, MARL} Enough?

**Short answer**: The scope is sufficient for SmartGridComm, but the current execution has critical gaps that must be addressed.

#### What Is Strong

| Strength | Why It Matters |
|----------|---------------|
| **Three diverse environments** (EV charging, building HVAC, cogeneration) | Covers heterogeneous energy domains—far broader than single-env papers |
| **Three evaluation axes** (standard RL, safe RL, MARL) | Multi-dimensional comparison is rare; most papers do only one axis |
| **Noise robustness evaluation** | Directly addresses the sim-to-real gap, a known challenge in energy RL |
| **Production-quality codebase** | VecNormalize sync, proper eval callbacks, SLURM scripts—demonstrates reproducibility |
| **Realistic environments** | Based on ACN-Data, EnergyPlus RC models, ONNX surrogate—not toy problems |
| **Multiple frameworks** (SB3 + OmniSafe + Ray RLlib) | Shows breadth; demonstrates interoperability |

#### What Is Weak / At Risk

| Weakness | Impact | Fix Difficulty |
|----------|--------|---------------|
| **No non-RL baselines** (rule-based, MPC, MILP) | Reviewers will ask "why is RL better than a thermostat?" | Medium |
| **Inconsistent experimental completeness** | EVCharging has extensive results; Building/Cogen have gaps | Medium-High |
| **No statistical rigor** (seeds, CIs) documented | Results without variance are unpublishable | Low |
| **No domain-specific metrics** beyond reward | Must report kWh, CO₂, temperature violation %, constraint satisfaction rate | Low-Medium |
| **Noise sweep only on obs/action, env noise incomplete** | Env noise (distribution shift) is the most practically relevant | Medium |
| **No cross-axis comparison** | Standard RL vs. Safe RL vs. MARL on same env not explicitly compared | Medium |
| **SustainGym itself is published** (NeurIPS 2023) | Cannot claim environments as novel; must claim the benchmark protocol as novel | N/A |

### 1.3 Verdict

**Publishable with targeted improvements**. The project has the right scope and engineering quality. The main risks are: (1) missing non-RL baselines, (2) incomplete experiments across all 3 envs, and (3) insufficient statistical reporting. These are all fixable before the June 19 deadline.

---

## 2. Literature Review

### 2.1 RL for Energy Management

#### EV Charging

| Ref | Citation | Summary |
|-----|----------|---------|
| [EV1] | Wan, Z., Li, H., He, H., Prokhorov, D. "Model-free real-time EV charging scheduling based on deep reinforcement learning." *IEEE Trans. Smart Grid*, 10(5):5246-5257, 2019. | DQN-based real-time EV scheduling without arrival/departure knowledge. Key single-agent EV baseline. |
| [EV2] | Qian, T., Shao, C., Wang, X., Shahidehpour, M. "Deep reinforcement learning for EV charging navigation by coordinating smart grid and intelligent transportation system." *IEEE Trans. Smart Grid*, 10(6):6545-6556, 2019. | Joint EV routing + charging coordination with grid constraints. |
| [EV3] | Lee, Z.J., Li, T., Low, S.H. "ACN-Data: Analysis and Applications of an Open EV Charging Dataset." *Proc. ACM e-Energy*, 2019. | **Must-cite**: The ACN-Data dataset that underlies EVChargingEnv. |
| [EV4] | Lee, Z.J., Sharma, G., Johansson, D., Low, S.H. "ACN-Sim: An Open-Source Simulator for Data-Driven EV Charging Research." *IEEE SmartGridComm*, 2020. | **Must-cite**: ACN-Sim backend for EVChargingEnv. Published at our target venue. |

#### Building Energy Management

| Ref | Citation | Summary |
|-----|----------|---------|
| [B1] | Vazquez-Canteli, J.R., Nagy, Z. "Reinforcement learning for demand response: A review of algorithms and modeling techniques." *Applied Energy*, 235:1072-1089, 2019. | Comprehensive RL for demand response survey. Foundational. |
| [B2] | Yu, L., Xu, Z., Zhao, P., Liu, T., Yue, D. "Multi-agent deep reinforcement learning for HVAC control in commercial buildings." *IEEE Trans. Smart Grid*, 12(1):407-419, 2021. | Multi-agent PPO/DDPG for zone-level HVAC. Direct parallel to your Building MARL. |
| [B3] | Zhang, Z., Chong, A., Pan, Y., Zhang, C., Lam, K.P. "Whole building energy model for HVAC optimal control: A practical framework based on DRL." *Energy and Buildings*, 199:472-490, 2019. | DDPG for EnergyPlus HVAC control. |
| [B4] | Brandi, S., Piscitelli, M.S., Martellacci, M., Capozzoli, A. "Deep reinforcement learning to optimise indoor temperature control and heating energy consumption in buildings." *Energy and Buildings*, 224:110225, 2020. | SAC/PPO comparison for building thermal comfort. |

#### Cogeneration / Industrial Energy

| Ref | Citation | Summary |
|-----|----------|---------|
| [CG1] | Perez, C.F. et al. "Deep reinforcement learning control of a cogeneration plant." *IFAC-PapersOnLine*, 54(6):354-359, 2021. | RL for CHP plant control—motivates the Cogen domain. |

#### Microgrid / General

| Ref | Citation | Summary |
|-----|----------|---------|
| [MG1] | Francois-Lavet, V., Taralla, D., Ernst, D., Fonteneau, R. "Deep Reinforcement Learning Solutions for Energy Microgrids Management." *European Workshop on RL (EWRL)*, 2016. | Foundational deep RL for microgrid management. |
| [MG2] | Duan, J. et al. "Deep-Reinforcement-Learning-Based Autonomous Voltage Control for Power Grid Operations." *IEEE Trans. Power Systems*, 36(1):266-276, 2021. | DRL for voltage control with safety constraints. |

### 2.2 Safe RL in Energy Systems

| Ref | Citation | Summary |
|-----|----------|---------|
| [SR1] | Garcia, J., Fernandez, F. "A Comprehensive Survey on Safe Reinforcement Learning." *JMLR*, 16(1):1437-1480, 2015. | Foundational safe RL survey defining the problem space. |
| [SR2] | Altman, E. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999. | **Must-cite**: CMDP theory underlying OmniSafe algorithms. |
| [SR3] | Achiam, J., Held, D., Tamar, A., Abbeel, P. "Constrained Policy Optimization." *ICML*, 2017. | CPO: foundational constrained RL algorithm used in OmniSafe. |
| [SR4] | Ray, A., Achiam, J., Amodei, D. "Benchmarking Safe Exploration in Deep Reinforcement Learning." *arXiv:1910.01708*, 2019. | Safety Gym: motivates structured safe RL benchmarking. |
| [SR5] | Ji, J. et al. "OmniSafe: An Infrastructure for Accelerating Safe Reinforcement Learning Research." *arXiv:2305.09304*, 2023. | **Must-cite**: The OmniSafe framework you use. |
| [SR6] | Tessler, C., Mankowitz, D.J., Mannor, S. "Reward Constrained Policy Optimization." *ICLR*, 2019. | RCPO: Lagrangian-based constrained RL in OmniSafe. |
| [SR7] | Xu, X., Jain, R., Shi, Y. "Safe Reinforcement Learning for Power Systems: A Survey." *arXiv:2302.08486*, 2023. | Recent survey on safe RL specifically in power systems. Very close to your Safe RL axis. |

### 2.3 MARL in Energy Systems

| Ref | Citation | Summary |
|-----|----------|---------|
| [MA1] | Ye, Y. et al. "Model-Free Real-Time Autonomous Control for a Residential Multi-Energy System Using Deep RL." *IEEE Trans. Smart Grid*, 11(4):3401-3412, 2020. | Decentralized deep RL for residential multi-energy systems. |
| [MA2] | Pigott, A., Crozier, C., Baker, K., Nagy, Z. "GridLearn: Multiagent RL for Grid-Aware Building Energy Management." *Electric Power Systems Research*, 213:108521, 2022. | MARL for grid-aware building control with PettingZoo-style API. |
| [MA3] | Wang, H. et al. "Multi-Agent Deep RL for Smart Charging of EV Fleets." *IEEE Trans. Smart Grid*, 14(2):1385-1396, 2023. | Multi-agent EV fleet charging comparable to your 54-agent EVCharging MARL. |
| [MA4] | Zhang, K., Yang, Z., Basar, T. "Multi-Agent Reinforcement Learning: A Selective Overview." *arXiv:1911.10635*, 2021. | MARL theory survey. Essential background. |
| [MA5] | Lowe, R. et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NeurIPS*, 2017. | MADDPG: influential cooperative MARL algorithm. |
| [MA6] | Qiu, D. et al. "Scalable Coordinated Management of Peer-to-Peer Energy Trading: A MADRL Approach." *Applied Energy*, 285:116390, 2021. | Scalability of MARL for energy—relevant to your 54-agent scaling. |

### 2.4 Benchmarking Frameworks for Energy RL

| Ref | Citation | Summary | Relationship |
|-----|----------|---------|-------------|
| [BF1] | Yeh, C. et al. "SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems." *NeurIPS Datasets & Benchmarks*, 2023. | **The SustainGym paper**—your base environments. Must-cite. | We extend with multi-axis benchmarking protocol |
| [BF2] | Vazquez-Canteli, J.R. et al. "CityLearn v1.0: An OpenAI Gym Environment for Demand Response with Deep RL." *BuildSys*, 2019. | Leading competing benchmark for building MARL. | We cover more domains (EV, Cogen) + safe RL axis |
| [BF3] | Nweye, K. et al. "CityLearn v2: Energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities." *JBPS*, 2024. | Updated CityLearn with expanded capabilities. | They focus on buildings only; we cover 3 domains |
| [BF4] | Blum, D. et al. "BOPTEST for Simulation-Based Benchmarking of Control Strategies." *JBPS*, 14(5):586-610, 2021. | Building control benchmark with Modelica. | Building-only; includes MPC baselines we should emulate |
| [BF5] | Mora, M.A. et al. "Sinergym: A Building Simulation and Control Framework for Training RL Agents." *BuildSys*, 2021. | EnergyPlus-based RL benchmark. | Building-only; our coverage is broader |
| [BF6] | "Characterizing MARL for Energy Control: A Multi-KPI Benchmark on the CityLearn Environment." *arXiv:2602.19223*, 2026. | Very recent MARL benchmark on CityLearn. | Closest competitor—but building-only, no safe RL axis |

### 2.5 Robustness & Reproducibility in RL

| Ref | Citation | Summary |
|-----|----------|---------|
| [RB1] | Dulac-Arnold, G. et al. "Challenges of Real-World RL: Definitions, Benchmarks and Analysis." *Machine Learning*, 110(9):2419-2468, 2021. | Defines real-world RL challenges (obs noise, action perturbation, non-stationarity). Frames your noise axis. |
| [RB2] | Henderson, P. et al. "Deep Reinforcement Learning That Matters." *AAAI*, 2018. | Establishes need for multiple seeds, statistical significance in DRL. Essential methodology reference. |
| [RB3] | Agarwal, R. et al. "Deep RL at the Edge of the Statistical Cliff." *NeurIPS*, 2021. | Stratified bootstrap CIs for RL evaluation. Best practice for your results. |
| [RB4] | Andrychowicz, M. et al. "What Matters in On-Policy RL? A Large-Scale Empirical Study." *ICLR*, 2021. | Large-scale ablation methodology. |
| [RB5] | Drgona, J. et al. "All You Need to Know About Model Predictive Control for Buildings." *Annual Reviews in Control*, 50:190-232, 2020. | MPC vs. RL for buildings; motivates non-RL baselines. |

### 2.6 Algorithm & Framework References (Must-Cite)

| Ref | Citation |
|-----|----------|
| [A1] | Schulman, J. et al. "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*, 2017. |
| [A2] | Haarnoja, T. et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor." *ICML*, 2018. |
| [A3] | Fujimoto, S. et al. "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*, 2018. (TD3) |
| [A4] | Raffin, A. et al. "Stable-Baselines3: Reliable RL Implementations." *JMLR*, 22(268):1-8, 2021. |
| [A5] | Liang, E. et al. "RLlib: Abstractions for Distributed Reinforcement Learning." *ICML*, 2018. |
| [A6] | Terry, J. et al. "PettingZoo: Gym for Multi-Agent Reinforcement Learning." *NeurIPS*, 2021. |
| [A7] | Towers, M. et al. "Gymnasium: A Standard Interface for RL Environments." Farama Foundation, 2023. |

### 2.7 Literature Gap Summary

| Dimension | Existing Work | Our Gap/Opportunity |
|-----------|--------------|-------------------|
| **Multi-env benchmark** | CityLearn (buildings only), Grid2Op (transmission), SustainGym (baselines only) | **No existing benchmark covers EV + Building + Cogen under unified protocol** |
| **Safe RL for energy** | Theory papers, single-env demonstrations | **No systematic safe RL benchmark across multiple energy domains** |
| **MARL for energy** | CityLearn MARL, individual papers | **No cross-domain MARL comparison (EV vs Building vs Cogen)** |
| **Noise robustness** | Dulac-Arnold framework, ad-hoc studies | **No structured obs/action/env noise decomposition for energy RL** |
| **Cross-paradigm comparison** | Each paradigm studied separately | **No paper compares standard RL, safe RL, and MARL on the same environments** |

---

## 3. Gap Analysis

### 3.1 What Is Missing (Must Fix)

| # | Gap | Why Critical | Effort | Priority |
|---|-----|-------------|--------|----------|
| 1 | **Non-RL baselines** (rule-based, MPC/optimization) | Reviewers demand: "Is RL even needed?" Without this, the paper is fatally flawed. | Medium | **P0** |
| 2 | **Statistical rigor**: ≥5 seeds per experiment, confidence intervals | Single-seed results are unpublishable per Henderson et al. (2018) [RB2] | Low (run time) | **P0** |
| 3 | **Domain-specific metrics** beyond reward | Must report: energy cost ($), CO₂ emissions (kg), temperature violation (%), constraint satisfaction rate (%), peak demand (kW) | Low (code) | **P0** |
| 4 | **Complete experiments across all 3 envs** | EVCharging is well-covered; Building and Cogen have gaps (esp. noise sweeps, safe RL, MARL) | High (compute) | **P0** |
| 5 | **Cross-axis comparison table** | A single table showing Standard RL vs. Safe RL vs. MARL per env is the paper's central deliverable | Low | **P1** |
| 6 | **Env noise / distribution shift** experiments | Obs/action noise done; env noise (the most practical concern) is incomplete | Medium | **P1** |
| 7 | **Normalized/comparable metrics** across envs | Envs have different reward scales; need a way to compare across them | Low | **P1** |
| 8 | **Ablation study on noise decomposition** | Show that obs noise, action noise, and env noise have different impacts—this is a key insight | Medium | **P2** |

### 3.2 What Is Unnecessary / Remove to Avoid Scope Creep

| Item | Why Remove |
|------|-----------|
| **15+ OmniSafe algorithm variants** for EVCharging | Too many. Pick 3-4 representative algorithms (OnCRPO, CPO, PPOLag, one Lagrangian PID). Reporting 15 clutters the paper. |
| **Exhaustive noise sweep (10+ levels per env)** | 3-5 well-chosen noise levels suffice for showing degradation curves. |
| **Multiple MARL algorithm variants per env** | One MARL algorithm per env with one strong baseline is enough for a 6-page paper. |
| **The `envs_copy/` directory and early prototype scripts** | Cleanup for release; not paper-relevant. |
| **`marl_training.py` (abandoned unified script)** | Dead code. Remove or ignore. |
| **Cogen env_marl.py alternate** | Pick one MARL env version per environment; the alternate adds confusion. |
| **TD3 for all envs** | TD3 is less commonly used for energy. Keep PPO + SAC as primary; TD3 only if it shows something interesting. |

### 3.3 Current Experiment Completion Status

| Experiment | EVCharging | Building | Cogen |
|------------|-----------|----------|-------|
| **Std RL (PPO)** | ✅ Complete | ⚠️ Partial | ⚠️ Partial |
| **Std RL (SAC)** | ✅ Complete | ⚠️ Partial | ⚠️ Partial |
| **Std RL (TD3)** | ✅ Complete | ⚠️ Partial | ⚠️ Partial |
| **Obs Noise Sweep** | ✅ Complete | ⚠️ Partial | ⚠️ Partial |
| **Action Noise Sweep** | ✅ Complete | ⚠️ Partial | ❌ Missing |
| **Env Noise Sweep** | ⚠️ In progress | ❌ Missing | ❌ Missing |
| **Safe RL (OmniSafe)** | ✅ Extensive (15+ algos) | ⚠️ Early results | ⚠️ Incomplete |
| **MARL (RLlib)** | ⚠️ APPO only | ⚠️ SAC only | ⚠️ PPO only (100 iters) |
| **Non-RL baselines** | ❌ Missing | ❌ Missing | ❌ Missing |
| **Multi-seed (≥5)** | ❓ Unknown | ❓ Unknown | ❓ Unknown |
| **Domain metrics** | ⚠️ Reward breakdown exists | ❌ Not extracted | ❌ Not extracted |

---

## 4.

### 4.1 Non-RL Baselines (CRITICAL)

Every benchmark paper needs baselines that answer: **"Why use RL at all?"**

| Baseline | EVCharging | Building | Cogen | Effort |
|----------|-----------|----------|-------|--------|
| **Do-Nothing / Fixed** | Charge at max rate always | Set thermostat to midpoint | Nominal operating point | Trivial |
| **Rule-Based / Heuristic** | Charge proportional to demand, LLLF (Least Laxity Last First) | Thermostat deadband (±1°C) | Load-following proportional | Low |
| **Optimization-Based** | LP/QP for min-cost charging (single-step) | MPC with RC model (1-step lookahead) | LP for fuel cost minimization | Medium |

**Recommendation**: Implement at least **Do-Nothing + Rule-Based** for all envs (trivial), and **one optimization baseline** for at least one env (medium effort).

### 4.2 Robustness Testing Protocol

**Structured noise decomposition** (your key differentiator):

```
For each env ∈ {EVCharging, Building, Cogen}:
  For each algo ∈ {PPO, SAC}:
    For noise_type ∈ {observation, action, environment}:
      For noise_level ∈ {0.0, 0.05, 0.10, 0.20, 0.40}:
        Run 5 seeds → report mean ± 95% CI
```

**Distribution shift test** (train on noise=0, test on increasing noise):
- Train clean policy → evaluate under noise
- This is the most practically relevant test (policies deployed in noisy real world)

### 4.3 Ablation Studies

| Ablation | What It Shows | Implementation |
|----------|--------------|---------------|
| **Noise type decomposition** | Obs vs. action vs. env noise have different impacts on performance | Already partially done; complete for all envs |
| **Safe RL cost threshold** | Sensitivity to constraint limit (`climit`) parameter | Run 3-4 cost limits per env |
| **MARL agent granularity** | Effect of number of agents (e.g., 54 stations vs. 10 clusters in EVCharging) | New experiment; may be too ambitious for deadline |
| **VecNormalize impact** | Whether obs normalization matters (yes/no) | Already have `--use-vecnormalize` flag |
| **Reward shaping** | Effect of `reward_beta` in Building env | Run 3 values: 0.3, 0.5, 0.7 |

### 4.4 Standardized Evaluation Protocol

**Seeds**: Minimum 5 seeds per experiment (seeds 42, 43, 44, 45, 46)

**Metrics to report per env**:

| Metric | EVCharging | Building | Cogen |
|--------|-----------|----------|-------|
| Cumulative reward | ✓ | ✓ | ✓ |
| Energy cost ($) | profit from reward_breakdown | sum of |action| * energy_price | fuel_cost from info |
| CO₂ / carbon (kg) | carbon_cost from reward_breakdown | indirect (GHI-based) | fuel_cost proxy |
| Constraint violation rate (%) | excess_charge / total_steps | temp_violation / total_steps | dyn_cv_costs / total_steps |
| Peak demand (kW) | max(pilot_signals) per episode | max(|action|) per episode | max(power_output) |
| Comfort / service quality | demand satisfaction rate | mean temp error (°C) | steam delivery rate |

**Confidence intervals**: Bootstrap 95% CI across seeds (per Agarwal et al. 2021 [RB3])

**Training curves**: Show learning progress with shaded interquartile range across seeds

### 4.5 Reporting & Visualization

**Required figures** (for 6-page paper, aim for 4-5 figures):

1. **Figure 1**: Taxonomy diagram — 3 envs × 3 paradigms × noise axes
2. **Figure 2**: Training curves for all algorithms per env (3 panels, one per env)
3. **Figure 3**: Noise degradation curves (reward vs. noise level, by noise type)
4. **Figure 4**: Safe RL constraint satisfaction (cost vs. reward Pareto fronts)
5. **Figure 5**: MARL vs. single-agent comparison (bar chart with error bars)

**Required tables**:

1. **Table I**: Environment specification summary (obs/action space, reward, episode length, # agents)
2. **Table II**: Main results — Mean reward ± CI for all {algo × env × paradigm} combinations
3. **Table III**: Domain-specific metrics (cost, CO₂, constraint violation, comfort)
4. **Table IV**: Robustness — Performance degradation under noise (% drop from clean baseline)

---

## 5. Novelty & Contributions

### 5.1 Current Novelty Assessment (Honest)

**Novelty is currently MODERATE**. The environments are not new (SustainGym, NeurIPS 2023). The algorithms are not new (PPO, SAC, OmniSafe, RLlib). The novelty must come from the **benchmark protocol and the insights it generates**.

### 5.2 What Can Be Claimed

| Claim | Type | Confidence | Requires |
|-------|------|-----------|----------|
| **First unified benchmark comparing standard RL, safe RL, and MARL across multiple energy domains** | Protocol novelty | High | Complete all 3×3 experiments |
| **First systematic noise robustness decomposition (obs/action/env) for energy RL** | Methodology novelty | High | Complete noise experiments for all 3 envs |
| **First safe RL benchmark for EV charging and cogeneration** | Application novelty | Medium-High | Verify no prior safe RL papers on these specific domains |
| **Actionable insights for practitioners** | Empirical contribution | High | Generate and articulate non-obvious findings |

### 5.3 Novelty Strengthening Strategies

| If We Add... | We Can Claim... | Effort |
|--------------|----------------|--------|
| **Non-RL baselines (rule-based + optimization)** | "RL outperforms/matches optimization under uncertainty" — this is the most impactful finding for the SmartGrid community | Medium |
| **Distribution shift experiments** (train clean, test noisy) | "Policy robustness under deployment uncertainty" — directly actionable for practitioners | Low-Medium |
| **Cross-paradigm comparison table** | "Safe RL achieves X% fewer violations at Y% cost; MARL scales to Z agents" — these are quotable results | Low |
| **Open-source benchmark suite release** | "We release a reproducible benchmark suite with standardized evaluation protocol" — artifact contribution | Low (code cleanup) |
| **Pareto analysis (reward vs. constraint cost)** for safe RL | "Safe RL enables tunable safety-reward tradeoff; we characterize the Pareto frontier" | Low |
| **Scalability analysis** for MARL (agents vs. performance) | "MARL performance degrades/improves with agent count" — practical insight | Medium |

### 5.4 Proposed Contribution Statement (for the paper)

> We present a comprehensive benchmark for reinforcement learning in sustainable energy systems, evaluating three paradigms—standard RL, safe (constrained) RL, and multi-agent RL—across three diverse SustainGym environments (EV charging, building HVAC, cogeneration). Our contributions are:
>
> 1. **A unified multi-paradigm evaluation protocol** for energy RL, including standardized metrics, noise decomposition, and statistical methodology.
> 2. **The first systematic robustness analysis** decomposing observation, action, and environment noise across energy domains, revealing that [key finding, e.g., "action noise degrades performance more severely than observation noise in actuator-heavy environments"].
> 3. **The first safe RL benchmark** for EV charging and cogeneration, demonstrating that constrained optimization can reduce constraint violations by X% with only Y% reward reduction.
> 4. **A comparative analysis of single-agent vs. multi-agent RL** across heterogeneous energy systems, showing that [key finding, e.g., "MARL improves scalability but requires careful reward decomposition"].
> 5. **An open-source benchmark suite** with reproducible training, evaluation, and reporting scripts.

---

## 6. Recommended Paper Structure

### 6-Page IEEE Format

| Section | Pages | Content |
|---------|-------|---------|
| **I. Introduction** | 0.75 | Problem statement, motivation, contributions (5 bullet points) |
| **II. Related Work** | 0.75 | RL for energy, safe RL, MARL, benchmarks (cite Table from §2) |
| **III. Environments & Problem Formulation** | 1.0 | Table I (env specs), noise model formalization, CMDP formulation |
| **IV. Experimental Setup** | 0.75 | Algorithms, hyperparameters, baselines, evaluation protocol, seeds |
| **V. Results & Analysis** | 2.0 | Main results (Table II-III), noise robustness (Fig 3), safe RL (Fig 4), MARL (Fig 5) |
| **VI. Discussion & Conclusion** | 0.75 | Key insights, limitations, future work |

### Key Insights to Target (Fill These In After Experiments)

These are the kinds of non-obvious findings that make a benchmark paper valuable:

1. **"Which algorithm wins?"** → Likely SAC for off-policy efficiency, PPO for stability under noise
2. **"Is safe RL worth the reward cost?"** → Quantify the Pareto tradeoff
3. **"Does MARL improve over single-agent?"** → Likely depends on env; EV charging may benefit, cogen may not
4. **"Which noise type matters most?"** → Environment noise (distribution shift) likely dominates
5. **"Does RL beat baselines?"** → RL likely beats rule-based but may not beat MPC in low-noise settings

---

## Appendix A: Concrete Action Items & Timeline

### Phase 1: Critical Fixes (March 3 – April 15, 2026)

- [ ] Implement Do-Nothing and Rule-Based baselines for all 3 envs
- [ ] Run all existing experiments with 5 seeds (42-46)
- [ ] Add domain-specific metric extraction to `stdrl_testing.py`
- [ ] Complete Building and Cogen standard RL training (all algos, all noise levels)
- [ ] Complete env noise experiments for EVCharging

### Phase 2: Core Experiments (April 15 – May 15, 2026)

- [ ] Run Safe RL (OmniSafe) for Building and Cogen (3 algorithms × 5 seeds)
- [ ] Run MARL training for all 3 envs with sufficient iterations (≥10k for Cogen)
- [ ] Distribution shift experiment (train clean → test noisy)
- [ ] Cross-paradigm comparison: Standard RL vs. Safe RL vs. MARL per env
- [ ] Implement one optimization baseline (LP for EVCharging or MPC for Building)

### Phase 3: Analysis & Writing (May 15 – June 12, 2026)

- [ ] Generate all figures and tables
- [ ] Write paper (6 pages + optional 2-page extension)
- [ ] Internal review and revision
- [ ] Code cleanup and prepare repository for release

### Phase 4: Submission (June 12 – June 19, 2026)

- [ ] Final proofreading
- [ ] Supplementary material preparation
- [ ] Submit by June 19, 2026

---

## Appendix B: Code Bugs / Issues Found During Review

| Issue | File | Impact | Fix |
|-------|------|--------|-----|
| `marl_tr_co.py` metrics filename says `building_env_metrics_` | `marl_tr_co.py` | Minor (misleading filename) | Rename to `cogen_env_metrics_` |
| `Omnisafe_Building.py` ignores `--noise-action` | `Omnisafe_Building.py` | Medium (action noise not applied in Building safe RL) | Pass `noise_action` to BuildingEnv |
| `Omnisafe_Cogen.py` has verbose debug prints on every step | `Omnisafe_Cogen.py` | Low (performance hit on ARC) | Remove or gate behind `--verbose` |
| `marl_training.py` entirely commented out | `marl_training.py` | None (dead code) | Delete or document as deprecated |
| `ttp/marl_plot.py` entirely commented out | `ttp/marl_plot.py` | Low (no MARL visualization) | Reactivate with current data paths |
| Multiple commented-out versions in `multiagent_env.py` files | Building, Cogen | Low (code cleanliness) | Clean up before public release |

---

## Appendix C: Full Citation List (BibTeX-Ready)

```
@article{wan2019ev,
  author={Wan, Zhongjing and Li, Hanping and He, Haibo and Prokhorov, Danil},
  title={Model-free real-time EV charging scheduling based on deep reinforcement learning},
  journal={IEEE Transactions on Smart Grid},
  volume={10}, number={5}, pages={5246--5257}, year={2019}}

@inproceedings{lee2019acndata,
  author={Lee, Zachary J. and Li, Tongxin and Low, Steven H.},
  title={{ACN-Data}: Analysis and Applications of an Open {EV} Charging Dataset},
  booktitle={Proc. ACM e-Energy}, year={2019}}

@inproceedings{lee2020acnsim,
  author={Lee, Zachary J. and Sharma, Gautam and Johansson, Daniel and Low, Steven H.},
  title={{ACN-Sim}: An Open-Source Simulator for Data-Driven {EV} Charging Research},
  booktitle={IEEE SmartGridComm}, year={2020}}

@article{vazquezcanteli2019rl,
  author={Vazquez-Canteli, Jose R. and Nagy, Zoltan},
  title={Reinforcement learning for demand response: A review},
  journal={Applied Energy}, volume={235}, pages={1072--1089}, year={2019}}

@article{yu2021marl,
  author={Yu, Liang and others},
  title={Multi-agent deep reinforcement learning for HVAC control in commercial buildings},
  journal={IEEE Transactions on Smart Grid},
  volume={12}, number={1}, pages={407--419}, year={2021}}

@book{altman1999cmdp,
  author={Altman, Eitan},
  title={Constrained Markov Decision Processes},
  publisher={Chapman \& Hall/CRC}, year={1999}}

@inproceedings{achiam2017cpo,
  author={Achiam, Joshua and Held, David and Tamar, Aviv and Abbeel, Pieter},
  title={Constrained Policy Optimization},
  booktitle={ICML}, year={2017}}

@article{ji2023omnisafe,
  author={Ji, Jiaming and others},
  title={{OmniSafe}: An Infrastructure for Accelerating Safe Reinforcement Learning Research},
  journal={arXiv:2305.09304}, year={2023}}

@inproceedings{yeh2023sustaingym,
  author={Yeh, Christopher and others},
  title={{SustainGym}: Reinforcement Learning Environments for Sustainable Energy Systems},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2023}}

@inproceedings{vazquezcanteli2019citylearn,
  author={Vazquez-Canteli, Jose R. and others},
  title={{CityLearn} v1.0: An {OpenAI Gym} Environment for Demand Response with Deep RL},
  booktitle={BuildSys}, year={2019}}

@article{blum2021boptest,
  author={Blum, David and others},
  title={Building Optimization Testing Framework ({BOPTEST})},
  journal={Journal of Building Performance Simulation},
  volume={14}, number={5}, pages={586--610}, year={2021}}

@inproceedings{schulman2017ppo,
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  title={Proximal Policy Optimization Algorithms},
  booktitle={arXiv:1707.06347}, year={2017}}

@inproceedings{haarnoja2018sac,
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL},
  booktitle={ICML}, year={2018}}

@inproceedings{fujimoto2018td3,
  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  booktitle={ICML}, year={2018}}

@article{raffin2021sb3,
  author={Raffin, Antonin and others},
  title={Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal={JMLR}, volume={22}, number={268}, pages={1--8}, year={2021}}

@inproceedings{liang2018rllib,
  author={Liang, Eric and others},
  title={{RLlib}: Abstractions for Distributed Reinforcement Learning},
  booktitle={ICML}, year={2018}}

@inproceedings{terry2021pettingzoo,
  author={Terry, J. and others},
  title={{PettingZoo}: Gym for Multi-Agent Reinforcement Learning},
  booktitle={NeurIPS}, year={2021}}

@inproceedings{henderson2018matters,
  author={Henderson, Peter and others},
  title={Deep Reinforcement Learning That Matters},
  booktitle={AAAI}, year={2018}}

@inproceedings{agarwal2021cliff,
  author={Agarwal, Rishabh and others},
  title={Deep Reinforcement Learning at the Edge of the Statistical Cliff},
  booktitle={NeurIPS}, year={2021}}

@article{dulacarnold2021challenges,
  author={Dulac-Arnold, Gabriel and others},
  title={Challenges of Real-World Reinforcement Learning},
  journal={Machine Learning}, volume={110}, number={9}, pages={2419--2468}, year={2021}}

@article{drgona2020mpc,
  author={Drgona, Jan and others},
  title={All You Need to Know About Model Predictive Control for Buildings},
  journal={Annual Reviews in Control}, volume={50}, pages={190--232}, year={2020}}
```

---

*Report generated March 3, 2026. All citations should be independently verified on Google Scholar before submission.*
