# Chapter 5: Experimental Results and Discussion — Structure

## 5.1 Noise Robustness

### 5.1.1 Baseline Performance (Clean Conditions)
- **Table:** `baseline_results` — 3 env × 3 algo (PPO, SAC, TD3), mean reward, std, 95% CI
- **Fig:** `baseline_ev` — EV Charging baseline training curve comparison (PPO, SAC, TD3)
- **Fig:** `baseline_bu` — Building baseline training curve comparison (PPO, SAC, TD3)
- **Fig:** `baseline_co` — Cogeneration baseline training curve comparison (PPO, SAC, TD3)
- **¶ EV Charging** — PPO best absolute reward, SAC higher profit ratio
- **¶ Building** — PPO best, TD3 high variance
- **¶ Cogeneration** — SAC/TD3 near-zero, PPO conservative fuel-minimizer

### 5.1.2 EV Charging: Noise Robustness

#### EV Charging — PPO
- **¶ Training curves**
  - Fig: `C_STDRL/ev/PPO/DS` (obs noise)
  - Fig: `C_STDRL/ev/PPO/DA` (action noise)
  - Fig: `C_STDRL/ev/PPO/DE` (env noise)
- **¶ Test-time robustness** — distribution stable, no mode shift
  - Fig: `C_POST/evcharging/PPO/obs_violin`
  - Fig: `C_POST/evcharging/PPO/action_violin`
  - Fig: `C_POST/evcharging/PPO/env_violin`
- Fig: `C_POST/evcharging/PPO/panel` — <1% degradation summary

#### EV Charging — SAC
- **¶ Training curves**
  - Fig: `C_STDRL/ev/SAC/DS`
  - Fig: `C_STDRL/ev/SAC/DA`
  - Fig: `C_STDRL/ev/SAC/DE`
- **¶ Test-time robustness** — implicit regularization finding (obs noise improves reward)
  - Fig: `C_POST/evcharging/SAC/obs_violin` — rightward shift
  - Fig: `C_POST/evcharging/SAC/action_violin`
  - Fig: `C_POST/evcharging/SAC/env_violin` — distribution broadens
- Fig: `C_POST/evcharging/SAC/panel`

### 5.1.3 Building: Noise Robustness

#### Building — PPO
- **¶ Training curves** — clear separation between noise levels
  - Fig: `C_STDRL/bu/PPO/DS`
  - Fig: `C_STDRL/bu/PPO/DA`
  - Fig: `C_STDRL/bu/PPO/DE`
- **¶ Test-time robustness** — bimodal collapse (obs), bang-bang tolerance (act), catastrophic (env)
  - Fig: `C_POST/building/PPO/obs_violin` — high-reward mode collapses at σ≥0.15
  - Fig: `C_POST/building/PPO/action_violin` — preserved up to σ=0.2
  - Fig: `C_POST/building/PPO/env_violin` — entire distribution collapses
- Fig: `C_POST/building/PPO/panel` — obs+env severe, action minimal

#### Building — SAC
- **¶ Training curves** — similar to PPO, lower absolute
  - Fig: `C_STDRL/bu/SAC/DS`
  - Fig: `C_STDRL/bu/SAC/DA`
  - Fig: `C_STDRL/bu/SAC/DE`
- **¶ Test-time robustness** — confirms env-driven (not algo-driven) sensitivity
  - Fig: `C_POST/building/SAC/obs_violin`
  - Fig: `C_POST/building/SAC/action_violin`
  - Fig: `C_POST/building/SAC/env_violin`
- Fig: `C_POST/building/SAC/panel`

### 5.1.4 Cogeneration: Noise Robustness

#### Cogeneration — PPO
- **¶ Training curves** — indistinguishable across noise levels
  - Fig: `C_STDRL/co/PPO/DS`
  - Fig: `C_STDRL/co/PPO/DA`
  - Fig: `C_STDRL/co/PPO/DE`
- **¶ Test-time robustness** — <0.3% degradation across full range (σ=0 to 5)
  - Fig: `C_POST/cogen/PPO/obs_violin` — tight, unchanged
  - Fig: `C_POST/cogen/PPO/action_violin`
  - Fig: `C_POST/cogen/PPO/env_violin`
- Fig: `C_POST/cogen/PPO/panel` — near-zero sensitivity

#### Cogeneration — SAC
- **¶ Training curves** — obs noise disrupts convergence
  - Fig: `C_STDRL/co/SAC/DS`
  - Fig: `C_STDRL/co/SAC/DA`
  - Fig: `C_STDRL/co/SAC/DE`
- **¶ Test-time robustness** — heavy left tails at σ≥2, >2000% degradation
  - Fig: `C_POST/cogen/SAC/obs_violin` — heavy left tails
  - Fig: `C_POST/cogen/SAC/env_violin` — moderate degradation
  - *(no action_violin — data not available)*
- Fig: `C_POST/cogen/SAC/panel` — obs severe, env moderate

### 5.1.5 Noise Robustness Findings
- **Finding:** Environment-Dependent Noise Sensitivity (RQ1, RQ6)
- **Finding:** Observation vs Action Noise Asymmetry (RQ2)
- **Finding:** Algorithm-Specific Robustness (RQ1)

---

## 5.2 Safe Reinforcement Learning

### 5.2.1 Cogeneration: Strong Constraint Differentiation
- Text: CMDP cost (ramp stress) orthogonal to reward

#### Cogeneration — PPOLag
- Text: clearest differentiation, monotonic costs (5, 12, 23, 43 for d=10,25,50,200)
- Fig: `C_SRL/cogen/PPOLag` — dual-panel (reward + cost)

#### Cogeneration — CPO
- Text: rapid convergence (~500 epochs), weak differentiation
- Fig: `C_SRL/cogen/CPO`

#### Cogeneration — OnCRPO
- Text: minimal limit separation (costs 12–22 across all limits)
- Fig: `C_SRL/cogen/OnCRPO`

#### Cogeneration — FOCOPS
- Text: good monotonic (d=10→8, d=25→17, d=50→28, d=200→51), slowest
- Fig: `C_SRL/cogen/FOCOPS`

#### Cogeneration — Cross-Algorithm Comparisons
- **Equation:** PPOLag ≻ OnCRPO ≈ CPO ≻ FOCOPS
- Fig: `C_SRL/cogen/compare_CL10`
- Fig: `C_SRL/cogen/compare_CL25`
- Fig: `C_SRL/cogen/compare_CL50`
- Fig: `C_SRL/cogen/compare_CL200`

### 5.2.2 EV Charging: Precise Constraint Tracking
- **Table:** `ev_saferl` — algo × limit, bold = satisfied (C ≤ d)

#### EV Charging — PPOLag
- Text: near-exact satisfaction at all levels
- Fig: `C_SRL/evcharging/PPOLag`

#### EV Charging — CPO
- Text: slight overshoot at d=3 (cost=3.3)
- Fig: `C_SRL/evcharging/CPO`

#### EV Charging — OnCRPO
- Text: consistent overshoot at all levels (d=3→4.5, d=5→6.9, d=25→27.5)
- Fig: `C_SRL/evcharging/OnCRPO`

#### EV Charging — FOCOPS
- Text: strong at tight (d=3→3.1), overshoot at loose (d=25→26.2)
- Fig: `C_SRL/evcharging/FOCOPS`

#### EV Charging — Cross-Algorithm Comparisons
- Fig: `C_SRL/evcharging/compare_CL3`
- Fig: `C_SRL/evcharging/compare_CL25`

### 5.2.3 Building: A Negative Finding
- **Table:** `building_saferl` — 3 cost functions × 4 limits (all converge)
- Text: reward–cost alignment problem (β=0.5 already optimizes comfort)

#### Building — PPOLag
- Text: identical curves, cost collapses to 2–5 regardless of limit
- Fig: `C_SRL/building/PPOLag`

#### Building — CPO
- Text: divergence at loose limits (d=200, 500), trust-region instability
- Fig: `C_SRL/building/CPO`

#### Building — OnCRPO
- Text: similar divergence to CPO
- Fig: `C_SRL/building/OnCRPO`

#### Building — FOCOPS
- Text: late-training cost increase, Lagrangian oscillation
- Fig: `C_SRL/building/FOCOPS`

### 5.2.4 Off-Policy Safe RL: SACLag Failure
- Text: replay buffer staleness → λ oscillation/divergence (no figs)

### 5.2.5 Safe RL Findings
- **Finding:** Reward–Cost Orthogonality Is Necessary (RQ4)
- **Finding:** PPOLag Best CMDP Algorithm (RQ4)
- **Finding:** Off-Policy CMDP Failure (RQ4)
- **Table:** `saferl_summary` — 3 env comparison (orthogonal?, differentiates?, best algo)

---

## 5.3 Multi-Agent Reinforcement Learning
- *(placeholder — experiments in progress)*

---

## 5.4 Cross-Axis Discussion

### 5.4.1 Environment Characteristics > Algorithm Choice
- EV: robust to noise + amenable to Safe RL
- Building: fragile to noise + resistant to Safe RL
- Cogen: algo-dependent noise + strong Safe RL

### 5.4.2 PPO's Dual Advantage
- Noise robustness + CMDP satisfaction from on-policy nature

### 5.4.3 Implications for Sustainable Energy Systems
1. Sensor investment > actuator precision
2. CMDP constraints must be orthogonal to reward
3. PPO/PPOLag best robustness–safety combination

---

## Figure Count Summary

| Section | Baseline | Training | Violin | Panel | Safe RL | Compare | Total |
|---------|----------|----------|--------|-------|---------|---------|-------|
| Baseline (5.1.1) | 3 | — | — | — | — | — | **3** |
| EV Noise (PPO+SAC) | — | 6 | 6 | 2 | — | — | **14** |
| Building Noise (PPO+SAC) | — | 6 | 6 | 2 | — | — | **14** |
| Cogen Noise (PPO+SAC) | — | 6 | 5* | 2 | — | — | **13** |
| Cogen Safe RL | — | — | — | — | 4 | 4 | **8** |
| EV Safe RL | — | — | — | — | 4 | 2 | **6** |
| Building Safe RL | — | — | — | — | 4 | 0 | **4** |
| **Total** | **3** | **18** | **17** | **6** | **12** | **6** | **62** |

*\*Cogen SAC has no action_violin (data not available)*

**Tables:** 4 (baseline, ev_saferl, building_saferl, saferl_summary)
**Finding boxes:** 6 (3 noise + 3 safe RL)
**Equations:** 1 (Cogen algo ranking)
