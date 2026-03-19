# SmartGridComm Paper TODO

## Conference Info
- **Conference**: IEEE SmartGridComm 2026
- **Source**: Thesis Chapter 4 (Methodology) + Chapter 5 (Results) → condensed 6-page paper
- **Deadline**: TBD (typically June/July submission)

---

## 🔴 CRITICAL — Multi-Seed Experiments

### 1. Multi-Seed Training Runs (IQM-Ready)
Agarwal et al. (2021) requires 5+ independent seeds for IQM to be meaningful.
Currently thesis has 1 seed per config. Paper needs proper multi-seed evaluation.

**Standard RL — Key configs to re-run with 5 seeds:**
- [ ] EV Charging PPO: baseline + noise={0.1, 0.3, 0.6} obs, {0.1, 0.3} act, {0.1, 0.3} env → 7 configs × 5 seeds = 35 runs
- [ ] EV Charging SAC: same noise selection → 35 runs
- [ ] Building PPO: baseline + noise={0.05, 0.15, 0.3} obs, {0.05, 0.15} act, {0.05, 0.15} env → 7 × 5 = 35 runs
- [ ] Building SAC: same → 35 runs
- [ ] Cogen PPO: baseline + noise={1.0, 3.0, 5.0} obs → 4 × 5 = 20 runs
- [ ] Cogen SAC: same → 20 runs
- **Total**: ~180 training runs (vs thesis's ~150 single-seed)
- **Seeds**: {42, 43, 44, 45, 46}
- **Priority**: Focus on PPO + SAC only (TD3 negative finding → 1 paragraph, no multi-seed needed)

**Safe RL — Key configs:**
- [ ] PPOLag only (best algo) × 3 envs × 2 representative limits each × 5 seeds = 30 runs
- [ ] EV: d={3, 25}, Building: d={50, 200}, Cogen: d={25, 200}

**MARL — If included:**
- [ ] Best algo per env × 5 seeds = 15 runs

### 2. IQM + Stratified Bootstrap Pipeline
- [ ] Script: `scripts/analysis/compute_iqm.py` — reads multi-seed test results, computes IQM + bootstrap CI
- [ ] Script: `scripts/plot/sgc_plot.py` — paper-ready figures with IQM, performance profiles, probability of improvement
- [ ] Use `rliable` library (Agarwal et al.) for standardized IQM computation

### 3. Multi-Seed Test Evaluation
- [ ] Script: `scripts/test/stdrl_testing_multiseed.py` — batch test all 5 seeds per config
- [ ] 100 episodes per seed × 5 seeds = 500 episodes per config (or 200 × 5 = 1000)
- [ ] Output: per-seed CSV files for stratified bootstrap

---

## 🔴 CRITICAL — Paper Structure

### 4. Paper Outline (6 pages, IEEE double-column)
- [ ] I. Introduction (~0.75 page) — motivation, contributions, RQs
- [ ] II. Related Work (~0.5 page) — noise robustness + safe RL in energy systems
- [ ] III. SustainRL-Bench Framework (~1 page) — 3 envs, noise injection, CMDP formulation
- [ ] IV. Experimental Setup (~0.5 page) — algos, hyperparams, multi-seed protocol
- [ ] V. Results (~2.5 pages) — noise robustness findings + safe RL findings + key figures
- [ ] VI. Conclusion (~0.5 page) — key takeaways, practical implications
- [ ] References (~0.25 page)

### 5. Key Figures for Paper (max 5-6 figures)
- [ ] Fig 1: Framework architecture diagram (reuse/create Figure 4.1 from thesis)
- [ ] Fig 2: Noise degradation curves — 3 envs × 1 algo (PPO), panel plot with IQM + CI
- [ ] Fig 3: Algorithm comparison violin plots — 1 per env, all 3 algos
- [ ] Fig 4: Safe RL dual-panel — Cogen PPOLag (best case, clear differentiation)
- [ ] Fig 5: Cross-environment summary heatmap (algo × env × noise type → degradation %)
- [ ] Fig 6: (optional) MARL single-agent vs multi-agent comparison

### 6. Key Tables
- [ ] Table I: Environment characteristics (obs/action dims, episode length, reward range)
- [ ] Table II: Baseline performance (IQM ± CI across 5 seeds) — replaces thesis Table 5.1
- [ ] Table III: Noise sensitivity summary (degradation % at key noise levels)
- [ ] Table IV: Safe RL constraint satisfaction (PPOLag across cost limits)

### 6b. Non-RL Baselines (Paper-Critical)
Reviewer'lar "RL ne kadar iyi?" sorusuna cevap bekler. Random/Do-Nothing çalıştırılmalı.
- [ ] **Random policy**: 3 env × 1000 episode (10 dk iş, training yok)
- [ ] **Do-Nothing** (zero action): 3 env × 1000 episode
- [ ] **Greedy** (EV only): max-rate charging, SustainGym'de mevcut
- [ ] **Thermostat** (Building only): bang-bang controller, basit script
- [ ] **Load-Following** (Cogen only): equal GT power distribution
- [ ] Sonuçları Table II'ye "Non-RL" satırları olarak ekle
- [ ] Script: `scripts/test/baseline_nonrl.py` — hepsi için tek script

---

## 🟡 IMPORTANT — Writing

### 7. Differentiation from Thesis
- [ ] Paper focuses on **noise robustness + safe RL** only (drop MARL if space constrained)
- [ ] Condense 3 envs into comparative narrative (not per-env subsections)
- [ ] Lead with cross-environment findings, not per-algorithm details
- [ ] Emphasize practical implications for energy systems deployment

### 8. Novel Contributions to Highlight
- [ ] Three-channel noise injection framework (obs/act/env independently)
- [ ] Cross-environment noise sensitivity comparison (EV robust, Building fragile, Cogen algo-dependent)
- [ ] Observation > action noise finding (sensor investment priority)
- [ ] Reward-cost orthogonality requirement for CMDP
- [ ] TD3 noise-as-regularization finding
- [ ] SACLag failure (off-policy CMDP negative result)

---

## 🟢 NICE-TO-HAVE

### 9. Additional Analysis for Paper
- [ ] **Noise sensitivity coefficient κ**: $\kappa = -\partial \Delta_\sigma / \partial \sigma |_{\sigma=0}$ — linearized sensitivity, compact summary for Table III
  - Hesapla: linear fit to degradation curve at low noise, extract slope
  - Her env × algo × noise channel = 1 κ değeri → 3×3×3 = 27 entries
  - Thesis'ten kaldırıldı, paper'a özel metrik olarak kullan
- [ ] Performance profiles (Agarwal et al. style) — cumulative distribution of normalized scores
- [ ] Probability of improvement plots — P(algo A > algo B) with bootstrap
- [ ] Noise sensitivity coefficient κ (defined in thesis chapter 4 but unused) — compute and report

### 10. Code Cleanup for Reproducibility
- [ ] GitHub release with clean README for paper submission
- [ ] requirements.txt / conda env export
- [ ] Single-command reproduction: `bash scripts/reproduce_sgc.sh`

---

## Timeline
- [ ] Multi-seed runs on ARC: ~1 week (180 runs × ~2-8 hours each, parallelized on SLURM)
- [ ] IQM analysis + figures: ~2 days
- [ ] Paper draft: ~3-4 days
- [ ] Internal review + revision: ~1 week
- [ ] **Target**: 3-4 weeks from start to submission-ready
