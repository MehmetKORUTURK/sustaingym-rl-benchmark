# ============================================================================
# StdRL Testing — Cogen Robustness (Baseline model vs increasing noise)
# ============================================================================
#
# Tests the noise=0 baseline model at all trained noise levels.
# Shows how a clean-trained model degrades under noise at inference time.
#
# Summary Table:
# ─────────────────────────────────────────────────────────────────────────────
#  Noise Type    Algos          Levels Tested
# ─────────────────────────────────────────────────────────────────────────────
#  Observation   PPO,SAC,TD3    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0
#  Action        PPO,SAC,TD3    0.05, 0.1, 0.15, 0.2, 0.3, 0.4
#  Environment   PPO,SAC,TD3    0.5, 1.0, 2.0, 3.0, 4.0, 5.0
# ─────────────────────────────────────────────────────────────────────────────
# Grand Total: 57 runs (3 algos x 19 noise levels)
#
# Notes:
# - UPDATE BASELINE_* paths below to point to your actual trained model dirs.
# - All runs use --rm 300 (renewables magnitude), --n-eval 100, --seed 42.
# - Cogen uses conda env stdrl_train_co (separate from EV/Building).
# ============================================================================

# ── Baseline model paths (UPDATE THESE) ──
BASELINE_PPO="logs_std_train/cogen_PPO/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_SAC="logs_std_train/cogen_SAC/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_TD3="logs_std_train/cogen_TD3/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

# ============================================================================
# Observation Noise Robustness — PPO (7 runs)
# ============================================================================
echo "========== Cogen — Obs Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 0.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise 5.0 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — SAC (7 runs)
# ============================================================================
echo "========== Cogen — Obs Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 0.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise 5.0 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — TD3 (7 runs)
# ============================================================================
echo "========== Cogen — Obs Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 0.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise 5.0 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — PPO (6 runs)
# ============================================================================
echo "========== Cogen — Act Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — SAC (6 runs)
# ============================================================================
echo "========== Cogen — Act Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — TD3 (6 runs)
# ============================================================================
echo "========== Cogen — Act Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — PPO (6 runs)
# ============================================================================
echo "========== Cogen — Env Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo PPO --rm 300 --model_path "$BASELINE_PPO" --noise-env 5.0 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — SAC (6 runs)
# ============================================================================
echo "========== Cogen — Env Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo SAC --rm 300 --model_path "$BASELINE_SAC" --noise-env 5.0 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — TD3 (6 runs)
# ============================================================================
echo "========== Cogen — Env Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 3.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 4.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env cogen --algo TD3 --rm 300 --model_path "$BASELINE_TD3" --noise-env 5.0 --n-eval 100 --seed 42
