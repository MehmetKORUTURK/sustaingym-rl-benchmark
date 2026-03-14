# ============================================================================
# StdRL Testing — Building Robustness (Baseline model vs increasing noise)
# ============================================================================
#
# Tests the noise=0 baseline model at all trained noise levels.
# Shows how a clean-trained model degrades under noise at inference time.
#
# Summary Table:
# ─────────────────────────────────────────────────────────────────────────────
#  Noise Type    Algos          Levels Tested
# ─────────────────────────────────────────────────────────────────────────────
#  Observation   PPO,SAC,TD3    0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40
#  Action        PPO,SAC,TD3    0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40
#  Environment   PPO,SAC,TD3    0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0
# ─────────────────────────────────────────────────────────────────────────────
# Grand Total: 72 runs (3 algos x 24 noise levels)
#
# Notes:
# - UPDATE BASELINE_* paths below to point to your actual trained model dirs.
# - All runs use --n-eval 100, --seed 42, deterministic policy.
# ============================================================================

# ── Baseline model paths (UPDATE THESE) ──
BASELINE_PPO="logs_std_train/building_PPO/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_SAC="logs_std_train/building_SAC/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_TD3="logs_std_train/building_TD3/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

# ============================================================================
# Observation Noise Robustness — PPO (9 runs)
# ============================================================================
echo "========== Building — Obs Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise 0.40 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — SAC (9 runs)
# ============================================================================
echo "========== Building — Obs Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise 0.40 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — TD3 (9 runs)
# ============================================================================
echo "========== Building — Obs Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise 0.40 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — PPO (8 runs)
# ============================================================================
echo "========== Building — Act Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — SAC (8 runs)
# ============================================================================
echo "========== Building — Act Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — TD3 (8 runs)
# ============================================================================
echo "========== Building — Act Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.03 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.40 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — PPO (7 runs)
# ============================================================================
echo "========== Building — Env Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.1 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.3 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 1.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo PPO --model_path "$BASELINE_PPO" --noise-env 3.0 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — SAC (7 runs)
# ============================================================================
echo "========== Building — Env Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.1 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.3 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 1.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo SAC --model_path "$BASELINE_SAC" --noise-env 3.0 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — TD3 (7 runs)
# ============================================================================
echo "========== Building — Env Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.1 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.3 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 1.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 1.5 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 2.0 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env building --algo TD3 --model_path "$BASELINE_TD3" --noise-env 3.0 --n-eval 100 --seed 42
