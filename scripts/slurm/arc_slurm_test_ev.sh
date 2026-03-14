# ============================================================================
# StdRL Testing — EVCharging Robustness (Baseline model vs increasing noise)
# ============================================================================
#
# Tests the noise=0 baseline model at all trained noise levels.
# Shows how a clean-trained model degrades under noise at inference time.
#
# Summary Table:
# ─────────────────────────────────────────────────────────────────────────────
#  Noise Type    Algos          Levels Tested
# ─────────────────────────────────────────────────────────────────────────────
#  Observation   PPO,SAC,TD3    0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60
#  Action        PPO,SAC,TD3    0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60
#  Environment   PPO,SAC,TD3    0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60
# ─────────────────────────────────────────────────────────────────────────────
# Grand Total: 81 runs (3 algos x 27 noise levels)
#
# Notes:
# - UPDATE BASELINE_* paths below to point to your actual trained model dirs.
# - All runs use --n-eval 100, --seed 42, deterministic policy.
# ============================================================================

# ── Baseline model paths (UPDATE THESE) ──
BASELINE_PPO="logs_std_train/evcharging_PPO/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_SAC="logs_std_train/evcharging_SAC/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"
BASELINE_TD3="logs_std_train/evcharging_TD3/<timestamp>_NOISE_0.0_ACT_0.0_ENV_0.0/best_model.zip"

# ============================================================================
# Observation Noise Robustness — PPO (10 runs)
# ============================================================================
echo "========== EVCharging — Obs Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise 0.60 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — SAC (10 runs)
# ============================================================================
echo "========== EVCharging — Obs Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise 0.60 --n-eval 100 --seed 42

# ============================================================================
# Observation Noise Robustness — TD3 (10 runs)
# ============================================================================
echo "========== EVCharging — Obs Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.0  --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise 0.60 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — PPO (8 runs)
# ============================================================================
echo "========== EVCharging — Act Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-action 0.60 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — SAC (8 runs)
# ============================================================================
echo "========== EVCharging — Act Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-action 0.60 --n-eval 100 --seed 42

# ============================================================================
# Action Noise Robustness — TD3 (8 runs)
# ============================================================================
echo "========== EVCharging — Act Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-action 0.60 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — PPO (9 runs)
# ============================================================================
echo "========== EVCharging — Env Noise Robustness (PPO) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo PPO --model_path "$BASELINE_PPO" --noise-env 0.60 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — SAC (9 runs)
# ============================================================================
echo "========== EVCharging — Env Noise Robustness (SAC) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo SAC --model_path "$BASELINE_SAC" --noise-env 0.60 --n-eval 100 --seed 42

# ============================================================================
# Environment Noise Robustness — TD3 (9 runs)
# ============================================================================
echo "========== EVCharging — Env Noise Robustness (TD3) =========="

python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.01 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.05 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.10 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.15 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.20 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.30 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.40 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.50 --n-eval 100 --seed 42
python scripts/test/stdrl_testing.py --env evcharging --algo TD3 --model_path "$BASELINE_TD3" --noise-env 0.60 --n-eval 100 --seed 42
