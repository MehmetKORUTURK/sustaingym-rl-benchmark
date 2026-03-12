# ============================================================================
# MARL Batch Run Script — all runs via marl_training.py (unified script)
# ============================================================================
#
# Summary Table (19 runs total):
# ─────────────────────────────────────────────────────────────────────────────
#  #   Env          Algo     Policy Mode    Iterations  Workers
# ─────────────────────────────────────────────────────────────────────────────
#  1   EVCharging   PPO      independent    32000       10
#  2   EVCharging   SAC      independent    32000       10
#  3   EVCharging   APPO     independent    32000       10
#  4   EVCharging   IMPALA   independent    32000       10
#  5   EVCharging   PPO      shared         32000       10
#  6   EVCharging   SAC      shared         32000       10
#  7   EVCharging   APPO     shared         32000       10
#  8   EVCharging   IMPALA   shared         32000       10
# ─────────────────────────────────────────────────────────────────────────────
#  9   Building     PPO      independent    32000       4
# 10   Building     SAC      independent    32000       4
# 11   Building     APPO     independent    32000       4
# 12   Building     IMPALA   independent    32000       4
# 13   Building     PPO      shared         32000       4
# 14   Building     SAC      shared         32000       4
# 15   Building     APPO     shared         32000       4
# 16   Building     IMPALA   shared         32000       4
# ─────────────────────────────────────────────────────────────────────────────
# 17   Cogen        PPO      independent    750         4
# 18   Cogen        APPO     independent    750         4
# 19   Cogen        IMPALA   independent    750         4
# ─────────────────────────────────────────────────────────────────────────────
# Grand Total: 19 runs
#
# Notes:
# - ALL runs use the unified marl_training.py script.
# - TD3 removed (deprecated in Ray 2.10, moved to discontinued rllib_contrib).
# - Cogen only supports PPO, APPO, IMPALA (Dict action spaces incompatible
#   with SAC). Shared policy not supported (agents have different action spaces).
# - EVCharging and Building support 4 algorithms in both independent and
#   shared policy modes.
# - All runs use --seed 42, --lr 3e-4 (defaults).
# - Cogen uses --rm 300 (renewables magnitude).
# - num-iterations and num-workers use script defaults (no need to specify).
# ============================================================================

# Suppress CUBLAS workspace config warnings from PyTorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# EVCharging - Independent Policies (4 runs)
# ============================================================================
echo "========== EVCharging - Independent Policies =========="

python marl_training.py --env evcharging --algo PPO --seed 42
python marl_training.py --env evcharging --algo SAC --seed 42
python marl_training.py --env evcharging --algo APPO --seed 42
python marl_training.py --env evcharging --algo IMPALA --seed 42

# ============================================================================
# EVCharging - Shared Policies (4 runs)
# ============================================================================
echo "========== EVCharging - Shared Policies =========="

python marl_training.py --env evcharging --algo PPO --shared-policy --seed 42
python marl_training.py --env evcharging --algo SAC --shared-policy --seed 42
python marl_training.py --env evcharging --algo APPO --shared-policy --seed 42
python marl_training.py --env evcharging --algo IMPALA --shared-policy --seed 42

# ============================================================================
# Building - Independent Policies (4 runs)
# ============================================================================
echo "========== Building - Independent Policies =========="

python marl_training.py --env building --algo PPO --seed 42
python marl_training.py --env building --algo SAC --seed 42
python marl_training.py --env building --algo APPO --seed 42
python marl_training.py --env building --algo IMPALA --seed 42

# ============================================================================
# Building - Shared Policies (4 runs)
# ============================================================================
echo "========== Building - Shared Policies =========="

python marl_training.py --env building --algo PPO --shared-policy --seed 42
python marl_training.py --env building --algo SAC --shared-policy --seed 42
python marl_training.py --env building --algo APPO --shared-policy --seed 42
python marl_training.py --env building --algo IMPALA --shared-policy --seed 42

# ============================================================================
# Cogen - Independent Policies Only (3 runs)
# No SAC (Dict action spaces). No shared policy (different action spaces).
# ============================================================================
echo "========== Cogen - Independent Policies =========="

python marl_training.py --env cogen --algo PPO --rm 300 --seed 42
python marl_training.py --env cogen --algo APPO --rm 300 --seed 42
python marl_training.py --env cogen --algo IMPALA --rm 300 --seed 42
