# Safe RL Training Commands (4 algo x 3 env x 4 limit = 48 runs)
# SACLag removed: off-policy Lagrangian fails across all envs (negative finding for thesis)

# ══════════════════════════════════════════════════════════════════════════════
# EVCharging (cost_scale=100, cost=excess_charge)
# ══════════════════════════════════════════════════════════════════════════════

# PPOLag
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 1 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 5 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 25 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo PPOLag --climit 1000 --seed 42

# CPO
python scripts/train/saferl_training.py --env evcharging --algo CPO --climit 1 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo CPO --climit 5 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo CPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo CPO --climit 1000 --seed 42

# OnCRPO
python scripts/train/saferl_training.py --env evcharging --algo OnCRPO --climit 1 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo OnCRPO --climit 5 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo OnCRPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo OnCRPO --climit 1000 --seed 42

# FOCOPS
python scripts/train/saferl_training.py --env evcharging --algo FOCOPS --climit 1 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo FOCOPS --climit 5 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo FOCOPS --climit 25 --seed 42
python scripts/train/saferl_training.py --env evcharging --algo FOCOPS --climit 1000 --seed 42

# ══════════════════════════════════════════════════════════════════════════════
# Building (cost_scale=1.0, cost=hvac_ramping)
# Initial episode cost ~210-240, min achievable TBD
# ══════════════════════════════════════════════════════════════════════════════

# PPOLag
python scripts/train/saferl_training.py --env building --algo PPOLag --climit 25 --seed 42
python scripts/train/saferl_training.py --env building --algo PPOLag --climit 50 --seed 42
python scripts/train/saferl_training.py --env building --algo PPOLag --climit 100 --seed 42
python scripts/train/saferl_training.py --env building --algo PPOLag --climit 200 --seed 42

# CPO
python scripts/train/saferl_training.py --env building --algo CPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env building --algo CPO --climit 50 --seed 42
python scripts/train/saferl_training.py --env building --algo CPO --climit 100 --seed 42
python scripts/train/saferl_training.py --env building --algo CPO --climit 200 --seed 42

# OnCRPO
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 50 --seed 42
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 100 --seed 42
python scripts/train/saferl_training.py --env building --algo OnCRPO --climit 200 --seed 42

# FOCOPS
python scripts/train/saferl_training.py --env building --algo FOCOPS --climit 25 --seed 42
python scripts/train/saferl_training.py --env building --algo FOCOPS --climit 50 --seed 42
python scripts/train/saferl_training.py --env building --algo FOCOPS --climit 100 --seed 42
python scripts/train/saferl_training.py --env building --algo FOCOPS --climit 200 --seed 42

# ══════════════════════════════════════════════════════════════════════════════
# Cogen (cost_scale=0.005, cost=ramp_costs, calibrated limits)
# Initial episode cost ~240, min achievable ~5-10
# ══════════════════════════════════════════════════════════════════════════════

# PPOLag
python scripts/train/saferl_training.py --env cogen --algo PPOLag --climit 10 --seed 42
python scripts/train/saferl_training.py --env cogen --algo PPOLag --climit 25 --seed 42
python scripts/train/saferl_training.py --env cogen --algo PPOLag --climit 50 --seed 42
python scripts/train/saferl_training.py --env cogen --algo PPOLag --climit 200 --seed 42

# CPO
python scripts/train/saferl_training.py --env cogen --algo CPO --climit 10 --seed 42
python scripts/train/saferl_training.py --env cogen --algo CPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env cogen --algo CPO --climit 50 --seed 42
python scripts/train/saferl_training.py --env cogen --algo CPO --climit 200 --seed 42

# OnCRPO
python scripts/train/saferl_training.py --env cogen --algo OnCRPO --climit 10 --seed 42
python scripts/train/saferl_training.py --env cogen --algo OnCRPO --climit 25 --seed 42
python scripts/train/saferl_training.py --env cogen --algo OnCRPO --climit 50 --seed 42
python scripts/train/saferl_training.py --env cogen --algo OnCRPO --climit 200 --seed 42

# FOCOPS
python scripts/train/saferl_training.py --env cogen --algo FOCOPS --climit 10 --seed 42
python scripts/train/saferl_training.py --env cogen --algo FOCOPS --climit 25 --seed 42
python scripts/train/saferl_training.py --env cogen --algo FOCOPS --climit 50 --seed 42
python scripts/train/saferl_training.py --env cogen --algo FOCOPS --climit 200 --seed 42
