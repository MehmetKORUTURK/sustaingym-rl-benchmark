#!/bin/bash -l
#SBATCH -J STDRL_Training_EV_DE_0.30_PPO
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

echo "========================================"
echo "Job started on : $(hostname)"
echo "Start time     : $(date)"
echo "Submit dir     : ${SLURM_SUBMIT_DIR}"
echo "Working dir    : $(pwd)"
echo "========================================"
echo

cd "${SLURM_SUBMIT_DIR}"

echo "========================================"
echo "Working dir after cd: $(pwd)"
echo "========================================"
echo

# ----------------------------------------------------------
# 1) Conda init: conda.sh yolunu otomatik bul
# ----------------------------------------------------------
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
fi

if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found."
  echo "conda in PATH? -> $(command -v conda || echo 'NO')"
  echo "conda base -> ${CONDA_BASE:-'EMPTY'}"
  echo "Tried -> $CONDA_BASE/etc/profile.d/conda.sh"
  exit 1
fi

# ----------------------------------------------------------
# 2) Env activate
# ----------------------------------------------------------
conda activate stdrl_train

# ----------------------------------------------------------
# 3) KRITIK: conda libstdc++ once gelsin
# ----------------------------------------------------------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Torch thread kontrolü (SB3/PyTorch CPU tarafı için)
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_DISTRIBUTED_DEBUG=OFF

# Loglar anlık aksın
export PYTHONUNBUFFERED=1
# ----------------------------------------------------------
# 4) Debug
# ----------------------------------------------------------
echo "Using python: $(which python)"
python -c "import scipy, sklearn; print('scipy', scipy.__version__, 'sklearn', sklearn.__version__)"

# ----------------------------------------------------------
# 5) Run
# ----------------------------------------------------------
# Baseline
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.0 --seed 42
# Light noise suite
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.01 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.05 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.10 --seed 42
# # Heavy noise suite
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.15 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.20 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.30 --seed 42

# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.40 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.50 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise 0.60 --seed 42

###_DA_
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.01 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.05 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.10 --seed 42
# # Heavy noise suite
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.15 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.20 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.30 --seed 42

# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.40 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.50 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo SAC --use-vecnormalize --noise-action 0.60 --seed 42

# "========== EVCharging noise_env sweep =========="

# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.01 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.02 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.05 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.10 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.15 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.20 --seed 42
python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env evcharging --algo PPO --use-vecnormalize --noise-env 0.30 --seed 42


echo
echo "========================================"
echo "End time       : $(date)"
echo "Job finished."
echo "========================================"
