#!/bin/bash -l
#SBATCH -J STDRL_Training_BU_DE_3_SAC
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


# ----------------------------------------------------------
# 4) Debug
# ----------------------------------------------------------
echo "Using python: $(which python)"
python -c "import scipy, sklearn; print('scipy', scipy.__version__, 'sklearn', sklearn.__version__)"

# ----------------------------------------------------------
# 5) Run
# ----------------------------------------------------------
# Baseline
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo PPO --use-vecnormalize --noise 0.0 --seed 42 
# # Temperature-focused suite 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.01 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.03 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.05 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.10 --seed 42 

# # Heavy noise
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.15 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.20 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.30 --seed 42 
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-action 0.40 --seed 42 


# ========== Building noise_env sweep ==========

# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 0.1 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 0.3 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 0.5 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 1.0 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 1.5 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 2.0 --seed 42
# python "${SLURM_SUBMIT_DIR}/scripts/train/stdrl_training.py" --env building --algo SAC --use-vecnormalize --noise-env 3.0 --seed 42

echo
echo "========================================"
echo "End time       : $(date)"
echo "Job finished."
echo "========================================"
