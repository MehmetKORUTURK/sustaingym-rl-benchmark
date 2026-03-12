import os
from sustaingym.envs.cogen import CogenEnv
from envs.cogen.MyCogenEnv import MyCogenEnv
from envs.evcharging import EVChargingEnv, GMMsTraceGenerator
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, sync_envs_normalization
from envs.building import BuildingEnv, ParameterGenerator
import numpy as np
import argparse
from ttp.colored import color_text, Colors
import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import time

# Import VecNormalize separately if not in the line above
try:
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    pass

parser = argparse.ArgumentParser(description='Train RL agents with SustainGym environments')
parser.add_argument("--env", type=str, default="evcharging",
                   choices=["cogen", "evcharging", "building"],
                   help="Environment to train on")
parser.add_argument("--rm", type=int, default=300,
                   help="Renewables magnitude for cogen environment")
parser.add_argument("--algo", type=str, default="PPO",
                   choices=["PPO", "SAC", "TD3"],
                   help="RL algorithm to use")
parser.add_argument("--noise", type=float, default=0.0,
                   help="Observation/state noise level (Cogen: 0-2.0 scale, Others: standard noise)")
parser.add_argument("--noise-action", type=float, default=0.0,
                   help="Action noise level (e.g., 0.05 for 5%% noise)")
parser.add_argument("--noise-env", type=float, default=0.0,
                   help="Environment noise scale (1.0=default: temp 2F, pressure 0.1psia, humidity 0.05)")
parser.add_argument("--cogen_train_steps", type=int, default=3_000_000,
                   help="Training steps for cogen environment")
parser.add_argument("--evcharging_train_steps", type=int, default=9_000_000,
                   help="Training steps for evcharging environment")
parser.add_argument("--building_train_steps", type=int, default=9_000_000,
                   help="Training steps for building environment")
# New options
parser.add_argument("--eval-freq", type=int, default=1_000_000,
                   help="Evaluation frequency (steps) - increase for faster training")
parser.add_argument("--n-eval-episodes", type=int, default=1,
                   help="Number of evaluation episodes - decrease for faster training")
parser.add_argument("--checkpoint-freq", type=int, default=2_500_000,
                   help="Checkpoint save frequency (steps) - increase for faster training")
parser.add_argument("--use-vecnormalize", action="store_true",
                   help="Use VecNormalize for observation/reward normalization")
parser.add_argument("--norm-reward", action="store_true",
                   help="Normalize rewards (only with --use-vecnormalize)")
parser.add_argument("--quick-test", action="store_true",
                   help="Quick test run with reduced steps")
parser.add_argument("--no-eval", action="store_true",
                   help="Disable evaluation callbacks")
parser.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
args = parser.parse_args()

ENV = args.env # "cogen", "evcharging", "building"
RM = args.rm
ALGO = args.algo # "PPO", "SAC" "TD3"
NOISE = args.noise  # Observation/state noise (all environments)
NOISE_ACTION = args.noise_action
NOISE_ENV = args.noise_env  # Environment noise scale
COGEN_TRAIN_STEPS = args.cogen_train_steps
EVCHARGING_TRAIN_STEPS = args.evcharging_train_steps
BUILDING_TRAIN_STEPS = args.building_train_steps
SEED = args.seed

# New configuration variables
EVAL_FREQ = args.eval_freq
N_EVAL_EPISODES = args.n_eval_episodes
CHECKPOINT_FREQ = args.checkpoint_freq
USE_VECNORMALIZE = args.use_vecnormalize
NORM_REWARD = args.norm_reward
NO_EVAL = args.no_eval

# Quick test mode
if args.quick_test:
    COGEN_TRAIN_STEPS = 10000
    EVCHARGING_TRAIN_STEPS = 10000
    BUILDING_TRAIN_STEPS = 10000
    EVAL_FREQ = 10000  # Evaluate only once at the end
    CHECKPOINT_FREQ = 10000  # Checkpoint only at the end
    N_EVAL_EPISODES = 1  # Minimal evaluation
    print(color_text("\n[INFO] Quick test mode enabled (10k steps, minimal evaluation)", Colors.GREEN))

# Generate a timestamp string
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Log directory naming
logdir = f"./logs_std_train/{ENV}_{ALGO}/{timestamp}_NOISE_{NOISE}_ACT_{NOISE_ACTION}_ENV_{NOISE_ENV}"
os.makedirs(logdir, exist_ok=True)

# Print detailed configuration summary
print("\n" + "="*80)
print(color_text("TRAINING CONFIGURATION", Colors.GREEN))
print("="*80)
print(f"Environment:            {ENV}")
print(f"Algorithm:              {ALGO}")
print(f"Renewables Magnitude:   {RM}" if ENV == "cogen" else "")
print(f"Random Seed:            {SEED}")
print(f"Noise Level:            {NOISE}")
print(f"Action Noise:           {NOISE_ACTION}")
print(f"Environment Noise:      {NOISE_ENV}")
print(f"Total Timesteps:        {COGEN_TRAIN_STEPS:,}" if ENV == "cogen" else
      f"Total Timesteps:        {EVCHARGING_TRAIN_STEPS:,}" if ENV == "evcharging" else
      f"Total Timesteps:        {BUILDING_TRAIN_STEPS:,}")
print(f"Evaluation Frequency:   {EVAL_FREQ:,} steps")
print(f"Eval Episodes:          {N_EVAL_EPISODES}")
print(f"Checkpoint Frequency:   {CHECKPOINT_FREQ:,} steps")
print(f"Use VecNormalize:       {'Yes' if USE_VECNORMALIZE else 'No'}")
print(f"Normalize Rewards:      {'Yes' if NORM_REWARD else 'No'}")
print(f"Evaluation Enabled:     {'No' if NO_EVAL else 'Yes'}")
print(f"Log Directory:          {logdir}")
print(f"TensorBoard Log:        {logdir}")
print("="*80 + "\n")

# Helper function to wrap environment with VecNormalize
def wrap_env_with_vecnormalize(env, normalize_obs=True, normalize_reward=False, training=True):
    """Wrap environment with DummyVecEnv and VecNormalize

    Args:
        env: Environment to wrap
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        training: If True, updates running mean/std during training.
                 If False, uses fixed stats (for evaluation)
    """
    env = DummyVecEnv([lambda: env])
    if USE_VECNORMALIZE:
        env = VecNormalize(env,
                          training=training,  # Critical: False for eval env!
                          norm_obs=normalize_obs,
                          norm_reward=normalize_reward,
                          clip_obs=10.0,
                          clip_reward=10.0,
                          gamma=0.99,
                          epsilon=1e-8)
        print(color_text(f"[INFO] VecNormalize enabled (obs={normalize_obs}, reward={normalize_reward}, training={training})", Colors.GREEN))
    return env


# Custom callback to sync VecNormalize stats before each evaluation
class SyncVecNormalizeCallback(BaseCallback):
    """
    Callback to synchronize VecNormalize statistics from training env to eval env.

    This is CRITICAL for correct evaluation when using VecNormalize:
    - Train env continuously updates its running mean/std during training
    - Eval env must use the SAME stats to properly normalize observations
    - Without sync, eval uses stale/wrong stats → incorrect best model selection
    """
    def __init__(self, train_env, eval_env, eval_freq, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        # Sync before each evaluation (use num_timesteps for robustness with vectorized envs)
        if self.eval_freq > 0 and self.model.num_timesteps % self.eval_freq == 0:
            if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(self.train_env, self.eval_env)
                if self.verbose > 0:
                    print(color_text(f"[Callback] Synced VecNormalize stats at timestep {self.model.num_timesteps:,}", Colors.GREEN))
        return True

# Set random seeds for reproducibility
print(color_text(f"\n[1/5] Setting random seed to {SEED}...", Colors.GREEN))
np.random.seed(SEED)

# Create training and evaluation environments
print(color_text("\n[2/5] Creating environments...", Colors.GREEN))

if ENV == "cogen":
    # Build noise_env dict from scalar scale
    cogen_noise_env = None
    if NOISE_ENV > 0:
        cogen_noise_env = {
            "temperature": 2.0 * NOISE_ENV,   # Base: 2 F std dev
            "pressure": 0.1 * NOISE_ENV,      # Base: 0.1 psia std dev
            "humidity": 0.05 * NOISE_ENV,      # Base: 0.05 fraction std dev
            "model": 0.15 * NOISE_ENV,         # Base: 15% ONNX output uncertainty (increased from 4%)
        }
        print(f"Environment noise config: {cogen_noise_env}")

    # Training environment
    base_train_env = MyCogenEnv(CogenEnv(renewables_magnitude=RM), noise=NOISE, noise_action=NOISE_ACTION, noise_env=cogen_noise_env)
    base_train_env.reset(seed=SEED)  # Seed environment before wrapping
    train_env = Monitor(base_train_env, logdir, allow_early_resets=False)
    train_env = wrap_env_with_vecnormalize(train_env, normalize_obs=True, normalize_reward=NORM_REWARD, training=True)

    # Evaluation environment
    base_eval_env = MyCogenEnv(CogenEnv(renewables_magnitude=RM), noise=NOISE, noise_action=NOISE_ACTION, noise_env=cogen_noise_env)
    base_eval_env.reset(seed=SEED + 1)  # Use different seed for evaluation
    eval_env = Monitor(base_eval_env, logdir + "/eval")
    eval_env = wrap_env_with_vecnormalize(eval_env, normalize_obs=True, normalize_reward=False, training=False)

    # CRITICAL: Sync normalization statistics from training to evaluation env
    if USE_VECNORMALIZE and isinstance(train_env, VecNormalize):
        sync_envs_normalization(train_env, eval_env)
        print(color_text("[INFO] Synced VecNormalize stats: train -> eval", Colors.GREEN))

    TOTAL_TIMESTEPS = COGEN_TRAIN_STEPS
    POLICY_TYPE = "MultiInputPolicy"
    
elif ENV == "evcharging":
    gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')

    # EVCharging noise_env: scalar std dev applied to MOER in reward computation
    ev_noise_env = NOISE_ENV if NOISE_ENV > 0 else None
    if ev_noise_env is not None:
        print(f"EVCharging environment noise (MOER std dev): {ev_noise_env}")

    # Training environment
    base_train_env = EVChargingEnv(gmmg, noise=NOISE, noise_action=NOISE_ACTION, noise_env=ev_noise_env)
    base_train_env.reset(seed=SEED)  # Seed environment before wrapping
    train_env = Monitor(base_train_env, logdir, allow_early_resets=False)
    train_env = wrap_env_with_vecnormalize(train_env, normalize_obs=True, normalize_reward=NORM_REWARD, training=True)

    # Evaluation environment
    base_eval_env = EVChargingEnv(gmmg, noise=NOISE, noise_action=NOISE_ACTION, noise_env=ev_noise_env)
    base_eval_env.reset(seed=SEED + 1)  # Use different seed for evaluation
    eval_env = Monitor(base_eval_env, logdir + "/eval")
    eval_env = wrap_env_with_vecnormalize(eval_env, normalize_obs=True, normalize_reward=False, training=False)

    # CRITICAL: Sync normalization statistics from training to evaluation env
    if USE_VECNORMALIZE and isinstance(train_env, VecNormalize):
        sync_envs_normalization(train_env, eval_env)
        print(color_text("[INFO] Synced VecNormalize stats: train -> eval", Colors.GREEN))

    TOTAL_TIMESTEPS = EVCHARGING_TRAIN_STEPS
    POLICY_TYPE = "MultiInputPolicy"

elif ENV == "building":
    params = ParameterGenerator(building='OfficeSmall', weather='Hot_Dry', location='Tucson', reward_beta=0.5)

    # Building noise_env: perturbs outdoor temp, ground temp, and solar irradiance
    building_noise_env = None
    if NOISE_ENV > 0:
        building_noise_env = {
            "out_temp": 1.0 * NOISE_ENV,      # Base: 1 °C std dev
            "ground_temp": 0.5 * NOISE_ENV,    # Base: 0.5 °C std dev
            "ghi": 50.0 * NOISE_ENV,           # Base: 50 W/m² std dev
        }
        print(f"Building environment noise config: {building_noise_env}")

    # Training environment
    base_train_env = BuildingEnv(params, noise=NOISE, noise_action=NOISE_ACTION, noise_env=building_noise_env)
    base_train_env.reset(seed=SEED)  # Seed environment before wrapping
    train_env = Monitor(base_train_env, logdir, allow_early_resets=False)
    train_env = wrap_env_with_vecnormalize(train_env, normalize_obs=True, normalize_reward=NORM_REWARD, training=True)

    # Evaluation environment
    base_eval_env = BuildingEnv(params, noise=NOISE, noise_action=NOISE_ACTION, noise_env=building_noise_env)
    base_eval_env.reset(seed=SEED + 1)  # Use different seed for evaluation
    eval_env = Monitor(base_eval_env, logdir + "/eval")
    eval_env = wrap_env_with_vecnormalize(eval_env, normalize_obs=True, normalize_reward=False, training=False)

    # CRITICAL: Sync normalization statistics from training to evaluation env
    if USE_VECNORMALIZE and isinstance(train_env, VecNormalize):
        sync_envs_normalization(train_env, eval_env)
        print(color_text("[INFO] Synced VecNormalize stats: train -> eval", Colors.GREEN))

    TOTAL_TIMESTEPS = BUILDING_TRAIN_STEPS
    POLICY_TYPE = "MlpPolicy"

# Create model with TensorBoard logging
print(color_text("\n[3/5] Creating model...", Colors.GREEN))

if ENV == "cogen":
    if ALGO == "PPO":
        model = PPO(POLICY_TYPE, train_env, 
                    learning_rate=5e-5, 
                    n_steps=1024, 
                    batch_size=256,
                    gamma=0.995,
                    gae_lambda=0.95,
                    ent_coef=0.001,
                    clip_range=0.2,
                    n_epochs=10,
                    max_grad_norm=0.5, 
                    verbose=1, 
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "TD3":
        model = TD3(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "SAC":
        model = SAC(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef="auto",
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)

elif ENV == "evcharging":
    if ALGO == "PPO":
        model = PPO(POLICY_TYPE, train_env,
                    learning_rate=5e-5,
                    n_steps=2048,
                    batch_size=256,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.005,
                    clip_range=0.2,
                    n_epochs=10,
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "TD3":
        model = TD3(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "SAC":
        model = SAC(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef="auto",
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)

elif ENV == "building":
    if ALGO == "PPO":
        model = PPO(POLICY_TYPE, train_env,
                    learning_rate=3e-5,
                    n_steps=2048,
                    batch_size=128,
                    gamma=0.995,
                    gae_lambda=0.95,
                    ent_coef=0.001,
                    clip_range=0.2,
                    n_epochs=10,
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "TD3":
        model = TD3(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)
    elif ALGO == "SAC":
        model = SAC(POLICY_TYPE, train_env,
                    learning_rate=3e-4,
                    gamma=0.99,
                    buffer_size=1_000_000,
                    learning_starts=10_000,
                    batch_size=256,
                    tau=0.005,
                    ent_coef="auto",
                    verbose=1,
                    tensorboard_log=logdir,
                    device="cpu",
                    seed=SEED)

print(color_text(f"[INFO] Model: {ALGO} with {POLICY_TYPE}", Colors.GREEN))
print(color_text(f"[INFO] TensorBoard logging enabled: {logdir}", Colors.GREEN))

# Create callbacks
print(color_text("\n[4/5] Setting up callbacks...", Colors.GREEN))
callbacks = []

# VecNormalize sync callback (CRITICAL for correct evaluation!)
if USE_VECNORMALIZE and not NO_EVAL and isinstance(train_env, VecNormalize):
    sync_callback = SyncVecNormalizeCallback(
        train_env=train_env,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        verbose=1
    )
    callbacks.append(sync_callback)
    print(color_text(f"[INFO] VecNormalize sync callback enabled (syncs before each eval)", Colors.GREEN))

if not NO_EVAL:
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=logdir,  # Best model in root (easy to find)
        log_path=logdir + "/eval",     # Eval logs in eval subfolder (clean organization)
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    print(color_text(f"[INFO] Evaluation callback enabled (freq={EVAL_FREQ:,}, episodes={N_EVAL_EPISODES})", Colors.GREEN))

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=logdir,
    name_prefix=f"{ENV}_{ALGO}_checkpoint",
    save_vecnormalize=True if USE_VECNORMALIZE else False
)
callbacks.append(checkpoint_callback)
print(color_text(f"[INFO] Checkpoint callback enabled (freq={CHECKPOINT_FREQ:,})", Colors.GREEN))

# Training with duration tracking and error handling
print(color_text(f"\n[5/5] Starting training for {TOTAL_TIMESTEPS:,} timesteps...", Colors.GREEN))
print(color_text(f"[INFO] Progress bar disabled", Colors.GREEN))
print(color_text(f"[INFO] Estimated time: ~{TOTAL_TIMESTEPS / 100000 * 10:.0f} minutes (varies by environment)", Colors.GREEN))
print()

start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks if callbacks else None,
        progress_bar=False
    )
    training_success = True
except KeyboardInterrupt:
    print(color_text("\n\n[WARNING] Training interrupted by user!", Colors.GREEN))
    training_success = False
except Exception as e:
    print(color_text(f"\n\n[ERROR] Training failed: {e}", Colors.GREEN))
    training_success = False
    raise

end_time = time.time()
duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60

print(color_text(f"\n[INFO] Training duration: {duration_minutes:.1f} minutes ({duration_seconds:.0f} seconds)", Colors.GREEN))

# Save final model with improved naming
model.save(f"{logdir}/{ENV}_{ALGO}_final")
print(color_text(f"Model saved in {logdir}/{ENV}_{ALGO}_final.zip", Colors.UNDERLINE + Colors.GREEN))

# Save normalization statistics
if USE_VECNORMALIZE and isinstance(train_env, VecNormalize):
    norm_stats_path = f"{logdir}/vec_normalize.pkl"
    train_env.save(norm_stats_path)
    print(color_text(f"[INFO] VecNormalize stats saved to {norm_stats_path}", Colors.GREEN))

# Quick evaluation after training
if training_success:
    print(color_text("\n" + "="*80, Colors.GREEN))
    print(color_text("POST-TRAINING EVALUATION", Colors.GREEN))
    print(color_text("="*80, Colors.GREEN))

    # Sync VecNormalize stats one final time before evaluation
    if USE_VECNORMALIZE and isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        sync_envs_normalization(train_env, eval_env)
        print(color_text("[INFO] Final VecNormalize sync for post-training eval", Colors.GREEN))

    # Load and evaluate BEST model (if available from EvalCallback)
    best_model_path = f"{logdir}/best_model.zip"

    if not NO_EVAL and os.path.exists(best_model_path):
        # ✅ Evaluate BEST model (selected by EvalCallback)
        print(color_text(f"[INFO] Loading best model from {best_model_path}", Colors.GREEN))

        # Get algorithm class dynamically
        algo_class = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[ALGO]
        best_model = algo_class.load(best_model_path, env=eval_env)

        print(color_text(f"\n[EVAL] Running evaluation on BEST model ({N_EVAL_EPISODES} episodes)...", Colors.GREEN))
        mean_reward, std_reward = evaluate_policy(
            best_model, eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )
        print(color_text(f"[EVAL] BEST model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", Colors.GREEN))
    else:
        # Fallback to final model if no best model available
        if NO_EVAL:
            print(color_text("[INFO] Evaluation was disabled, using final model", Colors.GREEN))
        else:
            print(color_text("[INFO] No best model found, using final model", Colors.GREEN))

        print(color_text(f"\n[EVAL] Running evaluation on final model ({N_EVAL_EPISODES} episodes)...", Colors.GREEN))
        mean_reward, std_reward = evaluate_policy(
            model, eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )
        print(color_text(f"[EVAL] Final model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}", Colors.GREEN))

    print()

print(color_text("\n" + "="*80, Colors.GREEN))
print(color_text("TRAINING COMPLETED!", Colors.GREEN))
print(color_text("="*80, Colors.GREEN))
print(f"Training duration:     {duration_minutes:.1f} minutes")
print(f"Final model:           {logdir}/{ENV}_{ALGO}_final.zip")
print(f"Best model:            {logdir}/best_model.zip" if not NO_EVAL else "")
print(f"Logs saved in:         {logdir}")
print(color_text("="*80 + "\n", Colors.GREEN))

# Next steps guidance
print(color_text("NEXT STEPS:", Colors.GREEN))
print(color_text("-" * 80, Colors.GREEN))
print(color_text("1. View training progress with TensorBoard:", Colors.GREEN))
print(f"   tensorboard --logdir={logdir}")
print()
print(color_text("2. Load and evaluate the trained model:", Colors.GREEN))
print(f"   from stable_baselines3 import {ALGO}")
print(f"   model = {ALGO}.load('{logdir}/{ENV}_{ALGO}_final.zip')")
if USE_VECNORMALIZE:
    print(f"   # Don't forget to load VecNormalize stats:")
    print(f"   from stable_baselines3.common.vec_env import VecNormalize")
    print(f"   env = VecNormalize.load('{logdir}/vec_normalize.pkl', env)")
print()
print(color_text("3. Compare different training runs:", Colors.GREEN))
print(f"   tensorboard --logdir=./logs_std_train/{ENV}_{ALGO}")
print()
print(color_text("4. Resume training from checkpoint:", Colors.GREEN))
print(f"   model = {ALGO}.load('{logdir}/{ENV}_{ALGO}_checkpoint_XXX_steps.zip')")
print(f"   model.learn(total_timesteps=additional_steps)")
print(color_text("-" * 80 + "\n", Colors.GREEN))