"""
Dark Store Project - PPO Training Experiment: LR=0.0001
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# Import our custom environment
BASE_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = BASE_DIR / "code"
sys.path.append(str(CODE_DIR))
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

EXPERIMENT_DIR = Path(__file__).resolve().parents[0]

print("=" * 60)
print("DARK STORE - PPO TRAINING EXPERIMENT: LR=0.0001")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
LAYOUT_FILE = str(BASE_DIR / "results" / "optimized_layout.json")
MODEL_SAVE_PATH = str(EXPERIMENT_DIR / "ppo_model")
LOG_PATH = str(EXPERIMENT_DIR / "training_log.csv")

# Training parameters
TOTAL_TIMESTEPS = 200000  # Shorter for experiments
LEARNING_RATE = 1e-4  # 0.0001
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

# ============================================
# CUSTOM CALLBACK - Track Progress
# ============================================

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_path=None, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.logs = []
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            info = self.locals['infos'][0]
            success = info.get('success', False)
            self.episode_successes.append(1 if success else 0)
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        if self.num_timesteps % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]
                recent_lengths = self.episode_lengths[-100:]
                recent_successes = self.episode_successes[-100:]
                
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"\nTimestep {self.num_timesteps}/{TOTAL_TIMESTEPS}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Avg Length (last 100): {avg_length:.1f}")
                print(f"  Success Rate: {success_rate:.1f}%")
                
                self.logs.append({
                    'timestep': self.num_timesteps,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'success_rate': success_rate
                })
        
        return True
    
    def _on_training_end(self) -> None:
        if self.log_path and len(self.logs) > 0:
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_path, index=False)
            print(f"\n✓ Training log saved to: {self.log_path}")

# ============================================
# STEP 1: Create Environment
# ============================================
print("\n[1/5] Creating training environment...")

try:
    env = DarkStoreEnv(LAYOUT_FILE, max_steps=250, max_items_per_order=3)
    env = DummyVecEnv([lambda: env])
    
    print("✓ Environment created successfully!")
    print(f"  Grid size: 12×13")
    print(f"  Max steps per episode: 250")
    
except Exception as e:
    print(f"❌ Error creating environment: {e}")
    exit()

# ============================================
# STEP 2: Create PPO Model
# ============================================
print("\n[2/5] Creating PPO model...")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1
)

print("✓ PPO model created!")
print(f"  Policy: MLP [256 → 256]")
print(f"  Learning rate: {LEARNING_RATE}")

# ============================================
# STEP 3: Train the Model
# ============================================
print("\n[3/5] Starting PPO training...")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Estimated time: 10-20 minutes")
print(f"\nTraining progress:")
print("=" * 60)

start_time = time.time()
callback = ProgressCallback(check_freq=1000, log_path=LOG_PATH)

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"  Total time: {training_time/60:.1f} minutes")
    
except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user")
    training_time = time.time() - start_time

# ============================================
# STEP 4: Save the Model
# ============================================
print("\n[4/5] Saving trained model...")

model.save(MODEL_SAVE_PATH)
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

# ============================================
# STEP 5: Quick Evaluation
# ============================================
print("\n[5/5] Running quick evaluation...")

# Evaluate for 100 episodes
eval_rewards = []
eval_lengths = []
eval_successes = []

for ep in range(100):
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
        info = info[0] if isinstance(info, (list, tuple)) else info
        episode_reward += reward[0]
        episode_length += 1
        
        if episode_length >= 500:  # Safety limit
            break
    
    eval_rewards.append(episode_reward)
    eval_lengths.append(episode_length)
    success = info.get('success', False)
    eval_successes.append(1 if success else 0)

avg_eval_reward = np.mean(eval_rewards)
avg_eval_length = np.mean(eval_lengths)
eval_success_rate = np.mean(eval_successes) * 100

print("✓ Evaluation complete!")
print(f"  Avg Reward: {avg_eval_reward:.2f}")
print(f"  Avg Length: {avg_eval_length:.1f}")
print(f"  Success Rate: {eval_success_rate:.1f}%")

# Save evaluation results
eval_results = {
    'experiment': 'lr_0.0001',
    'learning_rate': LEARNING_RATE,
    'avg_reward': avg_eval_reward,
    'avg_length': avg_eval_length,
    'success_rate': eval_success_rate,
    'training_time_min': training_time/60
}

eval_df = pd.DataFrame([eval_results])
eval_df.to_csv(str(EXPERIMENT_DIR / "eval_results.csv"), index=False)

print(f"✓ Results saved to: {EXPERIMENT_DIR / 'eval_results.csv'}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE! ✓")
print("=" * 60)