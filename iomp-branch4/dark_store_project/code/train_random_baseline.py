"""
Dark Store Project - PPO Training
Step 4: Train the reinforcement learning agent
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
sys.stdout.reconfigure(encoding='utf-8')

# Import our custom environment - FIXED for filename 3_gym_environment.py
import sys
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

print("=" * 60)
print("DARK STORE - PPO TRAINING")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
LAYOUT_FILE = "D:/iomp/dark_store_project/results/random_layout.json"
MODEL_SAVE_PATH = "D:/iomp/dark_store_project/results/ppo_model_random"
LOG_PATH = "D:/iomp/dark_store_project/results/training_log_random.csv"

# Training parameters
TOTAL_TIMESTEPS = 500000  # 500k for multi-agent training
LEARNING_RATE = 3e-4
N_STEPS = 2048  # Steps per update
BATCH_SIZE = 64
N_EPOCHS = 10

# ============================================
# CUSTOM CALLBACK - Track Progress
# ============================================

class ProgressCallback(BaseCallback):
    """
    Custom callback to track training progress and save logs
    """
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
        # Track current episode
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Use the success flag from env — True only when items done + at depot
            info = self.locals['infos'][0]
            success = info.get('success', False)
            self.episode_successes.append(1 if success else 0)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log progress every check_freq steps
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
                
                # Save log
                self.logs.append({
                    'timestep': self.num_timesteps,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    'success_rate': success_rate
                })
        
        return True
    
    def _on_training_end(self) -> None:
        """Save logs when training ends"""
        if self.log_path and len(self.logs) > 0:
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_path, index=False)
            print(f"\n✓ Training log saved to: {self.log_path}")

# ============================================
# STEP 1: Create Environment
# ============================================
print("\n[1/5] Creating training environment...")

try:
    # Create environment
    env = DarkStoreEnv(LAYOUT_FILE, max_steps=250, max_items_per_order=3)
    
    # Wrap in vectorized environment (required by Stable-Baselines3)
    env = DummyVecEnv([lambda: env])
    
    print("✓ Environment created successfully!")
    print(f"  Grid size: 12×13")
    print(f"  Max steps per episode: 350")
    
except Exception as e:
    print(f"❌ Error creating environment: {e}")
    exit()

# ============================================
# STEP 2: Create PPO Model (FRESH)
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

print("\u2713 PPO model created!")
print(f"  Policy: MLP [256 → 256]")
print(f"  Action space: 4 (Up/Down/Left/Right — auto-pick)")
print(f"  Learning rate: {LEARNING_RATE}")

# ============================================
# STEP 3: Train the Model
# ============================================
print("\n[3/5] Starting PPO training...")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Estimated time: 30-60 minutes")
print(f"\nTraining progress:")
print("=" * 60)

start_time = time.time()

# Create callback
callback = ProgressCallback(check_freq=1000, log_path=LOG_PATH)

try:
    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False  # Disabled to show real errors; re-enable after fix
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"  Total time: {training_time/60:.1f} minutes")
    
except KeyboardInterrupt:
    print("\n\n⚠ Training interrupted by user")
    print("  Saving current model...")
    training_time = time.time() - start_time

# ============================================
# STEP 4: Save the Model
# ============================================
print("\n[4/5] Saving trained model...")

model.save(MODEL_SAVE_PATH)
print(f"✓ Model saved to: {MODEL_SAVE_PATH}.zip")

# ============================================
# STEP 5: Evaluate the Model
# ============================================
print("\n[5/5] Evaluating trained model...")

# Test on 100 episodes
n_eval_episodes = 100
episode_rewards = []
episode_lengths = []
episode_successes = []

for i in range(n_eval_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)  # False avoids wall-loop trapping
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        episode_length += 1
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    # Check success (items picked == total items)
    # Note: info is a list with one dict
    success = info[0]['items_picked'] == info[0]['total_items']
    episode_successes.append(1 if success else 0)

# Calculate statistics
avg_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
avg_length = np.mean(episode_lengths)
success_rate = np.mean(episode_successes) * 100

print(f"\nEvaluation results (100 episodes):")
print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"  Average episode length: {avg_length:.1f} steps")
print(f"  Success rate: {success_rate:.1f}%")

# ============================================
# STEP 6: Create Learning Curve Visualization
# ============================================
print("\n[6/6] Creating learning curve visualization...")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average Reward
    axes[0, 0].plot(log_df['timestep'], log_df['avg_reward'], linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Timesteps', fontsize=12)
    axes[0, 0].set_ylabel('Average Reward', fontsize=12)
    axes[0, 0].set_title('PPO Learning Curve - Reward', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[0, 0].legend()
    
    # Plot 2: Success Rate
    axes[0, 1].plot(log_df['timestep'], log_df['success_rate'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Timesteps', fontsize=12)
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0, 1].set_title('PPO Learning Curve - Success Rate', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # Plot 3: Episode Length
    axes[1, 0].plot(log_df['timestep'], log_df['avg_length'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Timesteps', fontsize=12)
    axes[1, 0].set_ylabel('Average Episode Length', fontsize=12)
    axes[1, 0].set_title('PPO Learning Curve - Episode Length', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training Summary Text
    axes[1, 1].axis('off')
    summary_text = f"""
    TRAINING SUMMARY
    ═══════════════════════════════════
    
    Total Timesteps: {TOTAL_TIMESTEPS:,}
    Training Time: {training_time/60:.1f} min
    
    FINAL PERFORMANCE (100 episodes):
    ─────────────────────────────────────
    Average Reward: {avg_reward:.2f}
    Success Rate: {success_rate:.1f}%
    Avg Steps: {avg_length:.1f}
    
    IMPROVEMENT vs Random Agent:
    ─────────────────────────────────────
    Random Success Rate: 0%
    PPO Success Rate: {success_rate:.1f}%
    Improvement: {success_rate:.1f} percentage points
    
    STATUS: {'✓ EXCELLENT' if success_rate > 80 else '✓ GOOD' if success_rate > 50 else '⚠ NEEDS MORE TRAINING'}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                     verticalalignment='center', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = "D:/iomp/dark_store_project/results/ppo_learning_curve_random.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curve saved to: {plot_path}")
    
    plt.close()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("PPO TRAINING COMPLETE! ✓")
print("=" * 60)

print(f"\nFiles created:")
print(f"  1. {MODEL_SAVE_PATH}.zip (trained model)")
print(f"  2. {LOG_PATH} (training log)")
print(f"  3. D:/iomp/dark_store_project/results/ppo_learning_curve_random.png")

print(f"\nFinal performance:")
print(f"  Success rate: {success_rate:.1f}%")
print(f"  Average reward: {avg_reward:.2f}")
print(f"  Average steps: {avg_length:.1f}")

if success_rate > 80:
    print(f"\n✓ EXCELLENT! PPO learned to pick orders efficiently!")
elif success_rate > 50:
    print(f"\n✓ GOOD! PPO shows significant improvement!")
    print(f"  (You can train longer for better results)")
else:
    print(f"\n⚠ Model needs more training")
    print(f"  Consider increasing TOTAL_TIMESTEPS to 100,000")

print(f"\n✓ Ready for baseline comparison (Day 16-17)")
print("=" * 60)
