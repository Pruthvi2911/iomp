"""
Evaluate saved PPO model for LR=0.0005 experiment
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.stdout.reconfigure(encoding='utf-8')

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = BASE_DIR / "code"
sys.path.append(str(CODE_DIR))
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

EXPERIMENT_DIR = Path(__file__).resolve().parents[0]
LAYOUT_FILE = str(BASE_DIR / "results" / "optimized_layout.json")
MODEL_PATH = str(EXPERIMENT_DIR / "ppo_model")

print("=" * 60)
print("EVALUATING LR=0.0005 PPO MODEL")
print("=" * 60)

# Load environment
print("\nLoading environment...")
env = DarkStoreEnv(LAYOUT_FILE, max_steps=250, max_items_per_order=3)
env = DummyVecEnv([lambda: env])

# Load model
print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# Evaluate
print("\nRunning evaluation (100 episodes)...")
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
        obs, reward, done, info = env.step(action)
        done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
        info = info[0] if isinstance(info, (list, tuple)) else info
        episode_reward += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
        episode_length += 1
        
        if episode_length >= 500:  # Safety limit
            break
    
    eval_rewards.append(episode_reward)
    eval_lengths.append(episode_length)
    success = info.get('success', False)
    eval_successes.append(1 if success else 0)
    
    if (ep + 1) % 20 == 0:
        print(f"  Progress: {ep + 1}/100 episodes")

avg_eval_reward = np.mean(eval_rewards)
avg_eval_length = np.mean(eval_lengths)
eval_success_rate = np.mean(eval_successes) * 100

print("\n✓ Evaluation complete!")
print(f"  Avg Reward: {avg_eval_reward:.2f}")
print(f"  Avg Length: {avg_eval_length:.1f}")
print(f"  Success Rate: {eval_success_rate:.1f}%")

# Save evaluation results
eval_results = {
    'experiment': 'lr_0.0005',
    'learning_rate': 0.0005,
    'avg_reward': avg_eval_reward,
    'avg_length': avg_eval_length,
    'success_rate': eval_success_rate,
}

eval_df = pd.DataFrame([eval_results])
eval_csv = str(EXPERIMENT_DIR / "eval_results.csv")
eval_df.to_csv(eval_csv, index=False)

print(f"✓ Results saved to: {eval_csv}")
print("\n" + "=" * 60)
