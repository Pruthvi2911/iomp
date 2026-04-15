import sys
import os
import json
import numpy as np
from stable_baselines3 import PPO

base_dir = "D:/iomp/dark_store_project"
sys.path.append(os.path.join(base_dir, 'code'))
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

layout_path = os.path.join(base_dir, "results/optimized_layout.json")
model_path = os.path.join(base_dir, "results/ppo_model.zip")

env = DarkStoreEnv(layout_path, max_steps=250, max_items_per_order=3)
model = PPO.load(model_path, device='cpu')

n_eval_episodes = 500
episode_rewards = []
episode_lengths = []
episode_successes = []

print(f"Running {n_eval_episodes} evaluation episodes with deterministic=True...")
for i in range(n_eval_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        done = terminated or truncated
        
    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_length)
    episode_successes.append(1 if info.get('success', False) else 0)

success_rate = np.mean(episode_successes) * 100
avg_reward = np.mean(episode_rewards)
avg_length = np.mean(episode_lengths)

print(f"\n--- PROPER EVAL (500 episodes) ---")
print(f"Success Rate: {success_rate:.1f}%")
print(f"Avg Reward: {avg_reward:.2f}")
print(f"Avg Steps: {avg_length:.1f}")
