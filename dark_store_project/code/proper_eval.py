"""
Proper evaluation of trained PPO model
"""
import sys
import importlib
sys.path.append('D:/iomp/dark_store_project/code')
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv
from stable_baselines3 import PPO

# Load environment and model
layout_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
env = DarkStoreEnv(layout_file, max_steps=350)
model = PPO.load("D:/iomp/dark_store_project/results/ppo_model")

# Evaluate on 100 episodes
successes = 0
total_rewards = []
total_steps = []

print("Evaluating trained model on 100 episodes...\n")

for episode in range(100):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    while not done:
        # Use trained model to predict action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        done = terminated or truncated
    
    # Check if successful
    if info['items_picked'] == info['total_items']:
        successes += 1
    
    total_rewards.append(episode_reward)
    total_steps.append(episode_steps)
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/100: Success rate so far: {successes/(episode+1)*100:.1f}%")

# Final results
import numpy as np
print(f"\n{'='*60}")
print("FINAL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Success Rate: {successes}% ({successes}/100 episodes)")
print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
print(f"Average Steps: {np.mean(total_steps):.1f}")
print(f"Min Steps: {np.min(total_steps)}")
print(f"Max Steps: {np.max(total_steps)}")
print(f"{'='*60}")
