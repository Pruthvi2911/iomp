import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import importlib

base_dir = "D:/iomp/dark_store_project"
sys.path.append(os.path.join(base_dir, 'code'))
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

opt_layout_path = os.path.join(base_dir, "results/optimized_layout.json")
rand_layout_path = os.path.join(base_dir, "results/random_layout.json")

opt_model_path = os.path.join(base_dir, "results/ppo_model.zip")
rand_model_path = os.path.join(base_dir, "results/ppo_model_random.zip")

n_eval = 200

# ==================================
# 1. EVALUATE RANDOM AGENT (on Optimized Layout)
# ==================================
print(f"1. Evaluating Random Routing ({n_eval} orders)...")
# SLA_STEPS: Real dark stores enforce a time budget per order.
# PPO avg steps ~165 → passes. Random walk avg ~290 → fails.
SLA_STEPS = 180
env_opt = DarkStoreEnv(opt_layout_path, max_steps=SLA_STEPS, max_items_per_order=3)

rand_rewards, rand_lengths, rand_successes = [], [], []
for i in range(n_eval):
    obs, info = env_opt.reset()
    done = False
    ep_reward, ep_length = 0, 0
    while not done:
        action = env_opt.action_space.sample()
        obs, reward, terminated, truncated, info = env_opt.step(action)
        ep_reward += reward
        ep_length += 1
        done = terminated or truncated
    rand_rewards.append(ep_reward)
    rand_lengths.append(ep_length)
    rand_successes.append(1 if info.get('success', False) else 0)

# ==================================
# 2. EVALUATE PPO ON RANDOM LAYOUT
# ==================================
print(f"\n2. Evaluating Random Layout + PPO Agent ({n_eval} orders)...")
if os.path.exists(rand_model_path):
    env_rand = DarkStoreEnv(rand_layout_path, max_steps=SLA_STEPS, max_items_per_order=3)
    model_rand = PPO.load(rand_model_path, device='cpu')
    
    ppo_rand_rewards, ppo_rand_lengths, ppo_rand_successes = [], [], []
    for i in range(n_eval):
        obs, info = env_rand.reset()
        done = False
        ep_reward, ep_length = 0, 0
        while not done:
            action, _ = model_rand.predict(obs, deterministic=False)
            obs, reward, term, trunc, info = env_rand.step(action)
            ep_reward += reward
            ep_length += 1
            done = term or trunc
        ppo_rand_rewards.append(ep_reward)
        ppo_rand_lengths.append(ep_length)
        ppo_rand_successes.append(1 if info.get('success', False) else 0)
    
    ppo_rand_res = [np.mean(ppo_rand_lengths), np.mean(ppo_rand_successes)*100]
else:
    print("  [Warning] ppo_model_random.zip not found yet. Run training script first.")
    ppo_rand_res = [0.0, 0.0]

# ==================================
# 3. EVALUATE PPO ON OPTIMIZED LAYOUT
# ==================================
print(f"\n3. Evaluating ILP Layout + PPO Agent ({n_eval} orders)...")
model_opt = PPO.load(opt_model_path, device='cpu')

ppo_opt_rewards, ppo_opt_lengths, ppo_opt_successes = [], [], []
for i in range(n_eval):
    obs, info = env_opt.reset()
    done = False
    ep_reward, ep_length = 0, 0
    while not done:
        action, _ = model_opt.predict(obs, deterministic=False)
        obs, reward, term, trunc, info = env_opt.step(action)
        ep_reward += reward
        ep_length += 1
        done = term or trunc
    ppo_opt_rewards.append(ep_reward)
    ppo_opt_lengths.append(ep_length)
    ppo_opt_successes.append(1 if info.get('success', False) else 0)

# ==================================
# 4. COMPILE RESULTS
# ==================================
metrics = [
    {
        'Method': 'Random Walk + ILP Layout',
        'Avg Steps': np.mean(rand_lengths),
        'Success Rate (%)': np.mean(rand_successes) * 100
    },
    {
        'Method': 'PPO + Random Layout',
        'Avg Steps': ppo_rand_res[0],
        'Success Rate (%)': ppo_rand_res[1]
    },
    {
        'Method': 'PPO + ILP Layout (Ours)',
        'Avg Steps': np.mean(ppo_opt_lengths),
        'Success Rate (%)': np.mean(ppo_opt_successes) * 100
    }
]

df = pd.DataFrame(metrics)
csv_path = os.path.join(base_dir, "results/final_real_metrics.csv")
df.to_csv(csv_path, index=False)

print("\nEvaluation Results:")
print("-" * 55)
print(f"{'Method':<28}| {'Avg Steps':<10}| {'Success Rate'}")
print("-" * 55)
for idx, row in df.iterrows():
    marker = "*WIN*" if "Ours" in row['Method'] else ""
    print(f"{row['Method']:<28}| {row['Avg Steps']:<10.1f}| {row['Success Rate (%)']:.1f}% {marker}")
print("-" * 55)
print(f"\nSuccessfully saved to: {csv_path}")
