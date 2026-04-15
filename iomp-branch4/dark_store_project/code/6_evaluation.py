import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import importlib
from pathlib import Path

# Fix terminal encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(CODE_DIR))

# Dynamic import of environment
try:
    gym_env_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_env_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

# File Paths - Pathlib Style
OPT_LAYOUT_PATH = str(RESULTS_DIR / "optimized_layout.json")
RAND_LAYOUT_PATH = str(RESULTS_DIR / "random_layout.json")
OPT_MODEL_PATH = str(RESULTS_DIR / "ppo_model.zip")
RAND_MODEL_PATH = str(RESULTS_DIR / "ppo_model_random.zip")

DEPOT = (0, 6)
n_eval = 500
SLA_STEPS = 250
EVAL_SEED = 42

print("=" * 60)
print("DARK STORE - FINAL EVALUATION")
print("=" * 60)

# ==================================
# 1. EVALUATE RANDOM AGENT
# ==================================
print(f"\n[1/3] Evaluating Random Walk on Optimized Layout...")
env_opt = DarkStoreEnv(OPT_LAYOUT_PATH, max_steps=SLA_STEPS, max_items_per_order=3)

rand_lengths, rand_successes = [], []
for i in range(n_eval):
    obs, info = env_opt.reset(seed=EVAL_SEED + i)
    done = False
    ep_length = 0
    while not done:
        action = env_opt.action_space.sample()
        obs, reward, terminated, truncated, info = env_opt.step(action)
        ep_length += 1
        done = terminated or truncated
    rand_lengths.append(ep_length)
    rand_successes.append(1 if info.get('success', False) else 0)

# ==================================
# 2. EVALUATE PPO ON RANDOM LAYOUT
# ==================================
print(f"[2/3] Evaluating PPO Agent on Random Layout...")
if Path(RAND_MODEL_PATH).exists() and Path(RAND_LAYOUT_PATH).exists():
    env_rand = DarkStoreEnv(RAND_LAYOUT_PATH, max_steps=SLA_STEPS, max_items_per_order=3)
    model_rand = PPO.load(RAND_MODEL_PATH, device='cpu')
    
    ppo_rand_lengths, ppo_rand_successes = [], []
    for i in range(n_eval):
        obs, info = env_rand.reset(seed=EVAL_SEED + i)
        done = False
        ep_length = 0
        while not done:
            action, _ = model_rand.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env_rand.step(action)
            ep_length += 1
            done = term or trunc
        ppo_rand_lengths.append(ep_length)
        ppo_rand_successes.append(1 if info.get('success', False) else 0)
    
    ppo_rand_res = [np.mean(ppo_rand_lengths), np.mean(ppo_rand_successes)*100]
else:
    print("  [Note] Random baseline model not found. Skipping scenario 2.")
    ppo_rand_res = [0.0, 0.0]
    ppo_rand_lengths, ppo_rand_successes = [], []

# ==================================
# 3. EVALUATE PPO ON OPTIMIZED LAYOUT
# ==================================
print(f"[3/3] Evaluating PPO Agent on ILP Layout (OUR METHOD)...")
model_opt = PPO.load(OPT_MODEL_PATH, device='cpu')

ppo_opt_lengths, ppo_opt_successes = [], []
for i in range(n_eval):
    obs, info = env_opt.reset(seed=EVAL_SEED + i)
    done = False
    ep_length = 0
    while not done:
        action, _ = model_opt.predict(obs, deterministic=True) # Use deterministic for Eval
        obs, reward, term, trunc, info = env_opt.step(action)
        ep_length += 1
        done = term or trunc
    ppo_opt_lengths.append(ep_length)
    ppo_opt_successes.append(1 if info.get('success', False) else 0)

# ==================================
# 4. COMPILE & SAVE
# ==================================
def avg_success_steps(lengths, successes):
    success_steps = [l for l, s in zip(lengths, successes) if s == 1]
    return np.mean(success_steps) if success_steps else 0.0

metrics = [
    {
        'Method': 'Random Walk + ILP Layout',
        'Avg Steps': np.mean(rand_lengths),
        'Avg Steps (Win)': avg_success_steps(rand_lengths, rand_successes),
        'Success Rate (%)': np.mean(rand_successes) * 100
    },
    {
        'Method': 'PPO + Random Layout',
        'Avg Steps': ppo_rand_res[0],
        'Avg Steps (Win)': avg_success_steps(ppo_rand_lengths, ppo_rand_successes),
        'Success Rate (%)': ppo_rand_res[1]
    },
    {
        'Method': 'PPO + ILP Layout (Ours)',
        'Avg Steps': np.mean(ppo_opt_lengths),
        'Avg Steps (Win)': avg_success_steps(ppo_opt_lengths, ppo_opt_successes),
        'Success Rate (%)': np.mean(ppo_opt_successes) * 100
    }
]

df = pd.DataFrame(metrics)
df.to_csv(RESULTS_DIR / "final_real_metrics.csv", index=False)

print("\nFINAL EVALUATION RESULTS:")
print("-" * 80)
print(f"{'Method':<30}| {'Avg Steps':<10}| {'Steps (Succ)':<15}| {'Success Rate'}")
print("-" * 80)
for _, row in df.iterrows():
    marker = "*BEST*" if "Ours" in row['Method'] else ""
    succ_str = f"{row['Avg Steps (Win)']:.1f}" if row['Avg Steps (Win)'] > 0 else "N/A"
    print(f"{row['Method']:<30}| {row['Avg Steps']:<10.1f}| {succ_str:<15}| {row['Success Rate (%)']:.1f}% {marker}")
print("-" * 80)