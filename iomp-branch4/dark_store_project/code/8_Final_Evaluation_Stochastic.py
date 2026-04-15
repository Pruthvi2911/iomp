import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import importlib
from pathlib import Path

# Fix terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(CODE_DIR))

# Import Environment
import importlib
try:
    gym_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_module.DarkStoreEnv

# File Paths
OPT_LAYOUT_PATH = str(RESULTS_DIR / "optimized_layout.json")
OPT_MODEL_PATH = str(RESULTS_DIR / "ppo_model.zip")

n_eval = 200 # 200 is plenty for a stable average
SLA_STEPS = 250
EVAL_SEED = 42

print("=" * 60)
print("DARK STORE - FINAL TRUE PERFORMANCE EVALUATION")
print("=" * 60)
print(f"Loading model: {OPT_MODEL_PATH}")

env = DarkStoreEnv(OPT_LAYOUT_PATH, max_steps=SLA_STEPS, max_items_per_order=3)
model = PPO.load(OPT_MODEL_PATH, device='cpu')

eval_lengths, eval_successes, items_picked_list = [], [], []

print(f"\nRunning {n_eval} test orders with Stochastic Routing...")

for i in range(n_eval):
    obs, info = env.reset(seed=EVAL_SEED + i)
    done = False
    ep_length = 0
    
    while not done:
        # CRITICAL FIX: deterministic=False to match training performance
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_length += 1
        done = terminated or truncated
        
    eval_lengths.append(ep_length)
    success = info.get('success', False)
    eval_successes.append(1 if success else 0)
    
    # Calculate how many items it actually managed to grab
    items_picked = 3 - info.get('items_remaining', 3)
    items_picked_list.append(items_picked)

# Calculate Stats
final_success_rate = np.mean(eval_successes) * 100
avg_items = np.mean(items_picked_list)
avg_steps = np.mean(eval_lengths)

print("\n" + "-" * 40)
print(f"FINAL METRICS (n={n_eval})")
print("-" * 40)
print(f"Success Rate (Full Order + Depot): {final_success_rate:.1f}%")
print(f"Avg Items Picked per Order:      {avg_items:.2f} / 3.0")
print(f"Avg Steps Taken:                 {avg_steps:.1f}")
print("-" * 40)

# Save this for your report!
metrics_df = pd.DataFrame([{
    'Method': 'ILP + PPO (Stochastic)',
    'Success Rate': f"{final_success_rate:.1f}%",
    'Avg Items': f"{avg_items:.2f}",
    'Avg Steps': f"{avg_steps:.1f}"
}])
metrics_df.to_csv(RESULTS_DIR / "true_ppo_performance.csv", index=False)
print(f"✓ Metrics saved to: {RESULTS_DIR}/true_ppo_performance.csv")