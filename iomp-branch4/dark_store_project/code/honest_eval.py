"""
Honest evaluation of the trained ILP PPO model.
Uses STRICT success: all items picked AND returned to depot.
No tricks, no hardcoding.
"""
import sys, os
import numpy as np
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv
from stable_baselines3 import PPO

layout = "D:/iomp/dark_store_project/results/optimized_layout.json"
model_path = "D:/iomp/dark_store_project/results/ppo_model.zip"

env = DarkStoreEnv(layout, max_steps=350, max_items_per_order=3)
model = PPO.load(model_path, device='cpu')

N = 300
strict_successes = 0
lenient_successes = 0
steps_list = []

print(f"Running {N} episodes with STRICT success definition...")
print("(Strict = all items picked AND returned to depot)\n")

for i in range(N):
    obs, info = env.reset(seed=i)
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated

    # STRICT: must pick all items AND return to depot
    strict = info.get('success', False)
    # LENIENT: just all items picked (ignores depot return)
    lenient = info['items_picked'] == info['total_items']

    if strict:   strict_successes  += 1
    if lenient:  lenient_successes += 1
    steps_list.append(steps)

print(f"Results over {N} episodes:")
print(f"  LENIENT (items only)    : {lenient_successes/N*100:.1f}%  ← what 4_ppo_training.py showed")
print(f"  STRICT  (items + depot) : {strict_successes/N*100:.1f}%  ← real number")
print(f"  Avg steps               : {np.mean(steps_list):.1f}")
