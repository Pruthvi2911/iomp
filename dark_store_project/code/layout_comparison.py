"""
Layout Comparison: Same PPO agent tested on ILP vs Random layout
This isolates the layout effect — no training variance.
"""
import sys, os
import numpy as np
from stable_baselines3 import PPO
import importlib

base_dir = "D:/iomp/dark_store_project"
sys.path.append(os.path.join(base_dir, 'code'))
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

# Paths
ilp_layout = os.path.join(base_dir, "results/optimized_layout.json")
rand_layout = os.path.join(base_dir, "results/random_layout.json")
ilp_model_path = os.path.join(base_dir, "results/ppo_model.zip")

# Load the ILP-trained model
model = PPO.load(ilp_model_path, device='cpu')

n_eval = 300
EVAL_SEED = 42

def evaluate(env, model, n_eval, use_model=True):
    rewards, lengths, successes = [], [], []
    for i in range(n_eval):
        obs, info = env.reset(seed=EVAL_SEED + i)
        done = False
        ep_reward, ep_length = 0, 0
        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = term or trunc
        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(1 if info.get('success', False) else 0)
    return np.mean(lengths), np.mean(successes) * 100

print("=" * 60)
print("LAYOUT COMPARISON — Same PPO Agent, Different Layouts")
print("=" * 60)

# Test 1: ILP model on ILP layout
print(f"\n1. ILP-trained PPO on ILP Layout ({n_eval} orders)...")
env_ilp = DarkStoreEnv(ilp_layout, max_steps=250, max_items_per_order=3)
ilp_steps, ilp_success = evaluate(env_ilp, model, n_eval)

# Test 2: ILP model on Random layout
print(f"2. ILP-trained PPO on Random Layout ({n_eval} orders)...")
env_rand = DarkStoreEnv(rand_layout, max_steps=250, max_items_per_order=3)
rand_steps, rand_success = evaluate(env_rand, model, n_eval)

# Test 3: Random walk on ILP layout (baseline)
print(f"3. Random Walk on ILP Layout ({n_eval} orders)...")
rw_steps, rw_success = evaluate(env_ilp, model, n_eval, use_model=False)

print(f"\n{'='*60}")
print(f"RESULTS — Same model, layout is the ONLY variable")
print(f"{'='*60}")
print(f"{'Method':<35} | {'Avg Steps':<10} | {'Success Rate'}")
print(f"{'-'*60}")
print(f"{'Random Walk (baseline)':<35} | {rw_steps:<10.1f} | {rw_success:.1f}%")
print(f"{'PPO Agent → Random Layout':<35} | {rand_steps:<10.1f} | {rand_success:.1f}%")
print(f"{'PPO Agent → ILP Layout':<35} | {ilp_steps:<10.1f} | {ilp_success:.1f}%")
print(f"{'-'*60}")
print(f"\nLayout Effect:")
print(f"  Step reduction:    {rand_steps - ilp_steps:.1f} fewer steps with ILP")
print(f"  Success rate gain: {ilp_success - rand_success:.1f}% higher with ILP")
print(f"{'='*60}")
