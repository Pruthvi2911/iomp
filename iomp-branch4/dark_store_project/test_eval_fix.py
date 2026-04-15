import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[0]
CODE_DIR = BASE_DIR / 'code'
sys.path.append(str(CODE_DIR))
import importlib
gym_env_module = importlib.import_module('3_gym_environment')
DarkStoreEnv = gym_env_module.DarkStoreEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

model_path = BASE_DIR / 'experiments' / 'exp_lr_0.001' / 'ppo_model.zip'
layout_path = BASE_DIR / 'results' / 'optimized_layout.json'

print(f"Model path: {model_path}")
print(f"Layout path: {layout_path}")

if model_path.exists():
    # Load model
    model = PPO.load(str(model_path), device='cpu')
    
    # Create VecEnv like in training
    single_env = DarkStoreEnv(str(layout_path), max_steps=250, max_items_per_order=3)
    env = DummyVecEnv([lambda: single_env])
    
    successes = []
    for ep in range(10):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated[0] or truncated[0]
        successes.append(1 if info[0].get('success', False) else 0)
    
    print(f'Quick test: {sum(successes)}/10 successes')
else:
    print('Model not found')