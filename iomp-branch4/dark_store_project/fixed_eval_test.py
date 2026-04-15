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

if model_path.exists() and layout_path.exists():
    model = PPO.load(str(model_path), device='cpu')
    single_env = DarkStoreEnv(str(layout_path), max_steps=250, max_items_per_order=3)
    env = DummyVecEnv([lambda: single_env])
    
    successes = []
    for ep in range(10):
        obs = env.reset()
        done = False
        episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]
            episode_length += 1
            if episode_length >= 500:
                break
        success = info.get('success', False)
        successes.append(1 if success else 0)
    
    success_rate = np.mean(successes) * 100
    print(f'Fixed evaluation: {sum(successes)}/10 successes ({success_rate:.1f}%)')
else:
    print('Files not found')