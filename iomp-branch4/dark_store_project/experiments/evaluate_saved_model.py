import argparse
import sys
from pathlib import Path
import numpy as np
import importlib
from stable_baselines3 import PPO

BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
sys.path.append(str(CODE_DIR))

def get_default_layout():
    return str(BASE_DIR / "results" / "optimized_layout.json")

parser = argparse.ArgumentParser(description='Evaluate a saved PPO model in branch4 dark store env.')
parser.add_argument('--model', required=True, help='Path to the saved PPO model (.zip)')
parser.add_argument('--layout', default=get_default_layout(), help='Path to the optimized layout JSON')
parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
args = parser.parse_args()

gym_env_module = importlib.import_module('3_gym_environment')
DarkStoreEnv = gym_env_module.DarkStoreEnv

model = PPO.load(args.model, device='cpu')
env = DarkStoreEnv(args.layout, max_steps=250, max_items_per_order=3)

rewards = []
lengths = []
successes = []
for ep in range(args.episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    ep_length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        ep_length += 1
        done = terminated or truncated
    rewards.append(ep_reward)
    lengths.append(ep_length)
    successes.append(1 if info.get('success', False) else 0)

print(f'Avg Reward: {np.mean(rewards):.2f}')
print(f'Avg Length: {np.mean(lengths):.1f}')
print(f'Success Rate: {np.mean(successes) * 100:.1f}%')
