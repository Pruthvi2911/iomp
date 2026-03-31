import sys
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv
from stable_baselines3 import PPO

env = DarkStoreEnv("D:/iomp/dark_store_project/results/optimized_layout.json", max_steps=250, max_items_per_order=3)
model = PPO.load("D:/iomp/dark_store_project/results/ppo_model", env=env, device='cpu')

success_count = 0
total_rewards = []
total_steps = []

print("Running 500 PPO Eval episodes...")
for i in range(500):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        done = terminated or truncated
    
    total_rewards.append(ep_reward)
    total_steps.append(steps)
    if info.get('success', False):
        success_count += 1

print(f"\n--- PPO EVALUATION (500 episodes) ---")
print(f"Success Rate: {(success_count/500)*100:.1f}%")
print(f"Avg Reward: {sum(total_rewards)/500:.2f}")
print(f"Avg Steps: {sum(total_steps)/500:.2f}")
