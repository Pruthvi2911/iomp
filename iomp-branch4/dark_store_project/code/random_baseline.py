import sys
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

env = DarkStoreEnv("D:/iomp/dark_store_project/results/optimized_layout.json", max_steps=250, max_items_per_order=3)

success_count = 0
total_rewards = []
total_steps = []

print("Running 100 Random Agent baseline episodes...")
for i in range(100):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        done = terminated or truncated
        
    total_rewards.append(ep_reward)
    total_steps.append(steps)
    if info.get('success', False):
        success_count += 1

print(f"\n--- RANDOM BASELINE EVALUATION (100 episodes) ---")
print(f"Success Rate: {(success_count/100)*100:.1f}%")
print(f"Avg Reward: {sum(total_rewards)/100:.2f}")
print(f"Avg Steps: {sum(total_steps)/100:.2f}")
