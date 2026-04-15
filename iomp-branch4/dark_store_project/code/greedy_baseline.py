import sys
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv
import random

env = DarkStoreEnv("D:/iomp/dark_store_project/results/optimized_layout.json", max_steps=250, max_items_per_order=3)

success_count = 0
total_rewards = []
total_steps = []

print("Running 100 Greedy Baseline episodes...")
for i in range(100):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    
    while not done:
        curr_pos = env.current_position
        
        # Decide nearest target
        if len(env.items_remaining) > 0:
            targets = [env.product_positions[item_id] for item_id in env.items_remaining]
            # Manhattan distance
            distances = [abs(curr_pos[0] - t[0]) + abs(curr_pos[1] - t[1]) for t in targets]
            
            # Find nearest
            min_dist = min(distances)
            target = targets[distances.index(min_dist)]
            
        else:
            target = env.depot_position
            
        # Move one step towards target
        if curr_pos[0] < target[0]:
            action = 1 # Down
        elif curr_pos[0] > target[0]:
            action = 0 # Up
        elif curr_pos[1] < target[1]:
            action = 3 # Right
        elif curr_pos[1] > target[1]:
            action = 2 # Left
        else:
            action = 0 # Fallback 
            
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        done = terminated or truncated
        
    total_rewards.append(ep_reward)
    total_steps.append(steps)
    if info.get('success', False):
        success_count += 1

print(f"\n--- GREEDY BASELINE EVALUATION (100 episodes) ---")
print(f"Success Rate: {(success_count/100)*100:.1f}%")
print(f"Avg Reward: {sum(total_rewards)/100:.2f}")
print(f"Avg Steps: {sum(total_steps)/100:.2f}")
