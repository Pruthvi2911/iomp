import sys
sys.stdout = open('final_clean.txt', 'w', encoding='utf-8')
import os
from stable_baselines3 import PPO

base_dir = "D:/iomp/dark_store_project"
sys.path.append(os.path.join(base_dir, 'code'))
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

layout_path = os.path.join(base_dir, "results/optimized_layout.json")
model_path = os.path.join(base_dir, "results/ppo_model.zip")

env = DarkStoreEnv(layout_path, max_steps=250, max_items_per_order=3)
model = PPO.load(model_path, device='cpu')

print("Running 5 Final Diagnostic Episodes (deterministic=False)\n")

for i in range(5):
    obs, info = env.reset()
    done = False
    
    path_log = []
    previous_items = list(env.items_remaining)
    
    step_count = 0
    last_event_step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        current_items = list(env.items_remaining)
        if len(current_items) < len(previous_items):
            picked_items = [item for item in previous_items if item not in current_items]
            for p_item in picked_items:
                loc = env.product_positions[p_item]
                steps_taken = step_count - last_event_step
                path_log.append(f"item_at_{loc}({steps_taken} steps)")
                last_event_step = step_count
        previous_items = current_items
        
        done = terminated or truncated
    
    success = info.get('success', False)
    if success:
        steps_to_depot = step_count - last_event_step
        path_log.append(f"depot({steps_to_depot} steps)")
        status = "✓"
    else:
        path_log.append(f"TIMED_OUT({step_count - last_event_step} wandering steps)")
        status = "✗"
        
    path_str = " → ".join(path_log)
    print(f"Episode {i+1}:\n  depot → {path_str} {status}")
    print(f"  Total Steps taken: {step_count}\n")
