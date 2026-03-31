import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from matplotlib.colors import ListedColormap

# Setup paths
base_dir = "D:/iomp/dark_store_project"
sys.path.append(os.path.join(base_dir, 'code'))

import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

layout_path = os.path.join(base_dir, "results/optimized_layout.json")
model_path = os.path.join(base_dir, "results/ppo_model.zip")
output_path = os.path.join(base_dir, "results/agent_route.gif")

# 1. Load the model and finding a successful episode
print("Testing the PPO brain visually...")
env = DarkStoreEnv(layout_path, max_steps=250, max_items_per_order=3)
model = PPO.load(model_path, device='cpu')

success = False
for attempt in range(20):
    obs, info = env.reset()
    done = False
    
    # Store exact path coordinate history
    agent_path = [tuple(env.current_position)]
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_path.append(tuple(env.current_position))
        done = terminated or truncated
        
    print(f"Attempt {attempt+1}: {len(agent_path)} steps. Success: {info.get('success', False)}. Items remaining: {len(env.items_remaining)}")
    
    if info.get('success', False):
        print(f"✓ Found a perfect 3-item pickup route! (Finished in {len(agent_path)} steps)")
        success = True
        break

if not success:
    print("Could not find a perfect run in 20 random samples. Retraining recommended.")
    sys.exit(0)

print(f"Generating visual GIF of the agent's path to: {output_path}")

# 2. Build mathematical representation of grid layout
rows, cols = env.grid_rows, env.grid_cols
with open(layout_path) as f:
    layout = json.load(f)

# Find all items to map shelves
visited_shelves = set()
for item_data in layout.values():
    r, c = item_data['row'], item_data['col']
    visited_shelves.add((r, c))

item_positions = [env.product_positions[i] for i in env.order_items]
dr, dc = env.depot_position

# Base map initialization
grid_display = np.zeros((rows, cols))
for r, c in visited_shelves: grid_display[r, c] = 1.0 # Gray shelf

# 3. Create dynamic drawing logic
fig, ax = plt.subplots(figsize=(8, 8))
cmap = ListedColormap(['white', '#cccccc', 'red', 'blue', 'green'])

# 0=floor, 1=shelf, 2=item, 3=depot, 4=agent
img = ax.imshow(grid_display, cmap=cmap, vmin=0, vmax=4)
ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
ax.tick_params(which='minor', size=0)
ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title(f"PPO Agent Route - Finding 3 Items")

# Step-by-step frame updater
def update(frame):
    current_grid = grid_display.copy()
    
    # Check if agent picked item this frame
    for r, c in item_positions:
        touched = False
        for path_r, path_c in agent_path[:frame+1]:
            if path_r == r and path_c == c:
                touched = True
        
        # Keep item red until touched
        if not touched:
            current_grid[r, c] = 2.0
            
    # Always draw depot blue
    current_grid[dr, dc] = 3.0
    
    # Draw agent as green
    agent_r, agent_c = agent_path[frame]
    current_grid[agent_r, agent_c] = 4.0
    
    img.set_data(current_grid)
    title.set_text(f"PPO Routing... Step: {frame}/{len(agent_path)-1}")
    return img, title

# Render and compile to GIF
ani = animation.FuncAnimation(fig, update, frames=len(agent_path), interval=60, blit=True)

try:
    ani.save(output_path, writer='pillow', fps=15)
    print("\n==================================")
    print(f"SUCCESS! A visual animated GIF was saved to:")
    print(f"-> {output_path}")
    print("==================================")
except Exception as e:
    print(f"Error rendering gif: {e}")
