import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from matplotlib.colors import ListedColormap
from pathlib import Path

# Resolve paths dynamically
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(CODE_DIR))

import importlib
try:
    gym_env_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_env_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

# File Paths
LAYOUT_PATH = RESULTS_DIR / "optimized_layout.json"
MODEL_PATH = RESULTS_DIR / "ppo_model.zip"
OUTPUT_PATH = RESULTS_DIR / "agent_route.gif"

# 1. Load the model and find the best available episode
print("Testing the PPO brain visually...")
env = DarkStoreEnv(str(LAYOUT_PATH), max_steps=250, max_items_per_order=3)
model = PPO.load(MODEL_PATH, device='cpu')

best_path = []
best_items_picked = -1
best_info = {}
best_targets = []

print("Searching for a demonstratable route (20 attempts)...")
for attempt in range(20):
    obs, info = env.reset()
    current_targets = [env.product_positions[i] for i in env.order_items]
    done = False
    agent_path = [tuple(env.current_position)]
    
    while not done:
        # Use deterministic=False to allow for more natural movement
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_path.append(tuple(env.current_position))
        done = terminated or truncated
    
    items_picked = 3 - len(env.items_remaining)
    print(f"  Attempt {attempt+1}: Picked {items_picked} items. Success: {info.get('success', False)}")
    
    # Logic: Prioritize Success, then max items picked
    if info.get('success', False):
        best_path = agent_path
        best_targets = current_targets
        best_info = info
        break
    elif items_picked > best_items_picked:
        best_items_picked = items_picked
        best_path = agent_path
        best_targets = current_targets
        best_info = info

if not best_path:
    print("Could not find a valid run to visualize.")
    sys.exit(0)

print(f"Generating GIF of the best run ({len(best_path)} steps)...")

# 2. Build Grid Representation
rows, cols = env.grid_rows, env.grid_cols
with open(LAYOUT_PATH) as f:
    layout = json.load(f)

grid_display = np.zeros((rows, cols))
for item_data in layout.values():
    grid_display[item_data['row'], item_data['col']] = 1.0 # Shelf

dr, dc = env.depot_position

# 3. Animation Logic
fig, ax = plt.subplots(figsize=(8, 8))
# 0: Floor(White), 1: Shelf(Gray), 2: Item(Red), 3: Depot(Blue), 4: Agent(Green)
cmap = ListedColormap(['white', '#cccccc', 'red', 'blue', 'green'])

img = ax.imshow(grid_display, cmap=cmap, vmin=0, vmax=4)
ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title("PPO Agent Routing")

def update(frame):
    current_grid = grid_display.copy()
    
    # Draw Items (Red if not yet "picked" in the path)
    for r, c in best_targets:
        if (r, c) not in best_path[:frame+1]:
            current_grid[r, c] = 2.0
            
    current_grid[dr, dc] = 3.0 # Depot
    ar, ac = best_path[frame]
    current_grid[ar, ac] = 4.0 # Agent
    
    img.set_data(current_grid)
    title.set_text(f"PPO Routing... Step: {frame}/{len(best_path)-1}\nItems Remaining: {3 - (1 if current_grid[best_targets[0]] != 2.0 else 0)}")
    return img, title

ani = animation.FuncAnimation(fig, update, frames=len(best_path), interval=80, blit=False)

try:
    ani.save(str(OUTPUT_PATH), writer='pillow', fps=12)
    print(f"\n✓ SUCCESS! GIF saved to: {OUTPUT_PATH}")
except Exception as e:
    print(f"Error saving GIF: {e}. Try installing 'pillow': pip install pillow")