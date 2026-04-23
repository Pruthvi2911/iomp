import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import sys
from pathlib import Path
from stable_baselines3 import PPO
import importlib

# Ensure paths work
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"
sys.path.append(str(CODE_DIR))

# Dynamic import of your environment
try:
    gym_env_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_env_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

LAYOUT_PATH = RESULTS_DIR / "optimized_layout.json"
MODEL_PATH = RESULTS_DIR / "ppo_model.zip"

st.set_page_config(page_title="Dark Store Agent Simulator", layout="centered")

st.title("🤖 Dark Store Autonomous Picker")
st.markdown("Click the button to spawn a random order and watch the agent navigate live!")

# Cache the environment and model so they load instantly on button clicks
@st.cache_resource
def load_system():
    env = DarkStoreEnv(str(LAYOUT_PATH), max_steps=250, max_items_per_order=3)
    model = PPO.load(MODEL_PATH, device='cpu')
    return env, model

env, model = load_system()

# Add a button
if st.button("Generate Random Order & Simulate 🚀", type="primary"):
    
    # 1. Generate new order & run simulation behind the scenes
    obs, info = env.reset()
    current_targets = [env.product_positions[i] for i in env.order_items]
    done = False
    
    agent_path = [tuple(env.current_position)]
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, term, trunc, info = env.step(action)
        agent_path.append(tuple(env.current_position))
        done = term or trunc
        
    items_picked = 3 - info.get('items_remaining', 3)
    success = info.get('success', False)
    
    # 2. Setup visualization components
    with open(LAYOUT_PATH) as f:
        layout = json.load(f)

    rows, cols = env.grid_rows, env.grid_cols
    grid_display = np.zeros((rows, cols))
    
    for item_data in layout.values():
        grid_display[item_data['row'], item_data['col']] = 1.0 # Shelf
        
    dr, dc = env.depot_position

    cmap = ListedColormap(['white', '#cccccc', 'red', 'blue', 'green'])
    
    # 3. Live Animation Loop in Streamlit
    st.markdown(f"**Target Items:** 3 | **Result:** Picked {items_picked}/3")
    
    # Create an empty placeholder container
    plot_placeholder = st.empty()
    
    # Animate frame by frame directly to Chrome
    for frame in range(len(agent_path)):
        fig, ax = plt.subplots(figsize=(6, 6))
        current_grid = grid_display.copy()
        
        # Mark remaining targets as red
        for r, c in current_targets:
            if (r, c) not in agent_path[:frame+1]:
                current_grid[r, c] = 2.0
                
        # Mark Depot and Agent
        current_grid[dr, dc] = 3.0
        ar, ac = agent_path[frame]
        current_grid[ar, ac] = 4.0
        
        ax.imshow(current_grid, cmap=cmap, vmin=0, vmax=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Live Simulation - Step: {frame}/{len(agent_path)-1}")
        
        # Overwrite the placeholder with the new frame
        plot_placeholder.pyplot(fig)
        plt.close(fig) # Prevent memory leaks
        
        time.sleep(0.05) # Delay for animation speed

    if success:
        st.success(f"✅ Route successfully completed in {len(agent_path)} steps!")
    else:
        st.warning(f"⚠️ Route ended. Items picked: {items_picked}/3.")
        
    st.markdown("""
    **Legend:**
    🟩 Agent | 🟦 Depot | 🟥 Target Items | ⬜ Walkway | ⬛ Shelves
    """)
