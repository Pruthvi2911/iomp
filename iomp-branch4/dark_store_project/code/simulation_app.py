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
    # We run multiple stochastic attempts to find the *best path* for the worker
    import random
    order_seed = random.randint(0, 1000000)
    
    best_path = []
    best_items_picked = -1
    best_info = {}
    best_targets = []
    best_success = False
    
    with st.spinner("Agent is exploring to find the optimal path..."):
        for attempt in range(15):
            obs, info = env.reset(seed=order_seed)
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
            
            # Logic to keep the best run
            if success and not best_success:
                best_success = True
                best_path = agent_path
                best_items_picked = items_picked
                best_info = info
                best_targets = current_targets
            elif success and best_success:
                # Both succeeded, keep the shorter path
                if len(agent_path) < len(best_path):
                    best_path = agent_path
                    best_items_picked = items_picked
                    best_info = info
                    best_targets = current_targets
            elif not best_success:
                # Neither succeeded yet, keep the one with most items picked
                if items_picked > best_items_picked:
                    best_items_picked = items_picked
                    best_path = agent_path
                    best_info = info
                    best_targets = current_targets
                elif items_picked == best_items_picked:
                    if not best_path or len(agent_path) < len(best_path):
                        best_path = agent_path
                        best_info = info
                        best_targets = current_targets

    # Use the best results for visualization
    agent_path = best_path
    items_picked = best_items_picked
    success = best_success
    current_targets = best_targets
    
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

    # --- Generate Worker Directions ---
    def get_turn(current_facing, new_facing):
        if current_facing == new_facing: return "Go straight"
        elif current_facing == (1, 0): # South
            return "Turn left" if new_facing == (0, 1) else "Turn right" if new_facing == (0, -1) else "Turn around"
        elif current_facing == (-1, 0): # North
            return "Turn right" if new_facing == (0, 1) else "Turn left" if new_facing == (0, -1) else "Turn around"
        elif current_facing == (0, 1): # East
            return "Turn right" if new_facing == (1, 0) else "Turn left" if new_facing == (-1, 0) else "Turn around"
        elif current_facing == (0, -1): # West
            return "Turn left" if new_facing == (1, 0) else "Turn right" if new_facing == (-1, 0) else "Turn around"
        return "Move"

    if len(agent_path) > 1:
        instructions = []
        facing = (1, 0) # Start facing South (from depot)
        current_dir = None
        step_count = 0
        
        for i in range(1, len(agent_path)):
            dr = agent_path[i][0] - agent_path[i-1][0]
            dc = agent_path[i][1] - agent_path[i-1][1]
            
            if dr == 0 and dc == 0: continue
            
            step_dir = (dr, dc)
            
            if current_dir is None:
                current_dir = step_dir
                step_count = 1
            elif current_dir == step_dir:
                step_count += 1
            else:
                turn_text = get_turn(facing, current_dir)
                if turn_text == "Go straight": instructions.append(f"↑ **Go straight** for {step_count} step(s)")
                elif turn_text == "Turn left": instructions.append(f"↰ **Turn left** and go forward {step_count} step(s)")
                elif turn_text == "Turn right": instructions.append(f"↱ **Turn right** and go forward {step_count} step(s)")
                else: instructions.append(f"↻ **Turn around** and go forward {step_count} step(s)")
                
                facing = current_dir
                current_dir = step_dir
                step_count = 1
                
        if current_dir is not None:
            turn_text = get_turn(facing, current_dir)
            if turn_text == "Go straight": instructions.append(f"↑ **Go straight** for {step_count} step(s)")
            elif turn_text == "Turn left": instructions.append(f"↰ **Turn left** and go forward {step_count} step(s)")
            elif turn_text == "Turn right": instructions.append(f"↱ **Turn right** and go forward {step_count} step(s)")
            else: instructions.append(f"↻ **Turn around** and go forward {step_count} step(s)")
            
        instructions.append("🎯 **Pick up all items and return to Depot**")
        
        with st.expander("📝 Step-by-Step Directions for Worker", expanded=True):
            for idx, inst in enumerate(instructions):
                st.markdown(f"**{idx + 1}.** {inst}")
