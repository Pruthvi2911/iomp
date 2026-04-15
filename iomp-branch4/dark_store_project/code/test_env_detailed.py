"""
Detailed Environment Test - Verify it's working correctly
"""

import sys
sys.path.append('D:/iomp/dark_store_project/code')

# Import from the actual filename (3_gym_environment.py)
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

print("=" * 60)
print("DETAILED ENVIRONMENT DIAGNOSTIC TEST")
print("=" * 60)

layout_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
env = DarkStoreEnv(layout_file, max_steps=350, max_items_per_order=4)

print("\n[TEST 1] Can agent move correctly?")
obs, info = env.reset()
print(f"Starting position: {info['current_position']}")

# Move right
obs, reward, term, trunc, info = env.step(3)  # Action 3 = Right
print(f"After moving right: {info['current_position']}")
print(f"Reward: {reward:.2f}")

# Move down
obs, reward, term, trunc, info = env.step(1)  # Action 1 = Down
print(f"After moving down: {info['current_position']}")
print(f"Reward: {reward:.2f}")

if reward == -0.1:
    print("✓ Movement working correctly!")
else:
    print("❌ Movement reward incorrect")

print("\n[TEST 2] Can agent pick items?")
obs, info = env.reset()
print(f"Order items: {env.order_items}")
print(f"Total items in order: {len(env.order_items)}")

# Get first item position
first_item = env.order_items[0]
target_pos = env.product_positions[first_item]
print(f"First item location: {target_pos}")

# Manually move agent to that position
env.current_position = list(target_pos)
print(f"Moved agent to: {env.current_position}")

# Try to pick
obs, reward, term, trunc, info = env.step(4)  # Action 4 = Pick
print(f"After picking: Reward = {reward:.2f}")
print(f"Items picked: {info['items_picked']}/{info['total_items']}")

if reward > 5:
    print("✓ Picking working correctly!")
else:
    print("❌ Picking not working")

print("\n[TEST 3] Are items spread across grid?")
positions = list(env.product_positions.values())
unique_positions = set(positions)
print(f"Total products: {len(env.product_positions)}")
print(f"Unique shelf positions: {len(unique_positions)}")
print(f"Products per shelf (avg): {len(positions) / len(unique_positions):.1f}")

# Sample some positions
import random
sample_positions = random.sample(positions, 10)
print(f"\nSample item positions: {sample_positions[:10]}")

print("\n[TEST 4] Can a SMART agent succeed?")
# Create a simple greedy agent that actually tries

successes = 0
for episode in range(10):
    obs, info = env.reset()
    done = False
    steps = 0
    
    # Greedy strategy: Go to each item one by one
    while not done and steps < 350:
        current_pos = tuple(env.current_position)
        
        # If items remain, go to nearest one
        if len(env.items_remaining) > 0:
            # Find nearest item
            nearest_item = None
            nearest_dist = 999
            
            for item_id in env.items_remaining:
                item_pos = env.product_positions[item_id]
                dist = abs(current_pos[0] - item_pos[0]) + abs(current_pos[1] - item_pos[1])
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_item = item_id
            
            target_pos = env.product_positions[nearest_item]
            
            # Move towards target
            if current_pos[0] < target_pos[0]:
                action = 1  # Down
            elif current_pos[0] > target_pos[0]:
                action = 0  # Up
            elif current_pos[1] < target_pos[1]:
                action = 3  # Right
            elif current_pos[1] > target_pos[1]:
                action = 2  # Left
            else:
                action = 4  # Pick (we're at the position)
        else:
            # All items picked, go back to depot
            depot = env.depot_position
            if current_pos[0] < depot[0]:
                action = 1  # Down
            elif current_pos[0] > depot[0]:
                action = 0  # Up
            elif current_pos[1] < depot[1]:
                action = 3  # Right
            elif current_pos[1] > depot[1]:
                action = 2  # Left
            else:
                action = 4  # Pick (at depot, just end)
        
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
    
    if info['items_picked'] == info['total_items']:
        successes += 1

print(f"Greedy agent success rate: {successes}/10 episodes")

if successes >= 7:
    print("\n✓✓✓ ENVIRONMENT IS PERFECT! ✓✓✓")
    print("✓ Random agent 0% is NORMAL (problem is hard)")
    print("✓ PPO training will work great!")
    print("\n→ START TRAINING WITH CONFIDENCE!")
elif successes >= 3:
    print("\n⚠ Environment works but is VERY difficult")
    print("→ PPO might need more training time")
else:
    print("\n❌ Environment might have issues")
    print("→ Let's debug before training")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
