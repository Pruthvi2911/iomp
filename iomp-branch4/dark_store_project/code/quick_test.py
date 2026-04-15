"""Quick test to see if agent can pick items"""
import sys
sys.path.append('D:/iomp/dark_store_project/code')
import importlib
gym_env_module = importlib.import_module("3_gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

# Create environment
layout_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
env = DarkStoreEnv(layout_file, max_steps=350)

# Reset and check order
obs, info = env.reset()
print(f"\n=== NEW ORDER ===")
print(f"Total items: {info['total_items']}")
print(f"Items to pick: {env.order_items}")
print(f"Starting position: {info['current_position']}")
print(f"Depot: {env.depot_position}")

# Check where first item is
first_item = env.order_items[0]
first_item_pos = env.product_positions[first_item]
print(f"\nFirst item location: {first_item_pos}")

# Try to manually complete an order
print(f"\n=== MANUAL TEST ===")
steps = 0
total_reward = 0

# Move to first item position manually
env.current_position = list(first_item_pos)
print(f"Teleported to first item: {env.current_position}")

# Try to pick
obs, reward, term, trunc, info = env.step(4)  # Action 4 = Pick
print(f"Pick action reward: {reward}")
print(f"Items remaining: {info['items_remaining']}")
print(f"Items picked: {info['items_picked']}")

# Try picking all items
for item_id in list(env.order_items):
    item_pos = env.product_positions[item_id]
    env.current_position = list(item_pos)
    obs, reward, term, trunc, info = env.step(4)
    print(f"Picked item at {item_pos}, reward: {reward}, remaining: {info['items_remaining']}")

# Return to depot
env.current_position = list(env.depot_position)
obs, reward, term, trunc, info = env.step(0)  # Any action
print(f"\nBack at depot: {env.current_position}")
print(f"All items picked: {info['items_picked']} / {info['total_items']}")
print(f"Episode terminated: {term}")
