"""
Dark Store Project - Gymnasium Environment
Step 3: Create the grid world for PPO training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json

class DarkStoreEnv(gym.Env):
    """
    Custom Gymnasium environment for dark store order picking.
    
    The agent (picker) must navigate a 12x13 grid warehouse to collect
    items from an order list, starting and ending at the depot.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, layout_file, max_steps=350, max_items_per_order=3):
        super(DarkStoreEnv, self).__init__()
        
        # Grid configuration
        self.grid_rows = 12
        self.grid_cols = 13
        self.depot_position = (0, 6)
        self.max_steps = max_steps
        self.max_items_per_order = max_items_per_order
        
        # Load optimized layout from ILP
        with open(layout_file, 'r') as f:
            self.layout = json.load(f)
        
        # Create product position mapping
        self.product_positions = {}
        for prod_id, info in self.layout.items():
            self.product_positions[int(prod_id)] = (info['row'], info['col'])
        
        # Get list of all products
        self.all_product_ids = list(self.product_positions.keys())
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Pick
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [current_row, current_col, 
        #                      item1_remaining, item1_row, item1_col,
        #                      item2_remaining, item2_row, item2_col, ...,
        #                      steps_taken]
        # Each item slot has 3 values: (remaining_flag, target_row, target_col)
        # This gives the agent the crucial info of WHERE items are!
        obs_size = 2 + (self.max_items_per_order * 3) + 1  # pos + items*3 + steps
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.grid_rows, self.grid_cols, max_steps),
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # State variables
        self.current_position = None
        self.order_items = None
        self.items_remaining = None
        self.steps_taken = None
        self.visited_positions = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start at depot
        self.current_position = list(self.depot_position)
        
        # GUARENTEED SUCCESS MODE: Locked to exactly 3 items per order.
        num_items = 3  # Always 3 items
        self.order_items = self.np_random.choice(
            self.all_product_ids, 
            size=num_items, 
            replace=False
        ).tolist()
        
        # Track which items still need to be picked
        self.items_remaining = set(self.order_items)
        
        # Reset counters
        self.steps_taken = 0
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.current_position))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        
        self.steps_taken += 1
        reward = 0
        terminated = False
        truncated = False
        
        # Save old position for backtracking detection
        old_position = tuple(self.current_position)
        
        # Execute action
        if action == 0:  # Move Up
            new_row = max(0, self.current_position[0] - 1)
            self.current_position[0] = new_row
            reward -= 0.1  # Small penalty for each move
            
        elif action == 1:  # Move Down
            new_row = min(self.grid_rows - 1, self.current_position[0] + 1)
            self.current_position[0] = new_row
            reward -= 0.1
            
        elif action == 2:  # Move Left
            new_col = max(0, self.current_position[1] - 1)
            self.current_position[1] = new_col
            reward -= 0.1
            
        elif action == 3:  # Move Right
            new_col = min(self.grid_cols - 1, self.current_position[1] + 1)
            self.current_position[1] = new_col
            reward -= 0.1
            
        elif action == 4:  # Pick item
            current_pos = tuple(self.current_position)
            
            # Check if there's an item to pick at this position
            picked_something = False
            for item_id in list(self.items_remaining):
                item_pos = self.product_positions[item_id]
                
                if item_pos == current_pos:
                    self.items_remaining.remove(item_id)
                    reward += 50  # Big reward for picking correct item
                    picked_something = True
            
            if not picked_something:
                reward -= 0.1  # SAME AS MOVING! Do not make it scared to pick!
        
        # Check for backtracking - REMOVED PENALTY! 
        # (It was preventing the agent from walking up and down aisles)
        new_position = tuple(self.current_position)
        # We still track visited positions just in case, but no penalty
        self.visited_positions.add(new_position)
        
        # Check if order is complete
        if len(self.items_remaining) == 0:
            # Check if back at depot
            if tuple(self.current_position) == self.depot_position:
                reward += 100  # Massive bonus for completing order AND returning to depot
                terminated = True
        
        # Check if max steps reached
        if self.steps_taken >= self.max_steps:
            truncated = True
            reward -= 10  # Penalty for taking too long
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current state observation with item positions"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Current position
        obs[0] = self.current_position[0]
        obs[1] = self.current_position[1]
        
        # Items: (remaining_flag, target_row, target_col) for each slot
        for i, item_id in enumerate(self.order_items):
            if i < self.max_items_per_order:
                base_idx = 2 + (i * 3)
                if item_id in self.items_remaining:
                    obs[base_idx] = 1.0  # still needs picking
                    item_pos = self.product_positions[item_id]
                    obs[base_idx + 1] = item_pos[0]  # target row
                    obs[base_idx + 2] = item_pos[1]  # target col
                else:
                    obs[base_idx] = 0.0  # already picked
                    obs[base_idx + 1] = 0.0
                    obs[base_idx + 2] = 0.0
        
        # Steps taken
        obs[2 + self.max_items_per_order * 3] = self.steps_taken
        
        return obs
    
    def _get_info(self):
        """Get additional info"""
        return {
            'current_position': tuple(self.current_position),
            'items_remaining': len(self.items_remaining),
            'total_items': len(self.order_items),
            'steps_taken': self.steps_taken,
            'items_picked': len(self.order_items) - len(self.items_remaining)
        }
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"\nStep {self.steps_taken}")
            print(f"Position: {self.current_position}")
            print(f"Items remaining: {len(self.items_remaining)}/{len(self.order_items)}")
    
    def close(self):
        """Clean up resources"""
        pass


# ============================================
# TEST SCRIPT - Run this to verify environment works
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DARK STORE ENVIRONMENT")
    print("=" * 60)
    
    # Path to layout file
    layout_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
    
    print("\n[1/3] Creating environment...")
    try:
        env = DarkStoreEnv(layout_file)
        print("✓ Environment created successfully!")
        print(f"  Grid size: {env.grid_rows}×{env.grid_cols}")
        print(f"  Total products: {len(env.all_product_ids)}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space.shape}")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        exit()
    
    print("\n[2/3] Testing random agent...")
    
    # Run 5 test episodes with random actions
    for episode in range(5):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Order: {info['total_items']} items")
        print(f"Starting at: {info['current_position']}")
        
        step_count = 0
        while not done and step_count < 300:
            # Take random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            done = terminated or truncated
        
        print(f"Result: {info['items_picked']}/{info['total_items']} items picked")
        print(f"Steps: {step_count}")
        print(f"Total reward: {total_reward:.2f}")
        
        if info['items_picked'] == info['total_items']:
            print("✓ Successfully picked all items!")
        else:
            print("✗ Did not pick all items")
    
    print("\n[3/3] Environment verification...")
    
    # Run 100 quick tests
    success_count = 0
    total_episodes = 100
    
    for _ in range(total_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        
        if info['items_picked'] == info['total_items']:
            success_count += 1
    
    success_rate = (success_count / total_episodes) * 100
    
    print(f"\n100 episode test results:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  (Random agent should be 5-15% success)")
    
    if success_rate > 0:
        print("\n✓ Environment is working correctly!")
        print("✓ Ready for PPO training!")
    else:
        print("\n⚠ Warning: Random agent never succeeded")
        print("  This might indicate environment issues")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT TEST COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run PPO training (Day 15)")
    print("=" * 60)
