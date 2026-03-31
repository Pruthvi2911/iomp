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
    
    def __init__(self, layout_file, max_steps=300, max_items_per_order=3):
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
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right (picking is AUTOMATIC)
        self.action_space = spaces.Discrete(4)
        
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
        self._prev_min_dist = None  # for reward shaping
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start at depot
        self.current_position = list(self.depot_position)
        
        num_items = 3  # Fixed 3 items per order
        # Get probabilities based on real grocery dataset frequencies
        probs = []
        for pid in self.all_product_ids:
            # Use real grocery dataset frequencies directly (no amplification)
            freq = self.layout[str(pid)].get('frequency', 1)
            probs.append(freq)
            
        prob_sum = sum(probs)
        normalized_probs = [p/prob_sum for p in probs]

        # Sample 3 items based on how likely real humans are to order them!
        self.order_items = list(self.np_random.choice(
            self.all_product_ids, 
            size=num_items, 
            replace=False,
            p=normalized_probs
        ))
        
        # Track which items still need to be picked
        self.items_remaining = set(self.order_items)
        
        # Reset counters
        self.steps_taken = 0
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.current_position))
        
        # Initialise shaping distance
        self._prev_min_dist = self._min_dist_to_item()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _min_dist_to_item(self):
        """Manhattan distance to nearest remaining item, or to depot if all picked."""
        pos = self.current_position
        if self.items_remaining:
            return min(
                abs(pos[0] - self.product_positions[i][0]) + abs(pos[1] - self.product_positions[i][1])
                for i in self.items_remaining
            )
        else:
            # All items picked — now guide back to depot
            return abs(pos[0] - self.depot_position[0]) + abs(pos[1] - self.depot_position[1])

    def step(self, action):
        """Execute one step in the environment"""
        
        self.steps_taken += 1
        reward = 0
        terminated = False
        truncated = False
        
        # Agent's intended move
        intended_pos = list(self.current_position)
        if action == 0:  # Up
            intended_pos[0] = max(0, intended_pos[0] - 1)
        elif action == 1:  # Down
            intended_pos[0] = min(self.grid_rows - 1, intended_pos[0] + 1)
        elif action == 2:  # Left
            intended_pos[1] = max(0, intended_pos[1] - 1)
        elif action == 3:  # Right
            intended_pos[1] = min(self.grid_cols - 1, intended_pos[1] + 1)
            
        self.current_position = intended_pos
        
        # AUTO-PICK: if agent is on an item cell, pick it automatically
        current_pos = tuple(self.current_position)
        for item_id in list(self.items_remaining):
            if self.product_positions[item_id] == current_pos:
                self.items_remaining.remove(item_id)
                reward += 50  # Reward for reaching and picking item
                self._prev_min_dist = self._min_dist_to_item()  # recalibrate shaping
                if len(self.items_remaining) == 0:
                    reward += 1  # Encouragement: all items picked, now go home
        
        # Reward shaping: guide toward nearest remaining item
        curr_dist = self._min_dist_to_item()
        shaping = self._prev_min_dist - curr_dist
        reward += 0.8 * shaping
        self._prev_min_dist = curr_dist
        
        # Small step penalty
        reward -= 0.1
        
        self.visited_positions.add(current_pos)
        
        # Success: all items picked AND returned to depot
        if len(self.items_remaining) == 0:
            if tuple(self.current_position) == self.depot_position:
                reward += 100
                terminated = True
        
        # Timeout
        if self.steps_taken >= self.max_steps:
            truncated = True
            reward -= 10
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current state observation — NORMALISED to [0,1]"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Current position (normalised)
        obs[0] = self.current_position[0] / self.grid_rows
        obs[1] = self.current_position[1] / self.grid_cols
        
        # Items: (remaining_flag, target_row, target_col) — normalised
        for i, item_id in enumerate(self.order_items):
            if i < self.max_items_per_order:
                base_idx = 2 + (i * 3)
                if item_id in self.items_remaining:
                    obs[base_idx] = 1.0
                    item_pos = self.product_positions[item_id]
                    obs[base_idx + 1] = item_pos[0] / self.grid_rows
                    obs[base_idx + 2] = item_pos[1] / self.grid_cols
                else:
                    obs[base_idx] = 0.0
                    obs[base_idx + 1] = 0.0
                    obs[base_idx + 2] = 0.0
        
        # Steps taken (normalised)
        obs[2 + self.max_items_per_order * 3] = self.steps_taken / self.max_steps
        
        return obs
    
    def _get_info(self):
        """Get additional info"""
        all_picked = len(self.items_remaining) == 0
        at_depot = tuple(self.current_position) == self.depot_position
        return {
            'current_position': tuple(self.current_position),
            'items_remaining': len(self.items_remaining),
            'total_items': len(self.order_items),
            'steps_taken': self.steps_taken,
            'items_picked': len(self.order_items) - len(self.items_remaining),
            'success': all_picked and at_depot  # True ONLY when fully done
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
