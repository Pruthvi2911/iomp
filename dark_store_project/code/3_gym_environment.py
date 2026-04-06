"""
Dark Store Project - Gymnasium Environment
Step 3: Create the grid world for PPO training

UPDATED: Wall-aware navigation
  - Agent cannot walk through shelf cells (walls)
  - 13x13 grid with bottom cross-aisle at row 12
  - Walkways: cols 1, 4, 7, 10 (rows 1-11) + rows 0 and 12 (full cross-aisles)
  - Picking is AUTOMATIC when agent stands in walkway adjacent to item shelf
  - Reward shaping uses TRUE BFS navigation distance (not through-wall Manhattan)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from collections import deque


class DarkStoreEnv(gym.Env):
    """
    Custom Gymnasium environment for dark store order picking.

    13x13 grid warehouse with physical aisle structure:
      Row 0  : top cross-aisle (depot at col 6)
      Row 12 : bottom cross-aisle
      Rows 1-11, cols 1/4/7/10 : vertical walkways
      All other cells in rows 1-11 : shelf walls (impassable)

    Agent picks an item by standing in the walkway cell adjacent to its shelf.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # Walkway columns inside the aisle block (rows 1-11)
    WALKWAY_COLS = {1, 4, 7, 10}

    def __init__(self, layout_file, max_steps=350, max_items_per_order=3):
        super(DarkStoreEnv, self).__init__()

        # ── Grid configuration ──────────────────────────────────────────
        self.grid_rows = 13          # rows 0-12
        self.grid_cols = 13          # cols 0-12
        self.depot_position = (0, 6) # top cross-aisle, centre
        self.max_steps = max_steps
        self.max_items_per_order = max_items_per_order

        # ── Build walkable cell set ──────────────────────────────────────
        # Row 0 & 12 : full cross-aisles (all cols)
        # Rows 1-11  : only walkway cols 1, 4, 7, 10
        self.walkable = set()
        for c in range(self.grid_cols):
            self.walkable.add((0, c))   # top cross-aisle
            self.walkable.add((12, c))  # bottom cross-aisle
        for r in range(1, 12):
            for c in self.WALKWAY_COLS:
                self.walkable.add((r, c))

        # ── Load layout ─────────────────────────────────────────────────
        with open(layout_file, 'r') as f:
            self.layout = json.load(f)

        self.product_positions = {}
        for prod_id, info in self.layout.items():
            self.product_positions[int(prod_id)] = (info['row'], info['col'])

        self.all_product_ids = list(self.product_positions.keys())

        # ── Precompute BFS distance between every pair of walkable cells ─
        # Called once at init — fast (~70 cells, runs in <0.1 s)
        self._bfs_dist = self._precompute_bfs()

        # ── For each shelf cell, find its walkable picking neighbours ────
        # A picker at any of these neighbours can auto-pick the item.
        self._shelf_pick_spots = {}
        for pid, shelf in self.product_positions.items():
            r, c = shelf
            spots = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in self.walkable:
                    spots.append(nb)
            self._shelf_pick_spots[pid] = spots

        # ── Action & observation spaces ──────────────────────────────────
        self.action_space = spaces.Discrete(4)  # 0=Up 1=Down 2=Left 3=Right

        obs_size = 2 + (self.max_items_per_order * 3) + 1
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.grid_rows, self.grid_cols, max_steps),
            shape=(obs_size,),
            dtype=np.float32
        )

        # ── State ────────────────────────────────────────────────────────
        self.current_position = None
        self.order_items      = None
        self.items_remaining  = None
        self.steps_taken      = None
        self.visited_positions = None
        self._prev_min_dist   = None

    # ────────────────────────────────────────────────────────────────────
    # BFS precomputation
    # ────────────────────────────────────────────────────────────────────

    def _precompute_bfs(self):
        """
        BFS from every walkable cell to every other walkable cell.
        Returns dict: dist[start][(r,c)] = steps along walkways.
        """
        dist = {}
        for start in self.walkable:
            d = {start: 0}
            q = deque([start])
            while q:
                node = q.popleft()
                nr, nc = node
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (nr + dr, nc + dc)
                    if (0 <= nb[0] < self.grid_rows and
                            0 <= nb[1] < self.grid_cols and
                            nb in self.walkable and
                            nb not in d):
                        d[nb] = d[node] + 1
                        q.append(nb)
            dist[start] = d
        return dist

    # ────────────────────────────────────────────────────────────────────
    # Navigation distance helpers
    # ────────────────────────────────────────────────────────────────────

    def _nav_dist_to_item(self, from_pos, item_id):
        """
        Minimum BFS steps from from_pos (walkable) to any picking spot
        adjacent to item_id's shelf.
        """
        spots = self._shelf_pick_spots.get(item_id, [])
        if not spots:
            return 999
        from_pos = tuple(from_pos)
        bfs_from = self._bfs_dist.get(from_pos, {})
        return min((bfs_from.get(s, 999) for s in spots), default=999)

    def _nav_dist_to_depot(self, from_pos):
        from_pos = tuple(from_pos)
        bfs_from = self._bfs_dist.get(from_pos, {})
        return bfs_from.get(self.depot_position, 999)

    def _min_dist_to_target(self):
        """BFS distance to nearest remaining item, or to depot if all done."""
        pos = tuple(self.current_position)
        if self.items_remaining:
            return min(
                self._nav_dist_to_item(pos, i)
                for i in self.items_remaining
            )
        return self._nav_dist_to_depot(pos)

    # ────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_position = list(self.depot_position)

        # Sample 3 items weighted by real purchase frequency
        probs = [self.layout[str(pid)].get('frequency', 1)
                 for pid in self.all_product_ids]
        prob_sum = sum(probs)
        normalized_probs = [p / prob_sum for p in probs]

        self.order_items = list(self.np_random.choice(
            self.all_product_ids,
            size=3,
            replace=False,
            p=normalized_probs
        ))

        self.items_remaining  = set(self.order_items)
        self.steps_taken      = 0
        self.visited_positions = {tuple(self.current_position)}
        self._prev_min_dist   = self._min_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps_taken += 1
        reward     = 0.0
        terminated = False
        truncated  = False

        # ── Compute intended next cell ───────────────────────────────────
        r, c = self.current_position
        if   action == 0: nr, nc = r - 1, c
        elif action == 1: nr, nc = r + 1, c
        elif action == 2: nr, nc = r, c - 1
        elif action == 3: nr, nc = r, c + 1
        else:             nr, nc = r, c

        # Clip to grid boundary
        nr = max(0, min(self.grid_rows - 1, nr))
        nc = max(0, min(self.grid_cols - 1, nc))

        intended = (nr, nc)

        # ── Wall check ───────────────────────────────────────────────────
        if intended in self.walkable:
            self.current_position = [nr, nc]
            reward -= 0.1          # small step cost
        else:
            reward -= 0.3          # wall-bump penalty (agent stays put)

        current_pos = tuple(self.current_position)

        # ── Reward shaping: BFS progress toward nearest target ───────────
        curr_dist = self._min_dist_to_target()
        reward   += 0.8 * (self._prev_min_dist - curr_dist)

        # ── Auto-pick: agent adjacent to item shelf → pick it ───────────
        for item_id in list(self.items_remaining):
            if current_pos in self._shelf_pick_spots.get(item_id, []):
                self.items_remaining.remove(item_id)
                reward += 50
                if len(self.items_remaining) == 0:
                    reward += 25   # bonus: all items collected, now go home

        # Recalibrate shaping baseline AFTER picking
        self._prev_min_dist = self._min_dist_to_target()

        self.visited_positions.add(current_pos)

        # ── Termination: all picked + back at depot ──────────────────────
        if not self.items_remaining and current_pos == self.depot_position:
            reward    += 100
            terminated = True

        # ── Timeout ─────────────────────────────────────────────────────
        if self.steps_taken >= self.max_steps:
            truncated   = True
            items_picked = len(self.order_items) - len(self.items_remaining)
            reward -= 10
            reward += items_picked * 5   # partial credit

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        obs[0] = self.current_position[0] / self.grid_rows
        obs[1] = self.current_position[1] / self.grid_cols

        for i, item_id in enumerate(self.order_items):
            if i < self.max_items_per_order:
                base = 2 + i * 3
                if item_id in self.items_remaining:
                    item_pos = self.product_positions[item_id]
                    obs[base]     = 1.0
                    obs[base + 1] = item_pos[0] / self.grid_rows
                    obs[base + 2] = item_pos[1] / self.grid_cols
                # else: zeros (item already picked)

        obs[2 + self.max_items_per_order * 3] = self.steps_taken / self.max_steps
        return obs

    def _get_info(self):
        all_picked = len(self.items_remaining) == 0
        at_depot   = tuple(self.current_position) == self.depot_position
        return {
            'current_position': tuple(self.current_position),
            'items_remaining' : len(self.items_remaining),
            'total_items'     : len(self.order_items),
            'steps_taken'     : self.steps_taken,
            'items_picked'    : len(self.order_items) - len(self.items_remaining),
            'success'         : all_picked and at_depot
        }

    def render(self, mode='human'):
        if mode == 'human':
            grid = [['.' for _ in range(self.grid_cols)]
                    for _ in range(self.grid_rows)]
            # Mark shelves
            for r in range(1, 12):
                for c in range(self.grid_cols):
                    if (r, c) not in self.walkable:
                        grid[r][c] = '#'
            # Mark remaining items
            for pid in self.items_remaining:
                r, c = self.product_positions[pid]
                grid[r][c] = 'I'
            # Mark depot
            dr, dc = self.depot_position
            grid[dr][dc] = 'D'
            # Mark agent
            ar, ac = self.current_position
            grid[ar][ac] = 'A'

            print(f"\nStep {self.steps_taken} | Pos {self.current_position} "
                  f"| Items left {len(self.items_remaining)}/{len(self.order_items)}")
            print("  " + " ".join(str(c) for c in range(self.grid_cols)))
            for r, row in enumerate(grid):
                print(f"{r:2} " + " ".join(row))

    def close(self):
        pass


# ============================================================
# QUICK SANITY TEST
# ============================================================
if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("TESTING WALL-AWARE DARK STORE ENVIRONMENT")
    print("=" * 60)

    layout_file = "D:/iomp/dark_store_project/results/optimized_layout.json"

    try:
        env = DarkStoreEnv(layout_file)
        print(f"Grid  : {env.grid_rows} x {env.grid_cols}")
        print(f"Walkable cells : {len(env.walkable)}")
        print(f"Products : {len(env.all_product_ids)}")
        print(f"Action space   : {env.action_space}")
        print(f"Obs space      : {env.observation_space.shape}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Verify no product is at an unreachable shelf
    unreachable = []
    for pid, shelf in env.product_positions.items():
        spots = env._shelf_pick_spots.get(pid, [])
        if not spots:
            unreachable.append((pid, shelf))
    if unreachable:
        print(f"\nWARNING: {len(unreachable)} items have no reachable pick spot!")
        for pid, shelf in unreachable:
            print(f"  Product {pid} at {shelf}")
    else:
        print("\n All products have at least one reachable pick spot.")

    # Run 5 episodes with a random agent
    print("\nRunning 5 random episodes...")
    for ep in range(5):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        print(f"  Ep {ep+1}: {info['items_picked']}/{info['total_items']} items "
              f"in {steps} steps | success={info['success']}")

    print("\nEnvironment test complete!")
    print("=" * 60)
