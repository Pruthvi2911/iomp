"""
Dark Store Project - ILP Slotting Optimization
Step 2: Assign 500 SKUs to 88 shelf positions optimally
"""

import pickle
import json
import numpy as np
import pandas as pd
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import os

print("=" * 60)
print("DARK STORE - ILP SLOTTING OPTIMIZATION")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "results"

# Grid configuration (4 aisles, 13x13 grid with bottom cross-aisle)
GRID_ROWS = 13
GRID_COLS = 13
DEPOT_POSITION = (0, 6)  # Middle of first row

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# STEP 1: Load Preprocessed Data
# ============================================
print("\n[1/6] Loading preprocessed data...")

# FIXED: Changed '+' to '/'
try:
    with open(DATA_PATH / "processed_data.pkl", 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: 'processed_data.pkl' not found in {DATA_PATH}. Run Step 1 first!")
    exit()

products = data['products']

# 88 products to fill 88 shelf slots exactly
# (4 aisles x 2 sides x 11 rows = 88 positions)
products = products.sort_values('frequency', ascending=False).head(88).reset_index(drop=True)
affinity_pairs = data['affinity_pairs']

print(f"✓ Loaded {len(products)} products (filtered to top 88 for exact fit)")
print(f"✓ Loaded {len(affinity_pairs)} affinity pairs")

# ============================================
# STEP 2: Define Shelf Positions
# ============================================
print("\n[2/6] Defining shelf positions in 13×13 grid...")

def get_shelf_positions():
    """
    Layout per aisle row:
      col 0  : left shelf   | col 1  : walkway
      col 2  : right shelf  | col 3  : left shelf
      col 4  : walkway      | col 5  : right shelf
      ...
    Total: 4 aisles x 2 sides x 11 rows = 88 shelf positions.
    """
    shelves = []
    SHELF_ROWS = range(1, GRID_ROWS - 1)  # rows 1-11
    SHELF_COLS = range(GRID_COLS - 1)     # cols 0-11

    for row in SHELF_ROWS:
        for col in SHELF_COLS:
            pos_in_aisle = col % 3
            if pos_in_aisle == 0:    # Left shelf
                shelves.append((row, col, 'left'))
            elif pos_in_aisle == 2:  # Right shelf
                shelves.append((row, col, 'right'))
    return shelves

shelves = get_shelf_positions()
print(f"✓ Generated {len(shelves)} shelf positions")

# ============================================
# STEP 3: Calculate Distances
# ============================================
print("\n[3/6] Calculating Manhattan distances from depot...")

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

shelf_distances = {}
for i, (row, col, side) in enumerate(shelves):
    dist = manhattan_distance((row, col), DEPOT_POSITION)
    shelf_distances[i] = dist

print(f"✓ Calculated distances. Max distance: {max(shelf_distances.values())} units")

# ============================================
# STEP 5: Solve ILP Optimization
# ============================================
print("\n[5/6] Setting up ILP optimization problem...")
start_time = time.time()

# Create the optimization problem
prob = LpProblem("Dark_Store_Slotting", LpMinimize)

# Decision variables: x[i,s] = 1 if product i assigned to shelf s
product_ids = products['product_id'].tolist()
x = {}
for i in range(len(product_ids)):
    for s in range(len(shelves)):
        x[i, s] = LpVariable(f"x_{i}_{s}", cat='Binary')

# Objective function: Minimize sum of (Frequency * Distance)
freq_terms = []
for i in range(len(product_ids)):
    freq = products.iloc[i]['frequency']
    for s in range(len(shelves)):
        dist = shelf_distances[s]
        freq_terms.append(freq * dist * x[i, s])

prob += lpSum(freq_terms)

# Constraints
# 1. Each product assigned to exactly ONE shelf
for i in range(len(product_ids)):
    prob += lpSum([x[i, s] for s in range(len(shelves))]) == 1

# 2. Each shelf holds at most 1 product
for s in range(len(shelves)):
    prob += lpSum([x[i, s] for i in range(len(product_ids))]) <= 1

print("\n  Solving ILP with CBC solver...")
prob.solve(PULP_CBC_CMD(msg=0)) 

solve_time = time.time() - start_time
print(f"✓ Optimization complete! Status: {LpStatus[prob.status]}")

# ============================================
# STEP 6: Extract and Save Solution
# ============================================
print("\n[6/6] Extracting optimal layout...")

layout = {}
for i, prod_id in enumerate(product_ids):
    for s in range(len(shelves)):
        if value(x[i, s]) == 1:
            row, col, side = shelves[s]
            layout[int(prod_id)] = {
                'shelf_id': s,
                'row': int(row),
                'col': int(col),
                'side': side,
                'product_name': products.iloc[i]['product_name'],
                'frequency': int(products.iloc[i]['frequency']),
                'abc_class': products.iloc[i]['abc_class'],
                'distance_from_depot': float(shelf_distances[s])
            }
            break

# Save layout as JSON
# FIXED: Changed '+' to '/'
layout_file = OUTPUT_PATH / "optimized_layout.json"
with open(layout_file, 'w') as f:
    layout_str_keys = {str(k): v for k, v in layout.items()}
    json.dump(layout_str_keys, f, indent=2)

# ============================================
# STEP 7: Create Visualization
# ============================================
print("\n[7/7] Creating layout heatmap...")

grid = np.zeros((GRID_ROWS, GRID_COLS))
for prod_id, info in layout.items():
    grid[info['row'], info['col']] = np.log10(info['frequency'] + 1)

plt.figure(figsize=(12, 8))
sns.heatmap(grid, annot=False, cmap='YlOrRd', cbar_kws={'label': 'log10(Frequency)'})
plt.scatter(DEPOT_POSITION[1] + 0.5, DEPOT_POSITION[0] + 0.5, s=500, c='blue', marker='*', label='Depot')
plt.title('Optimal ILP Layout - Heatmap')

# FIXED: Changed '+' to '/'
heatmap_file = OUTPUT_PATH / "layout_heatmap.png"
plt.savefig(heatmap_file, dpi=300)
print(f"✓ Saved heatmap to: {heatmap_file}")

print("\n" + "=" * 60)
print("ILP SLOTTING COMPLETE! ✓")
print("=" * 60)