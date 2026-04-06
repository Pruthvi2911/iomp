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

print("=" * 60)
print("DARK STORE - ILP SLOTTING OPTIMIZATION")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = "D:/iomp/dark_store_project/data/"
OUTPUT_PATH = "D:/iomp/dark_store_project/results/"

# Grid configuration (4 aisles, 13x13 grid with bottom cross-aisle)
GRID_ROWS = 13
GRID_COLS = 13
DEPOT_POSITION = (0, 6)  # Middle of first row

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================
# STEP 1: Load Preprocessed Data
# ============================================
print("\n[1/6] Loading preprocessed data...")

with open(DATA_PATH + "processed_data.pkl", 'rb') as f:
    data = pickle.load(f)

products = data['products']

# Keep only top 100 products for new layout
# 88 products to fill 88 shelf slots exactly
# (4 aisles x 2 sides x 11 rows = 88 positions, col 12 reserved as boundary)
products = products.sort_values('frequency', ascending=False).head(88).reset_index(drop=True)

affinity_pairs = data['affinity_pairs']

print(f"✓ Loaded {len(products)} products (filtered to top 150)")
print(f"✓ Loaded {len(affinity_pairs)} affinity pairs")

# ============================================
# STEP 2: Define Shelf Positions
# ============================================
print("\n[2/6] Defining shelf positions in 12×13 grid...")

def get_shelf_positions():
    """
    Generate all valid shelf positions in the 13x13 wall-aware grid.

    Layout per aisle row:
      col 0  : left shelf   | col 1  : walkway
      col 2  : right shelf  | col 3  : left shelf
      col 4  : walkway      | col 5  : right shelf
      col 6  : left shelf   | col 7  : walkway
      col 8  : right shelf  | col 9  : left shelf
      col 10 : walkway      | col 11 : right shelf
      col 12 : boundary (no shelf, no walkway)

    Rows 0 and 12 are cross-aisles — no shelves placed there.
    Total: 4 aisles x 2 sides x 11 rows = 88 shelf positions.
    """
    shelves = []
    SHELF_ROWS = range(1, GRID_ROWS - 1)  # rows 1-11 (skip row 0 depot & row 12 cross-aisle)
    SHELF_COLS = range(GRID_COLS - 1)     # cols 0-11 (skip col 12 boundary)

    for row in SHELF_ROWS:
        for col in SHELF_COLS:
            pos_in_aisle = col % 3
            if pos_in_aisle == 0:    # Left shelf: cols 0, 3, 6, 9
                shelves.append((row, col, 'left'))
            elif pos_in_aisle == 2:  # Right shelf: cols 2, 5, 8, 11
                shelves.append((row, col, 'right'))

    return shelves

shelves = get_shelf_positions()
print(f"✓ Generated {len(shelves)} shelf positions")
print(f"  Grid: {GRID_ROWS} rows × {GRID_COLS} columns")
print(f"  Depot at: {DEPOT_POSITION}")

# ============================================
# STEP 3: Calculate Distances
# ============================================
print("\n[3/6] Calculating Manhattan distances from depot...")

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan (grid) distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Calculate distance for each shelf to depot
shelf_distances = {}
for i, (row, col, side) in enumerate(shelves):
    dist = manhattan_distance((row, col), DEPOT_POSITION)
    shelf_distances[i] = dist

print(f"✓ Calculated distances for {len(shelves)} shelves")
print(f"  Closest shelf: {min(shelf_distances.values())} units")
print(f"  Farthest shelf: {max(shelf_distances.values())} units")

# ============================================
# STEP 4: Create Affinity Matrix
# ============================================
print("\n[4/6] Building affinity matrix from co-purchase data...")

# Create affinity lookup (which products are bought together)
affinity_dict = {}
for _, row in affinity_pairs.iterrows():
    p1 = row['product_1_id']
    p2 = row['product_2_id']
    strength = row['co_purchase_count']
    
    affinity_dict[(p1, p2)] = strength
    affinity_dict[(p2, p1)] = strength  # Symmetric

print(f"✓ Built affinity matrix with {len(affinity_dict)} relationships")

# ============================================
# STEP 5: Solve ILP Optimization
# ============================================
print("\n[5/6] Setting up ILP optimization problem...")
print("  This may take 2-10 seconds depending on your CPU...")

start_time = time.time()

# Create the optimization problem
prob = LpProblem("Dark_Store_Slotting", LpMinimize)

# Decision variables: x[i,s] = 1 if product i assigned to shelf s
product_ids = products['product_id'].tolist()
x = {}
for i, prod_id in enumerate(product_ids):
    for s in range(len(shelves)):
        x[i, s] = LpVariable(f"x_{i}_{s}", cat='Binary')

print(f"✓ Created {len(x)} decision variables")

# Objective function: Minimize weighted distance + affinity penalty
print("  Building objective function...")

# Part 1: Frequency-weighted distance
freq_terms = []
for i, prod_id in enumerate(product_ids):
    freq = products.iloc[i]['frequency']
    for s in range(len(shelves)):
        dist = shelf_distances[s]
        freq_terms.append(freq * dist * x[i, s])

# Note: Affinity terms would create a quadratic problem (x[i,s1] * x[j,s2])
# which PuLP (a linear solver) cannot handle. Using frequency-weighted 
# distance alone works well enough for optimal slotting.
prob += lpSum(freq_terms)

print(f"✓ Objective function built")

# Constraints
print("  Adding constraints...")

# Constraint 1: Each product assigned to exactly ONE shelf
for i in range(len(product_ids)):
    prob += lpSum([x[i, s] for s in range(len(shelves))]) == 1

# Constraint 2: Each shelf holds at most 1 product (capacity)
# 100 products into ~100 distinct layout spaces
for s in range(len(shelves)):
    prob += lpSum([x[i, s] for i in range(len(product_ids))]) <= 1

print(f"✓ Added constraints")
print("  Constraint 1: Each product → 1 shelf")
print("  Constraint 2: Each shelf → max 1 product")

# Solve the problem
print("\n  Solving ILP with CP-SAT solver...")
print("  (This takes 2-10 seconds...)")

prob.solve(PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

solve_time = time.time() - start_time

print(f"\n✓ Optimization complete!")
print(f"  Status: {LpStatus[prob.status]}")
print(f"  Solver time: {solve_time:.2f} seconds")
print(f"  Objective value: {value(prob.objective):,.0f}")

# ============================================
# STEP 6: Extract and Save Solution
# ============================================
print("\n[6/6] Extracting optimal layout...")

# Extract solution
layout = {}
shelf_assignments = {s: [] for s in range(len(shelves))}

for i, prod_id in enumerate(product_ids):
    for s in range(len(shelves)):
        if x[i, s].varValue == 1:
            row, col, side = shelves[s]
            
            layout[prod_id] = {
                'shelf_id': s,
                'row': int(row),
                'col': int(col),
                'side': side,
                'product_name': products.iloc[i]['product_name'],
                'frequency': int(products.iloc[i]['frequency']),
                'abc_class': products.iloc[i]['abc_class'],
                'distance_from_depot': float(shelf_distances[s])
            }
            
            shelf_assignments[s].append({
                'product_id': int(prod_id),
                'product_name': products.iloc[i]['product_name'],
                'frequency': int(products.iloc[i]['frequency'])
            })
            break

print(f"✓ Extracted layout for {len(layout)} products")

# Show sample assignments
print(f"\nSample assignments:")
sorted_products = sorted(layout.items(), key=lambda x: x[1]['frequency'], reverse=True)
for prod_id, info in sorted_products[:5]:
    print(f"  {info['product_name']}: Row {info['row']}, Col {info['col']} "
          f"({info['distance_from_depot']:.0f}m from depot)")

# Save layout as JSON
layout_file = OUTPUT_PATH + "optimized_layout.json"
with open(layout_file, 'w') as f:
    # Convert keys to strings for JSON
    layout_str_keys = {str(k): v for k, v in layout.items()}
    json.dump(layout_str_keys, f, indent=2)

print(f"\n✓ Saved layout to: {layout_file}")

# ============================================
# STEP 7: Create Visualization
# ============================================
print("\n[7/7] Creating layout heatmap...")

# Create grid visualization
grid = np.zeros((GRID_ROWS, GRID_COLS))

for prod_id, info in layout.items():
    row, col = info['row'], info['col']
    freq = info['frequency']
    
    # Color by frequency (log scale for better visualization)
    grid[row, col] = np.log10(freq + 1)

# Plot
plt.figure(figsize=(14, 10))
sns.heatmap(grid, annot=False, cmap='YlOrRd', cbar_kws={'label': 'log10(Frequency)'})
plt.title('Dark Store Layout - Product Frequency Heatmap\n(Warmer colors = Higher frequency items)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Column (Aisles)', fontsize=12)
plt.ylabel('Row (Depth)', fontsize=12)

# Mark depot
plt.scatter(DEPOT_POSITION[1] + 0.5, DEPOT_POSITION[0] + 0.5, 
            s=500, c='blue', marker='*', edgecolors='white', linewidths=2,
            label='Depot/Packing Station')

plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

heatmap_file = OUTPUT_PATH + "layout_heatmap.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved heatmap to: {heatmap_file}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("ILP SLOTTING COMPLETE! ✓")
print("=" * 60)

print(f"\nFiles created:")
print(f"  1. {layout_file}")
print(f"  2. {heatmap_file}")

print(f"\nOptimization summary:")
print(f"  - Products: {len(product_ids)}")
print(f"  - Shelves: {len(shelves)}")
print(f"  - Solver time: {solve_time:.2f}s")
print(f"  - Status: {LpStatus[prob.status]}")

# ABC distribution check
abc_dist = {}
for prod_id, info in layout.items():
    abc_class = info['abc_class']
    dist = info['distance_from_depot']
    
    if abc_class not in abc_dist:
        abc_dist[abc_class] = []
    abc_dist[abc_class].append(dist)

print(f"\nAverage distance by class:")
for abc_class in ['A', 'B', 'C']:
    if abc_class in abc_dist:
        avg_dist = np.mean(abc_dist[abc_class])
        print(f"  {abc_class}-items: {avg_dist:.1f} units from depot")

print(f"\n✓ Ready for PPO training (Day 10-12)")
print("=" * 60)
