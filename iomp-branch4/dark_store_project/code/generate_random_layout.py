"""
Generate a random product layout using the SAME 88 products and 88 shelf slots
as the ILP-optimised layout, but scatter them randomly.

Wall-aware grid: 13x13
  Shelf rows : 1-11
  Shelf cols : 0, 2, 3, 5, 6, 8, 9, 11  (col % 3 != 1  AND  col < 12)
  Col 12 and rows 0 & 12 are cross-aisles / boundary — no shelves.
"""

import json
import random

opt_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
with open(opt_file, "r") as f:
    opt_layout = json.load(f)

# Use the exact same product IDs that ILP placed
prod_ids = list(opt_layout.keys())

# Build the same 88 shelf positions used by the updated ILP
GRID_ROWS, GRID_COLS = 13, 13
all_shelves = []
for r in range(1, GRID_ROWS - 1):       # rows 1-11
    for c in range(GRID_COLS - 1):       # cols 0-11
        if c % 3 != 1:                   # skip walkway cols 1, 4, 7, 10
            all_shelves.append((r, c))

assert len(all_shelves) == 88, f"Expected 88 shelves, got {len(all_shelves)}"
assert len(prod_ids) == 88,    f"Expected 88 products, got {len(prod_ids)}"

random.shuffle(all_shelves)

random_layout = {}
for i, prod_id in enumerate(prod_ids):
    r, c = all_shelves[i]
    random_layout[prod_id] = opt_layout[prod_id].copy()
    random_layout[prod_id]['row'] = r
    random_layout[prod_id]['col'] = c

with open("D:/iomp/dark_store_project/results/random_layout.json", "w") as f:
    json.dump(random_layout, f, indent=4)

print(f"Generated random_layout.json with {len(random_layout)} products "
      f"across {len(all_shelves)} shelf positions.")
