import json
import random

opt_file = "D:/iomp/dark_store_project/results/optimized_layout.json"
with open(opt_file, "r") as f:
    opt_layout = json.load(f)

# Extract the exact same 100 product IDs
prod_ids = list(opt_layout.keys())

# Recreate the 100 products uniformly scattered
grid_rows, grid_cols = 12, 13
depot = (0, 6)

all_shelves = []
for r in range(grid_rows):
    for c in range(grid_cols):
        # We need 100 random shelf locations
        if (r, c) != depot:
            all_shelves.append((r, c))

random.shuffle(all_shelves)
selected_shelves = all_shelves[:len(prod_ids)]

random_layout = {}
for i, (r, c) in enumerate(selected_shelves):
    prod_id = prod_ids[i]
    random_layout[prod_id] = opt_layout[prod_id].copy()
    random_layout[prod_id]['row'] = r
    random_layout[prod_id]['col'] = c

with open("D:/iomp/dark_store_project/results/random_layout.json", "w") as f:
    json.dump(random_layout, f, indent=4)

print("Generated random_layout.json with real product IDs successfully.")
