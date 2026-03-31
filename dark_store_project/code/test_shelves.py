GRID_ROWS = 16
GRID_COLS = 14
shelves = []
for row in range(1, GRID_ROWS):
    for col in range(GRID_COLS):
        pos_in_aisle = col % 3
        if pos_in_aisle == 0:  # Left shelf
            shelves.append((row, col, 'left'))
        elif pos_in_aisle == 2:  # Right shelf
            if col < 12:  # 4 aisles = cols 0..11
                shelves.append((row, col, 'right'))
print(f"Total shelves: {len(shelves)}")
