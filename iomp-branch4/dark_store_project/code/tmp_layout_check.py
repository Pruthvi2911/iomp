import json
from collections import Counter

d = json.load(open('results/optimized_layout.json'))
positions = [(v['row'], v['col']) for v in d.values()]
unique_pos = set(positions)
pos_counts = Counter(positions)
shared = {pos: cnt for pos, cnt in pos_counts.items() if cnt > 1}

print(f"Total products in layout : {len(d)}")
print(f"Unique grid positions    : {len(unique_pos)}")
print(f"Positions with >1 product: {len(shared)}")
print(f"Max products at one cell : {max(pos_counts.values())}")
print(f"\nSample entry:")
k, v = list(d.items())[0]
print(f"  Product {k}: {v}")
print(f"\nTop 5 most crowded cells:")
for pos, cnt in sorted(shared.items(), key=lambda x: -x[1])[:5]:
    print(f"  Position {pos}: {cnt} products")
