"""
Dark Store Project - Data Preprocessing
Step 1: Load Instacart data and prepare for ILP + PPO
"""

import pandas as pd
import pickle
from collections import Counter
from itertools import combinations
import os

print("=" * 60)
print("DARK STORE DATA PREPROCESSING")
print("=" * 60)

# ============================================
# CONFIGURATION (Change these paths if needed)
# ============================================
DATA_PATH = "d:/iomp/dark_store_project/data/"
OUTPUT_PATH = "d:/iomp/dark_store_project/data/"

# ============================================
# STEP 1: Load Instacart Data
# ============================================
print("\n[1/5] Loading Instacart data...")

try:
    # Load the three main files
    orders = pd.read_csv(DATA_PATH + "orders.csv")
    products = pd.read_csv(DATA_PATH + "products.csv")
    order_products = pd.read_csv(DATA_PATH + "order_products__prior.csv")
    
    print(f"✓ Loaded {len(orders):,} orders")
    print(f"✓ Loaded {len(products):,} products")
    print(f"✓ Loaded {len(order_products):,} order-product pairs")
except FileNotFoundError as e:
    print(f"❌ ERROR: Could not find data files!")
    print(f"   Make sure these files exist in {DATA_PATH}:")
    print("   - orders.csv")
    print("   - products.csv")
    print("   - order_products__prior.csv")
    exit()

# ============================================
# STEP 2: Calculate Pick Frequency
# ============================================
print("\n[2/5] Calculating pick frequency for each product...")

# Count how many times each product was ordered
product_frequency = order_products['product_id'].value_counts()

# Merge with product names
product_freq_df = pd.DataFrame({
    'product_id': product_frequency.index,
    'frequency': product_frequency.values
})

# Add product names
product_freq_df = product_freq_df.merge(
    products[['product_id', 'product_name']], 
    on='product_id'
)

# Sort by frequency (highest first)
product_freq_df = product_freq_df.sort_values('frequency', ascending=False)

print(f"✓ Calculated frequency for {len(product_freq_df):,} products")
print(f"\nTop 5 most ordered items:")
for i, row in product_freq_df.head(5).iterrows():
    print(f"  {row['product_name']}: {row['frequency']:,} orders")

# ============================================
# STEP 3: Select Top 500 SKUs
# ============================================
print("\n[3/5] Selecting top 500 SKUs...")

top_500 = product_freq_df.head(500).copy()
top_500 = top_500.reset_index(drop=True)

print(f"✓ Selected top 500 products")
print(f"  Range: {top_500['frequency'].max():,} to {top_500['frequency'].min():,} orders")

# ============================================
# STEP 4: ABC Classification
# ============================================
print("\n[4/5] Performing ABC classification...")

# A-items: Top 20% (100 items) - High frequency
# B-items: Next 30% (150 items) - Medium frequency  
# C-items: Bottom 50% (250 items) - Low frequency

top_500['abc_class'] = 'C'  # Default to C
top_500.loc[:99, 'abc_class'] = 'A'      # Top 100
top_500.loc[100:249, 'abc_class'] = 'B'  # Next 150

# Count items in each class
abc_counts = top_500['abc_class'].value_counts().sort_index()
print(f"✓ ABC Classification complete:")
print(f"  A-items (High freq): {abc_counts['A']} products")
print(f"  B-items (Medium):    {abc_counts['B']} products")
print(f"  C-items (Low freq):  {abc_counts['C']} products")

# Save ABC classification
abc_output = OUTPUT_PATH + "abc_classification.csv"
top_500.to_csv(abc_output, index=False)
print(f"✓ Saved to: {abc_output}")

# ============================================
# STEP 5: Find Top 50 Affinity Pairs
# ============================================
print("\n[5/5] Finding top 50 co-purchased item pairs...")

# Get only orders containing our top 500 products
top_500_ids = set(top_500['product_id'].values)
relevant_orders = order_products[
    order_products['product_id'].isin(top_500_ids)
]

# Group products by order
orders_grouped = relevant_orders.groupby('order_id')['product_id'].apply(list)

# Find all pairs of items bought together
print("  Analyzing co-purchase patterns...")
pair_counter = Counter()

for order_items in orders_grouped:
    if len(order_items) >= 2:  # Need at least 2 items
        # Get all unique pairs from this order
        for pair in combinations(sorted(order_items), 2):
            pair_counter[pair] += 1

# Get top 50 most common pairs
top_50_pairs = pair_counter.most_common(50)

print(f"✓ Found {len(pair_counter):,} unique item pairs")
print(f"✓ Selected top 50 pairs")

# Create affinity dataframe
affinity_data = []
for (prod1, prod2), count in top_50_pairs:
    name1 = products[products['product_id'] == prod1]['product_name'].values[0]
    name2 = products[products['product_id'] == prod2]['product_name'].values[0]
    
    affinity_data.append({
        'product_1_id': prod1,
        'product_1_name': name1,
        'product_2_id': prod2,
        'product_2_name': name2,
        'co_purchase_count': count
    })

affinity_df = pd.DataFrame(affinity_data)

print(f"\nTop 5 co-purchased pairs:")
for i, row in affinity_df.head(5).iterrows():
    print(f"  {row['product_1_name']} + {row['product_2_name']}: {row['co_purchase_count']:,} times")

# Save affinity pairs
affinity_output = OUTPUT_PATH + "affinity_pairs.csv"
affinity_df.to_csv(affinity_output, index=False)
print(f"✓ Saved to: {affinity_output}")

# ============================================
# STEP 6: Save Processed Data
# ============================================
print("\n[6/6] Saving processed data for ILP and PPO...")

# Prepare final data structure
processed_data = {
    'products': top_500,
    'affinity_pairs': affinity_df,
    'metadata': {
        'total_products': 500,
        'a_items': 100,
        'b_items': 150,
        'c_items': 250,
        'affinity_pairs': 50
    }
}

# Save as pickle file
pickle_output = OUTPUT_PATH + "processed_data.pkl"
with open(pickle_output, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"✓ Saved to: {pickle_output}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE! ✓")
print("=" * 60)
print(f"\nFiles created:")
print(f"  1. {abc_output}")
print(f"  2. {affinity_output}")
print(f"  3. {pickle_output}")
print(f"\nData summary:")
print(f"  - Total SKUs: 500")
print(f"  - A-items (high freq): 100")
print(f"  - B-items (medium): 150")
print(f"  - C-items (low freq): 250")
print(f"  - Top affinity pairs: 50")
print(f"\n✓ Ready for ILP slotting (Day 8-9)")
print("=" * 60)
