import pandas as pd
import os

# 1. Load Data
# Assuming the script is run from the project root
csv_path = 'Data Ads - Kompetisi Data Analyts by inSight Data Batch 01.csv'

if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    exit()

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# 2. Date Parsing
print("Parsing dates...")
df['created_date'] = pd.to_datetime(df['created_date'])

# 3. Handling Missing Values
print("Handling missing values (filling numerics with 0)...")
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# 4. Feature Engineering (Important)
print("Performing Feature Engineering...")

# CTR (Click Through Rate) = (clicks / impressions) * 100
# Handle division by zero
df['ctr'] = df.apply(lambda x: (x['clicks'] / x['impressions'] * 100) if x['impressions'] > 0 else 0, axis=1)

# CPC (Cost Per Click) = amount_spent / clicks
df['cpc'] = df.apply(lambda x: (x['amount_spent'] / x['clicks']) if x['clicks'] > 0 else 0, axis=1)

# CVR (Conversion Rate) = (purchase / link_clicks) * 100
# Note: Using link_clicks as denominator as per user request (intent clicks)
df['cvr'] = df.apply(lambda x: (x['purchase'] / x['link_clicks'] * 100) if x['link_clicks'] > 0 else 0, axis=1)

# ROAS (Return on Ad Spend) = purchase_value / amount_spent
df['roas'] = df.apply(lambda x: (x['purchase_value'] / x['amount_spent']) if x['amount_spent'] > 0 else 0, axis=1)

# AOV (Average Order Value) = purchase_value / purchase
df['aov'] = df.apply(lambda x: (x['purchase_value'] / x['purchase']) if x['purchase'] > 0 else 0, axis=1)

# Extract Month (YYYY-MM)
df['month'] = df['created_date'].dt.to_period('M')

# 5. Output
print("\n--- DataFrame Head ---")
print(df.head())

print("\n--- DataFrame Info ---")
print(df.info())

# Optional: Print basic stats for new columns
print("\n--- New Features Statistics ---")
print(df[['ctr', 'cpc', 'cvr', 'roas', 'aov']].describe())
