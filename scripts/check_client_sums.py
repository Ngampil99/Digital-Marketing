import pandas as pd
import os

def check_clients():
    path = 'Data Ads.xlsx'
    print(f"Reading {path}...")
    df = pd.read_excel(path)
    df = df.dropna(subset=['created_date'])
    
    # Ensure numeric
    df['amount_spent'] = pd.to_numeric(df['amount_spent'], errors='coerce')
    
    print("\n--- Spend by Client (Raw Data) ---")
    grouped = df.groupby('account_name')['amount_spent'].sum().sort_values(ascending=False)
    
    total = 0
    for client, spend in grouped.items():
        print(f"{client}: {spend:,.2f}")
        total += spend
        
    print(f"\nTotal Raw Spend: {total:,.2f}")

if __name__ == '__main__':
    check_clients()
