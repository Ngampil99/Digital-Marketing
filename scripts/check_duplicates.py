import pandas as pd
import os

def check_structure():
    path = 'Data Ads.xlsx'
    print(f"Reading {path}...")
    df = pd.read_excel(path)
    
    # Remove footer
    df = df.dropna(subset=['created_date'])
    
    print(f"Total Data Rows: {len(df)}")
    
    # Check duplicates on Key columns
    # We suspect semantic duplicates (same client, same day, same objective)
    keys = ['created_date', 'account_name', 'campaign_objective']
    
    duplicates = df[df.duplicated(subset=keys, keep=False)]
    
    if not duplicates.empty:
        print(f"\n[!] POSITIVE MATCH: Found {len(duplicates)} rows with duplicate Keys!")
        print("Example Duplicates:")
        print(duplicates.sort_values(by=keys).head(10)[keys + ['amount_spent']])
        
        # Calculate impact
        dup_spend = duplicates['amount_spent'].sum()
        total_spend = df['amount_spent'].sum()
        print(f"\nImpact Analysis:")
        print(f"duplicate_rows_spend: {dup_spend:,.2f}")
        print(f"total_spend (Python): {total_spend:,.2f}")
    else:
        print("\n[?] Negative: No rows share the same (Date, Account, Objective). Each row is unique.")

if __name__ == '__main__':
    check_structure()
