import pandas as pd
import os

def debug_tail():
    path = 'Data Ads.xlsx'
    if not os.path.exists(path):
        print("File not found.")
        return
        
    print(f"Reading {path}...")
    df = pd.read_excel(path)
    
    print("\n--- Las 5 Rows ---")
    print(df.tail(5))
    
    print("\n--- Specific Columns of Last Row ---")
    last_row = df.iloc[-1]
    print(last_row)

if __name__ == '__main__':
    debug_tail()
