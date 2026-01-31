import csv
import os

def verify_raw_csv():
    # File path
    csv_path = 'Data Ads - Kompetisi Data Analyts by inSight Data Batch 01.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"--- Verifying Raw File: {csv_path} ---")
    
    total_rev = 0.0
    total_spend = 0.0
    row_count = 0
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f: # Use utf-8-sig to handle BOM if present
        reader = csv.DictReader(f)
        print(f"Headers found: {reader.fieldnames}")
        
        for i, row in enumerate(reader):
            row_count += 1
            try:
                rev = float(row.get('purchase_value', 0) or 0)
                spend = float(row.get('amount_spent', 0) or 0)
                
                total_rev += rev
                total_spend += spend
                
                if i < 3: # Print first 3 rows
                    print(f"Row {i+1}: Rev={rev}, Spend={spend}")
            except ValueError:
                print(f"Error parsing row {i+1}: {row}")

    print("-" * 30)
    print(f"Total Rows: {row_count}")
    print(f"Total Revenue (Raw CSV): {total_rev:,.2f}")
    print(f"Total Spend   (Raw CSV): {total_spend:,.2f}")

if __name__ == "__main__":
    verify_raw_csv()
