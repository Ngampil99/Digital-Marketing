import os
import sys
import pandas as pd
import django
from django.db import transaction

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_marketer_tool.settings')
django.setup()

from ads_analyzer.models import AdPerformance

def ingest_excel():
    excel_path = 'Data Ads.xlsx'
    
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    print(f"Reading Excel: {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    print(f"Columns found: {list(df.columns)}")
    
    # Verify expected columns
    required_cols = ['created_date', 'account_name', 'campaign_objective', 'impressions', 'reach', 'clicks']
    if not all(col in df.columns for col in required_cols):
        print("Error: Missing required columns in Excel.")
        return

    # [REVERTED] Footer Normalization removed as per user confirmation (Excel locale issue).
    # We now trust the Raw Body Data as the Source of Truth.
    
    # Clean the body (remove footer rows from calculation)
    df = df.dropna(subset=['created_date']).copy()
    
    # Ensure numeric columns are actually numeric
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce')
    df['amount_spent'] = pd.to_numeric(df['amount_spent'], errors='coerce')
    
    print("Wiping existing data...")
    AdPerformance.objects.all().delete()

    created_count = 0
    anomalies_fixed = 0
    
    records_to_create = []

    print("Processing raw rows...")
    for index, row in df.iterrows():
        try:
            # Parse metrics (Raw Values)
            impressions = int(row.get('impressions', 0))
            reach = int(row.get('reach', 0))
            clicks = int(row.get('clicks', 0))
            link_clicks = int(row.get('link_clicks', 0))
            content_views = int(row.get('content_views', 0))
            add_to_cart = int(row.get('add_to_cart', 0))
            purchases = int(row.get('purchase', 0) if 'purchase' in row else row.get('purchases', 0))
            purchase_value = float(row.get('purchase_value', 0))
            amount_spent = float(row.get('amount_spent', 0))
            
            # === ANOMALY CLEANING ===
            # 1. Reach > Impressions
            if reach > impressions:
                reach = impressions
                anomalies_fixed += 1
            
            # 2. Clicks > Impressions
            if clicks > impressions:
                clicks = impressions
                if link_clicks > clicks:
                    link_clicks = clicks
            
            # 3. Link Clicks > Clicks
            if link_clicks > clicks:
                link_clicks = clicks
                
            # 4. Ghost Revenue
            if purchase_value > 0 and purchases == 0:
                purchases = 1

            # Extract Industry
            account_name = str(row['account_name'])
            industry = None
            if ' - ' in account_name:
                parts = account_name.split(' - ')
                if len(parts) > 1:
                    industry = parts[1]

            records_to_create.append(AdPerformance(
                created_date=pd.to_datetime(row['created_date']).date(),
                account_name=account_name,
                campaign_objective=row['campaign_objective'],
                industry=industry,
                impressions=impressions,
                reach=reach,
                clicks=clicks,
                link_clicks=link_clicks,
                content_views=content_views,
                add_to_cart=add_to_cart,
                purchases=purchases,
                purchase_value=purchase_value,
                amount_spent=amount_spent
            ))
            
            created_count += 1
            
            # Batch Insert every 1000
            if len(records_to_create) >= 1000:
                AdPerformance.objects.bulk_create(records_to_create)
                records_to_create = []

        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Insert remaining
    if records_to_create:
        AdPerformance.objects.bulk_create(records_to_create)

    print(f"Success! Imported {created_count} rows. Anomalies corrected: {anomalies_fixed}")

if __name__ == '__main__':
    ingest_excel()
