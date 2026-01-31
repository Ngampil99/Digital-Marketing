from django.core.management.base import BaseCommand
from django.db.models import Sum, Case, When, F, FloatField
from ads_analyzer.models import AdPerformance
import pandas as pd
import numpy as np

class Command(BaseCommand):
    help = 'Objective Health Check & Diagnosis Analysis'

    def handle(self, *args, **options):
        self.stdout.write("=== OBJECTIVE HEALTH CHECK: CLIENT DIAGNOSIS ===")
        
        # 1. Fetch Data grouped by Client and Objective
        # We need flexible grouping.
        # Fetch raw data to pandas for complex multi-level agg
        
        data_qs = AdPerformance.objects.values(
            'account_name', 'campaign_objective', 'created_date'
        ).annotate(
            spend=Sum('amount_spent'),
            impressions=Sum('impressions'),
            clicks=Sum('clicks'),
            link_clicks=Sum('link_clicks'),
            content_views=Sum('content_views'),
            purchases=Sum('purchases'),
            revenue=Sum('purchase_value')
        )
        
        df = pd.DataFrame(list(data_qs))
        if df.empty:
            self.stdout.write("No data found.")
            return

        # Ensure numeric
        numeric_cols = ['spend', 'impressions', 'clicks', 'link_clicks', 'content_views', 'purchases', 'revenue']
        for col in numeric_cols:
            df[col] = df[col].astype(float)

        # === TAHAP 1: ANALISIS KUALITAS TRAFFIC (Objective='Traffic') ===
        traffic_df = df[df['campaign_objective'] == 'Traffic'].groupby('account_name')[numeric_cols].sum().reset_index()
        traffic_df['cpm'] = (traffic_df['spend'] / traffic_df['impressions']) * 1000
        
        # Drop-off Rate: (Link Clicks - Content Views) / Link Clicks
        # Note: If content_views > link_clicks (anomaly), cap at 0
        traffic_df['drop_off_rate'] = ((traffic_df['link_clicks'] - traffic_df['content_views']) / traffic_df['link_clicks']) * 100
        traffic_df['drop_off_rate'] = traffic_df['drop_off_rate'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # === TAHAP 2: ANALISIS KEKUATAN SALES (Objective='Sales') ===
        sales_df = df[df['campaign_objective'] == 'Sales'].groupby('account_name')[numeric_cols].sum().reset_index()
        
        # CVR: Purchases / Content Views
        sales_df['cvr'] = (sales_df['purchases'] / sales_df['content_views']) * 100
        sales_df['cvr'] = sales_df['cvr'].fillna(0)
        
        # Sales ROAS: Revenue / Spend
        sales_df['sales_roas'] = sales_df['revenue'] / sales_df['spend']
        sales_df['sales_roas'] = sales_df['sales_roas'].fillna(0)
        
        # Merge Analysis
        analysis = pd.merge(traffic_df[['account_name', 'cpm', 'drop_off_rate']], 
                            sales_df[['account_name', 'cvr', 'sales_roas']], 
                            on='account_name', how='outer').fillna(0)

        # === TAHAP 3: DIAGNOSA STRATEGI (The Diagnosis) ===
        # Lagged Effect Logic (Simplified for print output, usually needs daily breakdown)
        # We will assume Lagged Correlation is generally strictly calculated in 'views.py'. 
        # Here we just flag Strategy based on metrics.
        
        print(f"{'CLIENT':<25} | {'CPM (T)':<10} | {'DROP-OFF':<10} | {'CVR (S)':<10} | {'ROAS (S)':<10} | {'DIAGNOSIS'}")
        print("-" * 110)
        
        for _, row in analysis.iterrows():
            client = row['account_name']
            cpm = row['cpm']
            drop_off = row['drop_off_rate']
            cvr = row['cvr']
            sales_roas = row['sales_roas']
            
            diagnosis = "Normal"
            recom = "Monitor"
            
            # Skenario 1: Website Bermasalah
            if drop_off > 50:
                diagnosis = "CRITICAL: TECH ISSUE"
                recom = "FIX LOADING SPEED (Drop-off > 50%)"
            
            # Skenario 2: Produk Tidak Menarik
            elif drop_off <= 50 and cvr < 1.0:
                diagnosis = "WEAK OFFER"
                recom = "FIX OFFER / BUNDLING (CVR < 1%)"
                
            # Skenario 3: The Winner (Scale Up)
            elif cvr > 3.0:
                diagnosis = "WINNER"
                recom = "SCALE UP SALES 30% (High CVR)"
            
            # Else: Standard Optimization
            else:
                 diagnosis = "STABLE"
                 recom = "Maintain & Optimize"

            print(f"{client:<25} | {cpm:,.0f}       | {drop_off:>6.1f}%    | {cvr:>6.2f}%    | {sales_roas:>6.2f}x    | {recom}")
