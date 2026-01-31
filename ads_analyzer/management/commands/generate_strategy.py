from django.core.management.base import BaseCommand
from django.db.models import Sum, F, Case, When, Value, FloatField
from django.db.models.functions import ExtractMonth, ExtractDay
from ads_analyzer.models import AdPerformance
import pandas as pd
import calendar

class Command(BaseCommand):
    help = 'Generates Data-Driven Strategic Inputs'

    def handle(self, *args, **options):
        self.stdout.write("=== PHASE 1: DATA INTERROGATION ===")
        
        # 1. Monthly CPM Analysis (Traffic Only)
        self.stdout.write("\n1. MONTHLY CPM ANALYSIS (Traffic Campaigns)")
        traffic_data = AdPerformance.objects.filter(campaign_objective='Traffic').annotate(
            month=ExtractMonth('created_date')
        ).values('month').annotate(
            total_spend=Sum('amount_spent'),
            total_impressions=Sum('impressions')
        ).order_by('month')
        
        df_cpm = pd.DataFrame(list(traffic_data))
        if not df_cpm.empty:
            df_cpm['cpm'] = (df_cpm['total_spend'] / df_cpm['total_impressions']) * 1000
            df_cpm['month_name'] = df_cpm['month'].apply(lambda x: calendar.month_name[x])
            
            lowest_cpm = df_cpm.loc[df_cpm['cpm'].idxmin()]
            highest_cpm = df_cpm.loc[df_cpm['cpm'].idxmax()]
            
            self.stdout.write(f"LOWEST CPM: {lowest_cpm['month_name']} (IDR {lowest_cpm['cpm']:,.2f})")
            self.stdout.write(f"HIGHEST CPM: {highest_cpm['month_name']} (IDR {highest_cpm['cpm']:,.2f})")
        
        # 2. Volume Revenue Analysis (All Objectives)
        self.stdout.write("\n2. VOLUME REVENUE ANALYSIS (High Season)")
        rev_data = AdPerformance.objects.annotate(
            month=ExtractMonth('created_date')
        ).values('month').annotate(
            total_revenue=Sum('purchase_value')
        ).order_by('month')
        
        df_rev = pd.DataFrame(list(rev_data))
        if not df_rev.empty:
            df_rev['month_name'] = df_rev['month'].apply(lambda x: calendar.month_name[x])
            highest_rev = df_rev.loc[df_rev['total_revenue'].idxmax()]
            
            self.stdout.write(f"HIGHEST REVENUE MONTH: {highest_rev['month_name']} (IDR {highest_rev['total_revenue']:,.0f})")

        # 3. Forensic Daily ROAS (Aggregation 1-31)
        self.stdout.write("\n3. FORENSIC DAILY ROAS (Day of Month 1-31)")
        daily_data = AdPerformance.objects.annotate(
            day=ExtractDay('created_date')
        ).values('day').annotate(
            total_spend=Sum('amount_spent'),
            total_revenue=Sum('purchase_value')
        ).order_by('total_revenue') # Order by whatever to get qs
        
        df_daily = pd.DataFrame(list(daily_data))
        if not df_daily.empty:
            df_daily['total_revenue'] = df_daily['total_revenue'].astype(float)
            df_daily['total_spend'] = df_daily['total_spend'].astype(float)
            df_daily['roas'] = df_daily['total_revenue'] / df_daily['total_spend']
            
            # Dead Zones (Lowest ROAS)
            dead_zones = df_daily.nsmallest(3, 'roas')
            self.stdout.write("DEAD ZONES (Worst ROAS):")
            for _, row in dead_zones.iterrows():
                self.stdout.write(f"  Day {int(row['day'])}: ROAS {row['roas']:.2f}")

            # Gold Zones (Highest ROAS)
            gold_zones = df_daily.nlargest(3, 'roas')
            self.stdout.write("GOLD ZONES (Best ROAS):")
            for _, row in gold_zones.iterrows():
                self.stdout.write(f"  Day {int(row['day'])}: ROAS {row['roas']:.2f}")
