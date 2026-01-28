from django.core.management.base import BaseCommand
from django.db.models import Sum, F, Avg, Case, When, DecimalField
from django.db.models.functions import TruncMonth, ExtractDay
from ads_analyzer.models import AdPerformance
import pandas as pd

class Command(BaseCommand):
    help = 'Perform Exploratory Data Analysis (EDA)'

    def handle(self, *args, **options):
        self.stdout.write("=== Phase 2: Exploratory Data Analysis ===")
        
        # 1. Traffic vs Sales Analysis
        self.analyze_objectives()
        
        # 2. Seasonality Analysis
        self.analyze_seasonality()
        
        # 3. Industry Benchmarking (Simple)
        self.benchmark_industry()

    def analyze_objectives(self):
        self.stdout.write("\n-- 3.1 Objective Analysis (Traffic vs Sales) --")
        stats = AdPerformance.objects.values('campaign_objective').annotate(
            total_spend=Sum('amount_spent'),
            total_revenue=Sum('purchase_value'),
            total_impressions=Sum('impressions'),
            total_clicks=Sum('clicks'),
            avg_ctr=Avg(F('clicks') * 100.0 / F('impressions')),
        )
        
        for s in stats:
            formatted_spend = f"{s['total_spend']:,.0f}"
            formatted_rev = f"{s['total_revenue']:,.0f}"
            roas = s['total_revenue'] / s['total_spend'] if s['total_spend'] > 0 else 0
            
            self.stdout.write(f"Objective: {s['campaign_objective']}")
            self.stdout.write(f"  Spend: IDR {formatted_spend}")
            self.stdout.write(f"  Revenue: IDR {formatted_rev}")
            self.stdout.write(f"  ROAS: {roas:.2f}x")
            self.stdout.write(f"  Avg CTR: {s['avg_ctr']:.2f}%")

    def analyze_seasonality(self):
        self.stdout.write("\n-- 3.2 Seasonality Analysis --")
        
        # Check Payday (25th of month)
        payday_stats = AdPerformance.objects.annotate(
            day=ExtractDay('created_date')
        ).filter(day__in=[25, 26, 27, 28]).values('day').annotate(
            avg_revenue=Avg('purchase_value'),
            avg_roas=Avg(F('purchase_value') / F('amount_spent'))
        ).order_by('day')
        
        self.stdout.write("Payday Performance (Avg Revenue & ROAS):")
        for p in payday_stats:
             self.stdout.write(f"  Day {p['day']}: IDR {p['avg_revenue']:,.0f} | ROAS: {p['avg_roas']:.2f}")

        # Check Ramadan Pre-hype (Late March)
        # Assuming data ends March 30, check trends in March
        march_stats = AdPerformance.objects.filter(
            created_date__month=3
        ).values('created_date').annotate(
            daily_revenue=Sum('purchase_value')
        ).order_by('created_date')
        
        # Just creating a simple textual summary here for the console
        self.stdout.write("March Daily Revenue Trend (Last 5 days of data):")
        for d in list(march_stats)[-5:]:
            self.stdout.write(f"  {d['created_date']}: IDR {d['daily_revenue']:,.0f}")

    def benchmark_industry(self):
        self.stdout.write("\n-- 3.3 Industry Benchmarking --")
        # Breakdown by industry
        industries = AdPerformance.objects.values('industry').annotate(
            avg_ctr=Avg(F('clicks') * 100.0 / F('impressions')),
            avg_cvr=Avg(F('purchases') * 100.0 / F('clicks')), # Simple CVR approximation
            avg_roas=Avg(F('purchase_value') / F('amount_spent'))
        )
        
        for ind in industries:
             industry_name = ind['industry'] if ind['industry'] else "Unknown"
             self.stdout.write(f"Industry: {industry_name}")
             self.stdout.write(f"  CTR: {ind['avg_ctr']:.2f}%")
             self.stdout.write(f"  CVR (Purch/Clicks): {ind['avg_cvr']:.2f}%")
             self.stdout.write(f"  ROAS: {ind['avg_roas']:.2f}")
