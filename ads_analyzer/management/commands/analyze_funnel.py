from django.core.management.base import BaseCommand
from django.db.models import Sum, F, Avg, Case, When
import pandas as pd
import numpy as np
from django.db import models
from ads_analyzer.models import AdPerformance

class Command(BaseCommand):
    help = 'Perform Funnel Analysis and Attribution (Phase 3)'

    def handle(self, *args, **options):
        self.stdout.write("=== Phase 3: Funnel Analysis & Attribution ===")
        
        # 1. Sales Funnel Visualization
        self.analyze_funnel()
        
        # 2. Lagged Correlation
        self.analyze_lagged_correlation()

    def analyze_funnel(self):
        self.stdout.write("\n-- 4.1 Sales Funnel Visualization --")
        
        # Aggregate metrics across all Sales campaigns (since Traffic has 0 purchases)
        funnel_data = AdPerformance.objects.filter(campaign_objective='Sales').aggregate(
            impressions=Sum('impressions'),
            clicks=Sum('clicks'),
            content_views=Sum('content_views'),
            add_to_cart=Sum('add_to_cart'),
            purchases=Sum('purchases')
        )
        
        imp = funnel_data['impressions']
        clicks = funnel_data['clicks']
        cv = funnel_data['content_views']
        atc = funnel_data['add_to_cart']
        purch = funnel_data['purchases']
        
        self.stdout.write(f"Impressions: {imp:,.0f} (100%)")
        self.stdout.write(f"Clicks: {clicks:,.0f} ({clicks/imp*100:.2f}% of Imp)")
        self.stdout.write(f"Content Views: {cv:,.0f} ({cv/clicks*100:.2f}% of Clicks)")
        self.stdout.write(f"Add to Cart: {atc:,.0f} ({atc/cv*100:.2f}% of Views)")
        self.stdout.write(f"Purchases: {purch:,.0f} ({purch/atc*100:.2f}% of ATC)")
        
        if cv < clicks * 0.5:
            self.stdout.write(self.style.WARNING("  ALERT: Massive drop-off from Click to View. Check site load speed/tracking."))

    def analyze_lagged_correlation(self):
        self.stdout.write("\n-- 4.2 Lagged Correlation Analysis (Traffic Spend -> Sales Revenue) --")
        
        # Get daily data
        daily_stats = AdPerformance.objects.values('created_date').annotate(
            traffic_spend=Sum(Case(When(campaign_objective='Traffic', then='amount_spent'), default=0, output_field=models.DecimalField())),
            sales_revenue=Sum(Case(When(campaign_objective='Sales', then='purchase_value'), default=0, output_field=models.DecimalField()))
        ).order_by('created_date')
        
        df = pd.DataFrame(list(daily_stats))
        
        if df.empty:
            self.stdout.write("No data available for correlation analysis.")
            return

        # Calculate correlations for lags 0 to 7 days
        self.stdout.write("Correlation between Daily Traffic Spend and Sales Revenue (shifted):")
        
        best_lag = 0
        best_corr = -1
        
        for lag in range(8):
            # Shift Sales Revenue backwards (or Traffic Spend forwards)
            # We want to see if Traffic Spend at T correlates with Revenue at T+lag
            
            # Shift traffic spend forward to match future revenue
            # Or simpler: correlation(Traffic[t], Revenue[t+lag])
            
            shifted_revenue = df['sales_revenue'].shift(-lag)
            corr = df['traffic_spend'].corr(shifted_revenue)
            
            self.stdout.write(f"  Lag {lag} Days: {corr:.4f}")
            
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
                
        self.stdout.write(f"\nStrongest correlation found at Lag {best_lag} days ({best_corr:.4f}).")
        if best_corr > 0.5:
             self.stdout.write(self.style.SUCCESS("  Significant positive correlation detected! Traffic campaigns drive future sales."))
        elif best_corr > 0.3:
             self.stdout.write("  Moderate correlation detected.")
        else:
             self.stdout.write("  Weak or no correlation detected.")
