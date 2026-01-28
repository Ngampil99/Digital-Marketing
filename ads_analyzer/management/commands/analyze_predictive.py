from django.core.management.base import BaseCommand
from django.db.models import Sum, Case, When, F
from django.db import models
import pandas as pd
import numpy as np
from ads_analyzer.models import AdPerformance

class Command(BaseCommand):
    help = 'Perform Predictive Modeling (Phase 4) - Pure NumPy/Pandas Implementation'

    def handle(self, *args, **options):
        self.stdout.write("=== Phase 4: Predictive Modeling & MMM (NumPy/Pandas) ===")
        
        # Prepare Data
        df = self.get_daily_data()
        
        if df.empty:
            self.stdout.write("No data available.")
            return

        # 1. Time-Series Decomposition (Manual)
        self.perform_decomposition(df)
        
        # 2. Marketing Mix Modeling (NumPy OLS)
        self.perform_mmm(df)

    def get_daily_data(self):
        daily_stats = AdPerformance.objects.values('created_date').annotate(
            traffic_spend=Sum(Case(When(campaign_objective='Traffic', then='amount_spent'), default=0, output_field=models.DecimalField())),
            sales_spend=Sum(Case(When(campaign_objective='Sales', then='amount_spent'), default=0, output_field=models.DecimalField())),
            revenue=Sum('purchase_value')
        ).order_by('created_date')
        
        df = pd.DataFrame(list(daily_stats))
        if not df.empty:
            df['created_date'] = pd.to_datetime(df['created_date'])
            df.set_index('created_date', inplace=True)
            # Ensure strictly daily frequency, fill missing with 0
            df = df.asfreq('D').fillna(0)
            
            # Convert Decimals to float for numpy
            df['traffic_spend'] = df['traffic_spend'].astype(float)
            df['sales_spend'] = df['sales_spend'].astype(float)
            df['revenue'] = df['revenue'].astype(float)
            
        return df

    def perform_decomposition(self, df):
        self.stdout.write("\n-- 5.1 Time-Series Decomposition (Moving Average) --")
        
        try:
            # Manual Decomposition
            # 1. Trend: 7-day rolling average
            df['trend'] = df['revenue'].rolling(window=7, center=True).mean()
            
            # 2. Detrend
            df['detrended'] = df['revenue'] - df['trend']
            
            # 3. Seasonal: Average of detrended for each day of week (0=Mon, 6=Sun)
            df['day_of_week'] = df.index.dayofweek
            seasonal_means = df.groupby('day_of_week')['detrended'].mean()
            df['seasonal'] = df['day_of_week'].map(seasonal_means)
            
            # 4. Residual
            df['resid'] = df['revenue'] - df['trend'] - df['seasonal']
            
            # Analysis
            valid_trend = df['trend'].dropna()
            if not valid_trend.empty:
                start_trend = valid_trend.iloc[0]
                end_trend = valid_trend.iloc[-1]
                
                self.stdout.write("Trend Analysis (7-Day Rolling Avg):")
                self.stdout.write(f"  Start: IDR {start_trend:,.0f}")
                self.stdout.write(f"  End:   IDR {end_trend:,.0f}")
                
                if end_trend > start_trend:
                    self.stdout.write("  Result: Positive Growth Trend")
                else:
                    self.stdout.write("  Result: Negative/Stable Trend")

            self.stdout.write("\nSeasonality (Weekly Pattern - Avg Deviation from Trend):")
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for i, val in seasonal_means.items():
                self.stdout.write(f"  {days[i]}: {val:+,.0f}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Decomposition failed: {e}"))

    def perform_mmm(self, df):
        self.stdout.write("\n-- 5.2 Marketing Mix Modeling (MMM using NumPy OLS) --")
        
        # Simple Linear Regression: Revenue ~ Traffic_Adstock + Sales_Spend
        # Lag 2 for Traffic
        df['traffic_shifted'] = df['traffic_spend'].shift(2).fillna(0)
        
        # Prepare X and y
        # X = [Intercept, Traffic_Shifted, Sales_Spend]
        X = df[['traffic_shifted', 'sales_spend']].copy()
        X['intercept'] = 1
        
        # Reorder so intercept is first for standard viewing (beta0)
        X = X[['intercept', 'traffic_shifted', 'sales_spend']]
        
        # Drop rows with NaN (from shifting or rolling) if any remains, strictly speaking shift fills 0 but good to be safe
        valid_data = X.join(df['revenue']).dropna()
        
        X_val = valid_data[['intercept', 'traffic_shifted', 'sales_spend']].values
        y_val = valid_data['revenue'].values
        
        try:
            # Coefficients: beta = (X.T X)^-1 X.T y  OR  np.linalg.lstsq
            beta, residuals, rank, s = np.linalg.lstsq(X_val, y_val, rcond=None)
            
            intercept = beta[0]
            coef_traffic = beta[1]
            coef_sales = beta[2]
            
            self.stdout.write("Model Equation: Revenue = Intercept + (Beta1 * Traffic_Lag2) + (Beta2 * Sales_Spend)")
            
            self.stdout.write(f"  Base Revenue (Intercept): IDR {intercept:,.0f}")
            self.stdout.write(f"  Traffic ROAS (Lag 2): {coef_traffic:.2f}")
            self.stdout.write(f"  Sales ROAS (Direct): {coef_sales:.2f}")
            
            # Simple R-squared calculation
            y_pred = X_val @ beta
            ss_tot = np.sum((y_val - np.mean(y_val))**2)
            ss_res = np.sum((y_val - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            self.stdout.write(f"  R-squared: {r_squared:.4f}")
            
            if coef_traffic > 0:
                self.stdout.write(self.style.SUCCESS("  Traffic Spend has a positive contribution to future Revenue."))
            else:
                self.stdout.write(self.style.WARNING("  Traffic Spend coefficient is non-positive."))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"MMM failed: {e}"))
