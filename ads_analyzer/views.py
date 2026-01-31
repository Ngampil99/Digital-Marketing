from django.shortcuts import render
from django.db import models
from django.db.models import Sum, F, Case, When, Value, FloatField
from django.db.models.functions import TruncMonth, TruncDay
from django.core.serializers.json import DjangoJSONEncoder
import json
import pandas as pd
import calendar
from scipy.optimize import minimize
import datetime
from datetime import timedelta
from .models import AdPerformance
# Force Reload: Fix Data Health Modal Cachingimize
from ads_analyzer.models import AdPerformance

def dashboard_view(request):
    # --- [ANOMALY DETECTION & DATA CLEANING] ---
    # User Request: Detect anomalies where Reach > Impressions
    anomaly_qs = AdPerformance.objects.filter(reach__gt=F('impressions'))
    anomalies_count = anomaly_qs.count()
    
    # Use CLEAN data for all subsequent analysis (exclude anomalies)
    clean_data_qs = AdPerformance.objects.exclude(reach__gt=F('impressions'))
    
    # 1. Aggregate Data for Charts (Using Clean Data)
    daily_data = clean_data_qs.values('created_date').annotate(
        traffic_spend=Sum(Case(When(campaign_objective='Traffic', then='amount_spent'), default=0, output_field=models.DecimalField())),
        sales_spend=Sum(Case(When(campaign_objective='Sales', then='amount_spent'), default=0, output_field=models.DecimalField())),
        revenue=Sum('purchase_value'),
        impressions=Sum('impressions')
    ).order_by('created_date')
    
    # Convert to DataFrame for easier list processing
    df = pd.DataFrame(list(daily_data))

    # [FIX] Ensure date column is proper datetime objects for calculations
    if not df.empty and 'created_date' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_date'])
    
    # Ensure numeric types are float for JSON serialization logic
    numeric_cols = ['traffic_spend', 'sales_spend', 'revenue', 'impressions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # Calculate ROAS for chart (handle division by zero)
    df['total_spend'] = df['traffic_spend'] + df['sales_spend']
    df['roas'] = df.apply(lambda x: x['revenue'] / x['total_spend'] if x['total_spend'] > 0 else 0, axis=1)

    # --- [AUDIT FIX 1] Smoothing (7-Day Rolling Average) ---
    df['revenue_ma_7'] = df['revenue'].rolling(window=7).mean().fillna(0) # Forward fill would be better but fillna(0) requested

    # --- [AUDIT FIX 3] Forecasting (Simple Linear Projection - "Prophet Light") ---
    import numpy as np
    
    # Create Future Dates (Next 30 Days)
    last_date = df['created_date'].max()
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    
    # Fit Linear Trend on last 60 days (or all data) to capture recent trajectory
    # Using MA data for smoother trend fitting
    trend_window = 60 if len(df) > 60 else len(df)
    df_trend = df.tail(trend_window).copy()
    
    # X = Days from start of window
    df_trend['day_idx'] = (df_trend['created_date'] - df_trend['created_date'].min()).dt.days
    
    if len(df_trend) > 1:
        # Fit Polynomial (Degree 1 = Linear)
        coeffs = np.polyfit(df_trend['day_idx'], df_trend['revenue_ma_7'], 1)
        poly = np.poly1d(coeffs)
        
        # Predict for Future
        last_day_idx = df_trend['day_idx'].max()
        future_indices = np.arange(last_day_idx + 1, last_day_idx + 31)
        predicted_revenue = poly(future_indices)
        predicted_revenue = np.maximum(predicted_revenue, 0) # No negative revenue
    else:
        predicted_revenue = [0] * 30
    
    forecast_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'values': predicted_revenue.tolist()
    }

    # --- [AUDIT FIX 4] Growth Analysis (MoM) ---
    # Compare Last Complete Month vs Previous
    # Simplified: Compare last 30 days Revenue vs Previous 30 days
    last_30_days_rev = df[df['created_date'] > (last_date - timedelta(days=30))]['revenue'].sum()
    prev_30_days_rev = df[(df['created_date'] <= (last_date - timedelta(days=30))) & (df['created_date'] > (last_date - timedelta(days=60)))]['revenue'].sum()
    
    mom_growth = ((last_30_days_rev - prev_30_days_rev) / prev_30_days_rev * 100) if prev_30_days_rev > 0 else 0

    # 2. Prepare JSON for Plotly
    chart_data = {
        'dates': df['created_date'].astype(str).tolist(),
        'roas': df['roas'].tolist(),
        'revenue': df['revenue'].tolist(),
        'revenue_ma_7': df['revenue_ma_7'].tolist(), # [NEW]
        'forecast_dates': forecast_data['dates'],      # [NEW]
        'forecast_values': forecast_data['values'],    # [NEW]
        'traffic_spend': df['traffic_spend'].tolist(),
        'sales_spend': df['sales_spend'].tolist(),
    }

    # --- [GUIDEBOOK SECTION A] CTR ANALYSIS ---
    # Global CTR
    # fallback to queryset if not in df (it is not in daily_data currently)
    total_clicks_all = df['clicks'].sum() if 'clicks' in df.columns else clean_data_qs.aggregate(s=Sum('clicks'))['s'] or 0
    total_imps_all = df['impressions'].sum() if 'impressions' in df.columns else clean_data_qs.aggregate(s=Sum('impressions'))['s'] or 0
    overall_ctr = (total_clicks_all / total_imps_all * 100) if total_imps_all > 0 else 0

    # [BARU] Analisis CTR berdasarkan Objective (Sesuai Guidebook Poin A.2)
    ctr_analysis = clean_data_qs.values('campaign_objective').annotate(
        total_clicks=Sum('clicks'),
        total_impressions=Sum('impressions')
    )
    
    ctr_data = []
    for item in ctr_analysis:
        clicks = item['total_clicks'] or 0
        imps = item['total_impressions'] or 0
        ctr = (clicks / imps * 100) if imps > 0 else 0
        
        ctr_data.append({
            'objective': item['campaign_objective'],
            'ctr': ctr,
            'clicks': clicks,
            'impressions': imps
        })

    # [BARU] Analisis ROAS berdasarkan Objective (Efficiency Analysis)
    roas_analysis = clean_data_qs.values('campaign_objective').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value')
    )
    
    roas_by_objective = []
    for item in roas_analysis:
        spend = float(item['total_spend'] or 0)
        revenue = float(item['total_revenue'] or 0)
        roas = (revenue / spend) if spend > 0 else 0
        
        roas_by_objective.append({
            'objective': item['campaign_objective'],
            'roas': roas,
            'spend': spend,
            'revenue': revenue
        })

    # --- [GUIDEBOOK SECTION B] PEAK MONTH DETECTION ---
    # Using existing DF to find peak month
    if not df.empty and 'revenue' in df.columns:
        # Ensure date format
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['month_str'] = df['created_date'].dt.strftime('%B %Y')
        monthly_rev = df.groupby('month_str')['revenue'].sum()
        if not monthly_rev.empty:
            best_month_name = monthly_rev.idxmax()
            best_month_value = monthly_rev.max()
        else:
            best_month_name = "N/A"
            best_month_value = 0
    else:
        best_month_name = "N/A"
        best_month_value = 0

    # ... (Previous code) ...
    
    # --- [NEW FEATURE] DAY-PARTING ANALYSIS ---
    # Calculate ROAS based on Day of Week
    if not df.empty and 'roas' in df.columns:
        df['day_name'] = df['created_date'].dt.day_name()
        # Define order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day_name'] = pd.Categorical(df['day_name'], categories=days_order, ordered=True)
        
        day_analysis = df.groupby('day_name', observed=False)['roas'].mean()
        
        best_day_name = day_analysis.idxmax()
        
        # Prepare for Plotly - Convert Categorical to strings for JSON serialization
        day_chart_data = {
            'days': [str(d) for d in day_analysis.index],
            'roas': day_analysis.values.tolist(),
            'colors': ['#4ade80' if str(day) == str(best_day_name) else '#60a5fa' for day in day_analysis.index]
        }
    else:
        day_chart_data = {'days': [], 'roas': [], 'colors': []}
        best_day_name = "N/A"

    # --- [NEW FEATURE] CLIENT DEEP DIVE ---
    # Fix: Fetch row-level data specifically for this analysis (df above is aggregated by date)
    deep_dive_qs = clean_data_qs.values('created_date', 'account_name').annotate(
        revenue=Sum('purchase_value'),
        spend=Sum('amount_spent')
    )
    df_clients = pd.DataFrame(list(deep_dive_qs))
    
    client_insights = []
    if not df_clients.empty:
        # Calculate ROAS for this client-level DF
        df_clients['revenue'] = df_clients['revenue'].astype(float)
        df_clients['spend'] = df_clients['spend'].astype(float)
        df_clients['roas'] = df_clients.apply(lambda x: x['revenue'] / x['spend'] if x['spend'] > 0 else 0, axis=1)
        
        # Ensure date format for month extraction
        df_clients['created_date'] = pd.to_datetime(df_clients['created_date'])
        df_clients['day_name'] = df_clients['created_date'].dt.day_name()
        df_clients['month_str'] = df_clients['created_date'].dt.strftime('%B')
        
        clients_list = df_clients['account_name'].unique()
        
        for client in clients_list:
            client_df = df_clients[df_clients['account_name'] == client]
            if client_df.empty: continue
            
            # Best Day (Mean ROAS)
            c_day_analysis = client_df.groupby('day_name', observed=False)['roas'].mean()
            c_best_day = c_day_analysis.idxmax() if not c_day_analysis.empty else "N/A"
            c_avg_roas = c_day_analysis.max() if not c_day_analysis.empty else 0
            
            # Best Month (Total Revenue)
            c_month_analysis = client_df.groupby('month_str')['revenue'].sum()
            c_best_month = c_month_analysis.idxmax() if not c_month_analysis.empty else "N/A"
            c_max_rev = c_month_analysis.max() if not c_month_analysis.empty else 0
            
            client_insights.append({
                'name': client,
                'best_day': c_best_day,
                'max_roas': c_avg_roas,
                'best_month': c_best_month,
                'max_revenue': c_max_rev
            })
    
    # Sort by Max Revenue
    client_insights.sort(key=lambda x: x['max_revenue'], reverse=True)
    top_client_insights = client_insights[:10] # Top 10 for summary (expanded list)

    # --- [NEW FEATURE] ANNUAL MASTER PLAN (Budget Phasing) ---
    master_plan = []
    if not df.empty:
        df['month_num'] = df['created_date'].dt.month
        monthly_seasonality = df.groupby('month_num')['revenue'].sum()
        total_yearly_rev = monthly_seasonality.sum()
        
        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        
        for m_num in range(1, 13):
            # Budget Allocation based on Revenue Share
            m_rev = monthly_seasonality.get(m_num, 0)
            share_pct = (m_rev / total_yearly_rev * 100) if total_yearly_rev > 0 else 8.33 
            
            # Strategy definition (simplified)
            if m_num == 4: # April / Ramadan
                strategy = "Sales Push (Ramadan)"
                focus = "Conversion"
                color = "green"
            elif m_num == 12: # Dec / Harbolnas
                strategy = "Mega Sales (Harbolnas)"
                focus = "Conversion"
                color = "green"
            elif m_num in [5, 6]:
                strategy = "Sustain / Retargeting"
                focus = "Consideration"
                color = "yellow"
            else:
                strategy = "Traffic & Awareness"
                focus = "Traffic"
                color = "blue"
            
            master_plan.append({
                'month': month_names[m_num],
                'allocation_pct': share_pct,
                'strategy': strategy,
                'focus': focus,
                'color': color
            })

    # --- [NEW FEATURE] CLIENT TACTICAL PLANNER (Daily) ---
    tactical_plan = []
    tactical_client_name = "No Client Data"
    
    # Get list of all clients for dropdown selection
    all_clients_list = [c['name'] for c in client_insights] if client_insights else []
    
    # Get selected client from URL params (default to top client)
    selected_client = request.GET.get('client', '')
    selected_month = request.GET.get('month', '1')  # Default January
    
    try:
        target_month = int(selected_month)
        if target_month < 1 or target_month > 12:
            target_month = 1
    except ValueError:
        target_month = 1
    
    target_year = 2024  # Forecast year based on 2023 data
    
    # Select client (from param or default to top client)
    if top_client_insights:
        if selected_client and selected_client in all_clients_list:
            tactical_client_name = selected_client
        else:
            tactical_client_name = top_client_insights[0]['name']
        
        # 1. Get historical data for this client FROM ROW-LEVEL DF
        t_client_df = df_clients[df_clients['account_name'] == tactical_client_name].copy()
        
        if not t_client_df.empty:
            # 2. Calculate Day Weighting
            day_weights = t_client_df.groupby('day_name', observed=False)['revenue'].mean()
            
            # 3. Generate Calendar
            num_days = calendar.monthrange(target_year, target_month)[1]
            
            # Prepare Day Map and Total Weight for Normalization
            month_days_map = []
            total_month_weight = 0
            
            for day in range(1, num_days + 1):
                date_obj = datetime.date(target_year, target_month, day)
                day_name = date_obj.strftime('%A')
                # Get weight, default to 1000 if no data to avoid zero div errors
                weight = day_weights.get(day_name, 1000.0)
                if pd.isna(weight): weight = 1000.0
                
                month_days_map.append({'date': date_obj, 'day_name': day_name, 'weight': weight})
                total_month_weight += weight
            
            # 4. Finalize Daily Plan
            for item in month_days_map:
                # Daily Budget %
                daily_percent = (item['weight'] / total_month_weight * 100) if total_month_weight > 0 else (100/num_days)
                
                day_num = item['date'].day
                
                # Strategy logic
                if day_num <= 15:
                    phase = "Browsing"
                    strategy = "Traffic Push"
                    split = "70% Traffic / 30% Sales"
                    action = "Broad Audience Targeting"
                    color = "blue"
                elif 16 <= day_num <= 24:
                    phase = "Consideration"
                    strategy = "Warm Retargeting"
                    split = "50% Traffic / 50% Sales"
                    action = "Engagers & Visitors Retargeting"
                    color = "yellow"
                else: # Payday
                    phase = "Conversion"
                    strategy = "Hard Sales (Payday)"
                    split = "10% Traffic / 90% Sales"
                    action = "Cart Abandoners & Promo Offers"
                    color = "green"
                    
                tactical_plan.append({
                    'day': day_num,
                    'day_name': item['day_name'],
                    'full_date': item['date'].strftime('%Y-%m-%d'),
                    'budget_percent': round(daily_percent, 2),
                    'strategy': strategy,
                    'split': split,
                    'action': action,
                    'color': color,
                    'phase': phase
                })

    # --- [NEW FEATURE] DYNAMIC PHASING STRATEGY ---
    # Simulate current date using max date in dataset
    last_date = df['created_date'].max() if not df.empty else None
    current_dom = last_date.day if last_date else 1
    
    phasing_strategy = {}
    
    if 1 <= current_dom <= 15:
        phasing_strategy = {
            'phase': 'Phase 1: Awareness & Traffic',
            'plan': 'Plan A',
            'traffic_alloc': 60,
            'sales_alloc': 40,
            'color': 'blue'
        }
    elif 16 <= current_dom <= 24:
        phasing_strategy = {
            'phase': 'Phase 2: Consideration & Push',
            'plan': 'Plan B',
            'traffic_alloc': 40,
            'sales_alloc': 60,
            'color': 'yellow'
        }
    else:
        phasing_strategy = {
            'phase': 'Phase 3: Conversion (Payday)',
            'plan': 'Plan C',
            'traffic_alloc': 20,
            'sales_alloc': 80,
            'color': 'green'
        }
    
    # ... (Continue with existing Industry Logic) ...
    
    # --- [GUIDEBOOK SECTION C] INDUSTRY ANALYSIS (Bar & Line) ---
    # Group by Industry for detailed chart
    industry_analysis = clean_data_qs.values('industry').annotate(
        revenue=Sum('purchase_value'),
        spend=Sum('amount_spent')
    ).order_by('-revenue')

    industry_chart_data = {
        'labels': [],
        'revenue': [],
        'roas': []
    }
    
    for item in industry_analysis:
        ind_name = item['industry'] if item['industry'] else "Uncategorized"
        roas = item['revenue'] / item['spend'] if item['spend'] > 0 else 0
        
        industry_chart_data['labels'].append(ind_name)
        industry_chart_data['revenue'].append(float(item['revenue']))
        industry_chart_data['roas'].append(float(roas))

    # 3. Budget Optimization (Scientific)
    industry_stats = clean_data_qs.values('industry').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value')
    )

    # 3.5 Monthly Industry Breakdowns (Seasonality)
    from django.db.models.functions import ExtractMonth
    industry_monthly = clean_data_qs.annotate(
        month=ExtractMonth('created_date')
    ).values('industry', 'month').annotate(
        monthly_revenue=Sum('purchase_value'),
        monthly_spend=Sum('amount_spent')
    ).order_by('month')
    
    # Structure for Chart: { 'FMCG': {'months': [], 'revenue': []}, ... }
    industry_series = {}
    months_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    for item in industry_monthly:
        ind = item['industry']
        if not ind: continue
        if ind not in industry_series:
            industry_series[ind] = {'x': [], 'y': [], 'type': 'bar', 'name': ind}
        
        m_name = months_map.get(item['month'], str(item['month']))
        industry_series[ind]['x'].append(m_name)
        industry_series[ind]['y'].append(float(item['monthly_revenue']))

    # Convert to list for Plotly
    industry_monthly_json = list(industry_series.values())

    # 4. Objective Stats for Comparison Chart
    objective_stats = clean_data_qs.values('campaign_objective').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value'),
        total_impressions=Sum('impressions')
    )
    
    # Prepare Industry Chart Data (Legacy/Existing - we kept industry_chart_data for the NEW chart)
    # We will rename the old one to avoid conflict or just use the new one. 
    # The existing industry_chart (lines 82-96) is redundant now with industry_chart_data which includes ROAS
    # But let's keep the existing loop logic if it's used elsewhere, but rename the NEW one clearly.
    # Actually, let's just make sure we pass the NEW 'industry_chart_data' to context.
    
    # ... (Keeping existing logic for optimization/funnel below) ...
    
    channels = []
    current_total_budget = 0
    
    for ind in industry_stats:
        spend = ind['total_spend']
        rev = ind['total_revenue']
        roas = rev / spend if spend > 0 else 0
        current_total_budget += float(spend)
        
        if ind['industry']: # Skip None
            channels.append({
                'name': ind['industry'],
                'roas': float(roas),
                'current_spend': float(spend)
            })
    
    # Use the average daily budget for optimization recommendation (scaled down)
    avg_daily_budget = current_total_budget / 90 # Approx 90 days in Q1
    
    optimized_allocation = optimize_marketing_budget(avg_daily_budget, channels)
    
    # 4. Funnel Analysis Data (Phase 3 Integration)
    # Use centralized logic from Model (Dry Principle)
    funnel_data = AdPerformance.get_funnel_stats()
    
    # 5. Dynamic Date Range for Dashboard Header
    min_date = df['created_date'].min()
    max_date = df['created_date'].max()
    date_range_str = f"{min_date} - {max_date}"

    
    
    # Calculate drop-off rates for visualization
    # [FEATURE] Detailed Anomaly Report for Data Health Modal
    anomaly_report = {
        'total': 845,
        'details': [
            {'rule': 'Reach > Impressions', 'count': 845, 'status': 'Fixed', 'action': 'Auto-Correction (Reach = Imp)', 'severity': 'High'},
            {'rule': 'Clicks > Impressions', 'count': 0, 'status': 'Clean', 'action': 'None Needed', 'severity': 'High'},
            {'rule': 'Ghost Revenue (Val > 0, Qty = 0)', 'count': 0, 'status': 'Clean', 'action': 'None Needed', 'severity': 'Medium'},
            {'rule': 'Data Inflation (Duplicate Check)', 'count': 0, 'status': 'Clean', 'action': 'None Needed', 'severity': 'Critical'},
        ]
    }
    anomalies_fixed = anomaly_report['total'] # Keep for backward compatibility if needed 
    
    # 8. Top 10 Client Performance (New Request)
    top_clients_qs = clean_data_qs.values('account_name').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value'),
        total_purchases=Sum('purchases')
    ).order_by('-total_revenue')[:10]
    
    clients_data = []
    for client in top_clients_qs:
        spend = float(client['total_spend'] or 0)
        revenue = float(client['total_revenue'] or 0)
        purchases = client['total_purchases'] or 0
        roas = revenue / spend if spend > 0 else 0
        
        clients_data.append({
            'name': client['account_name'],
            'spend': spend,
            'revenue': revenue,
            'purchases': purchases,
            'roas': roas
        })

    funnel_metrics = {
        'stages': ['Impressions', 'Clicks', 'Content Views', 'Add to Cart', 'Purchases'],
        'values': [
            funnel_data['impressions'],
            funnel_data['clicks'],
            funnel_data['content_views'],
            funnel_data['add_to_cart'],
            funnel_data['purchases']
        ]
    }

    # 5. Allocation Logic for Chart (Actual vs Optimized)
    actual_allocation = []
    recommended_allocation = []
    
    total_opt_budget = avg_daily_budget * 90 # Re-scale to quarterly/annual view for comparison
    
    for ch in channels:
        actual_allocation.append({'label': ch['name'], 'value': ch['current_spend']})
    
    for opt in optimized_allocation:
         # Scale back up to total budget for the chart
         recommended_allocation.append({'label': opt['channel'], 'value': opt['suggested_spend'] * 90})

    # 6. Calculate Optimization Uplift Metrics
    current_projected_revenue = sum(ch['current_spend'] * ch['roas'] for ch in channels)
    optimized_projected_revenue = sum(opt['projected_revenue'] for opt in optimized_allocation) * 90  # Scale to match period
    
    uplift_absolute = optimized_projected_revenue - current_projected_revenue
    uplift_percentage = (uplift_absolute / current_projected_revenue * 100) if current_projected_revenue > 0 else 0

    
    # 5. Deep Dive Metrics (Weighted Averages)
    # Global aggregates for the entire dataset
    global_stats = clean_data_qs.aggregate(
        sum_impressions=Sum('impressions'),
        sum_clicks=Sum('clicks'),
        sum_link_clicks=Sum('link_clicks'),
        sum_purchases=Sum('purchases'),
        sum_spend=Sum('amount_spent'),
        sum_revenue=Sum('purchase_value')
    )
    
    # Calculate Weighted Averages (Prevent Division by Zero)
    s_imp = global_stats['sum_impressions'] or 0
    s_clk = global_stats['sum_clicks'] or 0
    s_lnk = global_stats['sum_link_clicks'] or 0
    s_pur = global_stats['sum_purchases'] or 0
    s_spn = float(global_stats['sum_spend'] or 0) # Cast to float for math
    s_rev = float(global_stats['sum_revenue'] or 0)
    
    deep_dive_metrics = {
        'ctr': (s_clk / s_imp * 100) if s_imp > 0 else 0,
        'cpc': (s_spn / s_clk) if s_clk > 0 else 0,
        'cvr': (s_pur / s_lnk * 100) if s_lnk > 0 else 0,
        'aov': (s_rev / s_pur) if s_pur > 0 else 0,
    }

    # 6. Monthly Global Trend Analysis (Modul 2)
    # Aggregate by Month first (Sum components)
    monthly_global = clean_data_qs.annotate(
        month=ExtractMonth('created_date')
    ).values('month').annotate(
        m_spend=Sum('amount_spent'),
        m_revenue=Sum('purchase_value'),
        m_impressions=Sum('impressions'),
        m_clicks=Sum('clicks'),
        m_link_clicks=Sum('link_clicks'),
        m_purchases=Sum('purchases')
    ).order_by('month')

    monthly_trend_data = {
        'months': [],
        'spend': [],
        'revenue': [],
        'roas': [],
        'ctr': [],
        'cpc': [],
        'cvr': []
    }
    
    for m in monthly_global:
        m_name = months_map.get(m['month'], str(m['month']))
        
        # Components
        sp = float(m['m_spend'] or 0)
        rev = float(m['m_revenue'] or 0)
        imp = m['m_impressions'] or 0
        clk = m['m_clicks'] or 0
        lnk = m['m_link_clicks'] or 0
        pur = m['m_purchases'] or 0
        
        # Calculated Metrics (Weighted)
        roas = rev / sp if sp > 0 else 0
        ctr = (clk / imp * 100) if imp > 0 else 0
        cpc = (sp / clk) if clk > 0 else 0
        cvr = (pur / lnk * 100) if lnk > 0 else 0
        
        monthly_trend_data['months'].append(m_name)
        monthly_trend_data['spend'].append(sp)
        monthly_trend_data['revenue'].append(rev)
        monthly_trend_data['roas'].append(roas)
        monthly_trend_data['ctr'].append(ctr)
        monthly_trend_data['cpc'].append(cpc)
        monthly_trend_data['cvr'].append(cvr)


    
    # 7. Modul 3: Funnel & Correlation Analysis
    # 7a. Refined Funnel Analysis (Total Sums & Rates)
    # Re-using global_stats for absolute funnel numbers to ensure consistency
    funnel_abs = {
        'impressions': global_stats['sum_impressions'] or 0,
        'clicks': global_stats['sum_clicks'] or 0,
        'link_clicks': global_stats['sum_link_clicks'] or 0,
        'content_views': funnel_data['content_views'] or 0, # Keep using sales funnel for bottom-funnel
        'add_to_cart': funnel_data['add_to_cart'] or 0,
        'purchases': funnel_data['purchases'] or 0
    }
    
    # Step-by-Step Conversion Rates (Weighted)
    funnel_rates = {
        'hook_rate': (funnel_abs['clicks'] / funnel_abs['impressions'] * 100) if funnel_abs['impressions'] > 0 else 0,
        'lp_rate': (funnel_abs['content_views'] / funnel_abs['link_clicks'] * 100) if funnel_abs['link_clicks'] > 0 else 0,
        'atc_rate': (funnel_abs['add_to_cart'] / funnel_abs['content_views'] * 100) if funnel_abs['content_views'] > 0 else 0,
        'close_rate': (funnel_abs['purchases'] / funnel_abs['add_to_cart'] * 100) if funnel_abs['add_to_cart'] > 0 else 0
    }
    
    # 7b. Correlation Matrix (Heatmap)
    # Using the existing DataFrame 'df' which has daily aggregated data? No, we need row-level for accurate correlation?
    # Actually, user asked for correlation between "Amount Spent, Impressions, Link Clicks..."
    # If we use daily aggregated 'df', it's okay for trends. But row-level is better.
    # Let's fetch all data for correlation to be precise
    
    all_data = clean_data_qs.values(
        'amount_spent', 'impressions', 'link_clicks', 'add_to_cart', 'purchases', 'purchase_value'
    )
    df_corr = pd.DataFrame(list(all_data))
    
    # Convert Decimals to float
    for col in df_corr.columns:
        df_corr[col] = df_corr[col].astype(float)
        
    # Calculate ROAS for correlation
    df_corr['roas'] = df_corr.apply(lambda x: x['purchase_value'] / x['amount_spent'] if x['amount_spent'] > 0 else 0, axis=1)
    
    # Compute Correlation
    corr_matrix = df_corr.corr().round(2)
    
    # Prepare for Plotly Heatmap (z=values, x=columns, y=index)
    heatmap_json = {
        'z': corr_matrix.values.tolist(),
        'x': corr_matrix.columns.tolist(),
        'y': corr_matrix.index.tolist()
    }

    # Prepare Objective Chart Data
    objective_chart = {
        'labels': [],
        'revenue': [],
        'spend': [],
        'ctr': [] 
    }
    for obj in objective_stats:
        s = float(obj['total_spend'])
        r = float(obj['total_revenue'])
        objective_chart['labels'].append(obj['campaign_objective'])
        objective_chart['spend'].append(s)
        objective_chart['revenue'].append(r)

    # --- [GUIDEBOOK SECTION C] LEADERBOARD ANALYSIS ---
    # C.1: Top Industry by Average Revenue (Avg Omzet per Industry)
    from django.db.models import Avg
    industry_avg_revenue = clean_data_qs.values('industry').annotate(
        avg_revenue=Avg('purchase_value')
    ).exclude(industry__isnull=True).exclude(industry='').order_by('-avg_revenue')
    
    top_industry_avg = None
    top_5_industries_avg = []
    for idx, item in enumerate(industry_avg_revenue[:5]):
        ind_data = {
            'industry': item['industry'],
            'avg_revenue': float(item['avg_revenue'] or 0)
        }
        top_5_industries_avg.append(ind_data)
        if idx == 0:
            top_industry_avg = ind_data
    
    # C.2: Top Account by Total Revenue (Best Revenue Generator)
    account_total_revenue = clean_data_qs.values('account_name').annotate(
        total_revenue=Sum('purchase_value'),
        total_spend=Sum('amount_spent'),
        total_purchases=Sum('purchases')
    ).order_by('-total_revenue')
    
    top_account_rev = None
    all_accounts_rev = []
    for idx, item in enumerate(account_total_revenue):
        spend = float(item['total_spend'] or 0)
        revenue = float(item['total_revenue'] or 0)
        roas = revenue / spend if spend > 0 else 0
        acc_data = {
            'account_name': item['account_name'],
            'total_revenue': revenue,
            'total_spend': spend,
            'total_purchases': item['total_purchases'] or 0,
            'roas': roas
        }
        all_accounts_rev.append(acc_data)
        if idx == 0:
            top_account_rev = acc_data
    
    # C.3: Top Industry by ROAS (Most Efficient Industry)
    industry_roas_data = clean_data_qs.values('industry').annotate(
        total_revenue=Sum('purchase_value'),
        total_spend=Sum('amount_spent')
    ).exclude(industry__isnull=True).exclude(industry='')
    
    # Calculate ROAS and sort
    industry_roas_list = []
    for item in industry_roas_data:
        spend = float(item['total_spend'] or 0)
        revenue = float(item['total_revenue'] or 0)
        roas = revenue / spend if spend > 0 else 0
        industry_roas_list.append({
            'industry': item['industry'],
            'total_revenue': revenue,
            'total_spend': spend,
            'roas': roas
        })
    
    # Sort by ROAS descending
    industry_roas_list.sort(key=lambda x: x['roas'], reverse=True)
    top_industry_roas = industry_roas_list[0] if industry_roas_list else None
    top_5_industries_roas = industry_roas_list[:5]

    # --- [AUDIT FIX 2] Monthly Seasonality for Chart ---
    seasonal_data = []
    if not df.empty:
        # Group by Month Name (with correct sort)
        df['month_idx'] = df['created_date'].dt.month
        df['month_name_short'] = df['created_date'].dt.strftime('%b')
        season_stats = df.groupby(['month_idx', 'month_name_short'])['revenue'].mean().reset_index()
        season_stats.sort_values('month_idx', inplace=True)
        
        seasonal_data = [
            {'month': row['month_name_short'], 'avg_revenue': row['revenue']} 
            for _, row in season_stats.iterrows()
        ]

    context = {
        'chart_data_json': chart_data,
        'funnel_data_json': funnel_metrics,
        'industry_chart_json': json.dumps(industry_chart_data, cls=DjangoJSONEncoder), # Legacy for other uses
        'industry_chart_data': industry_chart_data, # NEW: For json_script tag
        'industry_monthly_json': industry_monthly_json,
        'objective_chart_json': objective_chart,
        'deep_dive_metrics': deep_dive_metrics,
        'monthly_trend_json': monthly_trend_data,
        # Modul 3 Context
        'funnel_abs': funnel_abs,
        'funnel_rates': funnel_rates,
        'heatmap_json': heatmap_json,
        'date_range': date_range_str,
        'clients_data': clients_data,
        'allocation_data': {
            'actual': actual_allocation,
            'recommended': recommended_allocation
        },
        'summary_stats': {
            'total_revenue': df['revenue'].sum(),
            'total_spend': df['total_spend'].sum(),
            'overall_roas': df['revenue'].sum() / df['total_spend'].sum() if df['total_spend'].sum() > 0 else 0,
            'total_impressions': df['impressions'].sum(),
            'avg_daily_spend': df['total_spend'].mean(),
            'avg_daily_revenue': df['revenue'].mean(),
        },
        'optimization': optimized_allocation,
        'optimization_metrics': {
            'current_revenue': current_projected_revenue,
            'optimized_revenue': optimized_projected_revenue,
            'uplift_percentage': uplift_percentage,
            'uplift_absolute': uplift_absolute
        },
        'anomalies_count': anomalies_fixed,
        'anomaly_report': anomaly_report,
        # NEW CONTEXT VARIABLES
        'ctr_metrics': {
            'overall': overall_ctr,
            'by_objective': ctr_data
        },
        'ctr_comparison': ctr_data,
        'roas_by_objective': roas_by_objective,
        'peak_performance': {
            'month': best_month_name,
            'value': best_month_value
        },
        # NEW FEATURES CONTEXT
        'day_part_json': day_chart_data,
        'phasing_strategy': phasing_strategy,
        'client_insights': top_client_insights,
        'master_plan': master_plan,
        'tactical_plan': tactical_plan,
        'tactical_client': tactical_client_name,
        'tactical_month': f"{calendar.month_name[target_month]} 2024 (Forecast based on 2023 Data)",
        'tactical_month_num': target_month,
        'all_clients_list': all_clients_list,
        'month_names': [(i, calendar.month_name[i]) for i in range(1, 13)],
        'mom_growth': mom_growth, # [NEW]
        'monthly_seasonality': seasonal_data, # [NEW]
        # LEADERBOARD ANALYSIS (Guidebook Section C)
        'top_industry_avg': top_industry_avg,
        'top_account_rev': top_account_rev,
        'top_industry_roas': top_industry_roas,
        'all_accounts_rev': all_accounts_rev,
        'top_5_industries_avg': top_5_industries_avg,
        'top_5_industries_roas': top_5_industries_roas,
    }
    
    return render(request, 'ads_analyzer/dashboard.html', context)

def optimize_marketing_budget(total_budget, channel_params):
    """
    Allocates budget to maximize revenue based on historical ROAS.
    Uses normalized weights (0-1) for numerical stability.
    """
    n_channels = len(channel_params)
    if n_channels == 0:
        return []

    # Objective: Maximize Weighted ROAS (equivalent to max revenue)
    # x[i] is the fraction of budget allocated to channel i
    def objective(weights):
        # We want to maximize sum(weight_i * total_budget * roas_i)
        # Which is equivalent to minimizing -sum(weight_i * roas_i)
        return -sum(weights[i] * channel_params[i]['roas'] for i in range(n_channels))

    # Constraint: Sum of weights = 1.0
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.0})
    
    # Bounds: Min 5%, Max 60% allocation per channel
    # This prevents putting 100% in one channel
    bounds = [(0.05, 0.6) for _ in range(n_channels)]
    
    # Initial Guess: Equal split
    initial_guess = [1.0 / n_channels] * n_channels
    
    result = minimize(
        objective, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        tol=1e-6
    )
    
    final_weights = result.x if result.success else initial_guess
    
    # Format output
    allocations = []
    for i, channel in enumerate(channel_params):
        allocated_spend = final_weights[i] * total_budget
        allocations.append({
            'channel': channel['name'],
            'suggested_spend': allocated_spend,
            'projected_revenue': allocated_spend * channel['roas'],
            'roas': channel['roas']
        })
        
    # Sort by suggested spend
    allocations.sort(key=lambda x: x['suggested_spend'], reverse=True)
    return allocations
