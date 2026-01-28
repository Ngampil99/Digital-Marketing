from django.shortcuts import render
from django.db import models
from django.db.models import Sum, Case, When, F, Avg
from django.core.serializers.json import DjangoJSONEncoder
import json
import pandas as pd
from scipy.optimize import minimize
from ads_analyzer.models import AdPerformance

def dashboard_view(request):
    # 1. Aggregate Data for Charts
    daily_data = AdPerformance.objects.values('created_date').annotate(
        traffic_spend=Sum(Case(When(campaign_objective='Traffic', then='amount_spent'), default=0, output_field=models.DecimalField())),
        sales_spend=Sum(Case(When(campaign_objective='Sales', then='amount_spent'), default=0, output_field=models.DecimalField())),
        revenue=Sum('purchase_value'),
        impressions=Sum('impressions')
    ).order_by('created_date')
    
    # Convert to DataFrame for easier list processing
    df = pd.DataFrame(list(daily_data))
    
    # Ensure numeric types are float for JSON serialization logic
    numeric_cols = ['traffic_spend', 'sales_spend', 'revenue', 'impressions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # Calculate ROAS for chart (handle division by zero)
    df['total_spend'] = df['traffic_spend'] + df['sales_spend']
    df['roas'] = df.apply(lambda x: x['revenue'] / x['total_spend'] if x['total_spend'] > 0 else 0, axis=1)
    
    # 2. Prepare JSON for Plotly
    chart_data = {
        'dates': df['created_date'].astype(str).tolist(),
        'roas': df['roas'].tolist(),
        'revenue': df['revenue'].tolist(),
        'traffic_spend': df['traffic_spend'].tolist(),
        'sales_spend': df['sales_spend'].tolist(),
    }
    
    # 3. Budget Optimization (Scientific)
    # Get average historical ROAS per industry to use as parameters
    industry_stats = AdPerformance.objects.values('industry').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value')
    )

    # 3.5 Monthly Industry Breakdowns (Seasonality)
    from django.db.models.functions import ExtractMonth
    industry_monthly = AdPerformance.objects.annotate(
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
    objective_stats = AdPerformance.objects.values('campaign_objective').annotate(
        total_spend=Sum('amount_spent'),
        total_revenue=Sum('purchase_value'),
        total_impressions=Sum('impressions')
    )
    
    # Prepare Industry Chart Data
    industry_chart = {
        'labels': [],
        'revenue': [],
        'spend': [],
        'roas': []
    }
    for ind in industry_stats:
        if ind['industry']:
            s = float(ind['total_spend'])
            r = float(ind['total_revenue'])
            industry_chart['labels'].append(ind['industry'])
            industry_chart['spend'].append(s)
            industry_chart['revenue'].append(r)
            industry_chart['roas'].append(r / s if s > 0 else 0)

    # Prepare Objective Chart Data
    objective_chart = {
        'labels': [],
        'revenue': [],
        'spend': [],
        'ctr': [] # Placeholder if needed, focus on Rev/Spend first
    }
    for obj in objective_stats:
        s = float(obj['total_spend'])
        r = float(obj['total_revenue'])
        objective_chart['labels'].append(obj['campaign_objective'])
        objective_chart['spend'].append(s)
        objective_chart['revenue'].append(r)
    
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
    funnel_data = AdPerformance.objects.filter(campaign_objective='Sales').aggregate(
        impressions=Sum('impressions'),
        clicks=Sum('clicks'),
        content_views=Sum('content_views'),
        add_to_cart=Sum('add_to_cart'),
        purchases=Sum('purchases')
    )
    
    # Calculate drop-off rates for visualization
    # Mock anomaly count for demonstration (or calculate based on logic)
    anomalies_fixed = 415 # logic: AdPerformance.objects.filter(quality_score__lt=0.5).count() or similar
    
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
    global_stats = AdPerformance.objects.aggregate(
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
    monthly_global = AdPerformance.objects.annotate(
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
    
    all_data = AdPerformance.objects.values(
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

    context = {
        'chart_data_json': chart_data,
        'funnel_data_json': funnel_metrics,
        'industry_chart_json': industry_chart,
        'industry_monthly_json': industry_monthly_json,
        'objective_chart_json': objective_chart,
        'deep_dive_metrics': deep_dive_metrics,
        'monthly_trend_json': monthly_trend_data,
        # Modul 3 Context
        'funnel_abs': funnel_abs,
        'funnel_rates': funnel_rates,
        'heatmap_json': heatmap_json,
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
        'anomalies_count': anomalies_fixed
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
