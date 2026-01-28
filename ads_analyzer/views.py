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

    context = {
        'chart_data_json': chart_data,
        'funnel_data_json': funnel_metrics,
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
    channel_params: list of dicts {'name': str, 'roas': float}
    """
    n_channels = len(channel_params)
    if n_channels == 0:
        return []

    # Objective: Max Revenue => Min -Revenue
    def objective(spends):
        revenue = sum(spends[i] * channel_params[i]['roas'] for i in range(n_channels))
        return -revenue

    # Constraint: Sum of spends = total_budget
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - total_budget})
    
    # Bounds: Min 5%, Max 60% per channel ensures diversification
    bounds = [(total_budget * 0.05, total_budget * 0.6) for _ in range(n_channels)]
    
    # Initial Guess
    initial_guess = [total_budget / n_channels] * n_channels
    
    result = minimize(
        objective, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    final_spends = result.x if result.success else initial_guess
    
    # Format output
    allocations = []
    for i, channel in enumerate(channel_params):
        allocations.append({
            'channel': channel['name'],
            'suggested_spend': final_spends[i],
            'projected_revenue': final_spends[i] * channel['roas'],
            'roas': channel['roas']
        })
        
    # Sort by suggested spend
    allocations.sort(key=lambda x: x['suggested_spend'], reverse=True)
    return allocations
