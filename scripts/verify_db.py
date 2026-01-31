import os
import sys
import django
from django.db.models import Sum

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_marketer_tool.settings')
django.setup()

from ads_analyzer.models import AdPerformance

def verify():
    print("--- Database Verification (All Columns) ---")
    
    sums = AdPerformance.objects.aggregate(
        rev=Sum('purchase_value'),
        spend=Sum('amount_spent'),
        imps=Sum('impressions'),
        reach=Sum('reach'),
        clicks=Sum('clicks'),
        vlc=Sum('link_clicks'),
        cv=Sum('content_views'),
        atc=Sum('add_to_cart'),
        pur=Sum('purchases')
    )
    
    for k, v in sums.items():
        print(f"Sum {k}: {v}")
    
    print("\n--- First 5 Rows ---")
    rows = AdPerformance.objects.all()[:5]
    for r in rows:
        print(f"{r.created_date} | {r.account_name} | {r.campaign_objective} | Spend: {r.amount_spent} | Rev: {r.purchase_value}")

if __name__ == '__main__':
    verify()
