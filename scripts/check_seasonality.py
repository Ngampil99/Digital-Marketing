import os
import sys
import django
from django.db.models import Sum
from django.db.models.functions import ExtractMonth

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_marketer_tool.settings')
django.setup()

from ads_analyzer.models import AdPerformance

def check():
    print("--- Monthly Verification ---")
    monthly = AdPerformance.objects.annotate(
        month=ExtractMonth('created_date')
    ).values('month').annotate(
        spend=Sum('amount_spent'),
        rev=Sum('purchase_value')
    ).order_by('month')
    
    for m in monthly:
        print(f"Month {m['month']}: Spend={m['spend']}, Rev={m['rev']}")

if __name__ == '__main__':
    check()
