import csv
import os
from datetime import datetime
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.conf import settings
from ads_analyzer.models import AdPerformance

class Command(BaseCommand):
    help = 'Ingest ads data from CSV'

    def handle(self, *args, **options):
        # Path to the CSV file - assuming it's in the project root based on file list
        # "D:/Lomba/Data Analisis/Data Marketer Tool/Data Ads - Kompetisi Data Analyts by inSight Data Batch 01.csv"
        # Since manage.py is in "D:/Lomba/Data Analisis/Data Marketer Tool", we can find it relative to that or absolute.
        # Let's use the known absolute path or relative from the project base.
        
        # The project base (manage.py location) is d:/Lomba/Data Analisis/Data Marketer Tool
        csv_path = 'Data Ads - Kompetisi Data Analyts by inSight Data Batch 01.csv'
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at {csv_path}'))
            return

        self.stdout.write(f'Reading CSV from {csv_path}...')
        
        ad_performances = []
        anomalies_fixed = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse basic fields
                created_date = datetime.strptime(row['created_date'], '%Y-%m-%d').date()
                account_name = row['account_name']
                campaign_objective = row['campaign_objective']
                
                # Metrics - Handle potential floats in CSV
                impressions = int(float(row['impressions']))
                reach = int(float(row['reach']))
                clicks = int(float(row['clicks']))
                link_clicks = int(float(row['link_clicks']))
                content_views = int(float(row['content_views']))
                add_to_cart = int(float(row['add_to_cart']))
                purchases = int(float(row['purchase'])) # CSV header says 'purchase', model says 'purchases'
                purchase_value = Decimal(row['purchase_value'])
                amount_spent = Decimal(row['amount_spent'])
                
                # 2.2 Anomaly Remediation: Reach > Impressions
                if reach > impressions:
                    reach = impressions
                    anomalies_fixed += 1
                
                # Extract Industry from account_name (e.g., "Client A - Fashion")
                industry = None
                if ' - ' in account_name:
                    parts = account_name.split(' - ')
                    if len(parts) > 1:
                        industry = parts[1]
                
                ad_performances.append(AdPerformance(
                    created_date=created_date,
                    account_name=account_name,
                    industry=industry,
                    campaign_objective=campaign_objective,
                    impressions=impressions,
                    reach=reach,
                    clicks=clicks,
                    link_clicks=link_clicks,
                    content_views=content_views,
                    add_to_cart=add_to_cart,
                    purchases=purchases,
                    purchase_value=purchase_value,
                    amount_spent=amount_spent
                ))
        
        from django.db import transaction
        # Bulk create proved problematic, switching to atomic transaction loop
        self.stdout.write(f'Ingesting {len(ad_performances)} records...')
        
        with transaction.atomic():
             for ad in ad_performances:
                 ad.save()
        
        self.stdout.write(self.style.SUCCESS(f'Successfully ingested {len(ad_performances)} records.'))
        self.stdout.write(self.style.SUCCESS(f'Fixed {anomalies_fixed} Reach > Impressions anomalies.'))
