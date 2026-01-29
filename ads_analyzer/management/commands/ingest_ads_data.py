import csv
import os
from datetime import datetime
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction
from ads_analyzer.models import AdPerformance

class Command(BaseCommand):
    help = 'Ingest ads data from CSV (Safe & Idempotent)'

    def handle(self, *args, **options):
        # Path to the CSV file - relative to project root
        csv_path = os.path.join(settings.BASE_DIR, 'Data Ads - Kompetisi Data Analyts by inSight Data Batch 01.csv')
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'CSV file not found at {csv_path}'))
            return

        self.stdout.write(f'Reading CSV from {csv_path}...')
        
        # [CRITICAL UPDATE] Wipe existing data to prevent duplicates/inflation
        self.stdout.write('Wiping existing data to ensure data integrity...')
        AdPerformance.objects.all().delete()
        
        created_count = 0
        updated_count = 0
        anomalies_fixed = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            with transaction.atomic():
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
                    purchases = int(float(row['purchase']))
                    purchase_value = Decimal(row['purchase_value'])
                    amount_spent = Decimal(row['amount_spent'])
                    
                    # === ANOMALY CLEANING AND LOGIC SANITIZATION ===
                    
                    # 1. Reach vs Impressions Logic (Physics: One cannot see what is not shown)
                    if reach > impressions:
                        reach = impressions
                        anomalies_fixed += 1
                        
                    # 2. Clicks vs Impressions (Physics: One cannot click what is not shown)
                    if clicks > impressions:
                        clicks = impressions
                        # If clicks capped, link_clicks must be checked too
                        if link_clicks > clicks:
                            link_clicks = clicks
                            
                    # 3. Link Clicks vs Clicks (Subset logic)
                    if link_clicks > clicks:
                        link_clicks = clicks
                        
                    # 4. Ghost Revenue (Money In, No Transaction Count)
                    if purchase_value > 0 and purchases == 0:
                        purchases = 1 # Assume at least 1 transaction occurred
                    
                    # Extract Industry
                    industry = None
                    if ' - ' in account_name:
                        parts = account_name.split(' - ')
                        if len(parts) > 1:
                            industry = parts[1]
                            
                    # IDEMPOTENT UPDATE OR CREATE
                    obj, created = AdPerformance.objects.update_or_create(
                        created_date=created_date,
                        account_name=account_name,
                        campaign_objective=campaign_objective,
                        defaults={
                            'industry': industry,
                            'impressions': impressions,
                            'reach': reach,
                            'clicks': clicks,
                            'link_clicks': link_clicks,
                            'content_views': content_views,
                            'add_to_cart': add_to_cart,
                            'purchases': purchases,
                            'purchase_value': purchase_value,
                            'amount_spent': amount_spent,
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1

        self.stdout.write(self.style.SUCCESS(f'Process Complete. Created: {created_count}, Updated: {updated_count}, Anomalies Fixed: {anomalies_fixed}'))
