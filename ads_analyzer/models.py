from django.db import models

class AdPerformance(models.Model):
    # Industry Choices
    INDUSTRY_CHOICES = [
        ('Fashion', 'Fashion'),
        ('Beauty', 'Beauty'),
        ('FMCG', 'FMCG'),
        ('Electronics', 'Electronics'),
        ('Other', 'Other'),
    ]

    # Objective Choices
    OBJECTIVE_CHOICES = [
        ('Traffic', 'Traffic'),
        ('Sales', 'Sales'),
    ]

    created_date = models.DateField(db_index=True, verbose_name="Tanggal Data")
    account_name = models.CharField(max_length=255)
    
    # We allow blank/null for industry if data cleaning doesn't perfectly map it, 
    # but based on prompt we should extract it. 
    # For now, we'll keep it simple as CharField but with choices for validation if needed.
    industry = models.CharField(max_length=50, choices=INDUSTRY_CHOICES, blank=True, null=True)
    
    campaign_objective = models.CharField(max_length=50, choices=OBJECTIVE_CHOICES)
    
    # Metrics
    impressions = models.PositiveIntegerField()
    reach = models.PositiveIntegerField()
    clicks = models.PositiveIntegerField()
    link_clicks = models.PositiveIntegerField()
    
    # Conversion Metrics
    content_views = models.PositiveIntegerField(default=0)
    add_to_cart = models.PositiveIntegerField(default=0)
    purchases = models.PositiveIntegerField(default=0)
    
    # Financial Metrics
    purchase_value = models.DecimalField(max_digits=19, decimal_places=2, default=0.00)
    amount_spent = models.DecimalField(max_digits=19, decimal_places=2, default=0.00)

    class Meta:
        indexes = [
            models.Index(fields=['account_name', 'created_date']),
        ]
        verbose_name = "Kinerja Iklan"
        verbose_name_plural = "Data Kinerja Iklan"

    def __str__(self):
        return f"{self.account_name} - {self.created_date}"
    
    @property
    def roas(self):
        if self.amount_spent > 0:
            return self.purchase_value / self.amount_spent
        return 0
