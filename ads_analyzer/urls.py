from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='home'), # Add root path
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('report/', views.strategic_report_view, name='strategic_report'),
]
