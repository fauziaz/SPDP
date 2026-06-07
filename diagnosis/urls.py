from django.urls import path
from . import views

urlpatterns = [
    path('', views.diagnosis_view, name='diagnosis'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('rules/', views.rules_view, name='rules'),
    path('evaluation/', views.evaluation_view, name='evaluation'),
]
