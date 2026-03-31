from django.urls import path
from .views import FinancialSimulationView

urlpatterns = [
    path("simulate/", FinancialSimulationView.as_view(), name="simulate"),
    path("simulate", FinancialSimulationView.as_view(), name="simulate_no_slash"),
    path("v1/simulations/", FinancialSimulationView.as_view(), name="simulations_v1"),
    path("v1/simulations", FinancialSimulationView.as_view(), name="simulations_v1_no_slash"),
]
