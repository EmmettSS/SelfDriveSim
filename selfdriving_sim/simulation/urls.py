from django.urls import path

from .views import SimulationView

app_name = "simulation"

urlpatterns = [
    path("", SimulationView.as_view(), name="index"),
]
