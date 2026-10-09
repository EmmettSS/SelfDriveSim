"""URL routes for the simulation app."""

from django.urls import path

from simulation.views import SimulatorView

app_name = "simulation"

urlpatterns = [
    path("", SimulatorView.as_view(), name="index"),
]
