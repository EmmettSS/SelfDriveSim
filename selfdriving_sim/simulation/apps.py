"""App configuration for the simulation app."""

from django.apps import AppConfig


class SimulationConfig(AppConfig):
    """The Three.js driving scene and its WebSocket render endpoint."""

    name = "simulation"
    verbose_name = "Simulation"
