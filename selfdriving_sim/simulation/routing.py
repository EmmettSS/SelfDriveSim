"""WebSocket URL routes for the simulation app."""

from django.urls import path

from simulation.consumers import ControlConsumer, RenderConsumer

websocket_urlpatterns = [
    path("ws/control/", ControlConsumer.as_asgi(), name="ws-control"),
    path("ws/render/", RenderConsumer.as_asgi(), name="ws-render"),
]
