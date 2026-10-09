from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path("ws/control/", consumers.ControlConsumer.as_asgi()),
    path("ws/render/", consumers.RenderConsumer.as_asgi()),
]
