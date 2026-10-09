"""ASGI entry point for Daphne.

HTTP requests are served by Django. WebSocket connections are routed to the
Channels consumers listed in ``simulation.routing``.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "selfdriving_sim.settings")

# Initialise Django (settings and app registry) before importing app modules.
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402

from simulation.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        # Rejects WebSocket handshakes whose Origin host is not in ALLOWED_HOSTS.
        "websocket": AllowedHostsOriginValidator(
            URLRouter(websocket_urlpatterns),
        ),
    }
)
