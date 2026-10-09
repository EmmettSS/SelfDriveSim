"""Root URL configuration.

- ``/admin/``: Django admin site.
- ``/health/``: liveness probe that reports the current development phase.
- The simulation app is included at the root. Its pages arrive in phase 2.
- The mobile client app is mounted under ``/client/``. Its pages arrive in phase 3.
"""

from django.contrib import admin
from django.urls import include, path

from simulation.views import health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health, name="health"),
    path("", include("simulation.urls")),
    path("client/", include("client.urls")),
]
