"""Root URL configuration.

- ``/admin/``: Django admin site.
- ``/health/``: liveness probe that reports the current development phase.
- The Three.js simulation is included at the root.
- The mobile camera dashboard is mounted under ``/dashboard/``.
"""

from django.contrib import admin
from django.urls import include, path

from simulation.views import health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health, name="health"),
    path("", include("simulation.urls")),
    path("dashboard/", include("client.urls")),
]
