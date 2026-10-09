"""Root URL configuration."""
from django.contrib import admin
from django.urls import include, path

from simulation.views import health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health, name="health"),
    path("", include("simulation.urls")),
    path("client/", include("client.urls")),
]
