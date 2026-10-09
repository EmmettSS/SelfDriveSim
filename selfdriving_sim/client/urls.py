"""URL routes for the mobile client app."""

from django.urls import path

from client.views import DashboardView

app_name = "client"

urlpatterns = [
    path("", DashboardView.as_view(), name="dashboard"),
]
