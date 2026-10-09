from django.urls import path

from .views import ClientView

app_name = "client"

urlpatterns = [
    path("", ClientView.as_view(), name="index"),
]
