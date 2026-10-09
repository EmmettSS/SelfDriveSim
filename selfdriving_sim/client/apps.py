"""App configuration for the mobile client app."""

from django.apps import AppConfig


class ClientConfig(AppConfig):
    """The phone-side page that drives the simulator."""

    name = "client"
    verbose_name = "Mobile client"
