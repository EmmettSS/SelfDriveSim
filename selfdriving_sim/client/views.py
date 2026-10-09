"""HTTP views for the phone-side dashboard."""

from django.views.generic import TemplateView


class DashboardView(TemplateView):
    """Serve the camera dashboard; inference and control stay in the browser."""

    template_name = "client/dashboard.html"
