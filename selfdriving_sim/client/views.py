"""HTTP views for the phone-side dashboard."""

import time

from django.views.generic import TemplateView


class DashboardView(TemplateView):
    """Serve the camera dashboard; inference and control stay in the browser."""

    template_name = "client/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["cache_buster"] = int(time.time())
        return context
