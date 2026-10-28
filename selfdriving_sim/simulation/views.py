"""HTTP views for the simulation app."""

import time

from django.http import JsonResponse
from django.views.decorators.http import require_safe
from django.views.generic import TemplateView

from simulation import connect

# Released version reported by the health endpoint.
VERSION = "1.0.0"


class SimulatorView(TemplateView):
    """Serve the Three.js scene page at the site root, with the phone QR panel."""

    template_name = "simulation/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dashboard_url = connect.dashboard_url_for(self.request)
        context["dashboard_url"] = dashboard_url
        context["qr_data_uri"] = connect.qr_svg_data_uri(dashboard_url)
        context["cache_buster"] = int(time.time())
        return context


@require_safe
def health(request):
    """Return a liveness payload for smoke tests and deployment checks."""
    return JsonResponse({"status": "ok", "version": VERSION})
