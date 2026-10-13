"""HTTP views for the simulation app."""

from django.http import JsonResponse
from django.views.decorators.http import require_safe
from django.views.generic import TemplateView

# Released version reported by the health endpoint.
VERSION = "1.0.0"


class SimulatorView(TemplateView):
    """Serve the Three.js scene page at the site root."""

    template_name = "simulation/index.html"


@require_safe
def health(request):
    """Return a liveness payload for smoke tests and deployment checks."""
    return JsonResponse({"status": "ok", "version": VERSION})
