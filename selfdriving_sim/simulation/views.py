"""HTTP views for the simulation app."""

from django.http import JsonResponse
from django.views.decorators.http import require_safe
from django.views.generic import TemplateView

# Development phase reported by the health endpoint.
PHASE = 3


class SimulatorView(TemplateView):
    """Serve the Three.js scene page at the site root."""

    template_name = "simulation/index.html"


@require_safe
def health(request):
    """Return a liveness payload for smoke tests and deployment checks."""
    return JsonResponse({"status": "ok", "phase": PHASE})
