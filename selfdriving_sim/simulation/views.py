from django.http import JsonResponse
from django.views.generic import TemplateView


def health(request):
    """Liveness probe."""
    return JsonResponse({"status": "ok", "phase": 1})


class SimulationView(TemplateView):
    """Placeholder for the Three.js scene (phase 2)."""

    template_name = "simulation/index.html"
