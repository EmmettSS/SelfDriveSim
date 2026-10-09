"""HTTP views for the simulation app."""

from django.http import JsonResponse
from django.views.decorators.http import require_safe

# Development phase reported by the health endpoint.
PHASE = 1


@require_safe
def health(request):
    """Return a liveness payload for smoke tests and deployment checks."""
    return JsonResponse({"status": "ok", "phase": PHASE})
