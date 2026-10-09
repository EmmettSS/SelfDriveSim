"""WSGI entry point.

Daphne uses the ASGI entry point (``selfdriving_sim.asgi``). This module is
kept for WSGI-only hosting environments.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "selfdriving_sim.settings")

application = get_wsgi_application()
