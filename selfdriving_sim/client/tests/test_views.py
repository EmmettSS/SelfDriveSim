"""Server-side contract for the camera dashboard and its local assets."""

from pathlib import Path

import pytest
from django.conf import settings
from django.contrib.staticfiles import finders
from django.urls import resolve, reverse

from client.views import DashboardView


def test_dashboard_route_uses_the_template_view():
    assert reverse("client:dashboard") == "/dashboard/"
    assert resolve("/dashboard/").func.view_class is DashboardView


def test_dashboard_page_is_rtl_and_has_camera_markup(client):
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "client/dashboard.html" in [template.name for template in response.templates]
    html = response.content.decode()
    assert '<html lang="fa" dir="rtl">' in html
    assert "user-scalable=no" in html
    assert '<video id="camera-feed" autoplay muted playsinline' in html
    assert '<canvas id="process-canvas" width="200" height="66" hidden' in html
    assert 'id="hud-bottom" class="driving-instruments" dir="ltr"' in html


def test_dashboard_has_all_controls_and_truthful_telemetry_labels(client):
    html = client.get("/dashboard/").content.decode()
    for element_id in (
        "hud-top", "hud-bottom", "ws-status", "speed-value", "fps-value",
        "steering-wheel", "steering-value", "indicator-left", "indicator-right",
        "indicator-throttle", "indicator-brake", "throttle-meter", "brake-meter",
        "throttle-value", "brake-value", "btn-start", "btn-stop", "btn-camera",
        "btn-reconnect", "status-message", "packet-count",
    ):
        assert f'id="{element_id}"' in html
    assert "MOCK MODE" in html
    assert "خودروی شبیه‌ساز" in html
    assert "شروع خودران" in html
    assert "توقف" in html


def test_initial_controls_are_disabled(client):
    html = client.get("/dashboard/").content.decode()
    for button in ("btn-start", "btn-stop"):
        tag = html.split(f'<button id="{button}"', 1)[1].split(">", 1)[0]
        assert "disabled" in tag


def test_dashboard_needs_no_remote_cdn(client):
    html = client.get("/dashboard/").content.decode()
    assert 'type="module"' in html
    for asset in ("css/dashboard.css", "js/inference.js", "fonts/Vazirmatn.woff2"):
        assert f"/static/client/{asset}" in html
    assert 'src="https://' not in html
    assert 'href="https://' not in html


@pytest.mark.parametrize("asset", [
    "css/dashboard.css", "js/inference.js", "fonts/Vazirmatn.woff2", "fonts/OFL.txt",
])
def test_dashboard_static_assets_exist(asset):
    path = finders.find(f"client/{asset}")
    assert path is not None
    assert Path(path).stat().st_size > 0


def test_css_is_compiled_tailwind_not_a_browser_cdn_loader():
    css = Path(finders.find("client/css/dashboard.css")).read_text()
    assert "tailwindcss v4" in css
    assert "@apply" not in css
    assert "@source" not in css
    assert "@font-face" in css
    assert "Vazirmatn.woff2" in css


def test_existing_cross_origin_opener_policy_is_preserved(client):
    response = client.get("/dashboard/", secure=True)
    assert settings.SECURE_CROSS_ORIGIN_OPENER_POLICY == "same-origin-allow-popups"
    assert response["Cross-Origin-Opener-Policy"] == "same-origin-allow-popups"
    if not settings.ALLOW_IFRAME_PREVIEW:
        assert response["X-Frame-Options"] == "DENY"


def test_head_dashboard_request(client):
    response = client.head("/dashboard/")
    assert response.status_code == 200
    assert response.content == b""
