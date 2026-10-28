"""Tests for the HTTP views of the simulation app."""

from django.contrib.staticfiles import finders
from django.urls import reverse

from simulation.views import VERSION


def test_health_endpoint_reports_the_released_version(client):
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": VERSION}
    assert VERSION == "1.0.0"


def test_root_path_is_named_index():
    assert reverse("simulation:index") == "/"


def test_root_serves_the_scene_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "simulation/index.html" in [t.name for t in response.templates]
    content = response.content.decode()
    assert 'id="scene-canvas"' in content
    assert 'id="hud-top"' in content


def test_scene_page_has_the_hud_elements(client):
    content = client.get("/").content.decode()

    for element_id in ("ws-status", "speed-value", "fps-value"):
        assert f'id="{element_id}"' in content


def test_scene_page_carries_the_screen_lock(client):
    """The fullscreen lock button ships in the legend, wired by inline JS."""
    content = client.get("/").content.decode()

    assert 'id="screen-lock"' in content
    assert "fullscreenchange" in content  # label sync + backbuffer refit


def test_scene_page_loads_the_module_scripts(client):
    content = client.get("/").content.decode()

    assert "/static/simulation/js/scene.js" in content
    assert 'type="module"' in content


def test_scene_page_carries_the_phone_qr_panel(client):
    """The QR panel ships even when the QR library is missing: URL text only."""

    content = client.get("/").content.decode()

    assert 'id="qr-panel" hidden' in content
    assert 'id="qr-toggle"' in content
    assert 'id="qr-url"' in content
    assert "/dashboard/" in content  # the URL the phone has to visit


def test_scene_page_honours_the_dashboard_url_override(client, monkeypatch):
    """SIMULATION_DASHBOARD_URL wins over the automatic LAN detection."""

    monkeypatch.setenv("SIMULATION_DASHBOARD_URL", "https://10.9.8.7:9443/dashboard/")

    content = client.get("/").content.decode()

    assert "https://10.9.8.7:9443/dashboard/" in content


def test_every_static_asset_the_page_refers_to_exists():
    assert finders.find("simulation/js/scene.js") is not None
    assert finders.find("simulation/js/vendor/three.module.min.js") is not None


def test_scene_page_carries_the_team_credit(client):
    content = client.get("/").content.decode()

    assert 'id="hud-legend"' in content
    assert "© Design &amp; Develop by Emmett Software Development Group" in content
