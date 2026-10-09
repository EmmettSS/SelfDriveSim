"""Tests for the HTTP views of the simulation app."""

from django.contrib.staticfiles import finders
from django.urls import reverse

from simulation.views import PHASE


def test_health_endpoint_reports_the_current_phase(client):
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": PHASE}
    assert PHASE == 2


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


def test_scene_page_loads_the_module_scripts(client):
    content = client.get("/").content.decode()

    assert "/static/simulation/js/scene.js" in content
    assert 'type="module"' in content


def test_every_static_asset_the_page_refers_to_exists():
    assert finders.find("simulation/js/scene.js") is not None
    assert finders.find("simulation/js/vendor/three.module.min.js") is not None
