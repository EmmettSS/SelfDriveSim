"""Tests for the HTTP views of the simulation app."""


def test_health_endpoint_reports_phase_one(client):
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": 1}
