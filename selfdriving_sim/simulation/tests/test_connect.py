"""Tests for the LAN address, dashboard URL and QR code helpers."""

import ipaddress

import pytest
from django.test import RequestFactory

from simulation import connect


class FakeRequest:
    """Just the two HttpRequest facts dashboard_url_for reads."""

    def __init__(self, secure: bool, host: str):
        self._secure = secure
        self._host = host

    def is_secure(self) -> bool:
        return self._secure

    def get_host(self) -> str:
        return self._host


@pytest.fixture(autouse=True)
def clean_tls_environment(monkeypatch):
    monkeypatch.delenv(connect.TLS_PORT_ENV, raising=False)
    monkeypatch.delenv(connect.DASHBOARD_URL_ENV, raising=False)


def test_lan_ip_detection_returns_a_valid_ipv4_list():
    addresses = connect.detect_lan_ips()

    assert addresses, "at least the loopback fallback must come out"
    assert all(ipaddress.IPv4Address(address) for address in addresses)


def test_ordering_prefers_home_wifi_then_private_vpns_then_public():
    ordered = connect.order_lan_candidates(
        [
            "8.8.8.8",        # public
            "172.23.144.1",   # WSL / Docker virtual adapter
            "10.1.2.3",       # corporate LAN or hotspot
            "192.168.1.108",  # the phone's Wi-Fi
            "127.0.0.1",      # loopback fallback
        ]
    )

    assert ordered == ["192.168.1.108", "10.1.2.3", "172.23.144.1", "8.8.8.8"]


def test_ordering_drops_unparseable_and_link_local_addresses():
    ordered = connect.order_lan_candidates(["not-an-ip", "169.254.10.9", "192.168.1.2", "127.0.0.1"])

    assert ordered == ["192.168.1.2"]


def test_ordering_demotes_virtual_nat_gateways_to_non_leading_addresses():
    """This machine's real case: Wi-Fi .108 vs two VMware NAT host adapters.

    VMware/Hyper-V host adapters own the ``.1`` of their virtual subnet, so
    when a non-``.1`` sibling exists in the same bucket the ``.1`` addresses
    lose, even the one the default-route probe returned.
    """
    ordered = connect.order_lan_candidates(
        [
            "192.168.244.1",  # probe answer / VMware NAT host adapter
            "192.168.235.1",  # VMware NAT host adapter
            "192.168.1.108",  # the Wi-Fi address the phone reaches
            "172.18.0.1",     # WSL
        ]
    )

    assert ordered[0] == "192.168.1.108"


def test_ordering_keeps_a_lone_gateway_address_unchanged():
    """A PC that genuinely sits at ``.1`` is not demoted into a worse bucket."""
    ordered = connect.order_lan_candidates(["192.168.1.1", "10.9.9.9"])

    assert ordered == ["192.168.1.1", "10.9.9.9"]


def test_detect_lan_ip_prefers_the_private_addresses():
    ips = connect.detect_lan_ips()

    # Sorted so the output is deterministic for the certificate SANs too.
    private = [a for a in ips if ipaddress.IPv4Address(a).is_private]
    non_private = [a for a in ips if not ipaddress.IPv4Address(a).is_private]
    if private and non_private:
        assert ips.index(private[0]) < ips.index(non_private[0])


def test_dashboard_url_uses_the_request_scheme_and_port_by_default():
    request = FakeRequest(secure=False, host="localhost:8000")
    ip = connect.detect_lan_ip()

    assert connect.dashboard_url_for(request) == f"http://{ip}:8000/dashboard/"


def test_dashboard_url_on_https_keeps_the_scheme_when_no_tls_env_is_set():
    request = FakeRequest(secure=True, host="localhost:8443")
    ip = connect.detect_lan_ip()

    assert connect.dashboard_url_for(request) == f"https://{ip}:8443/dashboard/"


def test_default_http_and_https_ports_are_dropped():
    request = FakeRequest(secure=True, host="localhost:443")
    ip = connect.detect_lan_ip()

    assert connect.dashboard_url_for(request) == f"https://{ip}/dashboard/"


def test_tls_environment_switches_the_url_to_https(monkeypatch):
    monkeypatch.setenv("SIMULATION_TLS_PORT", "9443")
    request = FakeRequest(secure=False, host="localhost:8000")
    ip = connect.detect_lan_ip()

    assert connect.dashboard_url_for(request) == f"https://{ip}:9443/dashboard/"


def test_dashboard_url_override_wins_over_everything(monkeypatch):
    monkeypatch.setenv("SIMULATION_TLS_PORT", "9443")
    monkeypatch.setenv(
        connect.DASHBOARD_URL_ENV, "https://explicit.host.example/dashboard/"
    )

    assert (
        connect.dashboard_url_for(FakeRequest(secure=False, host="localhost:8000"))
        == "https://explicit.host.example/dashboard/"
    )


def test_qr_svg_is_an_inline_image_of_the_input():
    from urllib.parse import unquote

    data_uri = connect.qr_svg_data_uri("https://192.168.1.42:8443/dashboard/")

    assert data_uri is not None
    assert data_uri.startswith("data:image/svg+xml")
    decoded = unquote(data_uri)
    assert "<svg" in decoded
    assert "height=" in decoded  # segno sets intrinsic SVG size, not viewBox
