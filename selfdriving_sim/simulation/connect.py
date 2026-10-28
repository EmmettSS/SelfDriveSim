"""How the phone reaches the dashboard: LAN address, URL and QR code.

The monitor page shows a QR code the phone scans, so connecting the phone is
one camera shot instead of typing an IP address. Two facts are needed for it:

* which LAN address of the PC the phone can reach (``detect_lan_ips``);
* which port and scheme to use (from the environment of the running server:
  ``runsecure`` sets ``SIMULATION_TLS_PORT`` when it serves HTTPS).

Everything is overridable with ``SIMULATION_DASHBOARD_URL``, for hosts with
several network adapters (VPNs, virtual machines) where automatic detection
picks the wrong one.
"""

import ipaddress
import os
import socket

# Full dashboard URL override, e.g. SIMULATION_DASHBOARD_URL=https://10.1.2.3:8443/dashboard/
DASHBOARD_URL_ENV = "SIMULATION_DASHBOARD_URL"
# Set (to the port number) by the runsecure command while it serves HTTPS.
TLS_PORT_ENV = "SIMULATION_TLS_PORT"

# Fallback when nothing better is known; still correct for PC-side testing.
LOOPBACK_IPV4 = "127.0.0.1"

# Gateway-shaped last octet: the host-side adapter of virtual NATs (VMware,
# Hyper-V, WSL) owns the ".1" of its virtual subnet almost by definition.
_LAST_OCTET_ONE = ".1"
_GATEWAY_PENALTY = 0.5

# UDP "connect" targets: the connect() on an unconnected UDP socket only picks
# the default-route interface, no packet is sent, no internet access needed.
_PROBE_TARGETS = (("8.8.8.8", 80), ("1.1.1.1", 53), ("9.9.9.9", 53))

_cached_lan_ips = None


def _address_bucket(address: str) -> int:
    """Rank one candidate address; lower bucket = more likely the phone's Wi-Fi.

    Home Wi-Fi networks are overwhelmingly 192.168.x.x; 10.x comes second
    (corporate LANs and hotspots); 172.16-31.x.x ranks last among private
    ranges because that is also where virtual adapters (WSL, Docker) live. A
    public address belongs to a real adapter but is unusual in this setup.
    """
    try:
        parsed = ipaddress.IPv4Address(address)
    except ValueError:
        return 9  # keep unknowns at the tail
    if parsed.is_loopback:
        return 8
    if parsed.is_link_local:
        return 7
    if address.startswith("192.168."):
        return 0
    if address.startswith("10."):
        return 1
    if address.startswith("172."):
        try:
            second_octet = int(address.split(".")[1])
        except (IndexError, ValueError):
            return 4
        if 16 <= second_octet <= 31:
            return 2  # matches virtual adapters (WSL, Docker) too — kept last
        return 4
    if parsed.is_private:
        return 3
    return 4


def order_lan_candidates(found: list[str]) -> list[str]:
    """Deduplicate and rank raw candidate addresses, best first.

    Any address that does not parse as IPv4 is dropped; nothing gracefully
    reduces to the loopback fallback handled in detect_lan_ips.

    Within the winning private bucket, addresses ending in ``.1`` are demoted
    below their non-``.1`` siblings: the host-side adapter of a virtual NAT
    (VMware, Hyper-V, WSL) virtually always owns the ``.1`` of its subnet,
    while the real Wi-Fi address the phone reaches usually does not. Without
    a non-``.1`` sibling the ``.1`` address keeps its rank.
    """
    ranked = {address: _address_bucket(address) for address in found}
    demote_gateways = any(
        bucket <= 1 and not address.endswith(_LAST_OCTET_ONE)
        for address, bucket in ranked.items()
    )
    ordered: list[str] = []
    for address in found:
        if address in ordered or ranked[address] >= 7:
            continue
        ordered.append(address)
    ordered.sort(
        key=lambda address: ranked[address] + (
            _GATEWAY_PENALTY
            if demote_gateways and address.endswith(_LAST_OCTET_ONE)
            else 0
        )
    )
    return ordered


def detect_lan_ips() -> list[str]:
    """IPv4 addresses of this machine, best first; loopback last as a fallback.

    The routing probe answers first because a PC can have many interfaces
    (Hyper-V, VPN, Bluetooth PAN) that all advertise an IPv4 address, and the
    default route is not always the Wi-Fi card a phone sits behind; the
    192.168/10/172 ranking in ``order_lan_candidates`` does the real choice.
    """
    found: list[str] = []

    for target in _PROBE_TARGETS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(target)
                address = sock.getsockname()[0]
        except OSError:
            continue
        if address:
            found.append(address)
            break

    try:
        _, _, resolver_addresses = socket.gethostbyname_ex(socket.gethostname())
        found.extend(resolver_addresses)
    except OSError:
        pass

    ordered = order_lan_candidates(found)
    if not ordered:
        ordered.append(LOOPBACK_IPV4)
    return ordered


def detect_lan_ip() -> str:
    """The one LAN address the phone is most likely to reach."""
    global _cached_lan_ips
    if _cached_lan_ips is None:
        _cached_lan_ips = detect_lan_ips()
    return _cached_lan_ips[0]


def dashboard_path() -> str:
    """URL path of the dashboard (checked at request time, not import time)."""
    from django.urls import reverse

    return reverse("client:dashboard")


def dashboard_url_for(request) -> str:
    """The dashboard URL for the QR code on the monitor page.

    ``SIMULATION_DASHBOARD_URL`` wins. Otherwise, while the server serves a
    TLS port (``runsecure``), the URL points there over HTTPS — that is the
    mode in which the phone gets camera access without any browser flags.
    Without it, the current request's scheme and port are mirrored onto the
    LAN address; a plain HTTP runserver therefore advertises exactly what it
    actually serves, and the panel can explain that this needs the browser
    flag on Android.
    """
    override = os.environ.get(DASHBOARD_URL_ENV)
    if override:
        return override

    path = dashboard_path()
    lan_ip = detect_lan_ip()

    tls_port = os.environ.get(TLS_PORT_ENV)
    if tls_port:
        return f"https://{lan_ip}:{tls_port}{path}"

    scheme = "https" if request.is_secure() else "http"
    host = request.get_host()
    # get_host may carry the port; render only non-default ports to keep the QR edge length small.
    if ":" in host and not host.endswith("]"):
        hostname, _, port = host.rpartition(":")
        if (scheme, port) not in (("http", "80"), ("https", "443")):
            return f"{scheme}://{lan_ip}:{port}{path}"
    return f"{scheme}://{lan_ip}{path}"


def qr_svg_data_uri(payload: str) -> str | None:
    """An inline SVG QR code, or ``None`` when segno is not installed.

    Standard black-on-white so phone cameras focus and decode reliably; the
    panel supplies the styling around the image, never behind it.
    """
    try:
        import segno
    except ImportError:
        return None
    return segno.make(payload, error="m").svg_data_uri(scale=8, border=2)
