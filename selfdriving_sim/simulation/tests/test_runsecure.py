"""Tests for the runsecure command's TLS endpoint plumbing.

Twisted's endpoint tokenizer treats ``\\`` as an escape character, so the
paths have to leave the builder POSIX-style with an escaped drive colon —
these tests pin both the string rules and a real ``serverFromString``
round-trip.
"""

from pathlib import Path

from simulation import selfsigned
from simulation.management.commands.runsecure import tls_endpoint_string


def test_backslashes_become_forward_slashes_and_colons_are_escaped(tmp_path):
    certfile = tmp_path / "certs" / "selfsigned.pem"
    keyfile = tmp_path / "certs" / "selfsigned.key"

    endpoint = tls_endpoint_string(8443, "0.0.0.0", certfile, keyfile)

    assert endpoint.startswith("ssl:port=8443:interface=0.0.0.0")
    assert "\\" not in endpoint.replace(r"\:", ""), "raw backslashes must not leak"
    assert "certs/selfsigned.pem" in endpoint
    assert "certs/selfsigned.key" in endpoint
    # The drive letter colon of the resolved Windows path, once per file.
    assert endpoint.count(r"\:") == 2


def test_twisted_parses_the_endpoint_with_a_real_pair(tmp_path):
    """Round-trip against Twisted's own parser, with loadable cert and key."""

    from twisted.internet import reactor
    from twisted.internet.endpoints import serverFromString

    certfile, keyfile = selfsigned.certificate_paths(tmp_path)
    selfsigned.generate_self_signed(certfile, keyfile, key_size=1024)

    endpoint = serverFromString(reactor, tls_endpoint_string(8443, "0.0.0.0", certfile, keyfile))

    assert endpoint._port == 8443
    assert type(endpoint._sslContextFactory).__name__ == "OpenSSLCertificateOptions"


def test_pick_free_tls_port_steps_past_an_occupied_port():
    import socket

    command = _make_command()

    # Occupy a port: the not-quite-standard Hyper-V/docker situation.
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    occupied = blocker.getsockname()[1]

    try:
        chosen = command._pick_free_tls_port(occupied)
    finally:
        blocker.close()

    assert chosen == occupied + 1
    printed = command.stdout.getvalue()
    assert "TLS port" in printed


def _make_command():
    from io import StringIO

    from simulation.management.commands.runsecure import Command

    command = Command()
    command.stdout = StringIO()
    return command
